import torch
from math import ceil
from typing import Generator, Iterable, Optional
from torch import Generator as TCGenerator, float64
import torch.nn.functional as F
import gzip


class Guesser:
    def __init__(
        self, model, params, data, lower_probability_threshold, output_file, device
    ):
        self.model = model
        self.data = data
        self.max_len = self.data.max_length
        self.params = params
        self.lower_probability_threshold = lower_probability_threshold
        self.chunk_size_guesser = self.params["eval"]["chunk_size_guesser"]
        self.n_generated_passwords = 0
        self.generated_passwords = []
        self.PASSWORD_END = "\n"
        self.pwd_end_idx = self.data.tokenizer.char_indices[self.PASSWORD_END]
        self.output_file = output_file
        self.device = device

    def generate(self, x_data: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            output = self.model(x_data)
            output = F.softmax(output, dim=1)
            output = output.to(dtype=torch.float64)
        return output

    def encode_passwords(self, astring_list: list[str]) -> torch.Tensor:
        # --- Inside your encode_passwords function ---

        max_len = self.max_len
        x_data_tensors = []  # A list to hold our tensors

        for password in astring_list:
            # 1. Python part (unavoidable):
            #    Map chars to indices, truncating to max_len
            indices = [
                self.data.tokenizer.char_indices.get(char, 0)
                for char in password[:max_len]
            ]

            # 2. Convert to tensor (still on CPU is fine)
            t = torch.tensor(indices, dtype=torch.long)

            # 3. Vectorized padding (replaces your 'while' loop)
            padding_len = max_len - t.shape[0]
            if padding_len > 0:
                # F.pad takes (left_pad, right_pad)
                t = F.pad(t, (0, padding_len), "constant", 0)

            x_data_tensors.append(t)

        # 4. Create the final batch tensor in one operation
        #    This is much faster than appending lists and then converting.
        x_data = torch.stack(x_data_tensors, dim=0).to(self.device)

        # 5. Now, continue with your one-hot encoding
        x_data = F.one_hot(x_data, self.data.tokenizer.vocab_size).to(torch.float32)
        return x_data

    def relevel_prediction(self, preds: torch.Tensor, astring: tuple[str] | str):
        """
        Destructive function which relevels preds
        """
        if isinstance(astring, tuple):
            astring_joined_len = sum(map(len, astring))
        else:
            astring_joined_len = 0

        if not self.pwd_is_valid(astring):
            preds[self.data.tokenizer.char_indices[self.PASSWORD_END]] = 0
        elif len(astring) == self.max_len or (
            isinstance(astring, tuple) and astring_joined_len == self.max_len
        ):
            multi = torch.zeros(len(preds), dtype=float64, device=self.device)
            multi[self.pwd_end_idx] = 1
            preds[self.pwd_end_idx] = 1
            preds.mul_(multi)

        sum_per = preds.sum()
        if sum_per > 0:
            preds /= sum_per  # In-place, vectorized division

    def pwd_is_valid(self, pwd):
        if isinstance(pwd, tuple):
            pwd = "".join(pwd)
        pwd = pwd.strip(self.PASSWORD_END)
        answer = (
            all(map(lambda c: c in self.data.char_bag, pwd))
            and len(pwd) <= self.max_len
            and len(pwd) >= 4
        )
        return answer

        """  def relevel_prediction_many(self, pred_list, str_list):
        if (self.pwd_is_valid(str_list[0]) and len(str_list[0]) != self.max_len):
            pwd = "".join(pwd)
        pwd = pwd.strip(self.PASSWORD_END)
        answer = (
            all(map(lambda c: c in self.data.char_bag, pwd))
            and len(pwd) <= self.max_len
            and len(pwd) >= 4
        )
        return answer
        """

    def relevel_prediction_many(self, pred_list: torch.Tensor, str_list: list[str]):
        for i, pred_item in enumerate(pred_list):
            self.relevel_prediction(pred_item[0], str_list[i])

    def conditional_probs_many(self, astring_list: list[str]) -> torch.Tensor:
        x_data = self.encode_passwords(astring_list)

        answer: torch.Tensor = self.generate(x_data)
        if len(answer.shape) == 2:
            answer = answer.unsqueeze(1)

        assert answer.shape == (len(astring_list), 1, self.data.tokenizer.vocab_size)

        self.relevel_prediction_many(answer, astring_list)
        return answer

    def choose(self, preds: torch.Tensor, rng: TCGenerator) -> int:
        idx = torch.multinomial(
            preds,
            num_samples=1,
            replacement=True,
            generator=rng,
        )
        return int(idx.item())

    # The batch size your GPU can comfortably handle (as per your docstring).

    def _find_optimal_batch_size(
        self, probe_batch_size: int = 32, safety_margin: float = 0.30
    ) -> int:
        if "cuda" not in str(self.device):
            print("CUDA not available. Defaulting to batch size 1024.")
            return 1024

        device = self.device
        print(
            f"Probing for optimal batch size on {torch.cuda.get_device_name(device)}..."
        )

        try:
            # 1) Baseline n=1
            _gc_cuda(device)
            torch.cuda.reset_peak_memory_stats(device)

            with torch.inference_mode():
                _consume(self.iid_sample_batched(1))

            stats1 = _sync_and_stats(device)
            baseline_peak_mem = stats1["peak_reserved"]

            # 2) Probe n=probe_batch_size
            _gc_cuda(device)
            torch.cuda.reset_peak_memory_stats(device)

            with torch.inference_mode():
                _consume(self.iid_sample_batched(probe_batch_size))

            stats_probe = _sync_and_stats(device)
            probe_peak_mem = stats_probe["peak_reserved"]

            # 3) Marginal per-sample (bytes)
            marginal_mem_for_batch = max(0, probe_peak_mem - baseline_peak_mem)
            mem_per_sample = marginal_mem_for_batch / max(1, (probe_batch_size - 1))
            if mem_per_sample <= 0:
                print(
                    "Warning: probe inconclusive; falling back to baseline per-sample."
                )
                mem_per_sample = max(1, baseline_peak_mem)  # very conservative

            # 4) Budget = current free * safety margin
            _gc_cuda(device)
            free_now, total = torch.cuda.mem_get_info(device)
            usable_budget = int(free_now * safety_margin)

            # Base model cost is roughly baseline minus one sample’s marginal cost
            base_cost = max(0, baseline_peak_mem - mem_per_sample)
            mem_for_samples = max(0, usable_budget - base_cost)

            max_samples = int(mem_for_samples // max(1, mem_per_sample))
            max_samples = max(1, max_samples)  # never return 0

            print(
                f"[probe] free={free_now / 2**30:.2f}GiB total={total / 2**30:.2f}GiB | "
                f"baseline_peak={baseline_peak_mem / 2**20:.1f}MiB "
                f"probe_peak={probe_peak_mem / 2**20:.1f}MiB "
                f"mem/sample={mem_per_sample / 2**20:.1f}MiB -> batch={max_samples}"
            )
            return max_samples

        except RuntimeError as e:
            # If the probe itself OOMs, fall back to 1
            print(f"Error during probing (likely OOM): {e}")
            return 1

    # ---------------------------------------------------------------------
    # Drop-in: batch driver (unchanged shape, clearer prints)
    # ---------------------------------------------------------------------
    def one_batch_to_control_them_all(self, n: int) -> Iterable[float]:
        BATCH_SIZE = self._find_optimal_batch_size()
        TOTAL = n
        num_batches = ceil(TOTAL / BATCH_SIZE)

        print(
            f"Generating {TOTAL:,} passwords in {num_batches:,} batches of {BATCH_SIZE}..."
        )
        generated = 0

        for i in range(num_batches):
            need = min(BATCH_SIZE, TOTAL - generated)
            if need <= 0:
                break

            for _, prob in self.iid_sample_batched(need):
                yield prob

            generated += need
            if (i + 1) % 10 == 0 or (i + 1) == num_batches:
                print(f"Progress: {generated:,} / {TOTAL:,}")

        print("Done.")

    # ---------------------------------------------------------------------
    # Core: MC i.i.d. sampler (unbiased, mask+relevel each step)
    #   Assumptions:
    #     • self.generate(x) RETURNS PROBABILITIES (NOT logits)
    #     • self.encode_passwords(prefixes) → tensor on correct device
    #     • self.pwd_is_valid(str) returns True iff the string is a valid FINAL password
    #     • self.data.tokenizer.char_list is your vocab (list of chars)
    #     • self.pwd_end_idx is the END token index
    #     • self.max_len is the max #symbols INCLUDING the END step
    # ---------------------------------------------------------------------
    def iid_sample_batched(
        self, n: int, treshold: Optional[float] = 0.0
    ) -> Iterable[tuple[str, float]]:
        LN2_INV = 1.4426950408889634  # 1 / ln(2)
        """
        Monte Carlo i.i.d. sampling from the (filtered) model distribution.
        Produces exactly `n` (password, probability) pairs drawn from the same
        distribution that will be used to score targets. No top-k/threshold truncation.
    
        Note: `treshold` kept for API compatibility but intentionally unused.
        """
        device = self.device
        chars = self.data.tokenizer.char_list
        END = self.pwd_end_idx

        prefixes = [""] * n
        logp2 = torch.zeros(n, device=device, dtype=torch.float32)
        is_finished = torch.zeros(n, device=device, dtype=torch.bool)

        with torch.inference_mode():
            for t in range(self.max_len):
                if is_finished.all():
                    break

                # Encode on right device
                x_data = self.encode_passwords(prefixes)

                # --- NEXT-STEP PROBABILITIES ---
                probs = self.generate(x_data)  # MUST be probabilities, shape [n, vocab]
                # If your generate() actually returns logits, uncomment:
                # probs = torch.softmax(probs, dim=1)

                # --- STEP-WISE VALIDITY MASK + RELEVEL ---
                # Allowed next tokens for each row (True/1 = allowed, False/0 = banned)
                valid = self.valid_next_mask(prefixes, allow_all_non_end=True).to(
                    device=probs.device
                )
                probs = probs * valid  # zero out disallowed tokens

                # Renormalize per row; if a row has no mass, fall back to END-only
                row_sums = probs.sum(dim=1, keepdim=True)
                end_only = torch.zeros_like(probs)
                end_only[:, END] = 1.0
                probs = torch.where(row_sums > 0, probs / row_sums, end_only)

                # --- FINISHED ROWS → DEGENERATE ON END ---
                if is_finished.any():
                    probs[is_finished] = 0.0
                    probs[is_finished, END] = 1.0

                # --- FORCE END AT MAX LENGTH (last iteration) ---
                if t == self.max_len - 1:
                    probs.zero_()
                    probs[:, END] = 1.0

                # --- SAMPLE + ACCUMULATE LOG2 PROB ---
                dist = torch.distributions.Categorical(probs=probs)
                next_idx = dist.sample()  # [n]
                # accumulate only for unfinished rows (finished rows add 0 anyway, but keep it clear)
                lp = dist.log_prob(next_idx) * LN2_INV
                logp2[~is_finished] += lp[~is_finished]

                # --- UPDATE PREFIXES / FINISHED FLAGS ---
                next_idx_cpu = next_idx.detach().cpu()
                for i in range(n):
                    if is_finished[i]:
                        continue
                    idx = int(next_idx_cpu[i])
                    if idx == END:
                        is_finished[i] = True
                    else:
                        prefixes[i] += chars[idx]

        # Convert to probabilities and yield EXACTLY n samples (all valid by construction)
        p_cpu = logp2.exp2().cpu()
        for i in range(n):
            # Safety: should always hold because we only allow END when valid
            # assert self.pwd_is_valid(prefixes[i]), (i, prefixes[i])
            yield (prefixes[i], float(p_cpu[i]))

    # ---------------------------------------------------------------------
    # Helper: build a step-wise validity mask
    #   • allow_all_non_end=True → all non-END tokens are allowed
    #   • END is allowed only when the current prefix already satisfies pwd_is_valid
    #   You can extend this to ban specific characters or implement prefix-level rules.
    # ---------------------------------------------------------------------
    def valid_next_mask(self, prefixes, allow_all_non_end: bool = True) -> torch.Tensor:
        """
        Returns a [n, vocab] mask (float tensor of 0/1) indicating allowed next tokens.

        By default:
          - All non-END tokens are allowed (1's).
          - END is allowed IFF pwd_is_valid(prefix) is True (and prefix not empty).
        """
        n = len(prefixes)
        vocab_size = len(self.data.tokenizer.char_list)
        END = self.pwd_end_idx

        mask = (
            torch.ones((n, vocab_size), dtype=torch.float32, device=self.device)
            if allow_all_non_end
            else torch.zeros((n, vocab_size), dtype=torch.float32, device=self.device)
        )

        # If you have any globally disallowed tokens, zero them here, e.g.:
        # mask[:, self.pad_idx] = 0.0

        # END handling: only allow END when the current prefix is a valid final password
        # (This enforces the constraint "don’t end until constraints satisfied")
        for i, pref in enumerate(prefixes):
            allow_end = (len(pref) > 0) and self.pwd_is_valid(pref)
            mask[i, END] = 1.0 if allow_end else 0.0

        return mask

    def sample_one_iid(self, rng: TCGenerator) -> tuple[str, float]:
        """Draw ONE password i.i.d. from the model; return (pwd, prob)."""
        prefix: str = ""
        prob: torch.Tensor = torch.scalar_tensor(0.0, dtype=float64, device=self.device)
        ch: str = ""
        idx: int = -1
        while True:
            x_data = self.encode_passwords([prefix])  # 1. Encode the prefix
            preds: torch.Tensor = self.generate(x_data)[0, :]
            preds_for_sampling = preds.clone()

            # 3. Normalize the COPY (preds_for_sampling)
            preds_for_sampling.clamp_min_(0.0)
            s = preds_for_sampling.sum()
            if not torch.isfinite(s) or s <= 0:
                preds_for_sampling = torch.zeros_like(preds_for_sampling)
                preds_for_sampling[self.pwd_end_idx] = 1.0
            else:
                preds_for_sampling /= s

            # 4. Choose using the safe, normalized copy
            idx = self.choose(preds_for_sampling, rng)
            prob += torch.log2(preds[idx])
            ch = self.data.tokenizer.char_list[idx]
            if ch == self.PASSWORD_END:
                return prefix, float(torch.exp2(prob).item())
            prefix += ch

    def iid_sampler(self, n: int) -> Generator[tuple[str, float], None, None]:
        """Yield n i.i.d. samples as (pwd, p)."""
        rng = torch.Generator(device=self.device)
        for _ in range(n):
            pwd, prob = self.sample_one_iid(rng)
            yield (pwd, prob)

    def next_nodes(
        self, astring: str, prob: float, prediction: torch.Tensor, file_buffer: list
    ) -> list:
        total_preds = prediction * prob
        max_len = self.max_len
        if len(astring) + 1 > max_len:
            prob_end = total_preds[self.pwd_end_idx]
            if prob_end >= self.lower_probability_threshold:
                file_buffer.append(f"{astring} {prob_end.item()}\n")
                self.n_generated_passwords += 1
            return []

        indexes = torch.arange(len(total_preds)).to(device=self.device)
        above_cutoff = total_preds >= self.lower_probability_threshold
        above_indices = indexes[above_cutoff]
        probs_above = total_preds[above_cutoff]
        answer = []
        above_indices_cpu = above_indices.cpu()

        for i, chain_prob in enumerate(probs_above):
            char = self.data.tokenizer.char_list[above_indices_cpu[i]]
            if char == self.PASSWORD_END:
                file_buffer.append(f"{astring} {float(chain_prob.item())}\n")
                self.n_generated_passwords += 1
            else:
                chain_pass = astring + char
                answer.append((chain_pass, chain_prob.item()))
        return answer

    def batch_prob(self, prefixes: list[str]) -> torch.Tensor:
        return self.conditional_probs_many(prefixes)

    def password_probability(self, target: str) -> float:
        """Probability that the model emits `target` then END, with the same mask+relevel policy as sampling."""
        if not self.pwd_is_valid(target):
            return 0.0

        prefixes = [""] + [target[:i] for i in range(1, len(target) + 1)]
        next_chars = list(target) + [self.PASSWORD_END]
        char_indices = self.data.tokenizer.char_indices
        END = (
            self.pwd_end_idx
            if isinstance(getattr(self, "pwd_end_idx", None), int)
            else char_indices[self.PASSWORD_END]
        )

        # Map next chars to indices; bail to 0 prob if any char isn't in vocab
        try:
            idx_list = [
                c if isinstance(c, int) else char_indices[c] for c in next_chars
            ]
        except KeyError:
            return 0.0

        with torch.inference_mode():
            x = self.encode_passwords(prefixes)
            probs = self.generate(x)  # MUST be probabilities [steps,vocab]
            # If generate() returns logits, uncomment:
            # probs = torch.softmax(probs, dim=1)

            # Step-wise validity mask + relevel (same as sampler)
            valid = self.valid_next_mask(prefixes, allow_all_non_end=True).to(
                device=probs.device
            )
            probs = probs * valid
            row_sums = probs.sum(dim=1, keepdim=True)
            end_only = torch.zeros_like(probs)
            end_only[:, END] = 1.0
            probs = torch.where(row_sums > 0, probs / row_sums, end_only)

            # Pick per-step probs and multiply in log2 space
            rows = torch.arange(probs.size(0), device=probs.device)
            idxs_t = torch.tensor(idx_list, dtype=torch.long, device=probs.device)
            step_p = probs[rows, idxs_t].to(torch.float64)
            step_p.clamp_min_(torch.finfo(step_p.dtype).tiny)

            log2p = step_p.log2().sum()
            return float(log2p.exp2().item())

    def extract_pwd_from_node(self, node_list):
        return map(lambda x: x[0], node_list)

    def super_node_recur(self, node_list, file):
        if len(node_list) == 0:
            return
        pwds_list = list(self.extract_pwd_from_node(node_list))
        predictions = self.batch_prob(pwds_list)
        node_batch = []
        file_buffer = []
        for i, cur_node in enumerate(node_list):
            astring, prob = cur_node
            for next_node in self.next_nodes(
                astring, prob, predictions[i][0], file_buffer
            ):
                node_batch.append(next_node)
                if len(node_batch) == self.chunk_size_guesser:
                    self.super_node_recur(node_batch, file)
                    node_batch = []

            if len(file_buffer) >= 1_000_000:
                file.writelines(file_buffer)
                file_buffer.clear()

        if len(file_buffer) > 0:
            file.writelines(file_buffer)
            file_buffer.clear()

        if len(node_batch) > 0:
            self.super_node_recur(node_batch, file)
            node_batch = []

    def _recur(self, file, astring="", prob=1):
        self.super_node_recur([(astring, prob)], file)

    def starting_node(self, default_value):
        return default_value

    def guess(self, astring="", prob=1):
        with gzip.open(self.output_file, "at") as file:
            self._recur(file, self.starting_node(astring), prob)

    def complete_guessing(self, start="", start_prob=1):
        self.guess(start, start_prob)
        return self.n_generated_passwords


def _consume(gen):
    # Consume an iterator without storing anything
    for _ in gen:
        pass


def _sync_and_stats(device):
    torch.cuda.synchronize(device)
    reserved = torch.cuda.memory_reserved(device)
    allocated = torch.cuda.memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)
    peak_allocated = torch.cuda.max_memory_allocated(device)
    free, total = torch.cuda.mem_get_info(device)
    return {
        "reserved": reserved,
        "allocated": allocated,
        "peak_reserved": peak_reserved,
        "peak_allocated": peak_allocated,
        "free": free,
        "total": total,
    }


def _as_cuda_device(d):
    if isinstance(d, torch.device):
        return d
    if isinstance(d, str):
        return torch.device(d)
    # assume integer index
    return torch.device(f"cuda:{int(d)}")


def _gc_cuda(device):
    import gc

    gc.collect()

    if not torch.cuda.is_available():
        return

    d = _as_cuda_device(device)
    if d.type != "cuda":
        return

    # Make sure we're operating on *this* device
    with torch.cuda.device(d):
        torch.cuda.synchronize()  # ensure pending kernels are done
        torch.cuda.empty_cache()  # clears cached blocks for current device
        torch.cuda.ipc_collect()  # reclaims any outstanding IPC allocations
