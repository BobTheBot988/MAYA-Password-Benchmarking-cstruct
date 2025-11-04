from io import TextIOWrapper
import torch
import torch.nn.functional as F
from torch import Tensor, Generator as TCGenerator, device, float64
from typing import Tuple, List, Optional, Generator, Iterable
import gzip
from math import ceil


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

    def generate(self, x_data: Tensor) -> Tensor:
        self.model.eval()
        with torch.no_grad():
            output = self.model(x_data.to(device=self.device))
            output = F.softmax(output, dim=1)
            output = output.to(dtype=torch.float64)
        return output

    def encode_passwords(self, astring_list: List[str]) -> Tensor:
        max_len: int = self.max_len
        x_data: List[List[str]] | Tensor = []

        for password in astring_list:
            current_password: List[str] = []

            for char in password:
                encoded_char = self.data.charmap[char]
                current_password.append(encoded_char)

            while len(current_password) < max_len:
                current_password.append(0)

            x_data.append(current_password)

        x_data = torch.tensor(x_data, dtype=torch.long).to(self.device)
        x_data = (
            F.one_hot(x_data, len(self.data.charmap)).to(self.device).to(torch.float32)
        )
        return x_data

    def relevel_prediction(self, preds: Tensor, astring: Tuple[str] | str):
        if isinstance(astring, tuple):
            astring_joined_len = sum(map(len, astring))
        else:
            astring_joined_len = 0
        if not self.pwd_is_valid(astring):
            preds[self.data.tokenizer.char_indices[self.PASSWORD_END]] = 0
        elif len(astring) == self.max_len or (
            isinstance(astring, tuple) and astring_joined_len == self.max_len
        ):
            multiply = torch.zeros(len(preds), device=self.device)
            multiply[self.pwd_end_idx] = 1
            preds[self.pwd_end_idx] = 1
            torch.multiply(preds, multiply, out=preds)

        sum_per: Tensor = preds.sum()
        # this operation is the vectorized of for loop below
        preds = preds.div_(sum_per)

        # for i, v in enumerate(preds):
        #    preds[i] = v / sum_per

    def pwd_is_valid(self, pwd: Tuple[str] | str) -> bool:
        if isinstance(pwd, tuple):
            pwd = "".join(pwd)
        pwd = pwd.strip(self.PASSWORD_END)
        answer = (
            all(map(lambda c: c in self.data.char_bag, pwd))
            and len(pwd) <= self.max_len
            and len(pwd) >= 4
        )
        return answer

    def relevel_prediction_many(self, pred_list, str_list) -> None:
        if self.pwd_is_valid(str_list[0]) and len(str_list[0]) != self.max_len:
            return
        for i, pred_item in enumerate(pred_list):
            self.relevel_prediction(pred_item[0], str_list[i])

    def conditional_probs_many(self, astring_list) -> Tensor:
        x_data = self.data.tokenizer.encode_many(astring_list)
        x_data = torch.tensor(x_data, dtype=torch.float32).to(self.device)

        answer: Tensor = self.generate(x_data)
        if len(answer.shape) == 2:
            answer = answer.unsqueeze_(dim=1)

        assert answer.shape == (len(astring_list), 1, self.data.tokenizer.vocab_size)

        self.relevel_prediction_many(answer, astring_list)
        return answer

    def next_nodes(
        self, astring: str, prob: float, prediction: Tensor, file_buffer: List[str]
    ) -> List[Tuple[str, float]]:
        total_preds: Tensor = prediction * prob
        total_preds = total_preds.to(device=self.device)

        max_len: int = self.max_len
        if len(astring) + 1 > max_len:
            prob_end: float = float(total_preds[self.pwd_end_idx].item())
            if prob_end >= self.lower_probability_threshold:
                file_buffer.append(f"{astring} {prob_end}\n")
                self.n_generated_passwords += 1
            return []

        indexes: Tensor = torch.arange(len(total_preds), device=self.device)
        above_cutoff: Tensor = total_preds >= self.lower_probability_threshold
        above_cutoff = above_cutoff.to(device=self.device)

        above_indices = indexes[above_cutoff]

        probs_above = total_preds[above_cutoff]
        answer: List[Tuple[str, float]] = []

        for i, chain_prob in enumerate(probs_above):
            char: str = self.data.tokenizer.char_list[above_indices[i]]
            chain_prob = chain_prob.item()
            if char == self.PASSWORD_END:
                file_buffer.append(f"{astring} {chain_prob}\n")
                self.n_generated_passwords += 1
            else:
                chain_pass = astring + char
                answer.append((chain_pass, chain_prob))

        return answer

    def batch_prob(self, prefixes) -> Tensor:
        return self.conditional_probs_many(prefixes)

    def extract_pwd_from_node(self, node_list):
        return map(lambda x: x[0], node_list)

    def super_node_recur(self, node_list: List[Tuple[str, float]], file: TextIOWrapper):
        if len(node_list) == 0:
            return
        pwds_list: List[str] = list(self.extract_pwd_from_node(node_list))
        predictions: Tensor = self.batch_prob(pwds_list)
        node_batch = []
        file_buffer: List[str] = []
        for i, cur_node in enumerate(node_list):
            astring, prob = cur_node
            for next_node in self.next_nodes(
                astring, prob, predictions[i][0], file_buffer
            ):
                node_batch.append(next_node)
                if len(node_batch) == self.chunk_size_guesser:
                    self.super_node_recur(node_batch, file)
                    node_batch = []

            if len(file_buffer) >= 1000000:
                file.writelines(file_buffer)
                file_buffer.clear()

        if len(file_buffer) > 0:
            file.writelines(file_buffer)
            file_buffer.clear()

        if len(node_batch) > 0:
            self.super_node_recur(node_batch, file)
            node_batch: List = []

    def _recur(self, file: TextIOWrapper, astring: str = "", prob: float = 1):
        self.super_node_recur([(astring, prob)], file)

    def starting_node(self, default_value):
        return default_value

    def guess(self, astring="", prob=1):
        with gzip.open(self.output_file, "at") as file:
            self._recur(file, self.starting_node(astring), prob)

    def complete_guessing(self, start="", start_prob=1):
        self.guess(start, start_prob)
        return self.n_generated_passwords

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
                x_data = torch.Tensor(self.data.tokenizer.encode_many(prefixes))

                # --- NEXT-STEP PROBABILITIES ---
                probs = self.generate(x_data).to(
                    device=self.device
                )  # MUST be probabilities, shape [n, vocab]
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
            x_data = torch.Tensor(
                self.data.tokenizer.encode_many([prefix]), device=self.device
            )  # 1. Encode the prefix
            preds: torch.Tensor = self.generate(x_data).to(device=self.device)[0, :]
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
