import torch
from math import ceil
from typing import Generator, Iterable
from torch import Generator as TCGenerator, Tensor, float64
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
        self, probe_batch_size: int = 32, safety_margin: float = 0.90
    ) -> int:
        """
        Probes the GPU to find the largest batch size that can fit in VRAM.

        It works by:
        1. Getting the baseline peak memory for n=1 (for the model + 1 sample's KV cache).
        2. Getting the peak memory for a small batch (e.g., n=32).
        3. Calculating the marginal memory cost for each *additional* sample.
        4. Dividing the available free memory by this marginal cost.
        """

        if "cuda" not in str(self.device):
            print("CUDA not available. Defaulting to batch size 1024.")
            return 1024

        device = self.device
        print(
            f"Probing for optimal batch size on {torch.cuda.get_device_name(device)}..."
        )

        try:
            # --- 1. Get Baseline Memory (n=1) ---
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)

            # We must *fully consume* the generator to get the *peak* memory
            # from the longest password (i.e., full KV cache).
            _ = list(self.iid_sample_batched(1))

            baseline_peak_mem = torch.cuda.max_memory_reserved(device)

            # --- 2. Get Delta Memory (n=probe_batch_size) ---
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)

            _ = list(self.iid_sample_batched(probe_batch_size))

            probe_peak_mem = torch.cuda.max_memory_reserved(device)

            # --- 3. Calculate Marginal Cost ---
            # Memory for (probe_batch_size - 1) additional samples
            marginal_mem_for_batch = probe_peak_mem - baseline_peak_mem

            # Memory for *one* additional sample
            mem_per_sample = marginal_mem_for_batch / (probe_batch_size - 1)

            if mem_per_sample <= 0:
                # This can happen if the batch size is too small to see a difference.
                # We'll just use the baseline.
                print("Warning: Probe was inconclusive. Using baseline.")
                mem_per_sample = baseline_peak_mem

            # --- 4. Calculate Max Batch Size ---
            # Get *total* free memory (not just reserved)
            _, total_mem = torch.cuda.mem_get_info(device)

            # Apply safety margin

            # We already have the baseline model loaded, so we just
            # see how many *additional* samples fit in the *remaining* usable memory.
            # (This is a simplified but safe estimation).

            # A more direct calculation:
            # How much memory do we have *in total* for this op?
            # Usable = Total_GPU_Memory * safety_margin
            # We subtract the memory *already used* just to load the model (baseline)

            # Let's use the total available VRAM as our budget.
            # This is safer.
            usable_budget = total_mem * safety_margin

            # How many samples can we fit in this budget?
            # We subtract the "base cost" of the model (baseline - mem_per_sample)
            # and then divide by the per-sample cost.
            base_cost = baseline_peak_mem - mem_per_sample

            # How much memory is left for samples?
            mem_for_samples = usable_budget - base_cost

            max_samples = int(mem_for_samples / mem_per_sample)

            print(f"Probe complete. Optimal batch size: {max_samples}")
            return max_samples

        except RuntimeError as e:
            # This happens if even the probe_batch_size OOMs.
            print(f"Error during probing (likely OOM): {e}")
            print("Defaulting to batch size 1.")
            return 1

    def one_batch_to_control_them_all(self, n: int) -> Iterable[float]:
        # Let's use 8192.
        # BATCH_SIZE = self._find_optimal_batch_size()
        BATCH_SIZE = 5_000
        TOTAL_PASSWORDS_NEEDED = n

        # Calculate how many batches (loops) you'll need.
        num_batches = ceil(TOTAL_PASSWORDS_NEEDED / BATCH_SIZE)

        print(
            f"Generating {TOTAL_PASSWORDS_NEEDED:,} passwords in {num_batches:,} batches of {BATCH_SIZE}..."
        )

        generated_count = 0

        # We will loop `num_batches` times.
        for i in range(num_batches):
            # Calculate how many to get in this specific batch.
            # It's usually BATCH_SIZE, except for the very last batch.
            n_this_batch = min(BATCH_SIZE, TOTAL_PASSWORDS_NEEDED - generated_count)

            if n_this_batch <= 0:
                break  # Just in case

            # Call your generator. It will yield `n_this_batch` passwords.
            # `self` would be an instance of your class.
            # We wrap this in a loop to consume the generator.
            for _, prob in self.iid_sample_batched(n_this_batch):
                yield prob

            generated_count += n_this_batch

            # Optional: Print progress
            if (i + 1) % 100 == 0:  # Print every 100 batches
                print(f"Progress: {generated_count:,} / {TOTAL_PASSWORDS_NEEDED:,}")

        print("Done.")

    def iid_sample_batched(self, n: int) -> Generator[tuple[str, float], None, None]:
        """
        Generates 'n' passwords in parallel using batching.
        'n' should be a number that can comfortably fit in GPU memory
        (e.g., 1000 or 8192).
        """

        # Create the RNG, just as the original iid_sampler does
        rng = torch.Generator(device=self.device)

        # `prefixes`: A list of 'n' strings, all starting empty.
        # We will build "pass", "123", "abc", etc. here.
        prefixes = [""] * n

        # running log2(probability) for each of the 'n' passwords.
        log_probs = torch.zeros(n, device=self.device, dtype=torch.float64)

        # It tracks which passwords have hit the END token.
        # Starts as all False.
        is_finished = torch.zeros(n, device=self.device, dtype=torch.bool)

        # `encode_passwords` will handle the START_TOKEN implicitly.

        # We loop up to self.max_len to prevent infinite loops.
        for _ in range(self.max_len):
            # Stop early if all 'n' passwords have been finished.
            if is_finished.all():
                break

            # 3. Encode
            # `self.encode_passwords` takes our list of 'n' prefixes
            # and returns a single tensor of shape [n, current_sequence_length].
            # This is the batch we feed to the model.
            x_data = self.encode_passwords(prefixes)

            # 4. Generate (The *SINGLE* batched call to the model)
            # `self.generate` calls the model.
            # It will return predictions for the *next* character for the whole batch.
            # `preds` shape: [n, vocab_size]
            preds = self.generate(x_data)

            # 5. Sample
            # Normalize probabilities for sampling, same as in `sample_one_iid`
            #
            preds_for_sampling = preds.clone()
            preds_for_sampling[preds_for_sampling < 0] = 0
            zero_sum_rows = preds_for_sampling.sum(dim=1) == 0
            if zero_sum_rows.any():
                # Set prob of END token to 1.0 for any row that is all zero
                preds_for_sampling[zero_sum_rows, self.pwd_end_idx] = 1.0

            # `torch.multinomial` samples 1 index from each of the 'n' rows.
            # `next_char_indices` shape: [n, 1]
            next_char_indices = torch.multinomial(preds_for_sampling, 1, generator=rng)

            # 6. Update Probabilities
            # We gather the *true* log-probabilities (from the original `preds`)
            # for the characters we just sampled.
            # `preds.gather(1, next_char_indices)` shape: [n, 1]
            # `.squeeze(-1)` shape: [n]
            chosen_probs = preds.gather(1, next_char_indices).squeeze(-1)

            # We only add the probability if the password is *not* already finished.
            # `(~is_finished)` is a boolean mask (e.g., [True, True, False, True])
            log_probs += torch.log2(chosen_probs) * (~is_finished)

            # 7. Update Prefixes and `is_finished` status
            next_char_indices_flat = next_char_indices.squeeze(-1)

            for i in range(n):
                # If this password *is* finished, we skip it.
                if is_finished[i]:
                    continue

                char_idx = next_char_indices_flat[i].item()

                # Check if the sampled character is the END token
                if char_idx == self.pwd_end_idx:
                    is_finished[i] = True
                    # We don't append the \n, the password is complete.
                else:
                    # It's a normal character, append it to the string.
                    prefixes[i] += self.data.tokenizer.char_list[char_idx]

            # --- End of `for _ in range(self.max_len)` loop ---

        # 3. Finalization
        # The loop is over. Convert all 'n' log-probs back to linear probs.
        final_probs = torch.exp2(log_probs)

        # Yield all 'n' results, one by one, to match the generator format.
        for i in range(n):
            yield (prefixes[i], final_probs[i].item())

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
        """Probability that the model emits `target` followed by PASSWORD_END."""
        if not self.pwd_is_valid(target):
            return 0.0

        prefixes = [""] + [target[:i] for i in range(1, len(target) + 1)]
        next_chars = list(target) + [self.PASSWORD_END]

        # Manually encode the prefixes
        x_data = self.encode_passwords(prefixes)
        # Call generate DIRECTLY to get pure probabilities
        preds: Tensor = self.generate(x_data)

        idxs = [self.data.tokenizer.char_indices[ch] for ch in next_chars]
        idxs_tensor = torch.tensor(idxs, dtype=torch.long, device=preds.device)
        rows = torch.arange(preds.shape[0], device=preds.device)
        step_probs = preds[rows, idxs_tensor]
        step_probs.clamp_(torch.finfo(float64).tiny, 1.0)
        step_probs.log2_()
        final_prob = step_probs.sum()
        final_prob.exp2_()
        return float(final_prob.item())

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
