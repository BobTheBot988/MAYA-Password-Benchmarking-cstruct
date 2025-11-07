import csv
import numpy
import torch
from torch import Tensor, device, float64
import torch.nn.functional as F
import gzip
import math
from typing import Iterable, Optional, Tuple, List


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

    def generate(self, x_data: Tensor, log=False) -> Tensor:
        self.model.eval()
        with torch.no_grad():
            output = self.model(x_data)
            if not log:
                output = F.softmax(output, dim=1)
            else:
                output = F.log_softmax(output, dim=1)

            output = output.to(dtype=float64)
        return output

    #    # TODO ASK why this exists if we don't even use it
    #    def encode_passwords(self, astring_list):
    #        max_len = self.max_len
    #        x_data = []
    #
    #        for password in astring_list:
    #            current_password = []
    #
    #            for char in password:
    #                encoded_char = self.data.charmap[char]
    #                current_password.append(encoded_char)
    #
    #            while len(current_password) < max_len:
    #                current_password.append(0)
    #
    #            x_data.append(current_password)
    #
    #        x_data = torch.tensor(np.array(x_data), dtype=torch.long).to(self.device)
    #        x_data = (
    #            F.one_hot(x_data, self.data.charmap_size).to(self.device).to(torch.float32)
    #        )
    #        return x_data

    def relevel_prediction(self, preds, astring):
        if isinstance(astring, tuple):
            astring_joined_len = sum(map(len, astring))
        else:
            astring_joined_len = 0
        if not self.pwd_is_valid(astring):
            preds[self.data.tokenizer.char_indices[self.PASSWORD_END]] = 0
        elif len(astring) == self.max_len or (
            isinstance(astring, tuple) and astring_joined_len == self.max_len
        ):
            multiply = torch.zeros_like(preds)
            multiply[self.pwd_end_idx] = 1
            preds[self.pwd_end_idx] = 1
            preds = preds.mul_(multiply)

        sum_per = sum(preds)
        preds /= sum_per
        sum_per = preds.sum()
        assert 0.9999 <= sum_per <= 1.0001, (
            f"The sum should be 1.0 actual sum:{sum_per}"
        )

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

    def relevel_prediction_many(self, pred_list, str_list):
        if self.pwd_is_valid(str_list[0]) and len(str_list[0]) != self.max_len:
            return
        for i, pred_item in enumerate(pred_list):
            self.relevel_prediction(pred_item[0], str_list[i])

    def relevel_prediction_many_log(self, pred_list, str_list):
        for i, pred_item in enumerate(pred_list):
            # pred_item[0] is the log-prob distribution for the i-th string
            self.relevel_prediction_log(pred_item[0], str_list[i])

    def relevel_prediction_log(self, preds, astring):
        if isinstance(astring, tuple):
            astring_joined_len = sum(map(len, astring))
        else:
            astring_joined_len = 0

        if not self.pwd_is_valid(astring):
            # Make it impossible to end the password
            preds[self.data.tokenizer.char_indices[self.PASSWORD_END]] = -torch.inf

        elif len(astring) == self.max_len or (
            isinstance(astring, tuple) and astring_joined_len == self.max_len
        ):
            # We MUST end the password now.
            # Set all log-probs to -inf (Prob = 0)
            preds[:] = -torch.inf
            # Set the END token log-prob to 0.0 (Prob = 1)
            preds[self.pwd_end_idx] = 0.0

    def extract_probability_for_pwd(self, pwd: str) -> float:
        """
        Calculates the full probability of a single password string
        using the chain rule: P(c1, c2, ..., cn) = P(c1) * P(c2|c1) * ...
        """
        tokenizer = self.data.tokenizer
        end_token = tokenizer.PASSWORD_END

        # 1. Create the list of all prefixes (the inputs)
        # For "be", this will be ["", "b", "be"]
        prefixes = [pwd[:i] for i in range(len(pwd) + 1)]

        # 2. Create the list of all target characters (the outputs)
        # For "be", this will be ['b', 'e', END_TOKEN]
        targets = list(pwd) + [end_token]

        # 3. Get all conditional probabilities in one batch
        # This calls encode_many(prefixes), which is what we want.
        # The model will predict P(next_char | prefix) for each prefix.
        # Shape of prob_dists_batch should be (len(prefixes), 1, vocab_size)
        prob_dists_batch = self.conditional_probs_many(prefixes, log=True)

        # Squeeze out the middle dimension (1) to get (batch_size, vocab_size)
        prob_dists_batch = prob_dists_batch.squeeze(dim=1).to(device=self.device)

        # total_probability = 1.0
        total_log_probability = torch.scalar_tensor(
            0.0, dtype=float64, device=self.device
        )

        # 4. Loop through each step, find the probability, and multiply
        for i, target_char in enumerate(targets):
            # Get the full probability distribution for this step
            # P( . | prefixes[i] )
            dist = prob_dists_batch[i]

            # Get the index of the character we *actually* observed
            try:
                target_index = tokenizer.get_char_index(target_char)
            except KeyError:
                print(f"Error: Character '{target_char}' not in tokenizer vocabulary.")
                return 0.0  # This string is impossible according to the model

            # Get the specific probability of that target character
            # prob_of_char = dist[target_index]
            log_prob_of_char = dist[target_index]

            # 5. Multiply it into our total
            # total_probability *= prob_of_char
            total_log_probability += log_prob_of_char
        total_probability = torch.exp(total_log_probability)

        assert 0.0 < total_probability < 1.0, (
            f"The probability is wrong:{total_probability}"
        )
        return total_probability.item()

    def conditional_probs_many(self, astring_list, log=False):
        x_data = self.data.tokenizer.encode_many(astring_list)
        x_data = torch.tensor(x_data, dtype=torch.float32).to(self.device)

        answer = self.generate(x_data, log=log)
        if len(answer.shape) == 2:
            answer.unsqueeze_(dim=1)

        assert answer.shape == (len(astring_list), 1, self.data.tokenizer.vocab_size)
        if not log:
            self.relevel_prediction_many(answer, astring_list)
        else:
            self.relevel_prediction_many_log(answer, astring_list)
        return answer

    def next_nodes(self, astring, prob, prediction, file_buffer):
        total_preds = prediction * prob
        max_len = self.max_len
        if len(astring) + 1 > max_len:
            prob_end = total_preds[self.pwd_end_idx]
            if prob_end >= self.lower_probability_threshold:
                file_buffer.append(f"{astring} {prob_end}\n")
                self.n_generated_passwords += 1
            return []

        indexes = torch.arange(len(total_preds), device=self.device)
        above_cutoff = total_preds >= self.lower_probability_threshold
        above_indices = indexes[above_cutoff]
        probs_above = total_preds[above_cutoff]
        answer = []
        for i, chain_prob in enumerate(probs_above):
            char = self.data.tokenizer.char_list[above_indices[i]]
            if char == self.PASSWORD_END:
                file_buffer.append(f"{astring} {chain_prob}\n")
                self.n_generated_passwords += 1
            else:
                chain_pass = astring + char
                answer.append((chain_pass, chain_prob))
        return answer

    def batch_prob(self, prefixes):
        return self.conditional_probs_many(prefixes)

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

            if len(file_buffer) >= 1000000:
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

    def one_batch_to_control_them_all(
        self, n: int, write_csv: bool = False, path: Optional[str] = None
    ) -> None:
        BATCH_SIZE = 2048
        TOTAL = n
        num_batches = math.ceil(TOTAL / BATCH_SIZE)

        print(
            f"Generating {TOTAL:,} passwords in {num_batches:,} batches of {BATCH_SIZE}..."
        )
        generated = 0

        lines: List[Tuple[str, float]] = []
        assert path is not None
        with gzip.open(path, "wt") as f:
            w = csv.writer(f)
            for i in range(num_batches):
                need = min(BATCH_SIZE, TOTAL - generated)
                if need <= 0:
                    break
                for pwd, prob in self.iid_sample_batched(need):
                    lines.append((pwd, prob))
                w.writerows(lines)
                lines.clear()
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
    def iid_sample_batched(self, n: int) -> Iterable[tuple[str, float]]:
        """
        Monte Carlo i.i.d. sampling from the (filtered) model distribution.
        Produces exactly `n` (password, probability) pairs drawn from the same
        distribution that will be used to score targets. No top-k/threshold truncation.

        """
        device = self.device
        chars = self.data.tokenizer.char_list
        END = self.pwd_end_idx

        prefixes = [""] * n
        logprob = torch.zeros(n, device=device, dtype=torch.float32)
        # prob = torch.zeros(n, device=device, dtype=torch.float64)
        is_finished = torch.zeros(n, device=device, dtype=torch.bool)

        with torch.inference_mode():
            for t in range(self.max_len):
                if is_finished.all():
                    break

                # Encode on right device
                x_data = torch.tensor(
                    self.data.tokenizer.encode_many(prefixes),
                    device=self.device,
                    dtype=torch.float32,
                )

                # --- NEXT-STEP PROBABILITIES ---
                probs = torch.tensor(
                    self.generate(x_data), device=self.device, dtype=torch.float64
                )  # MUST be probabilities, shape [n, vocab]
                # If your generate() actually returns logits, uncomment:
                # probs = torch.softmax(probs, dim=1)

                # --- STEP-WISE VALIDITY MASK + RELEVEL ---
                # Allowed next tokens for each row (True/1 = allowed, False/0 = banned)
                #
                #
                row_sums = probs.sum(dim=1)

                ones = torch.ones_like(row_sums)

                assert torch.allclose(row_sums, ones), (
                    f"Not all rows sum to 1.0. Min: {row_sums.min().item()}, Max: {row_sums.max().item()}"
                )

                valid = self.valid_next_mask(prefixes, allow_all_non_end=True).to(
                    device=probs.device
                )
                probs = probs * valid  # zero out disallowed tokens

                # Renormalize per row; if a row has no mass, fall back to END-only
                row_sums = probs.sum(dim=1, keepdim=True)

                for i in range(len(probs)):
                    probs[i] /= row_sums[i]
                # 1. Get the row sums. This is a vector of shape [k].
                row_sums = probs.sum(dim=1)

                # 2. Create a reference vector of all ones.
                ones = torch.ones_like(row_sums)

                # 3. Assert that *all* elements in 'row_sums' are close to 1.0.
                assert torch.allclose(row_sums, ones), (
                    f"Not all rows sum to 1.0. Min: {row_sums.min().item()}, Max: {row_sums.max().item()}"
                )
                # --- FINISHED ROWS → DEGENERATE ON END ---
                if is_finished.any():
                    probs[is_finished] = 0.0
                    probs[is_finished, END] = 1.0

                # --- FORCE END AT MAX LENGTH (last iteration) ---
                if t == self.max_len - 1:
                    print("[I] --- FORCE END AT MAX LENGTH (last iteration) ---")
                    probs.zero_()
                    probs[:, END] = 1.0

                # --- SAMPLE + ACCUMULATE LOGN PROB ---
                dist = torch.distributions.Categorical(probs=probs)
                next_idx: torch.Tensor = dist.sample()  # [n]
                # accumulate only for unfinished rows (finished rows add 0 anyway, but keep it clear)
                lp: torch.Tensor = dist.log_prob(next_idx)  # [n]
                logprob[~is_finished] += lp[~is_finished]
                # p = dist.log_prob(next_idx).exp_()
                # prob[~is_finished] *= p[~is_finished]

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
        p_cpu = logprob.to(dtype=torch.float64).exp().cpu()
        # p_cpu = prob.cpu()
        for i in range(n):
            # Safety: should always hold because we only allow END when valid
            # assert self.pwd_is_valid(prefixes[i]), (i, prefixes[i])
            yield (prefixes[i], float(p_cpu[i]))

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
