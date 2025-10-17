import torch
import numpy as np
from numpy.random import Generator as NPGenerator
from typing import Generator
from torch import Generator as TCGenerator
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
        self.chunk_size_guesser = self.params["eval"]["chunk_size_guesser"]
        self.n_generated_passwords = 0
        self.generated_passwords = []
        self.PASSWORD_END = "\n"
        self.pwd_end_idx = self.data.tokenizer.char_indices[self.PASSWORD_END]
        self.output_file = output_file
        self.device = device

    def generate(self, x_data):
        self.model.eval()
        with torch.no_grad():
            output = self.model(x_data)
            output = F.softmax(output, dim=1)
            output = np.array(output.to("cpu"), dtype=np.float64)
        return output

    def encode_passwords(self, astring_list):
        max_len = self.max_len
        x_data = []

        for password in astring_list:
            current_password = []

            for char in password:
                encoded_char = self.data.charmap[char]
                current_password.append(encoded_char)

            while len(current_password) < max_len:
                current_password.append(0)

            x_data.append(current_password)

        x_data = torch.tensor(np.array(x_data), dtype=torch.long).to(self.device)
        x_data = (
            F.one_hot(x_data, self.data.charmap_size).to(self.device).to(torch.float32)
        )
        x_data = (
            F.one_hot(x_data, self.data.charmap_size).to(self.device).to(torch.float32)
        )
        return x_data

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
            multiply = np.zeros(len(preds))
            multiply[self.pwd_end_idx] = 1
            preds[self.pwd_end_idx] = 1
            preds = np.multiply(preds, multiply, preds)

        sum_per = sum(preds)
        for i, v in enumerate(preds):
            preds[i] = v / sum_per

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

    def relevel_prediction_many(self, pred_list, str_list):
        for i, pred_item in enumerate(pred_list):
            self.relevel_prediction(pred_item[0], str_list[i])

    def conditional_probs_many(
        self, astring_list: list[str]
    ) -> np.typing.NDArray[np.float64]:
        x_data = self.data.tokenizer.encode_many(astring_list)
        x_data = torch.tensor(np.array(x_data), dtype=torch.float32).to(self.device)

        answer = self.generate(x_data)
        if len(answer.shape) == 2:
            answer = np.expand_dims(answer, axis=1)

        assert answer.shape == (len(astring_list), 1, self.data.tokenizer.vocab_size)

        self.relevel_prediction_many(answer, astring_list)
        return answer

    def choose(
        self,
        preds: np.typing.NDArray[np.float64] | torch.Tensor,
        rng: TCGenerator | NPGenerator,
    ) -> int:
        # Torch path (use if you pass a torch.Generator or preds is a torch.Tensor)
        if isinstance(rng, TCGen) or torch.is_tensor(preds):
            t = (
                preds
                if torch.is_tensor(preds)
                else torch.as_tensor(preds, dtype=torch.float64, device="cpu")
            )
            t = torch.nan_to_num(t, nan=0.0).clamp_min_(0)
            s = t.sum()
            if not torch.isfinite(s) or s <= 0:
                return int(self.pwd_end_idx)
            t = t / s
            idx = torch.multinomial(
                t,
                num_samples=1,
                replacement=True,
                generator=rng if isinstance(rng, TCGenerator) else None,
            )
            return int(idx.item())

        # NumPy path
        p = np.asarray(preds, dtype=np.float64)
        p = np.nan_to_num(p, nan=0.0)
        p[p < 0.0] = 0.0
        s = p.sum()
        if not np.isfinite(s) or s <= 0:
            return int(self.pwd_end_idx)
        p /= s
        if not isinstance(rng, NPGenerator):
            rng = np.random.default_rng()
        return int(rng.choice(p.size, p=p))

        """idx: int = -1
        if isinstance(rng, TCGenerator):
            tensor_preds: torch.Tensor = torch.tensor(preds)
            idx = int(torch.multinomial(tensor_preds, 1).item())
        elif isinstance(rng, NPGenerator):
            idx = rng.choice(len(preds), p=preds)  # sample next char

        return idx"""

    def sample_one_iid(
        self, rng: TCGenerator | NPGenerator = np.random.default_rng()
    ) -> tuple[str, float]:
        """Draw ONE password i.i.d. from the model; return (pwd, prob)."""
        prefix: str = ""
        prob: float = 0.0
        ch: str = ""
        idx: int = -1
        p: float = -1
        while True:
            preds: np.typing.NDArray[np.float64] = self.batch_prob([prefix])[
                0, 0, :
            ]  # probs over vocab

            preds = np.asarray(preds, dtype=np.float64).ravel()

            preds = np.clip(preds, 0.0, None)
            s = preds.sum()
            if not np.isfinite(s) or s <= 0:
                # fallback: force EOS or uniform over valid tokens
                preds = np.zeros_like(preds)
                preds[self.pwd_end_idx] = 1.0
            else:
                preds /= s
            # relevel_prediction_many already enforced EOS logic / validity
            idx = self.choose(preds, rng)
            p = float(np.log2(preds[idx]))
            prob += p
            ch = self.data.tokenizer.char_list[idx]
            if ch == self.PASSWORD_END:
                return prefix, np.exp2(prob)
            prefix += ch  # continue

    def iid_sampler(
        self, n: int, rng: TCGenerator | NPGenerator = np.random.default_rng()
    ) -> Generator[tuple[str, float], None, None]:
        """Yield n i.i.d. samples as (pwd, p)."""
        for _ in range(n):
            pwd, prob = self.sample_one_iid(rng=rng)
            yield (pwd, prob)

    def next_nodes(self, astring, prob, prediction, file_buffer):
        total_preds = prediction * prob
        max_len = self.max_len
        if len(astring) + 1 > max_len:
            prob_end = total_preds[self.pwd_end_idx]
            if prob_end >= self.lower_probability_threshold:
                file_buffer.append(f"{astring} {prob_end}\n")
                self.n_generated_passwords += 1
            return []

        indexes = np.arange(len(total_preds))
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

    def batch_prob(self, prefixes: list[str]) -> np.typing.NDArray[np.float64]:
        return self.conditional_probs_many(prefixes)

    def password_probability(self, target: str) -> float:
        """Probability that the model emits `target` followed by PASSWORD_END."""
        if not self.pwd_is_valid(target):
            return 0.0

        # prefixes to condition on: "", "c", "ci", "cia", "ciao"
        prefixes = [""] + [target[:i] for i in range(1, len(target) + 1)]
        # next tokens we want the model to choose at each step: "c","i","a","o","\n"
        next_chars = list(target) + [self.PASSWORD_END]

        preds = self.batch_prob(prefixes)  # shape: (len(prefixes), 1, vocab)
        preds = preds[:, 0, :]  # shape: (len(prefixes), vocab)

        # gather the probabilities for our desired next tokens
        idxs = [self.data.tokenizer.char_indices[ch] for ch in next_chars]
        step_probs = np.array(
            [preds[i, idxs[i]] for i in range(len(idxs))], dtype=np.float64
        )
        # numerical stability

        step_probs = np.clip(step_probs, np.finfo(np.float64).tiny, 1.0)
        target_prob: float = float(step_probs.sum())  # P(target)
        return target_prob

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
