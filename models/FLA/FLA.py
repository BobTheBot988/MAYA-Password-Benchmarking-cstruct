from io import BufferedRandom

# from math import log2
import os
from typing import Generator, Dict
import torch
import torch.nn.functional as F
from torch.optim import Optimizer
import numpy as np
from tqdm import tqdm
import gc
import gzip
import tempfile
import shutil
import heapcy

from script.test.model import Model

from models.FLA.architecture import LSTM
from models.FLA.guesser import Guesser
from models.FLA.fla_utils.dataloader import *  # noqa: F403


def get_lower_probability_threshold(n_samples):
    n_samples = int(n_samples)
    if n_samples <= 10**6:
        return 0.00000001
    elif n_samples <= 10**7:
        return 0.000000001
    elif n_samples <= 5 * (10**8):
        return 0.0000000001
    else:
        return 0.00000000001


class FLA(Model):
    def __init__(self, settings):
        self.model: LSTM
        self.optimizer: Optimizer

        super().__init__(settings)

    def prepare_data(self, train_passwords, test_passwords, max_length):
        return DataLoader(train_passwords, test_passwords, max_length, self.params)  # noqa: F405

    def load(self, file_name):
        try:
            self.init_model()
            state_dicts = torch.load(file_name, map_location=self.device)
            self.model.load_state_dict(state_dicts["model"])
            self.optimizer.load_state_dict(state_dicts["optimizer"])
            return 1
        except Exception as e:
            print(f"Exception: {e}")
            return 0

    def init_model(self):
        self.params["eval"]["evaluation_batch_size"] = self.n_samples + 1
        lstm_hidden_size = self.params["train"]["lstm_hidden_size"]
        dense_hidden_size = self.params["train"]["dense_hidden_size"]
        context_len = self.data.max_length
        vocab_size = self.data.tokenizer.vocab_size

        self.model = LSTM(
            lstm_hidden_size=lstm_hidden_size,
            dense_hidden_size=dense_hidden_size,
            vocab_size=vocab_size,
            context_len=context_len,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters())

    def train_step(self, x_train, y_train):
        self.model.train()
        self.optimizer.zero_grad()

        y_pred = self.model(x_train)

        train_loss = F.cross_entropy(y_pred, y_train)
        train_loss.backward()

        self.optimizer.step()

    def train(self):
        print("[I] - Launching training")

        epochs = self.params["train"]["epochs"]
        batch_size = self.params["train"]["batch_size"]

        current_epoch = 0
        n_matches = 0
        n_passwords = self.data.get_train_size()

        checkpoint_frequency = self.params["eval"]["checkpoint_frequency"]

        self.init_model()

        while current_epoch < epochs:
            print(f"Epoch: {current_epoch + 1} / {epochs}")

            progress_bar = tqdm(
                range(n_passwords), desc="Epoch {}/{}".format(current_epoch, epochs)
            )

            n_iter = 0
            for batch in self.data.get_batches(batch_size):
                x_train = np.array(batch[0])
                y_train = np.array(batch[1])

                x_train = torch.tensor(x_train, dtype=torch.float32).to(self.device)
                y_train = torch.tensor(y_train, dtype=torch.float32).to(self.device)

                self.train_step(x_train, y_train)
                progress_bar.update(batch_size)
                n_iter += 1

            if current_epoch % checkpoint_frequency == 0:
                matches, _, _ = self.evaluate(n_samples=10**6, validation_mode=True)
                if matches >= n_matches:
                    n_matches = matches
                    obj = {
                        "model": self.model.state_dict(),
                        "optimizer": self.optimizer.state_dict(),
                    }
                    self.save(obj)

            current_epoch += 1

    def eval_init(self, n_samples: int, evaluation_batch_size):
        self.model.eval()
        eval_dict = {
            "n_samples": n_samples,
            "output_file": os.path.join(self.path_to_guesses_dir, "total_guesses.gz"),
        }
        return eval_dict

    def get_string_probability(
        self, string: str, guess: Guesser | None = None
    ) -> float:
        if not guess:
            raise ValueError("Must pass a guesser")
        return guess.batch_prob([string])[0]

    def guesser_build(self, eval_dict: Dict[str, Model.T]) -> Guesser:
        return Guesser(
            model=self.model,
            params=self.params,
            data=self.data,
            lower_probability_threshold=get_lower_probability_threshold(
                eval_dict["n_samples"]
            ),
            output_file=eval_dict["output_file"],
            device=self.device,
        )

    def montecarlo_estimation(
        self, my_string: str, eval_dict: Dict[str, Model.T]
    ) -> float:
        """
        my_array: probabilities sorted in DESCENDING order
        target: p(alpha)
        returns: (1/n) * sum_{i: A[i] > target} 1/A[i]
        """

        # Needed for stability      target: float = -log2(self.get_string_probability(my_string, guesser))

        guesser = self.guesser_build(eval_dict)

        assert isinstance(eval_dict["output_file"], str)
        if not os.path.exists(eval_dict["output_file"]):
            assert isinstance(eval_dict["n_samples"], int)
            if eval_dict["n_samples"] > 100_000:
                raise ValueError(
                    "The number of samples for the montecarlo_estimation should not be higher than 10**5."
                )
            self.sample(0, eval_dict, guesser)

        print("[I] - Creating Temporary file")

        target: float = self.get_string_probability(my_string, guesser)
        with tempfile.TemporaryFile() as tmpfile:
            with gzip.open(eval_dict["output_file"]) as fopen:
                shutil.copyfileobj(fopen, tmpfile)

            my_array: Generator[float, None, None] = self.generator_tmp_file(tmpfile)
            size_of_array: int = get_n_of_lines(tmpfile)
            my_sum: float = 0
            prob: float = 0

            # This can be optimized in O(logn) if the file already exists, but in doing so we will use O(n) memory
            for prob in my_array:
                if prob == 0.0:
                    return -1.0
                if prob <= target:
                    break
                my_sum += 1.0 / prob

        return my_sum / size_of_array

    def generator_tmp_file(
        self, tmpfile: BufferedRandom, i: int = 0, encoding: str = "ascii"
    ) -> Generator[float, None, None]:
        while True:
            line: bytes = tmpfile.readline()
            if len(line) == 0 or line is EOFError:
                break

            if not line:
                yield 0.0  # offset past EOF
                continue
            # take bytes up to first space (or whole line), strip newline
            i = line.find(b" ")
            # take byets from space onwards so only the probability
            token = line[i + 1 :] if i != -1 else line.rstrip(b"\r\n")
            # Conversion needed to maintain stability this is according to the ccs15 paper on the montecarlo estimation
            """
            'The probabilities that we compute can be very small and may underflow:
            to avoid such problems, we store and compute the base-2 logarithms of probabilities rather than probabilities themselves'

            ccs15 page 5, implementation details https://www.dcs.gla.ac.uk/~maurizio/Publications/ccs15.pdf
            """
            # yield -log2(float(token.decode(encoding, errors="replace")))

            yield float(token.decode(encoding, errors="replace"))

    def generate_file(self, guesser: Guesser) -> int:
        return guesser.complete_guessing()

    def sample(
        self, evaluation_batch_size, eval_dict, guesser: Guesser | None = None
    ) -> Generator[str, None, None]:
        if not guesser:
            guesser = self.guesser_build(eval_dict)
        n_gen: int = guesser.complete_guessing()

        print(f"[I] - Generated {n_gen} passwords")
        print("[I] - Creating Temporary file")
        with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
            with gzip.open(eval_dict["output_file"]) as fopen:
                shutil.copyfileobj(fopen, tmpfile)
                temp_file_name: str = tmpfile.name

        print("[I] - Opening Temporary file")
        with open(temp_file_name, "rb") as f_open:
            min_heap_n_most_prob: heapcy.Heap = heapcy.Heap(eval_dict["n_samples"])

            while True:
                offset: int = f_open.tell()
                line: bytes = f_open.readline()
                if not line:
                    break

                parts: list[bytes] = line.rstrip(b"\b\n").split(b" ", 1)
                if len(parts) != 2:
                    continue

                prob: float = float(parts[1].decode(encoding="ascii"))
                heapcy.heappush(min_heap_n_most_prob, prob, offset)

        offsets: list[int] = []

                if len(min_heap_n_most_prob) < eval_dict["n_samples"]:
                    heapcy.heappush(min_heap_n_most_prob, prob, offset)
                else:
                    heapcy.heappushpop(min_heap_n_most_prob, prob, offset)

        offsets: list[int] = []

        print("[I] - Getting nlargest")
        for x in heapcy.nlargest(min_heap_n_most_prob, eval_dict["n_samples"]):
            offsets.append(x[1])

        del min_heap_n_most_prob
        gc.collect()

        eval_dict["tempfilename"] = temp_file_name

        eval_dict["tempfilename"] = temp_file_name

        print("[I] - Returning String Generator")
        return heapcy.string_generator(temp_file_name, offsets)

    def guessing_strategy(self, evaluation_batch_size, eval_dict):
        pass

    def post_sampling(self, eval_dict):
        gc.collect()
        os.remove(eval_dict["output_file"])
        os.remove(eval_dict["tempfilename"])
        pass


def get_n_of_lines(filename: BufferedRandom) -> int:
    i: int = 0
    while filename.readline():
        i += 1
    return i
