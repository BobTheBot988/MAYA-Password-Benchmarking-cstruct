import gc
import gzip

from math import ceil
import os
import shutil
import tempfile
from io import BufferedRandom
from typing import Dict, Generator, Iterable

import heapcy
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Optimizer
from tqdm import tqdm

from script.test.model import Model, SampleMode

from .architecture import LSTM
from .fla_utils.dataloader import DataLoader
from .guesser import Guesser
from models.FLA import guesser


def get_lower_probability_threshold(n_samples):
    n_samples = int(n_samples)
    if n_samples <= 10**5:
        return 0.000001
    elif n_samples <= 10**6:
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

    def prepare_data(self, train_passwords, test_passwords, max_length) -> DataLoader:
        return DataLoader(train_passwords, test_passwords, max_length, self.params)  # noqa: F405

    def load(self, file_name):
        try:
            self.init_model()
            state_dicts = torch.load(
                file_name, map_location=self.device, weights_only=False
            )
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

    def eval_init(self, n_samples: int, evaluation_batch_size) -> dict[str, Model.T]:
        self.model.eval()
        eval_dict = {
            "n_samples": self.n_samples,
            "output_file": os.path.join(self.path_to_guesses_dir, "total_guesses.gz"),
        }

        return eval_dict

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

    def get_string_probability(self) -> float:
        guesser: Guesser = self.guesser_build(self.eval_init(0, 0))
        return guesser.password_probability(self.estimate_pwd)

    def generate_file(self, guesser: Guesser) -> int:
        return guesser.complete_guessing()

    def temp_file_generate(self, eval_dict) -> str:
        print("[I] - Creating Temporary file")
        with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
            with gzip.open(eval_dict["output_file"]) as fopen:
                shutil.copyfileobj(fopen, tmpfile)
                temp_file_name: str = tmpfile.name
        return temp_file_name

    def generate_heap(
        self, temp_file_name: str, eval_dict: Dict[str, Model.T]
    ) -> heapcy.Heap:
        with open(temp_file_name, "rb") as f_open:
            print(f"[I] - Size of heap will be:{eval_dict['n_samples']}")
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

                if len(min_heap_n_most_prob) < eval_dict["n_samples"]:
                    heapcy.heappush(min_heap_n_most_prob, prob, offset)
                else:
                    heapcy.heappushpop(min_heap_n_most_prob, prob, offset)

        return min_heap_n_most_prob

    def generate_one_time_pwds(
        self, evaluation_batch_size: int = 8192
    ) -> Iterable[float]:
        # Let's use 8192.

        # Calculate how many batches (loops) you'll need.
        guess: Guesser = self.guesser_build(self.eval_init(0, 0))
        return guess.one_batch_to_control_them_all(self.n_samples)

    def sample(
        self,
        evaluation_batch_size,
        eval_dict: Dict[str, Model.T],
        guesser: Guesser | None = None,
    ) -> Generator[str, None, None] | Generator[tuple[str, float], None, None]:
        if not guesser:
            guesser = self.guesser_build(eval_dict)

        if self.mode is SampleMode.IID:
            print("[I] - Generating string so that we are iid")
            return guesser.iid_sample_batched(self.n_samples)

        print("[I] - Generating strings")
        n_gen: int = guesser.complete_guessing()

        print(f"[I] - Generated {n_gen} passwords")

        print("[I] - Opening Temporary file")
        temp_file_name: str = self.temp_file_generate(eval_dict)

        min_heap_n_most_prob: heapcy.Heap = self.generate_heap(
            temp_file_name, eval_dict
        )

        offsets: list[int] = []

        print("[I] - Getting nlargest")
        for x in heapcy.nlargest(min_heap_n_most_prob, eval_dict["n_samples"]):
            offsets.append(x[1])

        del min_heap_n_most_prob
        gc.collect()

        eval_dict["tempfilename"] = temp_file_name

        print("[I] - Returning String Generator")
        return heapcy.string_generator(temp_file_name, offsets)

    def guessing_strategy(self, evaluation_batch_size, eval_dict):
        pass

    def post_sampling(self, eval_dict):
        gc.collect()
        os.remove(eval_dict["output_file"])
        os.remove(eval_dict["tempfilename"])


def get_n_of_lines(filename: BufferedRandom) -> int:
    i: int = 0
    while filename.readline():
        i += 1
    return i
