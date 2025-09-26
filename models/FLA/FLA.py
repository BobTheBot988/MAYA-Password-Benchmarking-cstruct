import os
from typing import Generator
import torch
import torch.nn.functional as F
from tqdm import tqdm
import heapq
import gc
import gzip
import tempfile
import shutil
import heapcy

from script.test.model import Model

from models.FLA.architecture import LSTM
from models.FLA.guesser import Guesser
from models.FLA.fla_utils.dataloader import *


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
        self.model = None
        self.optimizer = None

        super().__init__(settings)

    def prepare_data(self, train_passwords, test_passwords, max_length):
        return DataLoader(train_passwords, test_passwords, max_length, self.params)

    def load(self, file_to_load):
        try:
            self.init_model()
            state_dicts = torch.load(file_to_load, map_location=self.device)
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

    def eval_init(self, n_samples, evaluation_batch_size):
        self.model.eval()
        eval_dict = {
            "n_samples": n_samples,
            "output_file": os.path.join(self.path_to_guesses_dir, "total_guesses.gz"),
        }
        return eval_dict

    def sample(self, evaluation_batch_size, eval_dict) -> Generator[str,None,None]:
        lower_probability_threshold: float = get_lower_probability_threshold(
            eval_dict["n_samples"]
        )
        guesser = Guesser(
            model=self.model,
            params=self.params,
            data=self.data,
            lower_probability_threshold=lower_probability_threshold,
            output_file=eval_dict["output_file"],
            device=self.device,
        )
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
                if len(min_heap_n_most_prob) < eval_dict['n_samples']:
                    heapcy.heappush(min_heap_n_most_prob, prob, offset)
                else:
                    heapcy.heappushpop(min_heap_n_most_prob,prob,offset)

        offsets: list[int] = []
        
        print("[I] - Getting nlargest")
        for x in heapcy.nlargest(min_heap_n_most_prob, eval_dict["n_samples"]):
            offsets.append(x[1])

        del min_heap_n_most_prob

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
