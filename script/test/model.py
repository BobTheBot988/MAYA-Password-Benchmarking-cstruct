import glob
import numpy as np
import gzip
import math
import os
import pickle
import shutil
import time
from datetime import timedelta
from typing import Dict, Generator, Iterable, Literal, Optional

from enum import Enum
import torch
from torch.types import Tensor
from tqdm import tqdm

from script.config.config import read_config, T
from script.utils.fast_eval import check_skip_generation, fast_eval, sub_sample
from script.utils.file_operations import redirect_stderr, redirect_stdout, write_to_csv
from script.utils.memory_watcher import MemoryWatcher


class SampleMode(Enum):
    TOPK = 0
    IID = 1


class Model:
    T = T

    def __init__(self, s):
        self.settings = s

        self._setup_device()
        self._parse_settings()
        self._prepare_paths()
        self._setup_logging()

        self.memory_watcher = MemoryWatcher(
            output_file=self.path_to_output_file, device=self.device
        )

        # Dictionary containing the model parameters loaded from the .yaml config file.
        self.params = read_config(self.path_to_config_file)

        train_passwords = read_dataset(self.path_to_train_dataset)
        test_passwords = read_dataset(self.path_to_test_dataset)

        self.data = self.prepare_data(train_passwords, test_passwords, self.max_length)

        self._setup_checkpoint()

        status = self._run_embedding()

        self.mode = (
            SampleMode.IID
            if (not status) and (self.estimate_pwd != "None")
            else SampleMode.TOPK
        )
        self.log_2 = False
        self._prepare_sample_paths()
        if (not status) and (self.estimate_pwd != "None"):
            print(
                f"[I] - Started the estimation of the given password:{self.estimate_pwd}"
            )
            self._run_training_and_estim()
            return
        elif not status:
            self._run_training_and_eval()

    def _parse_settings(self):
        # --- General settings ---
        self.autoload = int(self.settings["autoload"])
        self.estimate_pwd: str = str(self.settings["estimate_pwd"])
        self.path_to_checkpoint = self.settings["path_to_checkpoint"]
        self.overwrite = int(self.settings["overwrite"])
        self.display_logs = int(self.settings["display_logs"])
        self.save_guesses = int(self.settings["save_guesses"])
        self.save_matches = int(self.settings["save_matches"])
        self.save_samples = int(self.settings["save_samples"])

        # --- Dataset related settings ---
        self.train_hash = self.settings["train_hash"]
        self.test_hash = self.settings["test_hash"]
        self.max_length = int(self.settings["max_length"])
        self.n_samples = max(self.settings["n_samples"])
        self.thresholds = sorted(
            [s for s in self.settings["n_samples"] if s != self.n_samples]
        )

        # --- Model related settings ---
        self.model_name = str(self.settings["model_name"])

        self.keep_uniques = False

    def _prepare_paths(self):
        # --- Dataset related paths ---
        self.path_to_train_dataset = self.settings["train_path"]
        self.path_to_test_dataset = self.settings["test_path"]

        # --- Model related paths ---
        self.path_to_config_file = self.settings["config_file"]
        self.path_to_checkpoint_dir = os.path.join(
            "checkpoints", self.model_name, self.train_hash
        )
        os.makedirs(self.path_to_checkpoint_dir, exist_ok=True)

        # --- Output related paths ---
        self.path_to_results_dir = os.path.join(
            self.settings["output_path"], self.settings["test_hash"]
        )
        os.makedirs(self.path_to_results_dir, exist_ok=True)

        self.path_to_output_file = os.path.join(self.path_to_results_dir, "log.out")
        self.path_to_error_file = os.path.join(self.path_to_results_dir, "log.err")
        self.path_to_guesses_dir = os.path.join(self.path_to_results_dir, "guesses")
        self.path_to_guesses_file = os.path.join(self.path_to_guesses_dir, "guesses.gz")

        self.path_to_matches_dir = os.path.join(self.path_to_results_dir, "matches")
        self.path_to_matches_file = os.path.join(self.path_to_matches_dir, "matches.gz")

    def _prepare_sample_paths(self):
        self.path_to_samples_dir = os.path.join(self.path_to_results_dir, "samples")
        os.makedirs(self.path_to_samples_dir, exist_ok=True)
        self.path_to_samples_file = os.path.join(
            self.path_to_samples_dir,
            "samples.topk.gz" if self.mode is SampleMode.TOPK else "samples.iid.gz",
        )

    def _setup_logging(self):
        self.written_rows = {}
        # --- Redirect stderr ---
        if not self.display_logs:
            print(f"Redirecting stderr to {self.path_to_error_file}.")
            redirect_stderr(self.path_to_error_file)

        # --- Redirect stdout ---
        print(f"Redirecting stdout to {self.path_to_output_file}")
        redirect_stdout(self.path_to_output_file)
        print("-" * 40)

    def _setup_device(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Selected device: {self.device}.")

    def _setup_checkpoint(self):
        next_id, latest_checkpoint = get_checkpoint_id(self.path_to_checkpoint_dir)
        self.checkpoint_name = f"checkpoint{next_id}.pt"

        if self.path_to_checkpoint:
            self.checkpoint_name = use_specified_checkpoint(
                self.path_to_checkpoint,
                self.path_to_checkpoint_dir,
                self.checkpoint_name,
            )
            print(f"[I] - Using checkpoint: {self.path_to_checkpoint}")
        elif self.autoload and latest_checkpoint:
            self.checkpoint_name = latest_checkpoint
            print(f"[I] - Autoloading latest checkpoint: {latest_checkpoint}")
        else:
            print("[I] - No specific checkpoint or autoload passed.")
            print("[I] - No specific checkpoint or autoload passed.")

    def _run_fast_eval(self):
        n_samples_to_evaluate = sorted(self.settings.get("n_samples"))

        if not self.overwrite:
            if os.path.isfile(self.path_to_guesses_file):
                output = fast_eval(
                    self.path_to_test_dataset,
                    n_samples_to_evaluate,
                    self.path_to_guesses_file,
                )
                self.save_stats(output)
                return True

        sub_samples_from_file = str(self.settings.get("sub_samples_from_file", False))
        guesses_file = self.settings.get("guesses_file", False)

        sub_samples_from_file = check_skip_generation(sub_samples_from_file)
        guesses_file = check_skip_generation(guesses_file)

        if sub_samples_from_file:
            sub_sample(sub_samples_from_file, n_samples_to_evaluate)

        if guesses_file:
            output = fast_eval(
                self.path_to_test_dataset, n_samples_to_evaluate, guesses_file
            )
            self.save_stats(output)

        return sub_samples_from_file or guesses_file

    def _prepare_directories(self):
        if self.save_guesses:
            _create_and_clean_dir(self.path_to_guesses_dir)
        if self.save_matches:
            _create_and_clean_dir(self.path_to_matches_dir)
        if self.save_samples:
            _create_and_clean_dir(self.path_to_samples_dir)

    def _run_training_and_eval(self) -> None:
        self._prepare_directories()

        self.start_train(self.checkpoint_name)

        matches, match_percentage, test_size = self.start_eval(self.checkpoint_name)

        output = [[test_size, self.n_samples, matches, match_percentage]]
        self.save_stats(output)

        if len(self.thresholds) > 0:
            output = fast_eval(
                self.path_to_test_dataset, self.thresholds, self.path_to_guesses_file
            )
            self.save_stats(output)

    def _run_training_and_estim(self) -> None:
        self.start_train(self.checkpoint_name)

        rank = self.start_estimation(self.checkpoint_name)

        print(f"[I] - The rank of the password({self.estimate_pwd}) was:{rank}")

    def save_stats(self, output):
        if output:
            fieldnames = [
                "model",
                "train-dataset",
                "test-settings",
                "test-hash",
                "test-size",
                "n_samples",
                "matches",
                "match_percentage",
            ]

            infos = self.settings["output_path"].split("/")
            csv_path = os.path.join(infos[0], infos[1], f"{infos[1]}.csv")
            model_name = infos[2]
            if "-" in model_name:
                model_name = model_name.replace("-", "")
            fixed_values = [model_name, infos[3], infos[4], self.test_hash]

            rows = write_to_csv(
                csv_path,
                fieldnames=fieldnames,
                fixed_data=fixed_values,
                variable_data=output,
            )
            if csv_path not in self.written_rows:
                self.written_rows[csv_path] = []
            for row in rows:
                self.written_rows[csv_path].append(row)

    def plot_embedding(self, data, max_length):
        # you can skip implementing this
        raise NotImplementedError("This method should be implemented in the subclass.")

    def save(self, obj, mid=True):
        f_name = self.checkpoint_name if not mid else f"mid-{self.checkpoint_name}"
        save_path = os.path.join(self.path_to_checkpoint_dir, f_name)
        torch.save(obj, save_path)

    def finalize_checkpoint(self):
        source_path = os.path.join(
            self.path_to_checkpoint_dir, "mid-" + self.checkpoint_name
        )
        if os.path.isfile(source_path):
            output_path = os.path.join(
                self.path_to_checkpoint_dir, self.checkpoint_name
            )
            os.rename(source_path, output_path)

    def get_generator_for_sample_file(self) -> Generator[float, None, None]:
        """

        This method should be used to take the probability of a string from the distribution generated on the training dataset.
        This works for explicit models, for implicit models the dev has to make it possible, there are many ways to do so.

        The returned object must include at least the following attributes:
            - train_passwords (list): A list of training passwords.
            - test_passwords (set): A set of test passwords.

        Additionally, the object MUST implement the following methods:

        Parameters:
            - self (Model): The model instance. You can access all variables and methods defined in this class, including
              self.data (the object returned by prepare_data) and self.params (the configuration parameters).

            - string (str): string to get our probability.
        Returns:
             - the probability of the string to be generated by the model
        """

        with gzip.open(self.path_to_samples_file, "rb") as fd:
            while True:
                line: bytes = fd.readline()

                if not line:
                    break

                line_str: list[str] = (
                    line.rstrip(b"\r\n")
                    .decode(encoding="ascii", errors="replace")
                    .split(" ")
                )
                if len(line_str) != 2:
                    raise IndexError(f"The line should be composed of two parts:{line}")
                fl: float = float(line_str[1])

                yield fl

    def post_iid_sampling(self):
        if not self.save_samples and os.path.exists(self.path_to_samples_file):
            os.remove(self.path_to_samples_file)

    def generate_one_time_pwds(self) -> Iterable[float]:
        """
        **TO BE IMPLEMENTED BY SUBCLASS.**

        This method should be used to take the probability of a string from the distribution generated on the training dataset.
        This works for explicit models, for implicit models the dev has to make it possible, there are many ways to do so.

        The returned object must include at least the following attributes:
            - train_passwords (list): A list of training passwords.
            - test_passwords (set): A set of test passwords.

        Additionally, the object MUST implement the following methods:

        Parameters:
                - self (Model): The model instance. You can access all variables and methods defined in this class, including
                self.data (the object returned by prepare_data) and self.params (the configuration parameters).
            - string (str): string to get our probability.
        Returns:
             - the probability of the string to be generated by the model
        """
        raise NotImplementedError(
            "This method should be implemented by the subclass, if the model is explicit it's easy.\nOtherwise it can be difficult, consult the documentation."
        )

    def get_string_probability(self) -> float:
        """
        **TO BE IMPLEMENTED BY SUBCLASS.**

        This method should be used to take the probability of a string from the distribution generated on the training dataset.
        This works for explicit models, for implicit models the dev has to make it possible, there are many ways to do so.

        The returned object must include at least the following attributes:
            - train_passwords (list): A list of training passwords.
            - test_passwords (set): A set of test passwords.

        Additionally, the object MUST implement the following methods:

        Parameters:
                - self (Model): The model instance. You can access all variables and methods defined in this class, including
                self.data (the object returned by prepare_data) and self.params (the configuration parameters).
            - string (str): string to get our probability.
        Returns:
             - the probability of the string to be generated by the model
        """
        raise NotImplementedError(
            "This method should be implemented by the subclass, if the model is explicit it's easy.\nOtherwise it can be difficult, consult the documentation."
        )

    def error_range(self, target: float) -> float:
        return np.exp2(target - 1) / np.sqrt(self.n_samples)

    def montecarlo_estimation(self, gen: Optional[Iterable[float]]) -> float:
        """
        This method should be used to  get the rank of a string for the selected model.
        Parameters:
                - self (Model): The model instance. You can access all variables and methods defined in this class, including
                self.n_samples(the number of samples) and self.estimate_pwd(the string to estimate).
            - string (str): string to get our probability.
        Returns:
             - The rank of the string

        target: p(alpha)
        returns: (1/n) * sum_{i: A[i] > target} 1/A[i]
        """
        print("[I] - Running montecarlo_estimation")
        assert self.n_samples is not None and isinstance(self.n_samples, int)
        if self.n_samples < 1_000:
            raise ValueError(
                "The number of samples for the montecarlo_estimation should not be lowert than 10**3."
            )
        elif self.n_samples > 100_000:
            print("[W] - The generation could be slower")

        generator: Generator[float, None, None] | Iterable[float] = (
            self.get_generator_for_sample_file() if not gen else gen
        )
        print("[I] - Getting the target probability")
        target: float = self.get_string_probability()
        print(f"[I] - Target probability:{target}")
        print(f"[I] - Error range:{self.error_range(np.log2(target))}")
        target_tensor: torch.Tensor = torch.scalar_tensor(target, device=self.device)
        # The probability should be in the -log_2 form
        target_tensor = -torch.log2(target_tensor) if self.log_2 else target_tensor
        my_sum_tensor: torch.Tensor = torch.scalar_tensor(
            0.0, dtype=torch.float64, device=self.device
        )
        size_of_array: int = 0
        chunk: list[float] = list()
        chunk_size: int = 10_000
        for prob in generator:
            chunk.append(prob)
            size_of_array += 1
            if len(chunk) == chunk_size:
                gen_tensor: Tensor = torch.tensor(chunk, device=self.device)
                my_sum_tensor += _logic_of_log(self.log_2, gen_tensor, target_tensor)
                chunk.clear()

        if len(chunk) > 0:
            gen_tensor = torch.tensor(chunk, device=self.device)
            my_sum_tensor += _logic_of_log(self.log_2, gen_tensor, target_tensor)
            size_of_array += len(chunk) if size_of_array > chunk_size else 0
            chunk.clear()

        if size_of_array != self.n_samples:
            raise ValueError(
                f"[E] - The number of generated elements is different than the requested amount:{size_of_array},{self.n_samples}"
            )
        result: float = my_sum_tensor.item() / self.n_samples

        assert result >= 0
        return result

    def enumeration_of_pwd(self) -> int:
        """
        This method should be used to  get the rank of a string for the selected model.
        Parameters:
                - self (Model): The model instance. You can access all variables and methods defined in this class, including
                self.n_samples(the number of samples) and self.estimate_pwd(the string to estimate).
            - string (str): string to get our probability.
        Returns:
             - The rank of the string

        target: p(alpha)
        returns: (1/n) * sum_{i: A[i] > target} 1/A[i]
        """
        print("[I] - Running montecarlo_estimation")
        assert self.n_samples is not None and isinstance(self.n_samples, int)
        if self.n_samples > 100_000:
            raise ValueError(
                "The number of samples for the montecarlo_estimation should not be higher than 10**5."
            )
        elif self.n_samples < 1_000:
            raise ValueError(
                "The number of samples for the montecarlo_estimation should not be lowert than 10**3."
            )

        generator: Iterable[float] = self.get_generator_for_sample_file()
        print("[I] - Getting the target probability")
        # The probability should be in the -log_2 form
        target: float = self.get_string_probability()

        my_sum: float = 0
        for prob in generator:
            if prob >= target:
                break
            my_sum += 1

        assert my_sum >= 0
        return my_sum

    def start_train(self, checkpoint_name):
        if checkpoint_name:
            print("[I] - Train mode selected. Searching for a checkpoint...")
            file_to_load = os.path.join(
                self.path_to_checkpoint_dir, (self.checkpoint_name)
            )
            status = self.load(file_to_load)
            if status:
                print(
                    "[I] - Final checkpoint loaded successfully. Training already finished :)."
                )
                return
            else:
                print("[I] - No checkpoints found. Proceeding with normal training.")

        else:
            print("[I] - Checkpoint not specified. Starting training from scratch.")

        for _ in range(5):
            self.memory_watcher.reset()
            self.memory_watcher.start()
            train_start = time.time()

            self.train()
            self.finalize_checkpoint()

            train_end = time.time()
            time_delta = timedelta(seconds=train_end - train_start)
            print(f"[T] - Training completed after: {time_delta}")
            self.memory_watcher.stop()

    def start_eval(self, checkpoint_name):
        print("[I] - Searching for a checkpoint for evaluation...")

        file_to_load = os.path.join(self.path_to_checkpoint_dir, self.checkpoint_name)
        status = self.load(file_to_load)

        if not status:
            print("[I] - No checkpoint found. Starting the training model normally.")
            self.start_train(checkpoint_name)
            status = self.load(file_to_load)

        print("[I] - Checkpoint loaded successfully. Initiating model evaluation.")

        self.memory_watcher.reset()
        self.memory_watcher.start()
        eval_start = time.time()

        matches, match_percentage, test_size = self.evaluate(self.n_samples)

        eval_end = time.time()
        time_delta = timedelta(seconds=eval_end - eval_start)
        print(f"[T] - Evaluation completed after: {time_delta}")
        self.memory_watcher.stop()
        return matches, match_percentage, test_size

    def start_estimation(self, checkpoint_name: str) -> float:
        print("[I] - Searching for a checkpoint for estimation...")

        file_to_load = os.path.join(self.path_to_checkpoint_dir, self.checkpoint_name)
        status = self.load(file_to_load)

        if not status:
            print("[I] - No checkpoint found. Starting the training model normally.")
            self.start_train(checkpoint_name)
            status = self.load(file_to_load)

        print("[I] - Checkpoint loaded successfully. Initiating model estimation.")

        self.memory_watcher.reset()
        self.memory_watcher.start()
        estim_start = time.time()

        file_to_load = self.path_to_samples_file
        s = (
            "[I] - The sample file exists"
            if os.path.exists(file_to_load)
            else "[I] - The sample file does not exist"
        )

        print(s)
        gen: Iterable[float] = list()
        if True:
            # if self.generate_one_time:
            gen = self.generate_one_time_pwds()
        elif not os.path.exists(file_to_load):
            gen = self.iid_evaluation()
        else:
            gen = self.get_generator_for_sample_file()
        rank: float = self.montecarlo_estimation(gen)

        estim_end = time.time()
        time_delta = timedelta(seconds=estim_end - estim_start)
        print(f"[T] - Estimation completed after: {time_delta}")
        self.memory_watcher.stop()

        return rank

    def _run_embedding(self):
        if self.settings["data_to_embed"]:
            try:
                file_to_load = os.path.join(
                    self.path_to_checkpoint_dir, self.checkpoint_name
                )
                self.load(file_to_load)
                self.plot_embedding(self.settings["data_to_embed"], self.max_length)
                return True
            except NotImplementedError:
                print(
                    "[W] - plot_embedding method not implemented by the subclass. Skipping embedding."
                )
            except Exception as e:
                print(f"[E] - Error during embedding: {e}")
        return False

    def write_to_file(self, file, generated_data: list[str] | set[str]):
        with gzip.open(file, "at") as f:
            for password in generated_data:
                decoded_password = self.data.decode_password(password)
                if decoded_password is None:
                    continue
                password = self.data.remove_padding(decoded_password)
                f.write(password + "\n")

    def prepare_data(self, train_passwords, test_passwords, max_length):
        """
        **TO BE IMPLEMENTED BY SUBCLASS.**

        This method should be used to create an object that manages and processes the dataset.

        The returned object must include at least the following attributes:
            - train_passwords (list): A list of training passwords.
            - test_passwords (set): A set of test passwords.

        Additionally, the object MUST implement the following methods:
                - encode_password(password): Takes a password string and returns its tokenized representation.
                - decode_password(password): Takes a tokenized password and returns the corresponding string after detokenization.
            - remove_padding(password): Takes a padded password string and returns the same string with all padding tokens removed.

        Parameters:
                - self (Model): The model instance. You can access all variables and methods defined in this class, including
                self.data (the object returned by prepare_data) and self.params (the configuration parameters).
            - train_passwords (list): The list of passwords used for training.
            - test_passwords (list): The list of passwords used for testing.
            - max_length (int): The maximum allowed password length.
        Returns:
             - An object containing the required attributes and methods, which will be later accessible via self.data.
        """
        raise NotImplementedError("This method should be implemented in the subclass.")

    def load(self, file_name):
        """
        **TO BE IMPLEMENTED BY SUBCLASS.**

        This method should load the model's state from the specified checkpoint file.

        Parameters:
                - self (Model): The model instance. You can access all variables and methods defined in this class, including
                self.data (the object returned by prepare_data) and self.params (the configuration parameters).
            - file_name (str): Path to the checkpoint file.
        Returns:
            - int: Returns 1 if the model was successfully loaded, 0 otherwise.
        """
        raise NotImplementedError("This method should be implemented in the subclass.")

    def train(self):
        """
        **TO BE IMPLEMENTED BY SUBCLASS.**

        This method should train your model and save its state to a checkpoint file.

        To save a checkpoint, use the `self.save()` method from the base class by passing a dictionary containing
        all relevant model and optimizer states. For example:

        obj = {
            'generator_opt': self.generator_opt.state_dict(),
            'discriminator_opt': self.discriminator_opt.state_dict(),
            'Generator': self.Generator.state_dict(),
            'Discriminator': self.Discriminator.state_dict(),
        }
        self.save(obj)

        Parameters:
                - self (Model): The model instance. You can access all variables and methods defined in this class, including
                self.data (the object returned by prepare_data) and self.params (the configuration parameters).
        Returns:
            - None
        """
        raise NotImplementedError("This method should be implemented in the subclass.")

    def helper_write_to_file(self, save_every: int, save_guesses: bool | Literal[0]):
        if save_guesses and len(self.guesses) >= save_every:
            if not self.keep_uniques:
                self.write_to_file(self.path_to_guesses_file, self.guesses)
                self.guesses = []

    def iid_evaluation(self, *, validation_mode: bool = False) -> Iterable[float]:
        """
        Generate exactly self.n_samples i.i.d. samples and (optionally) write them to gzip.
        Returns the # of samples produced.
        """
        batch = int(self.params["eval"]["evaluation_batch_size"])
        n_full, rem = divmod(self.n_samples, batch)
        total_batches = n_full + (1 if rem else 0)

        # Init eval dict and ensure we're in IID mode
        eval_dict = self.eval_init(self.n_samples, batch)

        count = 0
        # Only open the file if we actually want to save
        f = (
            gzip.open(self.path_to_samples_file, "wt", compresslevel=1)
            if self.save_samples
            else None
        )
        try:
            with tqdm(total=self.n_samples, desc="Generating i.i.d. samples") as pbar:
                for b_ix in range(total_batches):
                    b = batch if b_ix < n_full else rem if rem else batch
                    if b == 0:  # possible when n is multiple of batch
                        continue

                    generated: (
                        list[str]
                        | Generator[str, None, None]
                        | Generator[tuple[str, float], None, None]
                    ) = self.sample(b, eval_dict)

                    if self.save_samples:
                        buf = []
                        for pwd, ell in generated:
                            buf.append(f"{pwd} {ell}\n")
                            count += 1
                        assert f is not None
                        f.writelines(buf)
                    else:
                        for _pwd, _ell in generated:
                            _ell = float(_ell)
                            yield _ell
                            count += 1

                    pbar.update(b)

                pbar.set_postfix(
                    {"sampled_%": round(100.0 * count / self.n_samples, 2)}
                )
        finally:
            if f is not None:
                f.close()

        if count != self.n_samples:
            # Make it loud if your generator under/over-produced
            raise RuntimeError(f"Expected {self.n_samples} samples, got {count}")

        self.post_iid_sampling()

    def evaluate(
        self, n_samples, evaluation_batch_sizedation_mode=False, validation_mode=False
    ) -> tuple[int, str, int]:
        print(f"Generating {n_samples} passwords...")
        save_every = 1000000
        save_guesses = self.save_guesses and not validation_mode
        save_matches = self.save_matches and not validation_mode

        evaluation_batch_size = int(self.params["eval"]["evaluation_batch_size"])
        if n_samples < evaluation_batch_size:
            n_batches, evaluation_batch_size = 1, int(n_samples)
        else:
            n_batches = math.floor(n_samples / evaluation_batch_size)

        eval_dict = self.eval_init(n_samples, evaluation_batch_size)

        progress_bar = tqdm(range(n_batches))
        progress_bar.set_description(desc="Generating sample batch")

        self.guesses: list[str] = []
        self.matches: set[str] = set()

        for _ in range(n_batches):
            generated_passwords: (
                list[str]
                | Generator[str, None, None]
                | Generator[tuple[str, float], None, None]
            ) = self.sample(evaluation_batch_size, eval_dict)
            if n_batches == 1:
                for sample in generated_passwords:
                    assert isinstance(sample, str)
                    self.guesses.append(sample)
                    if sample in self.data.test_passwords:
                        self.matches.add(sample)

                    self.guessing_strategy(evaluation_batch_size, eval_dict)
                    self.helper_write_to_file(save_every, save_guesses)

            else:
                assert isinstance(generated_passwords, list)
                self.guesses.extend(generated_passwords)
                self.matches.update(generated_passwords & self.data.test_passwords)

                self.guessing_strategy(evaluation_batch_size, eval_dict)

                self.helper_write_to_file(save_every, save_guesses)

            progress_bar.set_postfix(
                {
                    "Matches found": {len(self.matches)},
                    "Test set %": (
                        {len(self.matches) / len(self.data.test_passwords) * 100.0}
                    ),
                }
            )
            progress_bar.update(1)

        self.post_sampling(eval_dict)

        if save_guesses and len(self.guesses) > 0:
            self.write_to_file(self.path_to_guesses_file, self.guesses)

        if save_matches and len(self.matches) > 0:
            self.write_to_file(self.path_to_matches_file, self.matches)

        n_matches = len(self.matches)
        test_size = len(self.data.test_passwords)
        match_percentage = f"{(n_matches / test_size) * 100:.2f}%"
        print(f"{n_matches} matches found ({match_percentage} of test set).")
        return n_matches, match_percentage, test_size

    def eval_init(self, n_samples, evaluation_batch_size) -> Dict[str, T]:
        """
        **TO BE IMPLEMENTED BY SUBCLASS.**

        This method should initialize all variables required for the evaluation process, according to your model's needs.

        For example, if you require certain variables later during the evaluation, you can initialize them here and
        return them in a dictionary.

        Parameters:
                - self (Model): The model instance. You can access all variables and methods defined in this class, including
                self.data (the object returned by prepare_data) and self.params (the configuration parameters).
            - n_samples (int): The number of passwords to generate.
            - evaluation_batch_size (int): The batch size to use during evaluation.
        Returns:
            - eval_dict (dict): A dictionary containing all initialized resources.
        """
        raise NotImplementedError("This method should be implemented in the subclass.")

    def sample(
        self,
        evaluation_batch_size,
        eval_dict,
    ) -> (
        list[str]
        | Generator[str, None, None]
        | Generator[tuple[str, float], None, None]
    ):
        """
        **TO BE IMPLEMENTED BY SUBCLASS.**

        This method should generate and return 'evaluation_batch_size' passwords.

        Make sure that the returned passwords follow the same format as the test passwords defined in prepare_data;
        otherwise, no matches will be found during evaluation.
        Parameters:
                - self (Model): The model instance. You can access all variables and methods defined in this class, including
                self.data (the object returned by prepare_data) and self.params (the configuration parameters).
            - evaluation_batch_size (int): Number of passwords to generate in this batch.
            - eval_dict (dict): Dictionary returned in `self.eval_init`.
        Returns:
            - generated_passwords (list): A list of generated passwords, matching the format of the test passwords defined in
             prepare_data.
            -generated_passwords (Generator[str]): a generator for the passwords, use this if low on memory
        """
        raise NotImplementedError("This method should be implemented in the subclass.")

    def guessing_strategy(self, evaluation_batch_size, eval_dict):
        """
        **TO BE IMPLEMENTED BY SUBCLASS.**

        This method should implement your custom guessing strategy (see PassFlow with dynamic guessing as a reference).

        Implement this if you want to define a dynamic guessing strategy, that changes over time (e.g. changes in prior).

        Leave this method blank if your model does not require a specific guessing strategy.

        Parameters:
                - self (Model): The model instance. You can access all variables and methods defined in this class, including
                self.data (the object returned by prepare_data) and self.params (the configuration parameters).
            - evaluation_batch_size (int): Number of passwords generated in the current batch.
            - eval_dict (dict): Dictionary returned by `self.eval_init`.
        Returns:
            - None
        """
        raise NotImplementedError("This method should be implemented in the subclass.")

    def post_sampling(self, eval_dict):
        """
        **TO BE IMPLEMENTED BY SUBCLASS.**

        This method should handle any post-generation logic, such as cleaning up temporary files used during evaluation,
        resetting variables, or releasing resources.

        Parameters:
                - self (Model): The model instance. You can access all variables and methods defined in this class, including
                self.data (the object returned by prepare_data) and self.params (the configuration parameters).
            - eval_dict (dict): Dictionary returned by `self.eval_init`.
        Returns:
            - None
        """
        raise NotImplementedError("This method should be implemented in the subclass.")


def read_dataset(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    data = data.split("\n")
    return data


def get_checkpoint_id(path):
    next_id = 1
    checkpoints = []

    for filename in os.listdir(path):
        if filename.startswith("checkpoint") and filename.endswith(".pt"):
            file_path = os.path.join(path, filename)
            if not os.path.isfile(file_path):
                continue

            id_str = filename[len("checkpoint") : -3]  # remove prefix and '.pt'
            try:
                checkpoint_id = int(id_str) if id_str else 0
                checkpoints.append((filename, checkpoint_id))
                next_id = max(next_id, checkpoint_id + 1)
            except ValueError:
                continue  # Skip malformed IDs

    # Sort checkpoints by ID descending
    checkpoints.sort(key=lambda x: x[1], reverse=True)

    # Return next available ID and latest checkpoint filename (or None)
    latest_checkpoint = checkpoints[0][0] if checkpoints else None
    return next_id, latest_checkpoint


def use_specified_checkpoint(source_path, target_dir, file_name):
    if not os.path.exists(source_path):
        raise FileNotFoundError(f"Checkpoint file not found: {source_path}")

    dest_path = os.path.join(target_dir, file_name)

    if os.path.abspath(os.path.dirname(source_path)) != os.path.abspath(target_dir):
        shutil.copyfile(source_path, dest_path)
        return file_name
    else:
        return os.path.basename(source_path)


def _create_and_clean_dir(path):
    os.makedirs(path, exist_ok=True)

    for filename in glob.glob(os.path.join(path, "*")):
        os.remove(filename)


def _logic_of_log(log_2: bool, gen_tensor: Tensor, target_tensor: Tensor) -> Tensor:
    if log_2:
        # 1. Convert all sampled probs to log2
        gen_tensor.log2_()

        # 2. Create a mask of all samples more probable than the target
        mask = gen_tensor < target_tensor

        # 3. Get only those elements
        valid_elements = gen_tensor[mask]

        # 4. Convert back to linear space (exp2)
        valid_elements.exp2_()

    else:
        # 1. Create a mask of all samples more probable than the target
        mask = gen_tensor > target_tensor

        # 2. Get only those elements
        valid_elements = gen_tensor[mask]

        # 3. Safely calculate 1 / ell (the ^-1 part)
        valid_elements.pow_(-1)
    return valid_elements.sum()
