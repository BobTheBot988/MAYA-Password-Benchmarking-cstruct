import gzip
from numpy._core.numerictypes import float64
from numpy.typing import NDArray
from .toymodel_helper import make_zipf, ToySampler
from script.test.model import Model  # your base class
from typing import Iterable, Dict


class ToyModel(Model):
    sampler: ToySampler

    def __init__(self, settings):
        n: int = 10_000
        s: float = 1.15
        seed: int = 0
        p: NDArray[float64] = make_zipf(n=n, s=s)
        self.sampler: ToySampler = ToySampler(p, seed=seed)
        super().__init__(settings)

    def load(self, file_name: str):
        return 1

    def train(self):
        pass

    def eval_init(self, n_samples, evaluation_batch_size) -> Dict:
        return {}

    def guessing_strategy(self, evaluation_batch_size, eval_dict):
        pass

    def post_sampling(self, eval_dict):
        pass

    def prepare_data(self, train_passwords, test_passwords, max_length):
        pass

    def generate_one_time_pwds(self) -> Iterable[float]:
        m = int(self.n_samples)
        f = None

        if self.save_samples:
            f = gzip.open(self.path_to_samples_file, "at")

        produced = 0
        batch = 2048
        while produced < m:
            b = min(batch, m - produced)
            idxs = self.sampler.sample(b)
            pwds = [f"pw{i}" for i in idxs]
            probs = [self.sampler.prob_by_index(i) for i in idxs]

            if f:
                for pwd, p in zip(pwds, probs):
                    f.write(f"{pwd} {p}\n")
            for p in probs:
                yield float(p)

            produced += b
        if f:
            f.close()

    def get_string_probability(self) -> float:
        s: str = self.estimate_pwd
        p: float = self.sampler.prob_by_string(s)
        return p
