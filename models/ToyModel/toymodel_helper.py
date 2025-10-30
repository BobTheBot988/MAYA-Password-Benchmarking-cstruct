import numpy as np
from numpy._core.numerictypes import float64, int64
from numpy.typing import NDArray


def make_zipf(n: int = 10_000, s: float = 1.15) -> NDArray[float64]:
    i: NDArray[float64] = np.arange(1, n + 1, dtype=float)
    p: NDArray[float64] = np.pow(i, -s)
    p /= p.sum()
    return p


class ToySampler:
    def __init__(self, p: NDArray[float64], seed: int = 0) -> None:
        self.p = np.asarray(p, dtype=float64)
        self.n = len(p)
        self.rng = np.random.default_rng(seed)

    def sample(self, m: int) -> list[int]:
        idx: NDArray[int64] = self.rng.choice(self.n, size=m, replace=True, p=self.p)
        return idx.tolist()

    def prob_by_index(self, idx: int) -> float:
        return float(self.p[idx])

    def prob_by_string(self, s: str) -> float:
        # Encode toy passwords as "pw{index}", e.g. pw42
        if not s.startswith("pw"):
            return 0.0
        i: int = int(s[2:])
        if 0 <= i < self.n:
            return float(self.p[i])
        return 0.0
