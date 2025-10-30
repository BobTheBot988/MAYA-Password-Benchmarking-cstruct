import os
import csv
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from numpy.typing import NDArray

from models.ToyModel.ToyModel import make_zipf


def _true_ranks(p: NDArray) -> Tuple[NDArray, dict]:
    order = np.argsort(-p)
    rank = np.empty_like(order)
    last_p, r = None, -1
    for idx in order:
        if last_p is None or p[idx] < last_p:
            r += 1
            last_p = p[idx]
        rank[idx] = r
    # index -> strict rank; also build map "pw{i}" -> rank
    rank_map = {f"pw{i}": int(rank[i]) for i in range(len(p))}
    return rank.astype(int), rank_map


def _build_mc_curve(p: NDArray, n: int, seed: int = 0):
    """
    Sample once from p; return A (probs desc) and C (cum (1/n) * sum 1/A).
    """
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(p), size=int(n), replace=True, p=p)
    probs = p[idx]
    probs.sort()
    A = probs[::-1]  # descending
    C = np.cumsum(1.0 / np.clip(A, 1e-300, None)) / float(n)
    return A, C


def _mc_rank_from_curve(A: NDArray, C: NDArray, p_val: float) -> float:
    j = np.searchsorted(A, p_val, side="right") - 1
    return 0.0 if j < 0 else float(C[j])


def _ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def main(settings: dict) -> Dict[str, List[str]]:
    """
    Your Tester calls this and expects:
      - the script to write CSVs
      - return {csv_path: [row_as_csv_string, ...]}
    'settings' may be nested; normalize it below.
    """
    # ---------- read knobs ----------
    if (
        isinstance(settings, dict)
        and len(settings) == 1
        and isinstance(next(iter(settings.values())), dict)
    ):
        cfg = next(iter(settings.values()))
    else:
        cfg = settings

    n_universe = int(cfg.get("toy_n", 10_000))
    zipf_s = float(cfg.get("toy_s", 1.15))
    seed = int(cfg.get("seed", 0))
    ns_str = cfg.get("n_samples", [1_000, 3_000, 10_000, 30_000])
    if isinstance(ns_str, int):
        ns = [ns_str]
    else:
        ns = list(map(int, ns_str))

    # ---------- build toy dist ----------
    p = make_zipf(n=n_universe, s=zipf_s)
    ranks_true, rank_map = _true_ranks(p)

    # ---------- (A) MC accuracy vs true rank ----------
    rng = np.random.default_rng(seed)
    T = min(200, n_universe)
    targets = rng.choice(n_universe, size=T, replace=False)

    out_dir = os.path.join("results", "bench_toy")
    _ensure_dir(out_dir)

    acc_csv = os.path.join(out_dir, "mc_accuracy.csv")
    with open(acc_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n", "median_abs_err", "median_rel_err"])
        for n in ns:
            abs_err, rel_err = [], []
            for _ in range(20):
                A, C = _build_mc_curve(p, n=n, seed=seed)
                for i in targets:
                    est = _mc_rank_from_curve(A, C, float(p[i]))
                    tru = float(ranks_true[i])
                    ae = abs(est - tru)
                    re = ae / max(1.0, tru)
                    abs_err.append(ae)
                    rel_err.append(re)
            w.writerow([n, float(np.median(abs_err)), float(np.median(rel_err))])

    # ---------- (B) Top-K overlap ----------
    def topk_truth(K: int) -> set:
        return set(np.argsort(-p)[:K].tolist())

    def topk_from_sample(n_samples: int, K: int) -> set:
        idx = np.random.default_rng(seed).choice(
            n_universe, size=n_samples, replace=True, p=p
        )
        uniq = np.unique(idx)
        order = uniq[np.argsort(-p[uniq])]
        return set(order[:K].tolist())

    topk_csv = os.path.join(out_dir, "topk_overlap.csv")
    with open(topk_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["K", "recall_at_K", "n_samples"])
        for K in (100, 1_000, 10_000):
            n_samp = max(ns)
            rec = len(topk_truth(K) & topk_from_sample(n_samp, K)) / float(K)
            w.writerow([K, float(rec), n_samp])

    # ---------- return rows for Tester (optional but nice) ----------
    rows_map: Dict[str, List[str]] = {acc_csv: [], topk_csv: []}
    with open(acc_csv, newline="") as f:
        rows_map[acc_csv] = [line.rstrip("\n") for line in f]
    with open(topk_csv, newline="") as f:
        rows_map[topk_csv] = [line.rstrip("\n") for line in f]
    return rows_map
