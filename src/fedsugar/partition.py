"""Client partitioners: IID, Dirichlet label skew, and silos by a feature.

Each partitioner returns one index array per client. The arrays are disjoint
and together they hold every training row.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import CLASSES, FEATURES

SCHEMES = ("iid", "dirichlet", "silo")


def iid(n: int, clients: int, seed: int = 42) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [np.sort(part) for part in np.array_split(rng.permutation(n), clients)]


def dirichlet(y: np.ndarray, clients: int, alpha: float, seed: int = 42, min_size: int = 20, max_tries: int = 100) -> list[np.ndarray]:
    """Label skew: the share of each class on each client comes from Dirichlet(alpha).

    A small alpha gives strong skew. The draw repeats until each client has at least ``min_size`` rows.
    """
    if alpha <= 0:
        raise ValueError("alpha must be > 0")
    y = np.asarray(y)
    if clients * min_size > y.size:
        raise ValueError("min_size x clients is larger than the data")
    rng = np.random.default_rng(seed)
    for _ in range(max_tries):
        parts: list[list[int]] = [[] for _ in range(clients)]
        for cls in np.unique(y):
            members = rng.permutation(np.flatnonzero(y == cls))
            shares = rng.dirichlet(np.full(clients, alpha))
            cuts = (np.cumsum(shares)[:-1] * members.size).astype(int)
            for client, chunk in enumerate(np.split(members, cuts)):
                parts[client].extend(chunk.tolist())
        if min(len(p) for p in parts) >= min_size:
            return [np.sort(np.array(p, dtype=int)) for p in parts]
    raise RuntimeError(f"no Dirichlet draw gave {min_size} rows to every client in {max_tries} tries")


def silo(frame: pd.DataFrame, column: str, clients: int) -> list[np.ndarray]:
    """Silos by the quantiles of one feature (for example ``Age``): natural feature skew."""
    if column not in FEATURES:
        raise ValueError(f"unknown column {column!r}")
    ranks = frame[column].rank(method="first").to_numpy()
    bins = np.minimum((ranks - 1) * clients // len(ranks), clients - 1).astype(int)
    return [np.flatnonzero(bins == c) for c in range(clients)]


def make_partition(scheme: str, y: np.ndarray, frame: pd.DataFrame, clients: int, alpha: float = 0.5, column: str = "Age", seed: int = 42) -> list[np.ndarray]:
    if scheme == "iid":
        return iid(len(y), clients, seed)
    if scheme == "dirichlet":
        return dirichlet(y, clients, alpha, seed, min_size=max(5, min(20, len(y) // (4 * clients))))
    if scheme == "silo":
        return silo(frame, column, clients)
    raise ValueError(f"scheme must be one of {SCHEMES}")


def class_table(y: np.ndarray, parts: list[np.ndarray]) -> pd.DataFrame:
    """Rows per class for each client. A 0 shows a client that misses a class."""
    rows = []
    for i, part in enumerate(parts):
        counts = np.bincount(y[part], minlength=len(CLASSES))
        rows.append({"client": i, "rows": int(part.size), **{f"class_{c}": int(counts[c]) for c in CLASSES}})
    return pd.DataFrame(rows)


def label_skew(y: np.ndarray, parts: list[np.ndarray]) -> float:
    """Mean total-variation distance between each client's label mix and the global mix."""
    global_mix = np.bincount(y, minlength=len(CLASSES)) / y.size
    distances = []
    for part in parts:
        mix = np.bincount(y[part], minlength=len(CLASSES)) / max(part.size, 1)
        distances.append(0.5 * np.abs(mix - global_mix).sum())
    return float(np.mean(distances))
