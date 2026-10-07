"""Experiments: one comparison, and a grid of seeds x partitions x strategies x models from a TOML file.

Every run of the grid uses one data split per seed. The centralized model, the
federated model and the baselines see the same train, validation and test rows.
"""

from __future__ import annotations

import itertools
import tomllib
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from .data import Split, split
from .metrics import evaluate, mean_ci, paired_bootstrap_f1
from .partition import class_table, label_skew, make_partition
from .training import Client, TrainConfig, train_centralized, train_federated, train_local_only

GRID_KEYS = ("seeds", "schemes", "alphas", "clients", "strategies", "models")


def make_clients(data: Split, scheme: str, n_clients: int, alpha: float, seed: int) -> tuple[list[Client], pd.DataFrame, float]:
    parts = make_partition(scheme, data.y_train, data.train_frame, n_clients, alpha=alpha, seed=seed)
    clients = [Client(i, data.X_train[p], data.y_train[p]) for i, p in enumerate(parts)]
    return clients, class_table(data.y_train, parts), label_skew(data.y_train, parts)


def hgb_baseline(data: Split, seed: int) -> dict[str, float]:
    """Centralized gradient boosting with balanced class weights: a reference, not a federated model."""
    model = HistGradientBoostingClassifier(class_weight="balanced", random_state=seed, max_iter=200)
    model.fit(data.X_train, data.y_train)
    return evaluate(data.y_test, model.predict_proba(data.X_test))


def compare(frame: pd.DataFrame, cfg: TrainConfig, scheme: str = "dirichlet", n_clients: int = 5, alpha: float = 0.5,
            local_only: bool = True, gradient_boosting: bool = True) -> dict:
    """Centralized vs federated (same model and settings) plus baselines, on one split."""
    data = split(frame, seed=cfg.seed)
    clients, table, skew = make_clients(data, scheme, n_clients, alpha, cfg.seed)
    central = train_centralized(data.X_train, data.y_train, data.X_val, data.y_val, data.X_test, data.y_test, cfg)
    federated = train_federated(clients, data.X_val, data.y_val, data.X_test, data.y_test, cfg)
    rows = [central.row(), federated.row()]
    if local_only:
        rows.append({"setting": "local_only_mean", **train_local_only(clients, data.X_val, data.y_val, data.X_test, data.y_test, cfg)})
    if gradient_boosting:
        rows.append({"setting": "centralized_hgb", **hgb_baseline(data, cfg.seed)})
    diff = paired_bootstrap_f1(data.y_test, federated.test_proba.argmax(1), central.test_proba.argmax(1), seed=cfg.seed)
    return {
        "table": pd.DataFrame(rows),
        "clients": table,
        "label_skew": skew,
        "fed_minus_central_macro_f1": diff,
        "central": central,
        "federated": federated,
        "sizes": data.sizes(),
    }


def load_grid(path) -> dict:
    with open(path, "rb") as handle:
        config = tomllib.load(handle)
    grid = config.get("grid", {})
    missing = [k for k in GRID_KEYS if k not in grid]
    if missing:
        raise ValueError(f"{path}: [grid] needs {missing}")
    return config


def run_grid(frame: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (runs, summary). ``summary`` gives the mean and the 95 % t interval over seeds."""
    grid = config["grid"]
    train = config.get("train", {})
    rows = []
    for seed in grid["seeds"]:
        data = split(frame, seed=seed)
        for scheme, alpha, n_clients in itertools.product(grid["schemes"], grid["alphas"], grid["clients"]):
            if scheme != "dirichlet" and alpha != grid["alphas"][0]:
                continue  # alpha only matters for the Dirichlet partition
            clients, _, skew = make_clients(data, scheme, n_clients, alpha, seed)
            for model in grid["models"]:
                base = TrainConfig(**{**train, "model": model, "seed": seed})
                central = train_centralized(data.X_train, data.y_train, data.X_val, data.y_val, data.X_test, data.y_test, base)
                for strategy in grid["strategies"]:
                    fed = train_federated(clients, data.X_val, data.y_val, data.X_test, data.y_test, replace(base, strategy=strategy))
                    key = {"seed": seed, "scheme": scheme, "alpha": alpha if scheme == "dirichlet" else np.nan,
                           "clients": n_clients, "model": model, "strategy": strategy, "label_skew": skew}
                    rows.append({**key, "setting": "federated", **fed.test, "train_seconds": fed.train_seconds,
                                 "parallel_seconds": fed.extra["parallel_seconds"], "comm_megabytes": fed.extra["comm_megabytes"]})
                    rows.append({**key, "setting": "centralized", **central.test, "train_seconds": central.train_seconds})
    runs = pd.DataFrame(rows)
    return runs, summarize(runs)


def summarize(runs: pd.DataFrame, metric: str = "macro_f1") -> pd.DataFrame:
    keys = ["scheme", "alpha", "clients", "model", "strategy"]
    out = []
    for values, group in runs.groupby(keys, dropna=False, sort=True):
        fed = group[group["setting"] == "federated"].sort_values("seed")
        cen = group[group["setting"] == "centralized"].sort_values("seed")
        f_mean, f_low, f_high = mean_ci(fed[metric])
        c_mean, c_low, c_high = mean_ci(cen[metric])
        d_mean, d_low, d_high = mean_ci(fed[metric].to_numpy() - cen[metric].to_numpy())
        out.append({**dict(zip(keys, values)), "seeds": len(fed), f"fed_{metric}": f_mean, "fed_ci": f"[{f_low:.3f}, {f_high:.3f}]",
                    f"central_{metric}": c_mean, "central_ci": f"[{c_low:.3f}, {c_high:.3f}]",
                    "fed_minus_central": d_mean, "diff_ci_low": d_low, "diff_ci_high": d_high})
    return pd.DataFrame(out)


def write_grid(runs: pd.DataFrame, summary: pd.DataFrame, out_dir) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    runs.to_csv(out / "runs.csv", index=False)
    summary.to_csv(out / "summary.csv", index=False)
    return out
