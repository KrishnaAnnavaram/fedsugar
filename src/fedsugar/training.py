"""Centralized and federated training with one shared configuration.

Like-for-like rules:

* Both settings train the same model class with the same loss, the same
  optimizer, the same learning rate and the same class weights.
* The centralized trainer runs ``rounds x local_epochs`` epochs. Each federated
  client runs ``local_epochs`` epochs in each round.
* Both settings choose the round (or epoch) with the best validation macro-F1.
* Times are measured with ``time.perf_counter``. No scaling factor is applied.

Privacy rules for the federated simulation:

* The server makes the initial model from a seed only. It never sees a record.
* A client sends model weights and its row count. For ``class_weight="global"``
  it also sends its label counts once, before round 1.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from .data import CLASSES
from .metrics import evaluate
from .models import Model, balanced_class_weights, make_model, sgd_epochs

STRATEGIES = ("fedavg", "fedprox", "fedadam")
CLASS_WEIGHTING = ("global", "local", "none")
BYTES_PER_PARAM = 4  # float32 on the wire


@dataclass(frozen=True)
class TrainConfig:
    model: str = "logreg"
    hidden: int = 32
    rounds: int = 30
    local_epochs: int = 1
    lr: float = 0.05
    batch_size: int = 64
    class_weight: str = "global"
    strategy: str = "fedavg"
    prox_mu: float = 0.01
    server_lr: float = 0.01
    client_fraction: float = 1.0
    dp_clip: float | None = None
    dp_noise: float = 0.0
    seed: int = 42

    def __post_init__(self):
        if self.strategy not in STRATEGIES:
            raise ValueError(f"strategy must be one of {STRATEGIES}")
        if self.class_weight not in CLASS_WEIGHTING:
            raise ValueError(f"class_weight must be one of {CLASS_WEIGHTING}")
        if not 0 < self.client_fraction <= 1:
            raise ValueError("client_fraction must be in (0, 1]")
        if self.rounds < 1 or self.local_epochs < 1:
            raise ValueError("rounds and local_epochs must be >= 1")


@dataclass
class TrainResult:
    setting: str
    config: TrainConfig
    history: pd.DataFrame
    best_round: int
    model: Model
    test: dict[str, float]
    test_proba: np.ndarray
    train_seconds: float
    extra: dict = field(default_factory=dict)

    def row(self) -> dict:
        return {"setting": self.setting, "best_round": self.best_round, "train_seconds": self.train_seconds, **self.test, **self.extra}


def class_weights_for(counts: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return np.ones(len(CLASSES))
    return balanced_class_weights(counts)


def train_centralized(X, y, X_val, y_val, X_test, y_test, cfg: TrainConfig) -> TrainResult:
    model = make_model(cfg.model, X.shape[1], seed=cfg.seed, hidden=cfg.hidden)
    weights = class_weights_for(np.bincount(y, minlength=len(CLASSES)), "none" if cfg.class_weight == "none" else "global")
    rng = np.random.default_rng(cfg.seed + 1)
    history, best, best_weights, seconds = [], -1.0, model.get_weights(), 0.0
    for epoch in range(1, cfg.rounds * cfg.local_epochs + 1):
        start = time.perf_counter()
        loss = sgd_epochs(model, X, y, weights, 1, cfg.lr, cfg.batch_size, rng)
        seconds += time.perf_counter() - start
        val = evaluate(y_val, model.predict_proba(X_val))
        history.append({"round": epoch, "train_loss": loss, **{f"val_{k}": v for k, v in val.items()}})
        if val["macro_f1"] > best:
            best, best_weights, best_round = val["macro_f1"], model.get_weights(), epoch
    model.set_weights(best_weights)
    proba = model.predict_proba(X_test)
    return TrainResult("centralized", cfg, pd.DataFrame(history), best_round, model, evaluate(y_test, proba), proba, seconds,
                       {"class_weights": weights.round(4).tolist()})


class Client:
    """A data holder. It shares weights, its row count and (optionally) label counts, never rows."""

    def __init__(self, client_id: int, X: np.ndarray, y: np.ndarray):
        self.client_id = client_id
        self._X = X
        self._y = y

    @property
    def n(self) -> int:
        return int(self._y.size)

    def label_counts(self) -> np.ndarray:
        return np.bincount(self._y, minlength=len(CLASSES))

    def fit(self, global_weights: list[np.ndarray], cfg: TrainConfig, round_no: int, class_weights: np.ndarray | None) -> tuple[list[np.ndarray], float]:
        model = make_model(cfg.model, self._X.shape[1], seed=0, hidden=cfg.hidden)
        model.set_weights(global_weights)
        if class_weights is None:
            class_weights = class_weights_for(self.label_counts(), cfg.class_weight)
        rng = np.random.default_rng([cfg.seed, round_no, self.client_id])
        mu = cfg.prox_mu if cfg.strategy == "fedprox" else 0.0
        loss = sgd_epochs(model, self._X, self._y, class_weights, cfg.local_epochs, cfg.lr, cfg.batch_size, rng,
                          prox_mu=mu, prox_center=global_weights)
        return model.get_weights(), loss


def _clip_and_noise(update: list[np.ndarray], clip: float, noise: float, rng: np.random.Generator) -> list[np.ndarray]:
    norm = float(np.sqrt(sum(np.sum(u**2) for u in update)))
    factor = min(1.0, clip / norm) if norm > 0 else 1.0
    return [u * factor + rng.normal(0, noise * clip, size=u.shape) for u in update]


def train_federated(clients: list[Client], X_val, y_val, X_test, y_test, cfg: TrainConfig) -> TrainResult:
    n_features = X_val.shape[1]
    model = make_model(cfg.model, n_features, seed=cfg.seed, hidden=cfg.hidden)  # data-free initial model
    weights = model.get_weights()
    rng = np.random.default_rng(cfg.seed + 2)
    bytes_per_model = model.n_params * BYTES_PER_PARAM
    comm_bytes = 0

    shared_weights = None
    if cfg.class_weight == "global":
        counts = np.sum([c.label_counts() for c in clients], axis=0)
        comm_bytes += len(clients) * len(CLASSES) * 8
        shared_weights = balanced_class_weights(counts)
    elif cfg.class_weight == "none":
        shared_weights = np.ones(len(CLASSES))

    m_state = [np.zeros_like(w) for w in weights]
    v_state = [np.zeros_like(w) for w in weights]
    history, best, best_weights, best_round = [], -1.0, [w.copy() for w in weights], 0
    sequential, parallel = 0.0, 0.0
    n_select = max(1, int(round(cfg.client_fraction * len(clients))))
    for round_no in range(1, cfg.rounds + 1):
        selected = rng.choice(len(clients), size=n_select, replace=False)
        updates, sizes, times, losses = [], [], [], []
        for idx in selected:
            client = clients[idx]
            start = time.perf_counter()
            new, loss = client.fit(weights, cfg, round_no, shared_weights)
            times.append(time.perf_counter() - start)
            delta = [a - b for a, b in zip(new, weights)]
            if cfg.dp_clip:
                delta = _clip_and_noise(delta, cfg.dp_clip, cfg.dp_noise, rng)
            updates.append(delta)
            sizes.append(client.n)
            losses.append(loss)
        start = time.perf_counter()
        share = np.asarray(sizes, float) / np.sum(sizes)
        mean_delta = [sum(s * u[i] for s, u in zip(share, updates)) for i in range(len(weights))]
        if cfg.strategy == "fedadam":
            beta1, beta2, tau = 0.9, 0.99, 1e-3
            for i, d in enumerate(mean_delta):
                m_state[i] = beta1 * m_state[i] + (1 - beta1) * d
                v_state[i] = beta2 * v_state[i] + (1 - beta2) * d**2
                weights[i] = weights[i] + cfg.server_lr * m_state[i] / (np.sqrt(v_state[i]) + tau)
        else:
            weights = [w + d for w, d in zip(weights, mean_delta)]
        aggregate = time.perf_counter() - start
        sequential += sum(times) + aggregate
        parallel += max(times) + aggregate
        comm_bytes += 2 * len(selected) * bytes_per_model

        model.set_weights(weights)
        val = evaluate(y_val, model.predict_proba(X_val))
        history.append({"round": round_no, "train_loss": float(np.average(losses, weights=share)), "clients": len(selected),
                        **{f"val_{k}": v for k, v in val.items()}})
        if val["macro_f1"] > best:
            best, best_weights, best_round = val["macro_f1"], [w.copy() for w in weights], round_no

    model.set_weights(best_weights)
    proba = model.predict_proba(X_test)
    extra = {
        "parallel_seconds": parallel,
        "comm_megabytes": comm_bytes / 1e6,
        "n_params": model.n_params,
        "clients": len(clients),
        "class_weights": None if shared_weights is None else shared_weights.round(4).tolist(),
    }
    return TrainResult(f"federated_{cfg.strategy}", cfg, pd.DataFrame(history), best_round, model,
                       evaluate(y_test, proba), proba, sequential, extra)


def train_local_only(clients: list[Client], X_val, y_val, X_test, y_test, cfg: TrainConfig) -> dict[str, float]:
    """Each client trains alone with the centralized budget. Returns the mean test metrics over clients."""
    rows = []
    for client in clients:
        result = train_centralized(client._X, client._y, X_val, y_val, X_test, y_test, cfg)
        rows.append(result.test)
    return pd.DataFrame(rows).mean().to_dict()


def config_dict(cfg: TrainConfig) -> dict:
    return asdict(cfg)
