"""Optional Flower adapter: run the same NumPy clients with the Flower simulation engine.

Install with ``pip install 'fedsugar[flower]'``. This module imports ``flwr``
only inside its functions, so the core package works without it.
"""

from __future__ import annotations

import numpy as np

from .metrics import evaluate
from .models import make_model
from .training import Client, TrainConfig


def _flwr():
    try:
        import flwr
    except ImportError as exc:
        raise ImportError("the Flower adapter needs: pip install 'fedsugar[flower]'") from exc
    return flwr


def make_numpy_client(client: Client, cfg: TrainConfig, class_weights: np.ndarray | None = None):
    """Wrap a fedsugar ``Client`` as a Flower ``NumPyClient``."""
    fl = _flwr()

    class FedsugarNumPyClient(fl.client.NumPyClient):
        def get_parameters(self, config):
            return make_model(cfg.model, client._X.shape[1], seed=cfg.seed, hidden=cfg.hidden).get_weights()

        def fit(self, parameters, config):
            round_no = int(config.get("server_round", 1))
            weights, loss = client.fit([np.asarray(p) for p in parameters], cfg, round_no, class_weights)
            return weights, client.n, {"loss": float(loss)}

        def evaluate(self, parameters, config):
            model = make_model(cfg.model, client._X.shape[1], hidden=cfg.hidden)
            model.set_weights([np.asarray(p) for p in parameters])
            proba = model.predict_proba(client._X)
            scores = evaluate(client._y, proba)
            loss = float(-np.mean(np.log(np.clip(proba[np.arange(client.n), client._y], 1e-12, None))))
            return loss, client.n, {"macro_f1": scores["macro_f1"]}

    return FedsugarNumPyClient()


def run_simulation(clients: list[Client], cfg: TrainConfig):
    """Run FedAvg with the Flower simulation engine and return Flower's history object."""
    fl = _flwr()
    initial = make_model(cfg.model, clients[0]._X.shape[1], seed=cfg.seed, hidden=cfg.hidden).get_weights()
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=cfg.client_fraction,
        min_fit_clients=1,
        min_available_clients=len(clients),
        initial_parameters=fl.common.ndarrays_to_parameters(initial),
        on_fit_config_fn=lambda r: {"server_round": r},
    )
    return fl.simulation.start_simulation(
        client_fn=lambda cid: make_numpy_client(clients[int(cid)], cfg).to_client(),
        num_clients=len(clients),
        config=fl.server.ServerConfig(num_rounds=cfg.rounds),
        strategy=strategy,
    )
