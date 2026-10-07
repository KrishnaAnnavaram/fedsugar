"""Models, FedAvg dynamics (reference problem 4), like-for-like setup (problem 1),
honest timing (problem 3), data-free server (problem 5) and missing classes (problem 7)."""

import inspect

import numpy as np
import pytest

from fedsugar import training
from fedsugar.metrics import evaluate
from fedsugar.models import MLP, SoftmaxRegression, balanced_class_weights, make_model, sgd_epochs
from fedsugar.partition import dirichlet
from fedsugar.training import Client, TrainConfig, train_centralized, train_federated


@pytest.mark.parametrize("model", [SoftmaxRegression(5, seed=1), MLP(5, hidden=4, seed=1)])
def test_gradients_match_finite_differences(model):
    rng = np.random.default_rng(0)
    X, y = rng.normal(size=(12, 5)), rng.integers(0, 3, 12)
    w = rng.uniform(0.5, 2.0, 12)
    _, grads = model.loss_and_grad(X, y, w)
    for i, param in enumerate(model.weights):
        idx = tuple(rng.integers(0, s) for s in param.shape)
        old = param[idx]
        param[idx] = old + 1e-6
        up, _ = model.loss_and_grad(X, y, w)
        param[idx] = old - 1e-6
        down, _ = model.loss_and_grad(X, y, w)
        param[idx] = old
        assert grads[i][idx] == pytest.approx((up - down) / 2e-6, rel=1e-4, abs=1e-7)


def test_weights_round_trip_and_shape_check():
    model = make_model("mlp", 4, seed=3)
    copy = make_model("mlp", 4, seed=9)
    copy.set_weights(model.get_weights())
    assert all(np.array_equal(a, b) for a, b in zip(model.weights, copy.weights))
    with pytest.raises(ValueError):
        copy.set_weights(make_model("logreg", 4).get_weights())


def test_initial_model_depends_only_on_the_seed():
    a = make_model("logreg", 21, seed=7).get_weights()
    b = make_model("logreg", 21, seed=7).get_weights()
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    assert "X" not in inspect.signature(make_model).parameters


def test_balanced_weights_and_missing_class():
    weights = balanced_class_weights(np.array([80, 0, 20]))
    assert weights[1] == 0 and weights[0] < weights[2]


def test_one_fedavg_step_equals_one_centralized_step():
    """Full-batch, one local step, weights n_k / n: FedAvg is exactly gradient descent on the union."""
    rng = np.random.default_rng(1)
    X, y = rng.normal(size=(90, 6)), rng.integers(0, 3, 90)
    parts = [np.arange(0, 20), np.arange(20, 55), np.arange(55, 90)]
    cfg = TrainConfig(rounds=1, lr=0.1, batch_size=1000, class_weight="none", seed=4)
    central = make_model("logreg", 6, seed=4)
    sgd_epochs(central, X, y, np.ones(3), 1, cfg.lr, cfg.batch_size, rng)
    clients = [Client(i, X[p], y[p]) for i, p in enumerate(parts)]
    start = make_model("logreg", 6, seed=4).get_weights()
    new = [c.fit(start, cfg, 1, np.ones(3))[0] for c in clients]
    share = np.array([p.size for p in parts]) / 90
    fedavg = [sum(s * w[i] for s, w in zip(share, new)) for i in range(2)]
    for a, b in zip(fedavg, central.weights):
        assert np.allclose(a, b)


def test_rounds_change_the_global_model(data):
    parts = dirichlet(data.y_train, 3, 1.0, seed=2)
    clients = [Client(i, data.X_train[p], data.y_train[p]) for i, p in enumerate(parts)]
    result = train_federated(clients, data.X_val, data.y_val, data.X_test, data.y_test, TrainConfig(rounds=4, seed=2))
    losses = result.history["train_loss"].to_numpy()
    assert len(set(np.round(losses, 8))) == 4
    assert losses[-1] < losses[0]


def test_clients_without_a_class_aggregate_without_error(data):
    parts = dirichlet(data.y_train, 6, 0.05, seed=1, min_size=10)
    clients = [Client(i, data.X_train[p], data.y_train[p]) for i, p in enumerate(parts)]
    assert any(c.label_counts()[1] == 0 for c in clients)
    for weighting in ("global", "local"):
        result = train_federated(clients, data.X_val, data.y_val, data.X_test, data.y_test, TrainConfig(rounds=2, class_weight=weighting))
        assert result.test_proba.shape == (len(data.y_test), 3)


def test_centralized_and_federated_use_the_same_class_weights(data):
    cfg = TrainConfig(rounds=2)
    clients = [Client(i, data.X_train[p], data.y_train[p]) for i, p in enumerate(dirichlet(data.y_train, 4, 0.5))]
    central = train_centralized(data.X_train, data.y_train, data.X_val, data.y_val, data.X_test, data.y_test, cfg)
    fed = train_federated(clients, data.X_val, data.y_val, data.X_test, data.y_test, cfg)
    assert central.extra["class_weights"] == fed.extra["class_weights"]


def test_times_are_measured_and_no_scaling_factor_exists(data):
    clients = [Client(i, data.X_train[p], data.y_train[p]) for i, p in enumerate(dirichlet(data.y_train, 4, 0.5))]
    fed = train_federated(clients, data.X_val, data.y_val, data.X_test, data.y_test, TrainConfig(rounds=3))
    assert 0 < fed.extra["parallel_seconds"] <= fed.train_seconds
    assert "scaling_factor" not in inspect.getsource(training)
    expected = 3 * 2 * 4 * fed.extra["n_params"] * 4 + 4 * 3 * 8
    assert fed.extra["comm_megabytes"] * 1e6 == pytest.approx(expected)


@pytest.mark.parametrize("strategy", ["fedavg", "fedprox", "fedadam"])
def test_every_strategy_learns_more_than_the_majority_rule(data, strategy):
    clients = [Client(i, data.X_train[p], data.y_train[p]) for i, p in enumerate(dirichlet(data.y_train, 4, 1.0, seed=3))]
    fed = train_federated(clients, data.X_val, data.y_val, data.X_test, data.y_test, TrainConfig(rounds=10, lr=0.2, strategy=strategy, server_lr=0.05))
    majority = evaluate(data.y_test, np.tile([1.0, 0.0, 0.0], (len(data.y_test), 1)))
    assert fed.test["macro_f1"] > majority["macro_f1"] + 0.05


def test_client_fraction_and_dp_clipping(data):
    clients = [Client(i, data.X_train[p], data.y_train[p]) for i, p in enumerate(dirichlet(data.y_train, 5, 1.0))]
    fed = train_federated(clients, data.X_val, data.y_val, data.X_test, data.y_test,
                          TrainConfig(rounds=2, client_fraction=0.4, dp_clip=0.5, dp_noise=0.1))
    assert (fed.history["clients"] == 2).all()
    rng = np.random.default_rng(0)
    clipped = training._clip_and_noise([np.full(4, 10.0)], clip=1.0, noise=0.0, rng=rng)
    assert np.linalg.norm(clipped[0]) == pytest.approx(1.0)


def test_invalid_config_is_refused():
    with pytest.raises(ValueError):
        TrainConfig(strategy="fedsgd")
    with pytest.raises(ValueError):
        TrainConfig(client_fraction=0)
