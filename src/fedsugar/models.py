"""Differentiable models in NumPy, shared by the centralized trainer and the federated clients.

Both models expose their parameters as a list of arrays (``get_weights`` /
``set_weights``). A federated round sends these arrays, never data.

The loss is the class-weighted cross-entropy:
``L = sum_i w[y_i] * (-log p(y_i | x_i)) / n + l2 / 2 * ||W||^2``.
"""

from __future__ import annotations

import numpy as np

N_CLASSES = 3


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


class Model:
    name = "base"

    def get_weights(self) -> list[np.ndarray]:
        return [w.copy() for w in self.weights]

    def set_weights(self, weights: list[np.ndarray]) -> None:
        if len(weights) != len(self.weights) or any(a.shape != b.shape for a, b in zip(weights, self.weights)):
            raise ValueError("weight shapes do not match the model")
        self.weights = [np.array(w, dtype=float, copy=True) for w in weights]

    @property
    def n_params(self) -> int:
        return int(sum(w.size for w in self.weights))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)

    def loss_and_grad(self, X, y, sample_weight) -> tuple[float, list[np.ndarray]]:
        raise NotImplementedError


class SoftmaxRegression(Model):
    """Multinomial logistic regression."""

    name = "logreg"

    def __init__(self, n_features: int, n_classes: int = N_CLASSES, l2: float = 1e-4, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.l2 = l2
        self.weights = [rng.normal(0, 0.01, size=(n_features, n_classes)), np.zeros(n_classes)]

    def predict_proba(self, X):
        W, b = self.weights
        return softmax(X @ W + b)

    def loss_and_grad(self, X, y, sample_weight):
        W, b = self.weights
        n = X.shape[0]
        proba = softmax(X @ W + b)
        nll = -np.log(np.clip(proba[np.arange(n), y], 1e-12, None))
        loss = float(np.sum(sample_weight * nll) / n + 0.5 * self.l2 * np.sum(W**2))
        delta = proba.copy()
        delta[np.arange(n), y] -= 1.0
        delta *= sample_weight[:, None] / n
        return loss, [X.T @ delta + self.l2 * W, delta.sum(axis=0)]


class MLP(Model):
    """One hidden ReLU layer."""

    name = "mlp"

    def __init__(self, n_features: int, hidden: int = 32, n_classes: int = N_CLASSES, l2: float = 1e-4, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.l2 = l2
        self.weights = [
            rng.normal(0, np.sqrt(2.0 / n_features), size=(n_features, hidden)),
            np.zeros(hidden),
            rng.normal(0, np.sqrt(2.0 / hidden), size=(hidden, n_classes)),
            np.zeros(n_classes),
        ]

    def _forward(self, X):
        W1, b1, W2, b2 = self.weights
        pre = X @ W1 + b1
        hidden = np.maximum(pre, 0.0)
        return pre, hidden, softmax(hidden @ W2 + b2)

    def predict_proba(self, X):
        return self._forward(X)[2]

    def loss_and_grad(self, X, y, sample_weight):
        W1, b1, W2, b2 = self.weights
        n = X.shape[0]
        pre, hidden, proba = self._forward(X)
        nll = -np.log(np.clip(proba[np.arange(n), y], 1e-12, None))
        loss = float(np.sum(sample_weight * nll) / n + 0.5 * self.l2 * (np.sum(W1**2) + np.sum(W2**2)))
        delta = proba.copy()
        delta[np.arange(n), y] -= 1.0
        delta *= sample_weight[:, None] / n
        g_W2 = hidden.T @ delta + self.l2 * W2
        g_b2 = delta.sum(axis=0)
        d_hidden = (delta @ W2.T) * (pre > 0)
        g_W1 = X.T @ d_hidden + self.l2 * W1
        g_b1 = d_hidden.sum(axis=0)
        return loss, [g_W1, g_b1, g_W2, g_b2]


MODELS = ("logreg", "mlp")


def make_model(name: str, n_features: int, seed: int = 0, hidden: int = 32) -> Model:
    """A new model. The initial weights depend only on the seed, never on data."""
    if name == "logreg":
        return SoftmaxRegression(n_features, seed=seed)
    if name == "mlp":
        return MLP(n_features, hidden=hidden, seed=seed)
    raise ValueError(f"model must be one of {MODELS}")


def balanced_class_weights(counts: np.ndarray) -> np.ndarray:
    """``n / (K * n_k)`` for each class. A class with no sample gets weight 0."""
    counts = np.asarray(counts, dtype=float)
    total, k = counts.sum(), counts.size
    with np.errstate(divide="ignore"):
        weights = np.where(counts > 0, total / (k * counts), 0.0)
    return weights


def sgd_epochs(
    model: Model,
    X: np.ndarray,
    y: np.ndarray,
    class_weights: np.ndarray,
    epochs: int,
    lr: float,
    batch_size: int,
    rng: np.random.Generator,
    prox_mu: float = 0.0,
    prox_center: list[np.ndarray] | None = None,
) -> float:
    """Mini-batch SGD in place. With ``prox_mu > 0`` it adds the FedProx term ``mu/2 ||w - w_global||^2``."""
    n = X.shape[0]
    sample_weight = class_weights[y]
    last = float("nan")
    for _ in range(epochs):
        order = rng.permutation(n)
        for start in range(0, n, batch_size):
            batch = order[start : start + batch_size]
            last, grads = model.loss_and_grad(X[batch], y[batch], sample_weight[batch])
            for i, grad in enumerate(grads):
                if prox_mu > 0 and prox_center is not None:
                    grad = grad + prox_mu * (model.weights[i] - prox_center[i])
                model.weights[i] -= lr * grad
    return last
