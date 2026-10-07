"""Metrics for an imbalanced three-class problem. Macro-F1 is the main metric."""

from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
)

from .data import CLASS_NAMES, CLASSES


def expected_calibration_error(y: np.ndarray, proba: np.ndarray, bins: int = 10) -> float:
    """ECE of the top-class probability."""
    confidence = proba.max(axis=1)
    correct = proba.argmax(axis=1) == y
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        inside = (confidence > low) & (confidence <= high)
        if inside.any():
            ece += inside.mean() * abs(correct[inside].mean() - confidence[inside].mean())
    return float(ece)


def evaluate(y: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    pred = proba.argmax(axis=1)
    out = {
        "macro_f1": float(f1_score(y, pred, average="macro", labels=list(CLASSES), zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "accuracy": float(accuracy_score(y, pred)),
        "ece": expected_calibration_error(y, proba),
    }
    try:
        out["roc_auc_ovr"] = float(roc_auc_score(y, proba, multi_class="ovr", average="macro", labels=list(CLASSES)))
    except ValueError:
        out["roc_auc_ovr"] = float("nan")
    recalls = recall_score(y, pred, labels=list(CLASSES), average=None, zero_division=0)
    for name, value in zip(CLASS_NAMES, recalls):
        out[f"recall_{name}"] = float(value)
    return out


def confusion(y: np.ndarray, proba: np.ndarray) -> np.ndarray:
    return confusion_matrix(y, proba.argmax(axis=1), labels=list(CLASSES))


def paired_bootstrap_f1(y: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray, n_boot: int = 1000, seed: int = 42) -> dict[str, float]:
    """Macro-F1 of A minus B on the same test rows, with a 95 % bootstrap interval over rows."""
    rng = np.random.default_rng(seed)
    n = y.size
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[i] = f1_score(y[idx], pred_a[idx], average="macro", labels=list(CLASSES), zero_division=0) - f1_score(
            y[idx], pred_b[idx], average="macro", labels=list(CLASSES), zero_division=0
        )
    point = f1_score(y, pred_a, average="macro", labels=list(CLASSES), zero_division=0) - f1_score(
        y, pred_b, average="macro", labels=list(CLASSES), zero_division=0
    )
    return {"diff": float(point), "ci_low": float(np.percentile(diffs, 2.5)), "ci_high": float(np.percentile(diffs, 97.5))}


def mean_ci(values, level: float = 0.95) -> tuple[float, float, float]:
    """Mean and the t interval over repeated seeds."""
    values = np.asarray(values, float)
    mean = float(values.mean())
    if values.size < 2:
        return mean, mean, mean
    half = float(stats.t.ppf(0.5 + level / 2, values.size - 1) * values.std(ddof=1) / np.sqrt(values.size))
    return mean, mean - half, mean + half
