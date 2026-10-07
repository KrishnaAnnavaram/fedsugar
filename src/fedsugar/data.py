"""BRFSS 2015 diabetes health indicators: schema, loading, splits and scaling.

All 21 indicators are features. The scaler uses the documented value range of
each column (for example BMI 12 to 98). It needs no data, so neither the
centralized trainer nor the federated server has to see a raw record to fit it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

TARGET = "Diabetes_012"
CLASSES = (0, 1, 2)
CLASS_NAMES = ("no_diabetes", "prediabetes", "diabetes")

# column -> (min, max, kind)
SCHEMA: dict[str, tuple[float, float, str]] = {
    "HighBP": (0, 1, "binary"),
    "HighChol": (0, 1, "binary"),
    "CholCheck": (0, 1, "binary"),
    "BMI": (12, 98, "numeric"),
    "Smoker": (0, 1, "binary"),
    "Stroke": (0, 1, "binary"),
    "HeartDiseaseorAttack": (0, 1, "binary"),
    "PhysActivity": (0, 1, "binary"),
    "Fruits": (0, 1, "binary"),
    "Veggies": (0, 1, "binary"),
    "HvyAlcoholConsump": (0, 1, "binary"),
    "AnyHealthcare": (0, 1, "binary"),
    "NoDocbcCost": (0, 1, "binary"),
    "GenHlth": (1, 5, "ordinal"),
    "MentHlth": (0, 30, "numeric"),
    "PhysHlth": (0, 30, "numeric"),
    "DiffWalk": (0, 1, "binary"),
    "Sex": (0, 1, "binary"),
    "Age": (1, 13, "ordinal"),
    "Education": (1, 6, "ordinal"),
    "Income": (1, 8, "ordinal"),
}
FEATURES = tuple(SCHEMA)


class SchemaError(ValueError):
    """The table does not match the BRFSS column contract."""


def validate(frame: pd.DataFrame, source: str = "data") -> pd.DataFrame:
    """Check columns, types and ranges. Return the 21 features plus the target as floats."""
    missing = [c for c in (*FEATURES, TARGET) if c not in frame.columns]
    if missing:
        raise SchemaError(f"{source}: missing columns {missing}")
    out = frame[[*FEATURES, TARGET]].apply(pd.to_numeric, errors="coerce")
    bad = out.isna().any(axis=1)
    if bad.any():
        rows = (np.flatnonzero(bad.to_numpy()) + 2)[:5].tolist()
        raise SchemaError(f"{source}: {int(bad.sum())} row(s) with empty or non-numeric values, first file rows {rows}")
    for col, (low, high, kind) in SCHEMA.items():
        values = out[col]
        if ((values < low) | (values > high)).any():
            raise SchemaError(f"{source}: column {col!r} has values outside [{low}, {high}]")
        if kind != "numeric" and not np.allclose(values, np.round(values)):
            raise SchemaError(f"{source}: column {col!r} must hold whole numbers")
    if not out[TARGET].isin(CLASSES).all():
        raise SchemaError(f"{source}: {TARGET} must be 0, 1 or 2")
    return out.astype(float)


def load(path) -> pd.DataFrame:
    return validate(pd.read_csv(path), source=str(path))


def scale(frame: pd.DataFrame) -> np.ndarray:
    """Range scaling to [0, 1] with the documented limits. No statistic of the data is used."""
    low = np.array([SCHEMA[c][0] for c in FEATURES], dtype=float)
    high = np.array([SCHEMA[c][1] for c in FEATURES], dtype=float)
    return ((frame[list(FEATURES)].to_numpy(float) - low) / (high - low)).astype(np.float64)


@dataclass
class Split:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    train_frame: pd.DataFrame

    def sizes(self) -> dict[str, int]:
        return {"train": len(self.y_train), "val": len(self.y_val), "test": len(self.y_test)}


def split(frame: pd.DataFrame, seed: int = 42, val_size: float = 0.15, test_size: float = 0.15) -> Split:
    """Stratified train / validation / test split."""
    y = frame[TARGET].astype(int).to_numpy()
    idx = np.arange(len(frame))
    train_val, test = train_test_split(idx, test_size=test_size, stratify=y, random_state=seed)
    rel = val_size / (1 - test_size)
    train, val = train_test_split(train_val, test_size=rel, stratify=y[train_val], random_state=seed)
    X = scale(frame)
    return Split(X[train], y[train], X[val], y[val], X[test], y[test], frame.iloc[train].reset_index(drop=True))
