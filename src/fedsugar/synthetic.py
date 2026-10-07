"""Synthetic survey rows with the BRFSS columns, ranges and class shares. All values are generated.

The label comes from a multinomial logistic model of the strongest real risk
factors (blood pressure, cholesterol, BMI, general health, age, walking
difficulty, heart disease). The intercepts are tuned so the class shares are
close to the public data: about 84 % no diabetes, 2 % prediabetes, 14 % diabetes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .data import FEATURES, TARGET

TARGET_SHARES = np.array([0.842, 0.018, 0.140])


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def generate(n: int = 20_000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age = np.clip(np.round(rng.normal(8.0, 3.0, n)), 1, 13)
    bmi = np.clip(np.round(rng.normal(28.4, 6.5, n)), 12, 98)
    income = np.clip(np.round(rng.normal(6.0, 2.0, n)), 1, 8)
    education = np.clip(np.round(rng.normal(5.0, 1.0, n)), 1, 6)
    high_bp = rng.random(n) < _sigmoid(-2.2 + 0.22 * age + 0.07 * (bmi - 28))
    high_chol = rng.random(n) < _sigmoid(-1.6 + 0.16 * age + 0.03 * (bmi - 28))
    heart = rng.random(n) < _sigmoid(-4.4 + 0.25 * age + 0.6 * high_bp)
    stroke = rng.random(n) < _sigmoid(-5.0 + 0.2 * age + 0.5 * high_bp)
    phys_act = rng.random(n) < _sigmoid(1.6 - 0.04 * (bmi - 28) - 0.05 * age)
    gen_hlth = np.clip(np.round(1.2 + 0.09 * age + 0.05 * (bmi - 28) + 0.6 * heart - 0.4 * phys_act - 0.1 * (income - 6) + rng.normal(0, 0.8, n)), 1, 5)
    phys_hlth = np.where(rng.random(n) < _sigmoid(-1.5 + 0.6 * (gen_hlth - 2.5)), np.clip(np.round(rng.gamma(2, 6, n)), 0, 30), 0)
    ment_hlth = np.where(rng.random(n) < 0.3, np.clip(np.round(rng.gamma(1.5, 6, n)), 0, 30), 0)
    diff_walk = rng.random(n) < _sigmoid(-3.5 + 0.8 * (gen_hlth - 2.5) + 0.1 * age)
    frame = pd.DataFrame(
        {
            "HighBP": high_bp, "HighChol": high_chol, "CholCheck": rng.random(n) < 0.96, "BMI": bmi,
            "Smoker": rng.random(n) < 0.44, "Stroke": stroke, "HeartDiseaseorAttack": heart, "PhysActivity": phys_act,
            "Fruits": rng.random(n) < 0.63, "Veggies": rng.random(n) < 0.81, "HvyAlcoholConsump": rng.random(n) < 0.056,
            "AnyHealthcare": rng.random(n) < 0.95, "NoDocbcCost": rng.random(n) < 0.084, "GenHlth": gen_hlth,
            "MentHlth": ment_hlth, "PhysHlth": phys_hlth, "DiffWalk": diff_walk, "Sex": rng.random(n) < 0.44,
            "Age": age, "Education": education, "Income": income,
        }
    ).astype(float)

    risk = (
        0.9 * frame["HighBP"] + 0.6 * frame["HighChol"] + 0.07 * (frame["BMI"] - 28) + 0.5 * (frame["GenHlth"] - 2.5)
        + 0.14 * (frame["Age"] - 8) + 0.4 * frame["DiffWalk"] + 0.4 * frame["HeartDiseaseorAttack"]
        - 0.2 * frame["PhysActivity"] - 0.08 * (frame["Income"] - 6)
    ).to_numpy()
    slopes = np.array([0.0, 0.55, 1.0])
    intercepts = np.array([0.0, -4.0, -2.0])
    for _ in range(50):
        logits = intercepts[None, :] + risk[:, None] * slopes[None, :]
        proba = np.exp(logits - logits.max(axis=1, keepdims=True))
        proba /= proba.sum(axis=1, keepdims=True)
        intercepts[1:] += np.log(TARGET_SHARES[1:] / proba.mean(axis=0)[1:]) - np.log(TARGET_SHARES[0] / proba.mean(axis=0)[0])
    cumulative = proba.cumsum(axis=1)
    frame[TARGET] = (rng.random(n)[:, None] > cumulative).sum(axis=1).astype(float)
    return frame[[*FEATURES, TARGET]]


def write(path, n: int = 20_000, seed: int = 42) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    generate(n, seed).to_csv(path, index=False)
    return path
