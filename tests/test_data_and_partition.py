"""Data contract, splits, scaling (reference problems 5 and 6) and partitioners (problem 7)."""

import numpy as np
import pandas as pd
import pytest

from fedsugar.data import FEATURES, TARGET, SchemaError, scale, split, validate
from fedsugar.partition import class_table, dirichlet, iid, label_skew, make_partition, silo
from fedsugar.synthetic import generate


def test_all_21_indicators_are_features(data):
    assert len(FEATURES) == 21
    assert data.X_train.shape[1] == 21
    for strong in ("HighBP", "HighChol", "GenHlth", "HeartDiseaseorAttack", "DiffWalk"):
        assert strong in FEATURES


def test_synthetic_class_shares_are_close_to_the_survey(frame):
    shares = frame[TARGET].value_counts(normalize=True)
    assert shares[0.0] == pytest.approx(0.842, abs=0.03)
    assert shares[1.0] == pytest.approx(0.018, abs=0.01)
    assert shares[2.0] == pytest.approx(0.140, abs=0.03)


@pytest.mark.parametrize(
    "change, message",
    [
        (lambda f: f.drop(columns=["HighBP"]), "missing columns"),
        (lambda f: f.assign(BMI=200.0), "outside"),
        (lambda f: f.assign(HighBP=0.5), "whole numbers"),
        (lambda f: f.assign(Diabetes_012=3.0), "must be 0, 1 or 2"),
        (lambda f: f.assign(Age="old"), "non-numeric"),
    ],
)
def test_schema_errors(frame, change, message):
    with pytest.raises(SchemaError, match=message):
        validate(change(frame.head(50).copy()))


def test_scaling_uses_documented_ranges_not_data(frame):
    a = scale(frame.head(100))
    b = scale(frame.tail(100))
    assert a.min() >= 0 and a.max() <= 1
    row = frame.head(1)
    assert np.allclose(scale(row), scale(pd.concat([row, frame.tail(500)])).__getitem__(0))
    assert scale(row.assign(BMI=12.0))[0, FEATURES.index("BMI")] == 0.0
    assert b.shape == a.shape


def test_split_is_stratified_disjoint_and_seeded(frame):
    a, b = split(frame, seed=5), split(frame, seed=5)
    assert np.array_equal(a.y_test, b.y_test)
    sizes = a.sizes()
    assert sum(sizes.values()) == len(frame)
    for y in (a.y_train, a.y_val, a.y_test):
        assert np.bincount(y, minlength=3)[1] > 0


def test_partitions_are_disjoint_and_complete(data):
    y = data.y_train
    for parts in (iid(len(y), 4), dirichlet(y, 4, 0.5), silo(data.train_frame, "Age", 4)):
        joined = np.concatenate(parts)
        assert np.array_equal(np.sort(joined), np.arange(len(y)))


def test_small_alpha_gives_more_label_skew(data):
    y = data.y_train
    skew = {a: np.mean([label_skew(y, dirichlet(y, 5, a, seed=s)) for s in range(5)]) for a in (0.1, 100.0)}
    assert skew[0.1] > 3 * skew[100.0]


def test_dirichlet_respects_the_minimum_size(data):
    parts = dirichlet(data.y_train, 5, 0.1, seed=3, min_size=50)
    assert min(p.size for p in parts) >= 50
    with pytest.raises(ValueError):
        dirichlet(data.y_train, 5, 0.0)


def test_class_table_shows_missing_classes(data):
    parts = dirichlet(data.y_train, 6, 0.05, seed=1, min_size=10)
    table = class_table(data.y_train, parts)
    assert (table[["class_0", "class_1", "class_2"]] == 0).any().any()
    assert table["rows"].sum() == len(data.y_train)


def test_silo_partition_by_age_gives_feature_skew(data):
    parts = make_partition("silo", data.y_train, data.train_frame, 3, column="Age")
    means = [data.train_frame.iloc[p]["Age"].mean() for p in parts]
    assert means == sorted(means)
    with pytest.raises(ValueError):
        silo(data.train_frame, "Weight", 3)


def test_generator_is_seeded():
    assert generate(200, seed=1).equals(generate(200, seed=1))
