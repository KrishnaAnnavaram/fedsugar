"""Metrics (reference problem 2), experiments and the CLI (problem 8: one entry point that runs)."""

import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from fedsugar.cli import main
from fedsugar.experiments import compare, load_grid, run_grid
from fedsugar.metrics import evaluate, expected_calibration_error, mean_ci, paired_bootstrap_f1
from fedsugar.synthetic import write
from fedsugar.training import TrainConfig


def test_majority_predictor_has_high_accuracy_but_low_macro_f1():
    y = np.array([0] * 84 + [1] * 2 + [2] * 14)
    proba = np.tile([0.9, 0.05, 0.05], (100, 1))
    scores = evaluate(y, proba)
    assert scores["accuracy"] == pytest.approx(0.84)
    assert scores["macro_f1"] < 0.31
    assert scores["recall_prediabetes"] == 0 and scores["recall_diabetes"] == 0


def test_calibration_error_is_zero_for_perfect_confidence():
    y = np.array([0, 1, 2])
    assert expected_calibration_error(y, np.eye(3)) == pytest.approx(0.0)


def test_paired_bootstrap_and_mean_ci():
    y = np.array([0, 1, 2] * 30)
    diff = paired_bootstrap_f1(y, y, np.zeros_like(y), n_boot=200)
    assert diff["diff"] > 0 and diff["ci_low"] > 0
    mean, low, high = mean_ci([0.4, 0.42, 0.44])
    assert mean == pytest.approx(0.42)
    assert low < mean < high


def test_compare_returns_all_settings(frame):
    result = compare(frame, TrainConfig(rounds=3), n_clients=3, alpha=1.0)
    settings = set(result["table"]["setting"])
    assert settings == {"centralized", "federated_fedavg", "local_only_mean", "centralized_hgb"}
    assert result["fed_minus_central_macro_f1"]["ci_low"] <= result["fed_minus_central_macro_f1"]["diff"]


def test_grid_summary_has_intervals_over_seeds(frame, tmp_path):
    config_path = tmp_path / "g.toml"
    config_path.write_text(
        '[grid]\nseeds=[1,2]\nschemes=["iid","dirichlet"]\nalphas=[0.5]\nclients=[3]\nstrategies=["fedavg"]\nmodels=["logreg"]\n[train]\nrounds=2\n',
        encoding="utf-8",
    )
    runs, summary = run_grid(frame, load_grid(config_path))
    assert len(runs) == 2 * 2 * 2
    assert set(summary["scheme"]) == {"iid", "dirichlet"}
    assert (summary["seeds"] == 2).all()
    assert {"fed_minus_central", "diff_ci_low", "diff_ci_high"} <= set(summary.columns)


def test_grid_needs_all_keys(tmp_path):
    path = tmp_path / "bad.toml"
    path.write_text("[grid]\nseeds=[1]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="needs"):
        load_grid(path)


def test_shipped_configs_are_valid():
    for name in ("sweep.toml", "quick.toml"):
        assert load_grid(f"configs/{name}")["grid"]["seeds"]


@pytest.fixture(scope="module")
def csv(tmp_path_factory):
    return str(write(tmp_path_factory.mktemp("d") / "s.csv", n=3000, seed=4))


def test_cli_commands(csv, tmp_path, capsys):
    assert main(["partition", "--data", csv, "--clients", "3"]) == 0
    assert "label skew" in capsys.readouterr().out
    assert main(["centralized", "--data", csv, "--rounds", "2"]) == 0
    assert "macro_f1" in capsys.readouterr().out
    assert main(["federated", "--data", csv, "--rounds", "2", "--strategy", "fedadam", "--history", str(tmp_path / "h.csv")]) == 0
    assert len(pd.read_csv(tmp_path / "h.csv")) == 2
    assert main(["compare", "--data", csv, "--rounds", "2", "--clients", "3", "--out", str(tmp_path / "c")]) == 0
    assert (tmp_path / "c" / "compare.csv").exists()
    assert main(["sweep", "--data", csv, "--config", "configs/quick.toml", "--out", str(tmp_path / "s")]) == 0
    assert (tmp_path / "s" / "summary.csv").exists()


def test_missing_data_file_gives_a_clear_message(tmp_path):
    with pytest.raises(SystemExit, match="not found"):
        main(["centralized", "--data", str(tmp_path / "none.csv")])


def test_demo_runs_offline(tmp_path, capsys):
    assert main(["demo", "--out", str(tmp_path / "demo"), "--rows", "3000"]) == 0
    assert "SYNTHETIC DATA" in capsys.readouterr().out


def test_core_import_does_not_load_flower():
    code = "import fedsugar.cli, fedsugar.flower_app, sys; print('flwr' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip()
    assert out == "False"


def test_flower_adapter_wraps_the_numpy_client(data):
    pytest.importorskip("flwr")
    from fedsugar.flower_app import make_numpy_client
    from fedsugar.training import Client

    client = make_numpy_client(Client(0, data.X_train[:200], data.y_train[:200]), TrainConfig(rounds=1))
    params = client.get_parameters({})
    new, n, metrics = client.fit(params, {"server_round": 1})
    assert n == 200 and len(new) == len(params) and "loss" in metrics
