"""Command line interface: ``fedsugar <command>``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from . import __version__
from .config import Settings
from .data import load, split
from .experiments import compare, load_grid, make_clients, run_grid, write_grid
from .models import MODELS
from .partition import SCHEMES
from .synthetic import write as write_synthetic
from .training import CLASS_WEIGHTING, STRATEGIES, TrainConfig, train_centralized, train_federated

DEMO_GRID = {
    "grid": {"seeds": [1, 2, 3], "schemes": ["iid", "dirichlet"], "alphas": [0.1, 0.5], "clients": [5],
             "strategies": ["fedavg", "fedprox"], "models": ["logreg"]},
    "train": {"rounds": 20},
}


def _print(frame: pd.DataFrame) -> None:
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(frame.to_string(index=False))


def _config(args) -> TrainConfig:
    return TrainConfig(
        model=args.model, hidden=args.hidden, rounds=args.rounds, local_epochs=args.local_epochs, lr=args.lr,
        batch_size=args.batch_size, class_weight=args.class_weight, strategy=args.strategy, prox_mu=args.prox_mu,
        server_lr=args.server_lr, client_fraction=args.client_fraction, dp_clip=args.dp_clip, dp_noise=args.dp_noise,
        seed=args.seed if args.seed is not None else Settings.from_env().seed,
    )


def _data(args) -> pd.DataFrame:
    path = Path(args.data or Settings.from_env().data)
    if not path.exists():
        raise SystemExit(f"data file {path} not found: run 'fedsugar synth' or see data/README.md")
    return load(path)


def _metrics(result) -> dict:
    return {k: round(v, 4) if isinstance(v, float) else v for k, v in result.row().items()}


def cmd_synth(args) -> int:
    print(f"wrote {write_synthetic(args.out, n=args.rows, seed=args.seed)} (SYNTHETIC DATA)")
    return 0


def cmd_partition(args) -> int:
    frame = _data(args)
    data = split(frame, seed=args.seed or 42)
    _, table, skew = make_clients(data, args.scheme, args.clients, args.alpha, args.seed or 42)
    _print(table)
    print(f"label skew (mean total-variation distance): {skew:.3f}")
    return 0


def cmd_centralized(args) -> int:
    data = split(_data(args), seed=_config(args).seed)
    result = train_centralized(data.X_train, data.y_train, data.X_val, data.y_val, data.X_test, data.y_test, _config(args))
    print(json.dumps(_metrics(result), indent=2))
    return 0


def cmd_federated(args) -> int:
    cfg = _config(args)
    data = split(_data(args), seed=cfg.seed)
    clients, table, skew = make_clients(data, args.scheme, args.clients, args.alpha, cfg.seed)
    result = train_federated(clients, data.X_val, data.y_val, data.X_test, data.y_test, cfg)
    _print(table)
    print(f"label skew: {skew:.3f}")
    print(json.dumps(_metrics(result), indent=2))
    if args.history:
        Path(args.history).parent.mkdir(parents=True, exist_ok=True)
        result.history.to_csv(args.history, index=False)
    return 0


def cmd_compare(args) -> int:
    result = compare(_data(args), _config(args), scheme=args.scheme, n_clients=args.clients, alpha=args.alpha)
    _print(result["clients"])
    print(f"label skew: {result['label_skew']:.3f}   split sizes: {result['sizes']}\n")
    cols = ["setting", "macro_f1", "balanced_accuracy", "accuracy", "roc_auc_ovr", "ece",
            "recall_no_diabetes", "recall_prediabetes", "recall_diabetes", "train_seconds", "parallel_seconds", "comm_megabytes"]
    table = result["table"]
    _print(table[[c for c in cols if c in table.columns]].round(4))
    diff = result["fed_minus_central_macro_f1"]
    print(f"\nfederated minus centralized macro-F1: {diff['diff']:+.4f} (95 % bootstrap interval {diff['ci_low']:+.4f} to {diff['ci_high']:+.4f})")
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        table.to_csv(out / "compare.csv", index=False)
        result["central"].history.to_csv(out / "history_centralized.csv", index=False)
        result["federated"].history.to_csv(out / "history_federated.csv", index=False)
    return 0


def cmd_sweep(args) -> int:
    runs, summary = run_grid(_data(args), load_grid(args.config))
    _print(summary.round(4))
    print(f"wrote {write_grid(runs, summary, args.out)}")
    return 0


def cmd_demo(args) -> int:
    out = Path(args.out)
    path = write_synthetic(out / "synthetic.csv", n=args.rows, seed=42)
    frame = load(path)
    print("SYNTHETIC DATA: generated survey rows, not BRFSS records.\n")
    result = compare(frame, TrainConfig(rounds=20), scheme="dirichlet", n_clients=5, alpha=0.5)
    cols = ["setting", "macro_f1", "balanced_accuracy", "accuracy", "roc_auc_ovr", "recall_prediabetes", "recall_diabetes",
            "train_seconds", "parallel_seconds", "comm_megabytes"]
    _print(result["table"][[c for c in cols if c in result["table"].columns]].round(4))
    diff = result["fed_minus_central_macro_f1"]
    print(f"\nfederated minus centralized macro-F1: {diff['diff']:+.4f} ({diff['ci_low']:+.4f} to {diff['ci_high']:+.4f})\n")
    unweighted = compare(frame, TrainConfig(rounds=20, class_weight="none"), local_only=False, gradient_boosting=False)
    print("Without class weights (accuracy looks better, macro-F1 is worse):")
    _print(unweighted["table"][["setting", "macro_f1", "accuracy", "recall_prediabetes", "recall_diabetes"]].round(4))
    print()
    runs, summary = run_grid(frame, DEMO_GRID)
    _print(summary.round(4))
    write_grid(runs, summary, out)
    result["table"].to_csv(out / "compare.csv", index=False)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fedsugar", description="Federated vs centralized diabetes risk models")
    parser.add_argument("--version", action="version", version=f"fedsugar {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    def data_arg(p):
        p.add_argument("--data", help="CSV with the 21 BRFSS indicators and Diabetes_012")

    def train_args(p):
        data_arg(p)
        p.add_argument("--model", choices=MODELS, default="logreg")
        p.add_argument("--hidden", type=int, default=32)
        p.add_argument("--rounds", type=int, default=30)
        p.add_argument("--local-epochs", dest="local_epochs", type=int, default=1)
        p.add_argument("--lr", type=float, default=0.05)
        p.add_argument("--batch-size", dest="batch_size", type=int, default=64)
        p.add_argument("--class-weight", dest="class_weight", choices=CLASS_WEIGHTING, default="global")
        p.add_argument("--strategy", choices=STRATEGIES, default="fedavg")
        p.add_argument("--prox-mu", dest="prox_mu", type=float, default=0.01)
        p.add_argument("--server-lr", dest="server_lr", type=float, default=0.01)
        p.add_argument("--client-fraction", dest="client_fraction", type=float, default=1.0)
        p.add_argument("--dp-clip", dest="dp_clip", type=float)
        p.add_argument("--dp-noise", dest="dp_noise", type=float, default=0.0)
        p.add_argument("--seed", type=int)

    def partition_args(p):
        p.add_argument("--scheme", choices=SCHEMES, default="dirichlet")
        p.add_argument("--clients", type=int, default=5)
        p.add_argument("--alpha", type=float, default=0.5)

    p = sub.add_parser("synth", help="write synthetic survey rows with the BRFSS columns")
    p.add_argument("--out", default="data/synthetic.csv")
    p.add_argument("--rows", type=int, default=20_000)
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=cmd_synth)

    p = sub.add_parser("partition", help="show the rows per class of each client")
    data_arg(p)
    partition_args(p)
    p.add_argument("--seed", type=int)
    p.set_defaults(func=cmd_partition)

    p = sub.add_parser("centralized", help="train the centralized model")
    train_args(p)
    p.set_defaults(func=cmd_centralized)

    p = sub.add_parser("federated", help="train the federated model")
    train_args(p)
    partition_args(p)
    p.add_argument("--history")
    p.set_defaults(func=cmd_federated)

    p = sub.add_parser("compare", help="centralized vs federated vs baselines on one split")
    train_args(p)
    partition_args(p)
    p.add_argument("--out")
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("sweep", help="seed x partition x strategy x model grid from a TOML file")
    data_arg(p)
    p.add_argument("--config", default="configs/sweep.toml")
    p.add_argument("--out", default="runs/sweep")
    p.set_defaults(func=cmd_sweep)

    p = sub.add_parser("demo", help="synthetic data + comparison + small sweep, offline")
    p.add_argument("--out", default="runs/demo")
    p.add_argument("--rows", type=int, default=20_000)
    p.set_defaults(func=cmd_demo)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
