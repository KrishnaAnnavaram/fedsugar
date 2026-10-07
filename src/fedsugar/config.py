"""Settings from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    data: Path = Path("data/diabetes_012_health_indicators_BRFSS2015.csv")
    output_dir: Path = Path("runs")
    seed: int = 42

    @classmethod
    def from_env(cls) -> "Settings":
        raw_seed = os.environ.get("FEDSUGAR_SEED", "").strip()
        try:
            seed = int(raw_seed) if raw_seed else 42
        except ValueError as exc:
            raise ValueError(f"FEDSUGAR_SEED must be an integer, got {raw_seed!r}") from exc
        return cls(
            data=Path(os.environ.get("FEDSUGAR_DATA", "").strip() or cls.data),
            output_dir=Path(os.environ.get("FEDSUGAR_OUTPUT_DIR", "").strip() or cls.output_dir),
            seed=seed,
        )
