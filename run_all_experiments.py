
#!/usr/bin/env python3
"""
Run all experiments in sequence and write outputs to ./results.

Usage:
  python run_all_experiments.py
"""
from __future__ import annotations

from pathlib import Path

from purify_fanova.experiments import (
    convergence,
    boolean_example,
    log_puzzle,
    xgb_friedman,
    correlation_weights,
    german_credit,
    stress_dependence,
)


def main() -> None:
    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[01] Convergence ...")
    convergence.run(out_dir=out_dir)

    print("[02] Boolean identifiability ...")
    boolean_example.run(out_dir=out_dir)

    print("[03] Log puzzle ...")
    log_puzzle.run(out_dir=out_dir)

    print("[04] XGBoost (Friedman #1) ...")
    xgb_friedman.run(out_dir=out_dir)

    print("[05] Correlation vs weight estimator ...")
    correlation_weights.run(out_dir=out_dir)

    print("[06] German credit dataset ...")
    # Uses the attached CSV copied into ./data by default.
    german_credit.run(out_dir=out_dir, data_path=Path("data") / "german_credit_data.csv")

    print("[07] Dependence stress-test (quick mode) ...")
    # quick=True keeps runtime low; set quick=False in
    # purify_fanova/experiments/stress_dependence.py for the full matrix.
    stress_dependence.run(out_dir=out_dir, quick=True)

    print("Done. Outputs in", out_dir.resolve())


if __name__ == "__main__":
    main()
