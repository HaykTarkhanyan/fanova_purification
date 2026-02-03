
"""
Experiment 01: Convergence of mass-moving (Algorithm 1) on random matrices.

Replicates the qualitative behavior from Lengerich et al. (2020) Sec. 5.3 / Fig. 3:
most mass is moved in the first few sweeps, and uniform weights converge in 1 sweep.

Outputs:
- results/convergence_uniform.pdf
- results/convergence_random.pdf
- results/convergence_summary.csv
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from purify_fanova.metrics import purify_matrix_history


def run(
    out_dir: str | Path = "results",
    P: int = 50,
    sigmas: tuple[float, ...] = (1.0, 10.0),
    n_runs: int = 50,
    max_sweeps: int = 10,
    random_state: int = 0,
) -> pd.DataFrame:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(random_state)

    rows = []
    for sigma in sigmas:
        for weight_mode in ["uniform", "random"]:
            histories = []
            for _ in range(n_runs):
                T = rng.normal(0.0, sigma, size=(P, P))

                if weight_mode == "uniform":
                    W = np.ones((P, P), dtype=float)
                    W /= W.sum()
                else:
                    # positive random density (log-normal)
                    W = np.exp(rng.normal(0.0, 1.0, size=(P, P)))
                    W /= W.sum()

                _, hist = purify_matrix_history(T, W, max_sweeps=max_sweeps)
                # pad to length max_sweeps+1
                if len(hist) < max_sweeps + 1:
                    hist = hist + [hist[-1]] * (max_sweeps + 1 - len(hist))
                histories.append(hist[: max_sweeps + 1])

            H = np.array(histories)  # shape (n_runs, T)
            M0 = H[:, 0:1]
            cum_moved = 1.0 - (H / M0)
            mean = cum_moved.mean(axis=0)
            std = cum_moved.std(axis=0)
            for t in range(max_sweeps + 1):
                rows.append(
                    dict(
                        sigma=sigma,
                        weight_mode=weight_mode,
                        sweep=t,
                        cum_moved_mean=mean[t],
                        cum_moved_std=std[t],
                    )
                )

            # plot per setting
            fig, ax = plt.subplots(figsize=(6.0, 3.6))
            x = np.arange(max_sweeps + 1)
            ax.plot(x, mean * 100.0)
            ax.fill_between(x, (mean - std) * 100.0, (mean + std) * 100.0, alpha=0.2)
            ax.set_xlabel("Sweep (row/col zeroing)")
            ax.set_ylabel("Cumulative mass moved (%)")
            ax.set_title(f"Mass-moving convergence (P={P}, σ={sigma}, w={weight_mode})")
            ax.set_ylim(0, 100)
            fig.tight_layout()
            fig.savefig(out_dir / f"convergence_{weight_mode}_sigma{sigma}.pdf")
            plt.close(fig)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "convergence_summary.csv", index=False)
    return df


if __name__ == "__main__":
    run()
