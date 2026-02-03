
"""
Experiment 02: Boolean identifiability toy example (Fig. 1 style).

We construct two different additive-with-interaction parameterizations that represent the
same Boolean function, then purify both and show they collapse to the same fANOVA form.

Outputs:
- results/boolean_unpurified_vs_purified.pdf
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from purify_fanova.purify import PurifyConfig, purify_all
from purify_fanova.binning import compute_marginal_weights


def _predict_from_tensors(tensors, x1, x2):
    b1 = np.array(x1, dtype=int)
    b2 = np.array(x2, dtype=int)
    y = float(tensors.get((), 0.0))
    y += tensors[(0,)][b1]
    y += tensors[(1,)][b2]
    y += tensors[(0, 1)][b1, b2]
    return y


def run(out_dir: str | Path = "results") -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Domain: X1,X2 in {0,1}, uniform distribution
    Xb = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=int)
    y_and = np.array([0, 0, 0, 1], dtype=float)  # AND truth table in row-major order

    # Decomposition A: put everything into interaction (unpurified)
    T_A = {
        (): 0.0,
        (0,): np.zeros(2),
        (1,): np.zeros(2),
        (0, 1): y_and.reshape(2, 2),
    }

    # Decomposition B: shift mass into main effects + different interaction, but same total function
    # Pick arbitrary main effects, then set interaction to make the sum exactly equal to y_and.
    f1 = np.array([-0.1, 0.3])
    f2 = np.array([0.2, -0.05])
    f0 = 0.12
    interaction = y_and.reshape(2, 2) - f0 - f1[:, None] - f2[None, :]
    T_B = {
        (): float(f0),
        (0,): f1.copy(),
        (1,): f2.copy(),
        (0, 1): interaction.copy(),
    }

    # Verify equivalence pre-purification
    yA = np.array([_predict_from_tensors(T_A, x1, x2) for x1, x2 in Xb])
    yB = np.array([_predict_from_tensors(T_B, x1, x2) for x1, x2 in Xb])
    assert np.allclose(yA, y_and)
    assert np.allclose(yB, y_and)

    # Weights
    n_bins = {0: 2, 1: 2}
    w01 = compute_marginal_weights(Xb, (0, 1), n_bins, mode="uniform")
    w0 = compute_marginal_weights(Xb, (0,), n_bins, mode="uniform")
    w1 = compute_marginal_weights(Xb, (1,), n_bins, mode="uniform")
    weights = {(0, 1): w01, (0,): w0, (1,): w1}

    # Purify both
    cfg = PurifyConfig(tol=1e-12, max_iter=100, verbose=False)
    T_A_p = {k: (v.copy() if isinstance(v, np.ndarray) else float(v)) for k, v in T_A.items()}
    T_B_p = {k: (v.copy() if isinstance(v, np.ndarray) else float(v)) for k, v in T_B.items()}
    purify_all(T_A_p, weights, config=cfg)
    purify_all(T_B_p, weights, config=cfg)

    # Plot: unpurified vs purified (interaction matrices and main effects)
    fig, axes = plt.subplots(2, 3, figsize=(9.5, 6.0))

    def heat(ax, mat, title):
        im = ax.imshow(mat, aspect="equal")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["X2=0", "X2=1"])
        ax.set_yticklabels(["X1=0", "X1=1"])
        ax.set_title(title)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    heat(axes[0, 0], T_A[(0, 1)], "A: interaction (unpurified)")
    heat(axes[1, 0], T_B[(0, 1)], "B: interaction (unpurified)")
    heat(axes[0, 1], T_A_p[(0, 1)], "A: interaction (purified)")
    heat(axes[1, 1], T_B_p[(0, 1)], "B: interaction (purified)")

    axes[0, 2].plot([0, 1], T_A_p[(0,)], marker="o", label="f1 (A)")
    axes[0, 2].plot([0, 1], T_B_p[(0,)], marker="o", linestyle="--", label="f1 (B)")
    axes[0, 2].set_title("Purified main effect f1(X1)")
    axes[0, 2].set_xticks([0, 1]); axes[0, 2].set_xticklabels(["0", "1"])
    axes[0, 2].legend()

    axes[1, 2].plot([0, 1], T_A_p[(1,)], marker="o", label="f2 (A)")
    axes[1, 2].plot([0, 1], T_B_p[(1,)], marker="o", linestyle="--", label="f2 (B)")
    axes[1, 2].set_title("Purified main effect f2(X2)")
    axes[1, 2].set_xticks([0, 1]); axes[1, 2].set_xticklabels(["0", "1"])
    axes[1, 2].legend()

    fig.suptitle(
        "Different parameterizations (A,B) of AND collapse to the same fANOVA after purification",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "boolean_unpurified_vs_purified.pdf")
    plt.close(fig)


if __name__ == "__main__":
    run()
