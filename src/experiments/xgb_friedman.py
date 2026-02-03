
"""
Experiment 04: XGBoost depth-2 model -> tensor decomposition -> purification on Friedman #1.

This is a self-contained, offline proxy for the paper's GA2M / XGB2 experiments.

We train a depth-2 XGBoost regressor, decompose it into unpurified tensors, then purify
under different weight estimators:
  - uniform
  - empirical (train distribution)
  - laplace (empirical + uniform)

We then compare how the purified main/interaction terms change with the weighting.

Outputs:
- results/friedman_decomp_sanity.csv
- results/friedman_interaction_01_unpurified_vs_purified.pdf
- results/friedman_main_effect_feature0.pdf
- results/friedman_strength_by_weight.csv
"""
from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import make_friedman1
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from xgboost import XGBRegressor

from purify_fanova.xgb_decompose import decompose_xgb_depth2, predict_from_tensors
from purify_fanova.binning import discretize_matrix, compute_marginal_weights
from purify_fanova.purify import PurifyConfig, purify_all


def _variance_weighted(vals: np.ndarray, w: np.ndarray) -> float:
    vals = np.asarray(vals, dtype=float)
    w = np.asarray(w, dtype=float)
    w = w / w.sum()
    mean = float((vals * w).sum())
    var = float(((vals - mean) ** 2 * w).sum())
    return var


def run(
    out_dir: str | Path = "results",
    n_samples: int = 5000,
    noise: float = 1.0,
    n_estimators: int = 80,
    random_state: int = 0,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Data
    X, y = make_friedman1(n_samples=n_samples, n_features=5, noise=noise, random_state=random_state)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=random_state)

    # Model (depth-2)
    model = XGBRegressor(
        n_estimators=n_estimators,
        max_depth=2,
        learning_rate=0.1,
        subsample=1.0,
        colsample_bytree=1.0,
        reg_lambda=0.0,
        min_child_weight=1.0,
        objective="reg:squarederror",
        random_state=random_state,
        n_jobs=1,
    )
    model.fit(X_train, y_train)

    # Decompose
    dec = decompose_xgb_depth2(model, X_train)
    tensors0 = dec.tensors
    feature_bins = dec.feature_bins

    # Discretize train/test
    Xb_train = discretize_matrix(X_train, feature_bins)
    Xb_test = discretize_matrix(X_test, feature_bins)
    n_bins = {i: fb.n_bins for i, fb in enumerate(feature_bins)}

    # Sanity: predictions match exactly (unpurified)
    pred_model = model.predict(X_test)
    pred_tensor = predict_from_tensors(tensors0, Xb_test)
    max_abs_err = float(np.max(np.abs(pred_model - pred_tensor)))
    rmse = float(np.sqrt(mean_squared_error(y_test, pred_model)))

    pd.DataFrame(
        [dict(rmse_test=rmse, max_abs_pred_err=max_abs_err, n_estimators=n_estimators, noise=noise)]
    ).to_csv(out_dir / "friedman_decomp_sanity.csv", index=False)

    # Weight modes
    weight_modes = ["uniform", "empirical", "laplace"]
    cfg = PurifyConfig(tol=1e-10, max_iter=200, verbose=False)

    strength_rows = []
    purified_by_mode = {}

    for mode in weight_modes:
        # weights for every key in tensors (except intercept)
        weights = {}
        for key, val in tensors0.items():
            if len(key) == 0:
                continue
            weights[key] = compute_marginal_weights(Xb_train, key, n_bins, mode=mode)

        # Purify
        tensors = {k: (v.copy() if isinstance(v, np.ndarray) else float(v)) for k, v in tensors0.items()}
        purify_all(tensors, weights, config=cfg)
        purified_by_mode[mode] = tensors

        # verify predictions unchanged
        pred_p = predict_from_tensors(tensors, Xb_test)
        max_abs_err_p = float(np.max(np.abs(pred_model - pred_p)))

        # interaction strength for (0,1)
        key01 = (0, 1)
        if key01 in tensors:
            var_inter = _variance_weighted(np.asarray(tensors[key01]), weights[key01])
        else:
            var_inter = 0.0

        strength_rows.append(
            dict(
                weight_mode=mode,
                max_abs_pred_err=max_abs_err_p,
                var_interaction_01=var_inter,
            )
        )

    pd.DataFrame(strength_rows).to_csv(out_dir / "friedman_strength_by_weight.csv", index=False)

    # Plot interaction (0,1) unpurified vs purified (empirical)
    key01 = (0, 1)
    if key01 in tensors0:
        mat0 = np.asarray(tensors0[key01])
        mat_emp = np.asarray(purified_by_mode["empirical"][key01])
        fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))
        for ax, mat, title in zip(
            axes,
            [mat0, mat_emp],
            ["Unpurified interaction tensor T01", "Purified interaction f01 (empirical weights)"],
        ):
            im = ax.imshow(mat, origin="lower", aspect="auto")
            ax.set_title(title)
            ax.set_xlabel("Feature 1 bin")
            ax.set_ylabel("Feature 0 bin")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.suptitle("Friedman #1: interaction between X1 and X2 (feature 0 and 1)")
        fig.tight_layout()
        fig.savefig(out_dir / "friedman_interaction_01_unpurified_vs_purified.pdf")
        plt.close(fig)

    # Plot main effect for feature 0 under different weights
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    reps0 = feature_bins[0].reps
    for mode in weight_modes:
        ax.plot(reps0, np.asarray(purified_by_mode[mode][(0,)]), label=mode)
    ax.set_xlabel("Representative value (feature 0 bin)")
    ax.set_ylabel("Purified main effect f0(x0)")
    ax.set_title("Main effect of feature 0 under different weight estimators")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "friedman_main_effect_feature0.pdf")
    plt.close(fig)


if __name__ == "__main__":
    run()
