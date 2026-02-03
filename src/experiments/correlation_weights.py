
"""
Experiment 05: How feature dependence changes the purified decomposition.

We generate correlated (X1,X2), fit a depth-2 XGBoost regressor to a multiplicative target,
then compare purification under:
  - uniform weights (treat all bin combinations equally)
  - empirical weights (reflect the observed joint distribution)

This is an "additional experiment" extending the paper's discussion about weighting.

Outputs:
- results/correlation_weight_effect_summary.csv
- results/correlation_weight_effect_plot.pdf
- results/correlation_weight_effect_heatmaps.pdf
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from xgboost import XGBRegressor

from purify_fanova.xgb_decompose import decompose_xgb_depth2, predict_from_tensors
from purify_fanova.binning import discretize_matrix, compute_marginal_weights
from purify_fanova.purify import PurifyConfig, purify_all


def _make_correlated(n: int, rho: float, rng: np.random.Generator) -> np.ndarray:
    cov = np.array([[1.0, rho], [rho, 1.0]])
    L = np.linalg.cholesky(cov)
    Z = rng.normal(size=(n, 2))
    X = Z @ L.T
    return X


def _variance_weighted(vals: np.ndarray, w: np.ndarray) -> float:
    vals = np.asarray(vals, dtype=float)
    w = np.asarray(w, dtype=float)
    w = w / w.sum()
    mean = float((vals * w).sum())
    var = float(((vals - mean) ** 2 * w).sum())
    return var


def run(
    out_dir: str | Path = "results",
    n_samples: int = 4000,
    rhos: tuple[float, ...] = (0.0, 0.5, 0.9),
    n_estimators: int = 60,
    random_state: int = 0,
) -> pd.DataFrame:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(random_state)

    cfg = PurifyConfig(tol=1e-10, max_iter=200, verbose=False)

    rows = []
    mats_for_plot = {}

    for rho in rhos:
        X = _make_correlated(n_samples, rho, rng)
        # multiplicative target (pure interaction in independent case)
        y = X[:, 0] * X[:, 1] + 0.1 * rng.normal(size=n_samples)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=random_state)

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

        dec = decompose_xgb_depth2(model, X_train)
        tensors0 = dec.tensors
        fb = dec.feature_bins

        Xb_train = discretize_matrix(X_train, fb)
        Xb_test = discretize_matrix(X_test, fb)
        n_bins = {i: f.n_bins for i, f in enumerate(fb)}

        pred_model = model.predict(X_test)
        rmse = float(np.sqrt(mean_squared_error(y_test, pred_model)))

        # Purify under uniform and empirical
        purified = {}
        for mode in ["uniform", "empirical"]:
            weights = {}
            for key in tensors0.keys():
                if len(key) == 0:
                    continue
                weights[key] = compute_marginal_weights(Xb_train, key, n_bins, mode=mode)

            tens = {k: (v.copy() if isinstance(v, np.ndarray) else float(v)) for k, v in tensors0.items()}
            purify_all(tens, weights, config=cfg)
            purified[mode] = (tens, weights)

            pred_p = predict_from_tensors(tens, Xb_test)
            max_abs_err = float(np.max(np.abs(pred_model - pred_p)))

            if (0, 1) in tens:
                var_inter = _variance_weighted(np.asarray(tens[(0, 1)]), weights[(0, 1)])
            else:
                var_inter = 0.0

            rows.append(
                dict(
                    rho=rho,
                    weight_mode=mode,
                    rmse_test=rmse,
                    max_abs_pred_err=max_abs_err,
                    var_interaction=var_inter,
                )
            )

        # Compare purified interaction matrices
        if (0, 1) in purified["uniform"][0]:
            mat_u = np.asarray(purified["uniform"][0][(0, 1)])
            mat_e = np.asarray(purified["empirical"][0][(0, 1)])
            diff = float(np.linalg.norm(mat_u - mat_e) / (np.linalg.norm(mat_e) + 1e-12))
            rows.append(dict(rho=rho, weight_mode="diff_uniform_vs_empirical", rmse_test=rmse,
                             max_abs_pred_err=np.nan, var_interaction=diff))
            mats_for_plot[rho] = (mat_u, mat_e)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "correlation_weight_effect_summary.csv", index=False)

    # Plot summary metric (difference) vs rho
    df_diff = df[df["weight_mode"] == "diff_uniform_vs_empirical"].copy()
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    ax.plot(df_diff["rho"], df_diff["var_interaction"], marker="o")
    ax.set_xlabel("Correlation ρ between X1 and X2")
    ax.set_ylabel("Relative ||f01^unif - f01^emp||_2")
    ax.set_title("Purified interaction depends on weighting when features are dependent")
    fig.tight_layout()
    fig.savefig(out_dir / "correlation_weight_effect_plot.pdf")
    plt.close(fig)

    # Heatmaps for the largest rho
    rho_max = max(mats_for_plot.keys())
    mat_u, mat_e = mats_for_plot[rho_max]
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))
    for ax, mat, title in zip(
        axes,
        [mat_u, mat_e],
        ["Purified f01 (uniform weights)", "Purified f01 (empirical weights)"],
    ):
        im = ax.imshow(mat, origin="lower", aspect="auto")
        ax.set_title(f"{title} (ρ={rho_max})")
        ax.set_xlabel("X2 bin")
        ax.set_ylabel("X1 bin")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_dir / "correlation_weight_effect_heatmaps.pdf")
    plt.close(fig)

    return df


if __name__ == "__main__":
    run()
