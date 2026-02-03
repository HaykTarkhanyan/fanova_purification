"""purify_fanova.experiments.stress_dependence

Experiment: Stress-test purification under strong dependence (no copulas).

Core question
-------------
When features are strongly dependent, how much do purified main/interaction
effects change when we purify under:

  (W1) empirical joint weights (≈ p(x))
  (W2) uniform weights
  (W3) independence-assumed weights (product of empirical marginals)
  (W4) permutation joint weights (shuffle columns to break dependence)

This directly probes the paper's point that explanations depend on the
weighting distribution "w" and that specifying the data distribution is
essential for identifiability/interpretability.

Outputs
-------
Written to out_dir:

* stress_dependence_raw.csv
    Per-seed results (metrics for each rho/function/weight_mode)

* stress_dependence_summary.csv
    Aggregated (mean±std) curves vs rho.

* stress_dependence_metrics_F1.pdf, stress_dependence_metrics_F2.pdf
    Curves vs rho (L2 diff, sign flip rate, interaction variance share).

* stress_dependence_heatmaps_F1_rho0.97.pdf, stress_dependence_heatmaps_F2_rho0.97.pdf
    Interaction surface heatmaps under each weight mode (one representative seed).

* stress_dependence_mains_F1_rho0.97.pdf, stress_dependence_mains_F2_rho0.97.pdf
    Main effect curves f1 and f2 under each weight mode (one representative seed).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from xgboost import XGBRegressor

from purify_fanova.binning import compute_marginal_weights, discretize_matrix
from purify_fanova.metrics import (
    sign_flip_rate,
    weighted_l2,
    weighted_mean,
    weighted_variance,
)
from purify_fanova.purify import PurifyConfig, purify_all
from purify_fanova.xgb_decompose import decompose_xgb_depth2, predict_from_tensors


WeightMode = Literal["empirical", "uniform", "indep", "perm_joint"]
FunctionName = Literal["F1", "F2"]


def _make_linear_gaussian(
    n: int,
    rho: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Option A from the spec (exact correlation control).

    X1 ~ N(0,1)
    X2 = rho X1 + sqrt(1-rho^2) eps, eps~N(0,1).
    """
    x1 = rng.normal(size=n)
    eps = rng.normal(size=n)
    x2 = rho * x1 + np.sqrt(max(0.0, 1.0 - rho**2)) * eps
    return np.column_stack([x1, x2])


def _truth_F1(X: np.ndarray, rng: np.random.Generator, noise: float) -> np.ndarray:
    """Multiplicative interaction + main effects."""
    x1 = X[:, 0]
    x2 = X[:, 1]
    beta1, beta2, gamma = 1.0, 1.0, 1.5
    y = beta1 * x1 + beta2 * x2 + gamma * (x1 * x2)
    if noise > 0:
        y = y + noise * rng.normal(size=X.shape[0])
    return y


def _truth_F2(X: np.ndarray, rng: np.random.Generator, noise: float) -> np.ndarray:
    """XOR-like sign interaction + (small) mains."""
    x1 = X[:, 0]
    x2 = X[:, 1]
    beta1, beta2, gamma = 0.5, 0.5, 2.0
    y = gamma * np.sign(x1) * np.sign(x2) + beta1 * x1 + beta2 * x2
    if noise > 0:
        y = y + noise * rng.normal(size=X.shape[0])
    return y


def _quadrant_mask(reps1: np.ndarray, reps2: np.ndarray, quad: str) -> np.ndarray:
    if quad not in {"pp", "pn", "np", "nn"}:
        raise ValueError("quad must be one of pp/pn/np/nn")
    s1 = reps1 > 0
    s2 = reps2 > 0
    if quad == "pp":
        return np.outer(s1, s2)
    if quad == "pn":
        return np.outer(s1, ~s2)
    if quad == "np":
        return np.outer(~s1, s2)
    return np.outer(~s1, ~s2)


def _interaction_quadrant_sign(
    f12: np.ndarray,
    w_ref: np.ndarray,
    reps1: np.ndarray,
    reps2: np.ndarray,
    quad: str = "pp",
    tol: float = 1e-12,
) -> int:
    """Sign of mean interaction in a quadrant under reference weights."""
    mask = _quadrant_mask(reps1, reps2, quad)
    w = np.asarray(w_ref, dtype=float) * mask
    if float(w.sum()) <= 0:
        return 0
    m = weighted_mean(np.asarray(f12, dtype=float), w)
    if m > tol:
        return 1
    if m < -tol:
        return -1
    return 0


def _monotonic_fraction(f: np.ndarray) -> float:
    f = np.asarray(f, dtype=float)
    if f.ndim != 1 or f.shape[0] < 2:
        return 1.0
    d = np.diff(f)
    return float((d >= 0).mean())


def _build_weights(
    Xb_train: np.ndarray,
    tensors0: Dict[Tuple[int, ...], np.ndarray | float],
    n_bins: Dict[int, int],
    mode: WeightMode,
    rng: np.random.Generator,
) -> Dict[Tuple[int, ...], np.ndarray]:
    weights: Dict[Tuple[int, ...], np.ndarray] = {}
    for key in tensors0.keys():
        if len(key) == 0:
            continue
        weights[key] = compute_marginal_weights(Xb_train, key, n_bins, mode=mode, rng=rng)
    return weights


def _interaction_variance_share(
    intercept: float,
    f1: np.ndarray,
    f2: np.ndarray,
    f12: np.ndarray,
    w12: np.ndarray,
) -> float:
    """Var_w(f12) / Var_w(F) on the discrete bin grid."""
    f1 = np.asarray(f1, dtype=float)
    f2 = np.asarray(f2, dtype=float)
    f12 = np.asarray(f12, dtype=float)
    w12 = np.asarray(w12, dtype=float)
    if f12.shape != w12.shape:
        raise ValueError("Shape mismatch in _interaction_variance_share")

    F = intercept + f12 + f1[:, None] + f2[None, :]
    var_f12 = weighted_variance(f12, w12)
    var_F = weighted_variance(F, w12)
    return float(var_f12 / (var_F + 1e-18))


@dataclass
class _PlotCache:
    tensors0: Dict[Tuple[int, ...], np.ndarray | float]
    purified: Dict[WeightMode, Dict[Tuple[int, ...], np.ndarray | float]]
    reps1: np.ndarray
    reps2: np.ndarray


def run(
    out_dir: str | Path = "results",
    n_samples: int = 4000,
    test_size: float = 0.25,
    rhos: Sequence[float] = (0.0, 0.3, 0.6, 0.9, 0.97),
    functions: Sequence[FunctionName] = ("F1", "F2"),
    weight_modes: Sequence[WeightMode] = ("empirical", "uniform", "indep", "perm_joint"),
    seeds: Sequence[int] = tuple(range(10)),
    noise: float = 0.2,
    n_estimators: int = 60,
    random_state: int = 0,
    quick: bool = False,
) -> pd.DataFrame:
    """Run the dependence stress-test.

    If quick=True, we reduce runtime by using 3 seeds and fewer trees.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if quick:
        seeds = tuple(seeds)[:3]
        n_estimators = min(n_estimators, 40)

    cfg = PurifyConfig(tol=1e-10, max_iter=300, verbose=False)

    rows: List[dict] = []
    plot_cache: Dict[Tuple[FunctionName, float], _PlotCache] = {}

    for fn in functions:
        for rho in rhos:
            for seed in seeds:
                rng = np.random.default_rng(seed)
                X = _make_linear_gaussian(n_samples, float(rho), rng)
                if fn == "F1":
                    y = _truth_F1(X, rng=rng, noise=noise)
                elif fn == "F2":
                    y = _truth_F2(X, rng=rng, noise=noise)
                else:
                    raise ValueError(f"Unknown function {fn}")

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=seed
                )

                # Piecewise-constant learner (depth-2 XGBoost)
                model = XGBRegressor(
                    n_estimators=int(n_estimators),
                    max_depth=2,
                    learning_rate=0.1,
                    subsample=1.0,
                    colsample_bytree=1.0,
                    reg_lambda=0.0,
                    min_child_weight=1.0,
                    objective="reg:squarederror",
                    random_state=int(random_state + seed),
                    n_jobs=1,
                )
                model.fit(X_train, y_train)

                # Decompose model to unpurified tensors
                dec = decompose_xgb_depth2(model, X_train, feature_names=["X1", "X2"])
                tensors0 = dec.tensors
                fb = dec.feature_bins
                n_bins = {i: f.n_bins for i, f in enumerate(fb)}

                Xb_train = discretize_matrix(X_train, fb)
                Xb_test = discretize_matrix(X_test, fb)

                # Sanity: tensor predictions match model predictions
                pred_model = model.predict(X_test)
                pred_tensor0 = predict_from_tensors(tensors0, Xb_test)
                max_abs_pred_err0 = float(np.max(np.abs(pred_model - pred_tensor0)))
                rmse = float(np.sqrt(mean_squared_error(y_test, pred_model)))

                # Reference weights (empirical) for comparability
                w_ref_1 = compute_marginal_weights(Xb_train, (0,), n_bins, mode="empirical")
                w_ref_2 = compute_marginal_weights(Xb_train, (1,), n_bins, mode="empirical")
                w_ref_12 = compute_marginal_weights(Xb_train, (0, 1), n_bins, mode="empirical")

                # Purify under each weighting choice
                purified_by_mode: Dict[WeightMode, Dict[Tuple[int, ...], np.ndarray | float]] = {}
                weights_by_mode: Dict[WeightMode, Dict[Tuple[int, ...], np.ndarray]] = {}

                # Separate RNG stream for weight perturbations (perm_joint)
                rng_w = np.random.default_rng(10_000 + seed)

                for mode in weight_modes:
                    weights = _build_weights(Xb_train, tensors0, n_bins, mode=mode, rng=rng_w)

                    tens = {k: (v.copy() if isinstance(v, np.ndarray) else float(v)) for k, v in tensors0.items()}
                    purify_all(tens, weights, config=cfg)
                    purified_by_mode[mode] = tens
                    weights_by_mode[mode] = weights

                    # Prediction invariance check
                    pred_p = predict_from_tensors(tens, Xb_test)
                    max_abs_pred_err = float(np.max(np.abs(pred_model - pred_p)))

                    # Components of interest
                    f1 = np.asarray(tens.get((0,), 0.0), dtype=float)
                    f2 = np.asarray(tens.get((1,), 0.0), dtype=float)
                    f12 = np.asarray(tens.get((0, 1), np.zeros((len(f1), len(f2)))), dtype=float)
                    w12 = weights.get((0, 1), w_ref_12)

                    inter_share = _interaction_variance_share(
                        intercept=float(tens.get((), 0.0)),
                        f1=f1,
                        f2=f2,
                        f12=f12,
                        w12=w12,
                    )

                    rows.append(
                        dict(
                            function=fn,
                            rho=float(rho),
                            seed=int(seed),
                            weight_mode=mode,
                            rmse_test=rmse,
                            max_abs_pred_err0=max_abs_pred_err0,
                            max_abs_pred_err=max_abs_pred_err,
                            inter_var_share=inter_share,
                        )
                    )

                # Comparison metrics vs empirical baseline (same seed/rho/function)
                base = purified_by_mode["empirical"]
                f1_base = np.asarray(base.get((0,), 0.0), dtype=float)
                f2_base = np.asarray(base.get((1,), 0.0), dtype=float)
                f12_base = np.asarray(
                    base.get((0, 1), np.zeros((len(f1_base), len(f2_base)))), dtype=float
                )
                reps1 = fb[0].reps
                reps2 = fb[1].reps

                quad_base = _interaction_quadrant_sign(f12_base, w_ref_12, reps1, reps2, quad="pp")
                mono_base = _monotonic_fraction(f1_base) >= 0.9

                for mode in weight_modes:
                    if mode == "empirical":
                        continue
                    tens = purified_by_mode[mode]
                    f1 = np.asarray(tens.get((0,), 0.0), dtype=float)
                    f2 = np.asarray(tens.get((1,), 0.0), dtype=float)
                    f12 = np.asarray(tens.get((0, 1), np.zeros_like(f12_base)), dtype=float)

                    rows.append(
                        dict(
                            function=fn,
                            rho=float(rho),
                            seed=int(seed),
                            weight_mode=f"diff_vs_empirical::{mode}",
                            l2_f1=weighted_l2(f1, f1_base, w_ref_1),
                            l2_f2=weighted_l2(f2, f2_base, w_ref_2),
                            l2_f12=weighted_l2(f12, f12_base, w_ref_12),
                            signflip_f1=sign_flip_rate(f1, f1_base),
                            signflip_f12=sign_flip_rate(f12, f12_base),
                            quad_pp_sign=_interaction_quadrant_sign(f12, w_ref_12, reps1, reps2, quad="pp"),
                            quad_pp_changed=int(_interaction_quadrant_sign(f12, w_ref_12, reps1, reps2, quad="pp") != quad_base),
                            mono_frac=_monotonic_fraction(f1),
                            mono_changed=int((_monotonic_fraction(f1) >= 0.9) != mono_base),
                        )
                    )

                # Cache one representative run (strong dependence) for plots
                if seed == seeds[0] and float(rho) == max(rhos):
                    plot_cache[(fn, float(rho))] = _PlotCache(
                        tensors0=tensors0,
                        purified=purified_by_mode,
                        reps1=reps1,
                        reps2=reps2,
                    )

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "stress_dependence_raw.csv", index=False)

    # Aggregated curves for diff metrics
    df_diff = df[df["weight_mode"].astype(str).str.startswith("diff_vs_empirical::")].copy()
    if not df_diff.empty:
        df_diff["mode"] = df_diff["weight_mode"].astype(str).str.split("::", n=1).str[1]
        summary = (
            df_diff.groupby(["function", "mode", "rho"], as_index=False)
            .agg(
                l2_f1_mean=("l2_f1", "mean"),
                l2_f1_std=("l2_f1", "std"),
                l2_f2_mean=("l2_f2", "mean"),
                l2_f2_std=("l2_f2", "std"),
                l2_f12_mean=("l2_f12", "mean"),
                l2_f12_std=("l2_f12", "std"),
                signflip_f1_mean=("signflip_f1", "mean"),
                signflip_f12_mean=("signflip_f12", "mean"),
                quad_pp_changed_mean=("quad_pp_changed", "mean"),
                mono_changed_mean=("mono_changed", "mean"),
            )
        )
        summary.to_csv(out_dir / "stress_dependence_summary.csv", index=False)

        # Also save interaction-variance-share curves (by actual weight_mode)
        df_share = df[df["weight_mode"].isin(weight_modes)].copy()
        share = (
            df_share.groupby(["function", "weight_mode", "rho"], as_index=False)
            .agg(inter_var_share_mean=("inter_var_share", "mean"), inter_var_share_std=("inter_var_share", "std"))
        )
        share.to_csv(out_dir / "stress_dependence_interaction_share.csv", index=False)

        # Plots: curves vs rho
        for fn in functions:
            fig, ax = plt.subplots(figsize=(6.6, 4.0))
            sub = summary[summary["function"] == fn]
            for mode in sorted(sub["mode"].unique()):
                s = sub[sub["mode"] == mode]
                ax.plot(s["rho"], s["l2_f12_mean"], marker="o", label=str(mode))
            ax.set_xlabel("Dependence strength ρ")
            ax.set_ylabel("Weighted L2 distance of interaction (vs empirical)")
            ax.set_title(f"{fn}: ||f12(w) - f12(emp)|| under w_emp measure")
            ax.legend(title="purification weights")
            fig.tight_layout()
            fig.savefig(out_dir / f"stress_dependence_l2_interaction_{fn}.pdf")
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(6.6, 4.0))
            for mode in sorted(sub["mode"].unique()):
                s = sub[sub["mode"] == mode]
                ax.plot(s["rho"], s["signflip_f12_mean"], marker="o", label=str(mode))
            ax.set_xlabel("Dependence strength ρ")
            ax.set_ylabel("Sign flip rate (interaction bins)")
            ax.set_title(f"{fn}: Sign instability of f12 vs empirical")
            ax.legend(title="purification weights")
            fig.tight_layout()
            fig.savefig(out_dir / f"stress_dependence_signflip_interaction_{fn}.pdf")
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(6.6, 4.0))
            sub2 = share[share["function"] == fn]
            for mode in weight_modes:
                s = sub2[sub2["weight_mode"] == mode]
                ax.plot(s["rho"], s["inter_var_share_mean"], marker="o", label=str(mode))
            ax.set_xlabel("Dependence strength ρ")
            ax.set_ylabel("Var_w(f12) / Var_w(F)")
            ax.set_title(f"{fn}: Interaction variance share depends on w")
            ax.legend(title="weights")
            fig.tight_layout()
            fig.savefig(out_dir / f"stress_dependence_interaction_share_{fn}.pdf")
            plt.close(fig)

    # Visualizations for the strongest dependence setting
    for (fn, rho), cache in plot_cache.items():
        tens0 = cache.tensors0
        reps1, reps2 = cache.reps1, cache.reps2

        mat0 = np.asarray(tens0.get((0, 1), np.zeros((len(reps1), len(reps2)))), dtype=float)

        # Heatmaps: unpurified + each weight_mode
        cols = ["unpurified"] + list(weight_modes)
        ncols = len(cols)
        fig, axes = plt.subplots(1, ncols, figsize=(3.2 * ncols, 3.8))
        if ncols == 1:
            axes = [axes]
        for ax, name in zip(axes, cols):
            if name == "unpurified":
                mat = mat0
                title = "unpurified"
            else:
                mat = np.asarray(cache.purified[name].get((0, 1), np.zeros_like(mat0)), dtype=float)
                title = str(name)
            im = ax.imshow(mat, origin="lower", aspect="auto")
            ax.set_title(title)
            ax.set_xlabel("X2 bin")
            ax.set_ylabel("X1 bin")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.suptitle(f"{fn}: interaction surface at ρ={rho} under different w")
        fig.tight_layout()
        fig.savefig(out_dir / f"stress_dependence_heatmaps_{fn}_rho{rho}.pdf")
        plt.close(fig)

        # Main effects for X1 and X2
        fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.6))
        for ax, feat, reps in zip(axes, [0, 1], [reps1, reps2]):
            for mode in weight_modes:
                f = np.asarray(cache.purified[mode].get((feat,), np.zeros_like(reps)), dtype=float)
                ax.plot(reps, f, label=str(mode))
            ax.set_xlabel(f"Representative value (X{feat+1} bin)")
            ax.set_ylabel(f"Purified f{feat+1}(x)")
            ax.set_title(f"Main effect f{feat+1} under different w")
            ax.legend()
        fig.suptitle(f"{fn}: main effects at ρ={rho}")
        fig.tight_layout()
        fig.savefig(out_dir / f"stress_dependence_mains_{fn}_rho{rho}.pdf")
        plt.close(fig)

    return df


if __name__ == "__main__":
    run()
