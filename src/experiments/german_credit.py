"""purify_fanova.experiments.german_credit

Experiment: Run tensor decomposition + purification on the (attached) German
Credit dataset.

The goal is to provide a small, *offline* real-data experiment demonstrating
that purified effects can differ substantially depending on the weighting
distribution w.

Important note about the attached CSV
------------------------------------
The provided `german_credit_data.csv` does not include an explicit label column
in some variants. This script therefore:

1) Uses a label column if it exists (common names: Risk/target/y/...)
2) Otherwise, defaults to predicting `Credit amount` as a regression task.

Outputs
-------
Written to out_dir:

* german_credit_summary.csv
    Overall fit metrics + prediction invariance checks.

* german_credit_top_interactions.csv
    Interaction variances (empirical weights) for the purified model.

* german_credit_top_interaction_heatmap.pdf
    Unpurified vs purified interaction surface for the strongest interaction.

* german_credit_top_mains.pdf
    Main effect curves for the top-variance features (empirical weights).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, roc_auc_score

from xgboost import XGBClassifier, XGBRegressor

from purify_fanova.binning import compute_marginal_weights, discretize_matrix
from purify_fanova.metrics import weighted_variance
from purify_fanova.purify import PurifyConfig, purify_all
from purify_fanova.xgb_decompose import decompose_xgb_depth2, predict_from_tensors


WeightMode = Literal["empirical", "uniform", "indep"]


def _detect_label_column(df: pd.DataFrame) -> Optional[str]:
    candidates = [
        "Risk",
        "risk",
        "target",
        "Target",
        "y",
        "Y",
        "default",
        "Default",
        "class",
        "Class",
        "label",
        "Label",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _prep_features_and_target(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, np.ndarray, str, str]:
    """Return X_df, y, task_type, y_name."""
    df = df.copy()

    # Drop a common index column if present
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    y_col = _detect_label_column(df)
    if y_col is not None:
        y_raw = df.pop(y_col)
        y_name = y_col
        # Heuristic: treat as binary classification if few unique values
        uniq = pd.unique(y_raw.dropna())
        if len(uniq) <= 10:
            # map to 0/1 if needed
            if y_raw.dtype == object:
                y = (y_raw.astype(str).str.lower().isin(["bad", "1", "true", "yes"])) .astype(int).to_numpy()
            else:
                y = y_raw.to_numpy()
                # ensure 0/1
                if set(np.unique(y)) not in ({0, 1}, {1, 2}):
                    y = (y > np.median(y)).astype(int)
                elif set(np.unique(y)) == {1, 2}:
                    y = (y == 2).astype(int)
            task = "classification"
        else:
            y = y_raw.to_numpy(dtype=float)
            task = "regression"
    else:
        # Fallback: predict Credit amount
        if "Credit amount" not in df.columns:
            raise ValueError(
                "No label column found and no 'Credit amount' column to use as a fallback target."
            )
        y_name = "Credit amount"
        y = df.pop("Credit amount").to_numpy(dtype=float)
        task = "regression"

    # Identify categorical columns (even if stored as ints)
    cat_cols: List[str] = []
    for c in df.columns:
        if df[c].dtype == object:
            cat_cols.append(c)
    # Heuristic: treat some known integer-coded categoricals as categorical
    for c in ["Job", "Housing"]:
        if c in df.columns and c not in cat_cols:
            cat_cols.append(c)

    # Fill missing categoricals
    for c in cat_cols:
        df[c] = df[c].astype("string").fillna("unknown")

    # One-hot encode categoricals
    X_df = pd.get_dummies(df, columns=cat_cols, drop_first=False)
    # Ensure numeric dtype for xgboost
    X_df = X_df.astype(float)

    return X_df, y, task, y_name


def run(
    out_dir: str | Path = "results",
    data_path: str | Path = "data/german_credit_data.csv",
    test_size: float = 0.25,
    weight_modes: Sequence[WeightMode] = ("empirical", "uniform", "indep"),
    n_estimators: int = 120,
    random_state: int = 0,
) -> pd.DataFrame:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Could not find {data_path}. Place the CSV at that path or pass data_path=..."
        )

    df = pd.read_csv(data_path)
    X_df, y, task, y_name = _prep_features_and_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X_df.to_numpy(), y, test_size=test_size, random_state=random_state
    )

    feature_names = list(X_df.columns)

    if task == "classification":
        model = XGBClassifier(
            n_estimators=int(n_estimators),
            max_depth=2,
            learning_rate=0.1,
            subsample=1.0,
            colsample_bytree=1.0,
            reg_lambda=0.0,
            min_child_weight=1.0,
            objective="binary:logistic",
            random_state=int(random_state),
            n_jobs=1,
        )
    else:
        model = XGBRegressor(
            n_estimators=int(n_estimators),
            max_depth=2,
            learning_rate=0.1,
            subsample=1.0,
            colsample_bytree=1.0,
            reg_lambda=0.0,
            min_child_weight=1.0,
            objective="reg:squarederror",
            random_state=int(random_state),
            n_jobs=1,
        )

    model.fit(X_train, y_train)

    # Performance on test
    # NOTE: For XGBoost classification with objective="binary:logistic", the
    # tree leaf values live in *logit / margin* space, and probabilities are
    # obtained by applying the sigmoid link. The tensor decomposition sums leaf
    # contributions (i.e., produces margins), so prediction invariance is checked
    # in margin space.
    if task == "classification":
        proba = model.predict_proba(X_test)[:, 1]
        metric_name = "auc"
        metric_value = float(roc_auc_score(y_test, proba))
        pred_model = model.predict(X_test, output_margin=True)
    else:
        pred_model = model.predict(X_test)
        metric_name = "rmse"
        metric_value = float(np.sqrt(mean_squared_error(y_test, pred_model)))

    # Decompose
    dec = decompose_xgb_depth2(model, X_train, feature_names=feature_names)
    tensors0 = dec.tensors
    fb = dec.feature_bins
    n_bins = {i: f.n_bins for i, f in enumerate(fb)}

    # Discretize
    Xb_train = discretize_matrix(X_train, fb)
    Xb_test = discretize_matrix(X_test, fb)

    # Sanity: prediction matches tensor sum (unpurified)
    pred_tensor0 = predict_from_tensors(tensors0, Xb_test)
    max_abs_pred_err0 = float(np.max(np.abs(pred_model - pred_tensor0)))

    cfg = PurifyConfig(tol=1e-10, max_iter=400, verbose=False)
    rng_w = np.random.default_rng(123)

    per_mode_rows: List[dict] = []
    purified_by_mode: Dict[WeightMode, Dict[Tuple[int, ...], np.ndarray | float]] = {}

    for mode in weight_modes:
        weights = {}
        for key in tensors0.keys():
            if len(key) == 0:
                continue
            weights[key] = compute_marginal_weights(Xb_train, key, n_bins, mode=mode, rng=rng_w)

        tens = {k: (v.copy() if isinstance(v, np.ndarray) else float(v)) for k, v in tensors0.items()}
        purify_all(tens, weights, config=cfg)
        purified_by_mode[mode] = tens

        pred_p = predict_from_tensors(tens, Xb_test)
        max_abs_pred_err = float(np.max(np.abs(pred_model - pred_p)))

        per_mode_rows.append(
            dict(
                dataset="german_credit",
                task=task,
                target=y_name,
                metric_name=metric_name,
                metric_value=metric_value,
                weight_mode=mode,
                max_abs_pred_err0=max_abs_pred_err0,
                max_abs_pred_err=max_abs_pred_err,
                n_features=X_train.shape[1],
                n_samples_train=X_train.shape[0],
            )
        )

    summary = pd.DataFrame(per_mode_rows)
    summary.to_csv(out_dir / "german_credit_summary.csv", index=False)

    # Find strongest interactions (by empirical-weight variance) under empirical purification
    if "empirical" in purified_by_mode:
        tens_emp = purified_by_mode["empirical"]
    else:
        tens_emp = next(iter(purified_by_mode.values()))

    inter_rows: List[dict] = []
    for key, val in tens_emp.items():
        if len(key) != 2:
            continue
        w_ref = compute_marginal_weights(Xb_train, key, n_bins, mode="empirical")
        var = weighted_variance(np.asarray(val, dtype=float), w_ref)
        i, j = key
        inter_rows.append(
            dict(
                i=i,
                j=j,
                feature_i=feature_names[i],
                feature_j=feature_names[j],
                var_emp=var,
                shape=str(np.asarray(val).shape),
            )
        )

    inter_df = pd.DataFrame(inter_rows).sort_values("var_emp", ascending=False)
    inter_df.to_csv(out_dir / "german_credit_top_interactions.csv", index=False)

    # Plot top interaction surface across modes (if any interactions exist)
    if not inter_df.empty:
        top = inter_df.iloc[0]
        key = (int(top["i"]), int(top["j"]))
        mats = []
        titles = []

        mats.append(np.asarray(tensors0.get(key, 0.0), dtype=float))
        titles.append("unpurified")

        for mode in weight_modes:
            mats.append(np.asarray(purified_by_mode[mode].get(key, 0.0), dtype=float))
            titles.append(str(mode))

        ncols = len(mats)
        fig, axes = plt.subplots(1, ncols, figsize=(3.2 * ncols, 3.6))
        if ncols == 1:
            axes = [axes]
        for ax, mat, title in zip(axes, mats, titles):
            im = ax.imshow(mat, origin="lower", aspect="auto")
            ax.set_title(title)
            ax.set_xlabel(feature_names[key[1]] + " bin")
            ax.set_ylabel(feature_names[key[0]] + " bin")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.suptitle(f"German credit: top interaction {feature_names[key[0]]} × {feature_names[key[1]]}")
        fig.tight_layout()
        fig.savefig(out_dir / "german_credit_top_interaction_heatmap.pdf")
        plt.close(fig)

    # Plot top main effects (by variance under empirical marginal)
    main_rows: List[dict] = []
    for i in range(X_train.shape[1]):
        key = (i,)
        if key not in tens_emp:
            continue
        w_ref = compute_marginal_weights(Xb_train, key, n_bins, mode="empirical")
        var = weighted_variance(np.asarray(tens_emp[key], dtype=float), w_ref)
        main_rows.append(dict(i=i, feature=feature_names[i], var_emp=var))

    main_df = pd.DataFrame(main_rows).sort_values("var_emp", ascending=False)
    main_df.to_csv(out_dir / "german_credit_top_mains.csv", index=False)

    if not main_df.empty:
        top_k = main_df.head(6)
        fig, axes = plt.subplots(len(top_k), 1, figsize=(8.5, 2.2 * len(top_k)))
        if len(top_k) == 1:
            axes = [axes]
        for ax, (_, row) in zip(axes, top_k.iterrows()):
            i = int(row["i"])
            reps = fb[i].reps
            for mode in weight_modes:
                f = np.asarray(purified_by_mode[mode].get((i,), np.zeros_like(reps)), dtype=float)
                ax.plot(reps, f, label=str(mode))
            ax.set_title(f"Main effect: {feature_names[i]}")
            ax.set_xlabel("Bin representative")
            ax.set_ylabel("Effect")
            ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "german_credit_top_mains.pdf")
        plt.close(fig)

    return summary


if __name__ == "__main__":
    run()
