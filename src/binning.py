
"""
Utilities for discretizing continuous features into bins aligned with tree split thresholds.

We use bins of the form:
  (-inf, t1), [t1, t2), ..., [tK, +inf)

and map a value x to bin index:
  idx = searchsorted([t1, ..., tK], x, side="right")

This matches the common decision rule in tree learners (left if x < t, right otherwise).

The purification algorithm in this repo expects discrete bins for each feature.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class FeatureBins:
    thresholds: np.ndarray  # sorted unique thresholds, shape (K,)
    reps: np.ndarray        # representative value per bin, shape (K+1,)
    n_bins: int


def build_feature_bins(
    thresholds: Sequence[float],
    data: Optional[np.ndarray] = None,
    pad_frac: float = 1e-3,
) -> FeatureBins:
    """
    Build FeatureBins from split thresholds.

    Parameters
    ----------
    thresholds:
        Iterable of split thresholds for a feature.
    data:
        Optional 1D array of observed feature values, used to set representatives for
        the extreme bins (-inf, t1) and [tK, +inf). If omitted, the reps for extreme bins
        are set using +/- 1 around thresholds.
    pad_frac:
        Padding fraction of (max-min) used for extreme bin representatives.

    Returns
    -------
    FeatureBins with K+1 bins.
    """
    thr = np.array(sorted(set([float(t) for t in thresholds])), dtype=float)
    K = thr.shape[0]
    n_bins = K + 1

    # Determine min/max for representatives
    if data is not None and len(data) > 0:
        dmin = float(np.nanmin(data))
        dmax = float(np.nanmax(data))
        span = dmax - dmin
        pad = pad_frac * (span if span > 0 else 1.0)
        left_rep = dmin - pad
        right_rep = dmax + pad
    else:
        # fallback
        left_rep = float(thr[0] - 1.0) if K > 0 else -1.0
        right_rep = float(thr[-1] + 1.0) if K > 0 else 1.0

    reps = np.zeros(n_bins, dtype=float)
    if K == 0:
        reps[0] = 0.0 if data is None else float(np.nanmean(data))
        return FeatureBins(thresholds=thr, reps=reps, n_bins=1)

    reps[0] = left_rep
    for b in range(1, n_bins - 1):
        reps[b] = 0.5 * (thr[b - 1] + thr[b])
    reps[-1] = right_rep
    return FeatureBins(thresholds=thr, reps=reps, n_bins=n_bins)


def bin_index(thresholds: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Map values x to bin indices according to thresholds.

    thresholds: shape (K,)
    x: array-like

    Returns indices in [0, K].
    """
    thr = np.asarray(thresholds, dtype=float)
    return np.searchsorted(thr, x, side="right").astype(int)


def discretize_matrix(
    X: np.ndarray,
    feature_bins: Sequence[FeatureBins],
) -> np.ndarray:
    """
    Discretize a 2D array X into integer bin indices using per-feature bins.

    Returns:
      Xb of shape (n_samples, n_features) with integer bin IDs per feature.
    """
    X = np.asarray(X)
    n, d = X.shape
    if len(feature_bins) != d:
        raise ValueError("feature_bins length must match number of columns in X.")
    Xb = np.zeros((n, d), dtype=int)
    for j, fb in enumerate(feature_bins):
        Xb[:, j] = bin_index(fb.thresholds, X[:, j])
    return Xb


def compute_marginal_weights(
    Xb: np.ndarray,
    u: Tuple[int, ...],
    n_bins: Mapping[int, int],
    mode: str = "empirical",
    alpha: float = 1.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Compute marginal weights w_u over bins for subset u from binned data Xb.

    Parameters
    ----------
    Xb:
        Binned data of shape (n_samples, d) with integer bin indices.
    u:
        Subset of feature indices.
    n_bins:
        Mapping feature -> number of bins.
    mode:
        "empirical" | "uniform" | "laplace" | "indep" | "perm_joint"
    alpha:
        For laplace: w = w_emp + alpha * w_unif (then normalized).
        alpha=1 matches the paper's description (empirical + uniform).

    Returns
    -------
    w_u ndarray of shape (n_bins[u0], ..., n_bins[uk]) normalized to sum 1.

    Notes
    -----
    - "indep" (independence-assumed) keeps empirical marginals but destroys dependence:
        w_indep(x_u) = \prod_{i in u} w_emp(x_i).
      For |u|=1, this is identical to empirical weights.

    - "perm_joint" (permutation joint) estimates the joint on data where dependence
      is broken by shuffling all but the first variable in u across samples.
      For |u|=1, this is identical to empirical weights.
    """
    if len(u) == 0:
        raise ValueError("u must be non-empty for marginal weights.")

    u = tuple(u)
    shape = tuple([n_bins[i] for i in u])
    n = int(Xb.shape[0])

    if rng is None:
        rng = np.random.default_rng(0)

    def _empirical_joint(data_subset: np.ndarray) -> np.ndarray:
        """Empirical joint distribution over bins for columns already restricted to u."""
        if data_subset.ndim != 2 or data_subset.shape[1] != len(u):
            raise ValueError("data_subset must be (n_samples, |u|)")
        # Vectorized bincount via ravel_multi_index.
        flat = np.ravel_multi_index(data_subset.T, dims=shape, mode="raise")
        counts_flat = np.bincount(flat, minlength=int(np.prod(shape))).astype(float)
        counts = counts_flat.reshape(shape)
        w = counts / max(1.0, float(n))
        s = w.sum()
        return w / s if s > 0 else w

    X_u = Xb[:, list(u)].astype(int, copy=False)
    w_emp = _empirical_joint(X_u)

    if mode == "empirical":
        w = w_emp
    elif mode == "uniform":
        w = np.ones(shape, dtype=float)
        w /= w.sum()
    elif mode == "laplace":
        w_unif = np.ones(shape, dtype=float)
        w_unif /= w_unif.sum()
        w = w_emp + alpha * w_unif
        w /= w.sum()
    elif mode == "indep":
        # Independence-assumed weights: product of empirical 1D marginals.
        if len(u) == 1:
            w = w_emp
        else:
            # Compute 1D marginals for each variable in u.
            w = None
            for i in u:
                wi = compute_marginal_weights(Xb, (i,), n_bins, mode="empirical", rng=rng)
                w = wi if w is None else w[..., None] * wi
            s = w.sum()
            w = w / s if s > 0 else w
    elif mode == "perm_joint":
        # Permutation-based joint: break dependence by shuffling all but the first column.
        if len(u) == 1:
            w = w_emp
        else:
            X_perm = X_u.copy()
            for k in range(1, X_perm.shape[1]):
                perm = rng.permutation(n)
                X_perm[:, k] = X_perm[perm, k]
            w = _empirical_joint(X_perm)
    else:
        raise ValueError(f"Unknown mode={mode}")

    # Normalize defensively
    s = w.sum()
    if s > 0:
        w = w / s
    return w
