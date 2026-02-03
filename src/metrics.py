
"""purify_fanova.metrics

Small, dependency-free metrics used across the experiments.

The original project focused on convergence metrics for 2D tensors.
This module now also includes a few convenience metrics that are useful
for studying how *different weighting distributions* ("w") change the
purified main/interaction effects.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np


def _normalize_weights(w: np.ndarray) -> np.ndarray:
    w = np.asarray(w, dtype=float)
    s = float(w.sum())
    if s <= 0:
        return w
    return w / s


def weighted_l2(a: np.ndarray, b: np.ndarray, w: np.ndarray) -> float:
    """Weighted L2 distance: sqrt(E_w[(a-b)^2])."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    w = _normalize_weights(np.asarray(w, dtype=float))
    if a.shape != b.shape or a.shape != w.shape:
        raise ValueError("Shape mismatch in weighted_l2")
    return float(np.sqrt(((a - b) ** 2 * w).sum()))


def sign_flip_rate(
    a: np.ndarray,
    b: np.ndarray,
    tol: float = 1e-12,
    mask: Optional[np.ndarray] = None,
) -> float:
    """Fraction of entries whose sign differs between a and b.

    We ignore entries where either |a| <= tol or |b| <= tol (to avoid
    counting numerical jitter around 0 as a flip).
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("Shape mismatch in sign_flip_rate")

    valid = (np.abs(a) > tol) & (np.abs(b) > tol)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)

    denom = int(valid.sum())
    if denom == 0:
        return 0.0

    sa = np.sign(a[valid])
    sb = np.sign(b[valid])
    return float((sa != sb).mean())


def weighted_mean(vals: np.ndarray, w: np.ndarray) -> float:
    vals = np.asarray(vals, dtype=float)
    w = _normalize_weights(np.asarray(w, dtype=float))
    if vals.shape != w.shape:
        raise ValueError("Shape mismatch in weighted_mean")
    return float((vals * w).sum())


def weighted_variance(vals: np.ndarray, w: np.ndarray) -> float:
    """Weighted variance under discrete weights w (normalized internally)."""
    vals = np.asarray(vals, dtype=float)
    w = _normalize_weights(np.asarray(w, dtype=float))
    if vals.shape != w.shape:
        raise ValueError("Shape mismatch in weighted_variance")
    mu = float((vals * w).sum())
    return float((((vals - mu) ** 2) * w).sum())


def mass_measure_matrix(T: np.ndarray, W: np.ndarray) -> float:
    """
    Compute the "unpurified mass" measure M_t used in Lengerich et al. (2020),
    specialized to a 2D interaction matrix.

    Definitions (paper Eq. 11):
      c_j = sum_i w_{i,j} T_{i,j}
      r_i = sum_j w_{i,j} T_{i,j}
      M = sum_{i,j} w_{i,j} (|r_i| + |c_j|)

    Note: W is assumed normalized to sum 1, but normalization is not required.

    Returns
    -------
    Scalar M >= 0.
    """
    T = np.asarray(T, dtype=float)
    W = np.asarray(W, dtype=float)
    if T.shape != W.shape:
        raise ValueError("Shape mismatch in mass_measure_matrix")

    # r_i: shape (n_rows,)
    r = (W * T).sum(axis=1)
    # c_j: shape (n_cols,)
    c = (W * T).sum(axis=0)

    w_row = W.sum(axis=1)
    w_col = W.sum(axis=0)

    M = float((w_row * np.abs(r)).sum() + (w_col * np.abs(c)).sum())
    return M


def purify_matrix_history(
    T: np.ndarray,
    W: np.ndarray,
    tol: float = 1e-12,
    max_sweeps: int = 50,
) -> Tuple[np.ndarray, List[float]]:
    """
    Run mass-moving purification on a 2D matrix and return residual mass history.

    We perform "sweeps" that zero:
      1) column means (conditional expectation along rows)
      2) row means (conditional expectation along columns)

    This is equivalent to alternating axis updates in the general algorithm.

    Returns
    -------
    T_purified, history where history[t] = M_t after sweep t.
    """
    T = np.asarray(T, dtype=float).copy()
    W = np.asarray(W, dtype=float).copy()
    if T.shape != W.shape:
        raise ValueError("Shape mismatch in purify_matrix_history")

    # normalize weights to sum 1 for comparability
    s = W.sum()
    if s > 0:
        W = W / s

    hist: List[float] = [mass_measure_matrix(T, W)]

    n_rows, n_cols = T.shape

    for _ in range(max_sweeps):
        # Column updates: for each col j, subtract weighted mean over rows
        for j in range(n_cols):
            wcol = W[:, j]
            denom = wcol.sum()
            if denom <= 0:
                continue
            m = float((T[:, j] * wcol).sum() / denom)
            if abs(m) > tol:
                T[:, j] -= m  # move to main effect of column-feature (ignored here)

        # Row updates: for each row i, subtract weighted mean over cols
        for i in range(n_rows):
            wrow = W[i, :]
            denom = wrow.sum()
            if denom <= 0:
                continue
            m = float((T[i, :] * wrow).sum() / denom)
            if abs(m) > tol:
                T[i, :] -= m

        M = mass_measure_matrix(T, W)
        hist.append(M)
        if M <= tol:
            break

    return T, hist
