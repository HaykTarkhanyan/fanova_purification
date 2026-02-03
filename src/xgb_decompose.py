
"""
XGBoost depth<=2 model decomposition into additive main and pairwise interaction tensors.

This module provides a lightweight way to obtain the tensor representation T used by the
mass-moving purification algorithm (purify_fanova.purify).

Key points
----------
1) We **do not** require each tree to use at most 2 distinct features globally.
   Depth-2 trees can use up to 3 features across different branches, but *each root-to-leaf
   path* uses at most 2 features. Such trees can therefore be represented as a sum of
   main/pairwise tensors by assigning each leaf's constant prediction to the tensor
   corresponding to the features used along its path.

2) Leaf values from XGBoost JSON dumps are already scaled by learning_rate (eta),
   so we do NOT multiply by eta again.

3) The decomposition is exact on the discrete bin grid defined by the union of all split
   thresholds in the fitted model.

This produces an *unpurified* additive-with-interactions model; purification recovers the
identifiable fANOVA decomposition.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import json
import math
import numpy as np

from .binning import FeatureBins, build_feature_bins

Key = Tuple[int, ...]


def _feature_id(split_name: str) -> int:
    # split_name like "f3" -> 3
    if split_name.startswith("f"):
        return int(split_name[1:])
    return int(split_name)


def _traverse_nodes(tree: dict) -> Iterable[dict]:
    stack = [tree]
    while stack:
        node = stack.pop()
        yield node
        if "children" in node:
            stack.extend(node["children"])


def collect_thresholds(trees: Sequence[dict], n_features: int) -> Dict[int, List[float]]:
    thr: Dict[int, List[float]] = {i: [] for i in range(n_features)}
    for tree in trees:
        for node in _traverse_nodes(tree):
            if "split" in node:
                fid = _feature_id(node["split"])
                thr[fid].append(float(node["split_condition"]))
    return thr


def eval_tree(tree: dict, x: np.ndarray) -> float:
    """Evaluate a single XGBoost tree (JSON dict) at one input vector x."""
    node = tree
    while "leaf" not in node:
        fid = _feature_id(node["split"])
        thr = float(node["split_condition"])

        val = float(x[fid])
        if np.isnan(val):
            next_id = node["missing"]
        else:
            next_id = node["yes"] if val < thr else node["no"]

        children = node["children"]
        if children[0]["nodeid"] == next_id:
            node = children[0]
        else:
            node = children[1]
    return float(node["leaf"])


def _child_by_nodeid(node: dict, nodeid: int) -> dict:
    children = node.get("children", [])
    if not children:
        raise KeyError("Node has no children")
    if children[0]["nodeid"] == nodeid:
        return children[0]
    if len(children) > 1 and children[1]["nodeid"] == nodeid:
        return children[1]
    # Fallback search
    for ch in children:
        if ch.get("nodeid") == nodeid:
            return ch
    raise KeyError(f"Child with nodeid={nodeid} not found")


def leaf_paths(tree: dict) -> List[Tuple[float, List[Tuple[int, float, str]]]]:
    """
    Extract all leaf paths.

    Returns list of (leaf_value, conditions) where conditions is list of
      (feature_id, threshold, op) with op in {"lt", "ge"} representing x < thr or x >= thr.
    """
    paths: List[Tuple[float, List[Tuple[int, float, str]]]] = []

    def rec(node: dict, conds: List[Tuple[int, float, str]]):
        if "leaf" in node:
            paths.append((float(node["leaf"]), list(conds)))
            return

        fid = _feature_id(node["split"])
        thr = float(node["split_condition"])
        yes_id = node["yes"]
        no_id = node["no"]
        # Missing direction is ignored here; if NaNs exist, you'd need a third branch.
        yes_child = _child_by_nodeid(node, yes_id)
        no_child = _child_by_nodeid(node, no_id)

        rec(yes_child, conds + [(fid, thr, "lt")])
        rec(no_child, conds + [(fid, thr, "ge")])

    rec(tree, [])
    return paths


def _conds_to_bounds(conds: List[Tuple[int, float, str]]) -> Dict[int, Tuple[float, float]]:
    """
    Convert a list of split conditions to per-feature bounds [lower, upper).

    lower starts at -inf, upper at +inf.
    """
    bounds: Dict[int, Tuple[float, float]] = {}
    for fid, thr, op in conds:
        lo, hi = bounds.get(fid, (-math.inf, math.inf))
        if op == "lt":
            hi = min(hi, thr)
        elif op == "ge":
            lo = max(lo, thr)
        else:
            raise ValueError(f"Unknown op={op}")
        bounds[fid] = (lo, hi)
    return bounds


def _allowed_bins(fb: FeatureBins, lo: float, hi: float) -> np.ndarray:
    reps = fb.reps
    mask = np.ones_like(reps, dtype=bool)
    if not math.isinf(lo):
        mask &= reps >= lo
    if not math.isinf(hi):
        mask &= reps < hi
    return np.where(mask)[0]


@dataclass
class Decomposition:
    tensors: Dict[Key, np.ndarray | float]
    feature_bins: List[FeatureBins]
    feature_names: Optional[List[str]] = None


def decompose_xgb_depth2(
    model,
    X_train: np.ndarray,
    feature_names: Optional[List[str]] = None,
) -> Decomposition:
    """
    Decompose an XGBoost model into an unpurified additive model with main and pairwise terms.

    Parameters
    ----------
    model:
        Fitted xgboost.XGBRegressor or xgboost.XGBClassifier.
    X_train:
        Training data used for bin range estimation (representative points).
    feature_names:
        Optional names (length d).

    Returns
    -------
    Decomposition object with:
      - tensors dict: {(): intercept, (i,): main array, (i,j): interaction matrix}
      - feature_bins: list of FeatureBins for each feature.
    """
    booster = model.get_booster()
    config = json.loads(booster.save_config())
    base_score = float(config["learner"]["learner_model_param"]["base_score"])
    d = int(config["learner"]["learner_model_param"]["num_feature"])

    # Parse trees
    dump = booster.get_dump(dump_format="json")
    trees = [json.loads(s) for s in dump]

    # Global bins from all thresholds
    thr = collect_thresholds(trees, n_features=d)
    X_train = np.asarray(X_train, dtype=float)
    f_bins: List[FeatureBins] = []
    for j in range(d):
        f_bins.append(build_feature_bins(thr[j], data=X_train[:, j]))
    n_bins = {j: f_bins[j].n_bins for j in range(d)}

    # Determine which interaction pairs are needed (from leaf paths)
    pairs = set()
    for tree in trees:
        for _, conds in leaf_paths(tree):
            feats = sorted(set([fid for fid, _, _ in conds]))
            if len(feats) == 2:
                pairs.add((feats[0], feats[1]))

    # Initialize tensors
    tensors: Dict[Key, np.ndarray | float] = {(): base_score}
    for j in range(d):
        tensors[(j,)] = np.zeros(n_bins[j], dtype=float)
    for (a, b) in sorted(pairs):
        tensors[(a, b)] = np.zeros((n_bins[a], n_bins[b]), dtype=float)

    # Fill tensors leaf-by-leaf
    for tree in trees:
        for leaf_val, conds in leaf_paths(tree):
            bounds = _conds_to_bounds(conds)
            feats = sorted(bounds.keys())

            if len(feats) == 0:
                tensors[()] = float(tensors[()]) + leaf_val
                continue

            if len(feats) == 1:
                f = feats[0]
                lo, hi = bounds[f]
                idx = _allowed_bins(f_bins[f], lo, hi)
                tensors[(f,)][idx] += leaf_val
                continue

            if len(feats) == 2:
                a, b = feats
                lo_a, hi_a = bounds[a]
                lo_b, hi_b = bounds[b]
                idx_a = _allowed_bins(f_bins[a], lo_a, hi_a)
                idx_b = _allowed_bins(f_bins[b], lo_b, hi_b)
                tensors[(a, b)][np.ix_(idx_a, idx_b)] += leaf_val
                continue

            # For depth<=2, each path should involve <=2 features.
            raise ValueError(f"Leaf path uses {len(feats)} features {feats}; expected <=2.")

    return Decomposition(tensors=tensors, feature_bins=f_bins, feature_names=feature_names)


def predict_from_tensors(
    tensors: Mapping[Key, np.ndarray | float],
    Xb: np.ndarray,
) -> np.ndarray:
    """
    Compute predictions for binned data Xb from tensor representation.

    tensors must include () intercept and (i,) mains; interactions optional.

    Xb: integer bins of shape (n_samples, d).
    """
    n, d = Xb.shape
    y = np.full(n, float(tensors.get((), 0.0)), dtype=float)

    # mains
    for i in range(d):
        key = (i,)
        if key in tensors:
            fi = np.asarray(tensors[key], dtype=float)
            y += fi[Xb[:, i]]

    # interactions
    for key, Tij in tensors.items():
        if len(key) == 2:
            i, j = key
            mat = np.asarray(Tij, dtype=float)
            y += mat[Xb[:, i], Xb[:, j]]

    return y
