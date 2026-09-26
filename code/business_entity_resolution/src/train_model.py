"""
train_model.py — Build training pairs, train LightGBM classifier, tune threshold.

Key design decisions for a winning F_0.5 score:

1. NEGATIVE SAMPLING STRATEGY
   Hard negatives (close-but-wrong pairs from blocking) are far more informative
   than random negatives. We use the blocking output itself for negatives: any
   candidate pair that is NOT a true match is a hard negative.
   We balance positives:negatives at roughly 1:5 — enough negatives for good
   precision without drowning the signal.

2. CANDIDATE-RELATIVE FEATURES
   After the base feature vector, we add per-S1-entity rank features:
   - rank of this pair's name score within all candidates for this S1 entity
   - z-score of name score within the candidate group
   - is_top1: whether this is the best-scoring candidate
   These help the model abstain when no candidate is clearly better than others.

3. THRESHOLD TUNED DIRECTLY ON F_0.5
   We sweep threshold on a held-out validation set and pick the cutoff that
   maximises macro-averaged F_0.5, NOT AUC/F1/accuracy.
   F_0.5 is precision-heavy, so the optimal threshold is typically higher
   (more conservative) than what you'd pick for F1.

4. SINGLETON HANDLING
   Singletons are in the training data implicitly: any S1 entity with no
   true matches contributes only hard negatives. The model learns to assign
   low scores when no candidate is a strong match.
   Post-threshold: if no candidate clears the threshold, the S1 entity gets
   an empty prediction → correct prediction for a singleton → contributes 1.0.
"""

import logging
import json
import math
import pickle
from pathlib import Path
from collections import defaultdict
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from features import compute_pair_features, FEATURE_NAMES

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# F_0.5 helpers
# ---------------------------------------------------------------------------

def f05_score(precision: float, recall: float) -> float:
    denom = 0.25 * precision + recall
    if denom == 0:
        return 0.0
    return 1.25 * precision * recall / denom


def macro_f05(
    predictions: dict[str, set[str]],
    ground_truth: dict[str, set[str]],
    all_s1_ids: list[str],
) -> float:
    """
    Macro-averaged F_0.5 over ALL Source 1 entities (including singletons).
    predictions and ground_truth are dicts: s1_id → set of matched IDs.
    """
    scores = []
    for s1_id in all_s1_ids:
        pred = predictions.get(s1_id, set())
        true = ground_truth.get(s1_id, set())
        if not pred and not true:
            scores.append(1.0)
        elif not pred:
            scores.append(0.0)  # missed all true matches
        elif not true:
            scores.append(0.0)  # false merge on singleton
        else:
            tp = len(pred & true)
            prec = tp / len(pred)
            rec  = tp / len(true)
            scores.append(f05_score(prec, rec))
    return float(np.mean(scores))


# ---------------------------------------------------------------------------
# Training pair construction
# ---------------------------------------------------------------------------

def build_training_pairs(
    s1_df: pd.DataFrame,
    pool_df: pd.DataFrame,
    ground_truth: dict[str, set[str]],
    candidates: dict[str, set[str]],
    neg_ratio: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Builds a labelled DataFrame of (s1_id, cand_id, features…, label).

    Positives: all true match pairs (s1_id, matched_id) that survived blocking.
    Negatives: hard negatives sampled from blocking candidates that are NOT true matches.
    neg_ratio: max number of negatives per positive.
    """
    rng = np.random.default_rng(seed)

    # Index pool by entity_id for fast lookup
    pool_idx: dict[str, pd.Series] = {row.entity_id: row for row in pool_df.itertuples(index=False)}
    s1_idx: dict[str, pd.Series]   = {row.entity_id: row for row in s1_df.itertuples(index=False)}

    rows = []
    n_pos = n_neg = 0
    n_pos_lost_blocking = 0  # true matches not in candidate set (blocking miss)

    for s1_id, true_matches in ground_truth.items():
        s1_row = s1_idx.get(s1_id)
        if s1_row is None:
            continue
        cands = candidates.get(s1_id, set())

        # Positives
        pos_ids = [m for m in true_matches if m in cands]
        n_pos_lost_blocking += len(true_matches) - len(pos_ids)

        # Hard negatives: candidates that are not true matches
        neg_pool = [c for c in cands if c not in true_matches]
        max_neg = max(len(pos_ids) * neg_ratio, 1) if pos_ids else min(len(neg_pool), 3)
        if len(neg_pool) > max_neg:
            neg_ids = rng.choice(neg_pool, size=max_neg, replace=False).tolist()
        else:
            neg_ids = neg_pool

        for cid, label in [(p, 1) for p in pos_ids] + [(n, 0) for n in neg_ids]:
            cand_row = pool_idx.get(cid)
            if cand_row is None:
                continue
            feats = compute_pair_features(
                s1_name_raw=getattr(s1_row, "business_name", "") or "",
                s1_addr_raw=getattr(s1_row, "business_address", "") or "",
                s1_country=getattr(s1_row, "country", "") or "",
                sx_name_raw=getattr(cand_row, "business_name", "") or "",
                sx_addr_raw=getattr(cand_row, "business_address", "") or "",
                sx_country=getattr(cand_row, "country", "") or "",
            )
            row_dict = {"s1_id": s1_id, "cand_id": cid, "label": label}
            row_dict.update(feats)
            rows.append(row_dict)
            if label == 1:
                n_pos += 1
            else:
                n_neg += 1

    log.info(
        "Training pairs: %d positives, %d negatives. "
        "Blocking missed %d true matches (%.1f%% of all positives).",
        n_pos, n_neg, n_pos_lost_blocking,
        100 * n_pos_lost_blocking / max(n_pos + n_pos_lost_blocking, 1),
    )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Candidate-relative features (added after pair construction)
# ---------------------------------------------------------------------------

def add_relative_features(pair_df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds within-group rank/z-score features based on name similarity score.
    Must be called *after* base features are computed (FEATURE_NAMES present).
    """
    df = pair_df.copy()
    # Use name_safe (best name signal) as the ranking score
    df["_name_score"] = df.get("name_safe", df.get("name_jaccard_3g", 0.0))

    grouped = df.groupby("s1_id")["_name_score"]
    df["rel_name_rank"]   = grouped.rank(ascending=False, method="average")
    df["rel_name_max"]    = grouped.transform("max")
    df["rel_name_mean"]   = grouped.transform("mean")
    df["rel_name_std"]    = grouped.transform("std").fillna(0.0)
    df["rel_name_zscore"] = (
        (df["_name_score"] - df["rel_name_mean"]) / (df["rel_name_std"] + 1e-6)
    )
    df["is_top1"]         = (df["rel_name_rank"] == 1.0).astype(float)
    df["margin_to_2nd"]   = df["rel_name_max"] - df["_name_score"]  # 0 for top1

    return df.drop(columns=["_name_score"])


RELATIVE_FEATURE_NAMES = [
    "rel_name_rank", "rel_name_max", "rel_name_mean",
    "rel_name_std", "rel_name_zscore", "is_top1", "margin_to_2nd",
]

ALL_FEATURE_NAMES = FEATURE_NAMES + RELATIVE_FEATURE_NAMES


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

def train_model(pair_df: pd.DataFrame, val_pair_df: Optional[pd.DataFrame] = None):
    """Train HistGradientBoostingClassifier on pair-level features."""
    feat_cols = [c for c in ALL_FEATURE_NAMES if c in pair_df.columns]
    X = pair_df[feat_cols].to_numpy(dtype=np.float32)
    y = pair_df["label"].to_numpy(dtype=np.int32)

    if (y == 1).sum() == 0:
        raise ValueError("Training pairs must include at least one positive example.")

    pos_weight = (y == 0).sum() / max((y == 1).sum(), 1)
    sample_weight = np.where(y == 1, pos_weight, 1.0).astype(np.float32)

    model = HistGradientBoostingClassifier(
        max_iter=300,
        max_leaf_nodes=63,
        learning_rate=0.05,
        min_samples_leaf=50,
        l2_regularization=0.1,
        early_stopping=False,
        validation_fraction=None,
        random_state=42,
        verbose=0,
    )

    if val_pair_df is not None and len(val_pair_df) > 0:
        model.fit(X, y, sample_weight=sample_weight)
        model._val_probs = model.predict_proba(
            val_pair_df[[c for c in feat_cols if c in val_pair_df.columns]].to_numpy(dtype=np.float32)
        )[:, 1]
    else:
        model.fit(X, y, sample_weight=sample_weight)

    log.info("Training complete. n_iter_=%d", model.n_iter_)
    model._feature_cols = feat_cols
    return model


# ---------------------------------------------------------------------------
# Threshold selection
# ---------------------------------------------------------------------------

def _model_predict_proba(model, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        return 1.0 / (1.0 + np.exp(-scores))
    return model.predict(X).astype(float)


def tune_threshold(
    model,
    val_pair_df: pd.DataFrame,
    val_s1_ids: list[str],
    val_ground_truth: dict[str, set[str]],
    thresholds: Optional[list[float]] = None,
) -> tuple[float, float]:
    """Sweep thresholds on the validation set and pick the best F_0.5 cutoff."""
    if thresholds is None:
        thresholds = [i / 100 for i in range(10, 95, 2)]

    feat_cols = [c for c in ALL_FEATURE_NAMES if c in val_pair_df.columns]
    X_val = val_pair_df[feat_cols].to_numpy(dtype=np.float32)
    probs = _model_predict_proba(model, X_val)

    df = val_pair_df.copy()
    df["prob"] = probs

    best_thresh, best_f05 = 0.5, 0.0
    results = []

    for thresh in thresholds:
        predictions: dict[str, set[str]] = defaultdict(set)
        for _, row in df[df["prob"] >= thresh].iterrows():
            predictions[row["s1_id"]].add(row["cand_id"])

        score = macro_f05(predictions, val_ground_truth, val_s1_ids)
        results.append((thresh, score))
        if score > best_f05:
            best_f05 = score
            best_thresh = thresh

    log.info("Threshold sweep (top 5):")
    for t, s in sorted(results, key=lambda x: -x[1])[:5]:
        log.info("  thresh=%.2f  F_0.5=%.4f", t, s)
    log.info("Selected threshold: %.2f  (F_0.5 = %.4f)", best_thresh, best_f05)
    return best_thresh, best_f05


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def predict(
    model,
    s1_df: pd.DataFrame,
    pool_df: pd.DataFrame,
    candidates: dict[str, set[str]],
    threshold: float,
    feat_cols: Optional[list[str]] = None,
) -> dict[str, set[str]]:
    """Run inference on all S1/candidate pairs using the trained model."""
    if feat_cols is None:
        feat_cols = getattr(model, "_feature_cols", ALL_FEATURE_NAMES)

    pool_idx: dict[str, pd.Series] = {row.entity_id: row for row in pool_df.itertuples(index=False)}

    predictions: dict[str, set[str]] = {}

    n = len(s1_df)
    for i, s1_row in enumerate(s1_df.itertuples(index=False)):
        s1_id = s1_row.entity_id
        if i % 500 == 0:
            log.info("Inference: %d / %d", i, n)

        cands = candidates.get(s1_id, set())
        if not cands:
            predictions[s1_id] = set()
            continue

        pair_rows = []
        cand_ids = []
        for cid in cands:
            cand_row = pool_idx.get(cid)
            if cand_row is None:
                continue
            feats = compute_pair_features(
                s1_name_raw=getattr(s1_row, "business_name", "") or "",
                s1_addr_raw=getattr(s1_row, "business_address", "") or "",
                s1_country=getattr(s1_row, "country", "") or "",
                sx_name_raw=getattr(cand_row, "business_name", "") or "",
                sx_addr_raw=getattr(cand_row, "business_address", "") or "",
                sx_country=getattr(cand_row, "country", "") or "",
            )
            pair_rows.append(feats)
            cand_ids.append(cid)

        if not pair_rows:
            predictions[s1_id] = set()
            continue

        pair_df = pd.DataFrame(pair_rows)
        pair_df["s1_id"] = s1_id
        pair_df["cand_id"] = cand_ids
        pair_df = add_relative_features(pair_df)

        feat_cols_present = [c for c in feat_cols if c in pair_df.columns]
        X = pair_df[feat_cols_present].to_numpy(dtype=np.float32)
        probs = _model_predict_proba(model, X)

        matched = {cid for cid, p in zip(cand_ids, probs) if p >= threshold}
        predictions[s1_id] = matched

    return predictions


# ---------------------------------------------------------------------------
# Model persistence
# ---------------------------------------------------------------------------

def save_model(model, *args):
    if len(args) == 2:
        threshold, path = args
        feat_cols = getattr(model, "_feature_cols", ALL_FEATURE_NAMES)
    elif len(args) == 3:
        feat_cols, threshold, path = args
    else:
        raise TypeError("save_model(model, threshold, path) or save_model(model, feat_cols, threshold, path)")

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path + '.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open(path + '_meta.json', 'w') as f:
        json.dump({'threshold': threshold, 'feat_cols': feat_cols}, f)
    log.info('Model saved to %s.pkl', path)


def load_model(path):
    with open(path + '.pkl', 'rb') as f:
        model = pickle.load(f)
    with open(path + '_meta.json') as f:
        meta = json.load(f)
    feat_cols = meta.get('feat_cols', getattr(model, '_feature_cols', ALL_FEATURE_NAMES))
    setattr(model, '_feature_cols', feat_cols)
    return model, meta['threshold']
