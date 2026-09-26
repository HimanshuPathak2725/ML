"""
error_analysis.py — Deep-dive into model errors on the validation split.

Prints:
  - Top false positives (wrong merges) with their feature values
  - Top false negatives (missed true matches) with their feature values
  - Singleton false merges (the most costly error for F_0.5)
  - Feature importance from the trained LightGBM model

Usage:
  python error_analysis.py --model-path models/entity_resolution --val-fraction 0.2
"""

import sys
import argparse
import logging
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from blocking import generate_candidates
from train_model import (
    build_training_pairs, add_relative_features,
    load_model, macro_f05, ALL_FEATURE_NAMES,
)
from features import compute_pair_features, normalize_name, normalize_address

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("error_analysis")

TRAIN_S1  = "dataset/train/train_source1.tsv"
TRAIN_S2  = "dataset/train/train_source2.tsv"
TRAIN_S3  = "dataset/train/train_source3.tsv"
TRAIN_GT  = "dataset/train/train_ground_truth.tsv"


def load_tsv(path):
    return pd.read_csv(path, sep="\t", dtype=str).fillna("")


def load_gt(path):
    df = pd.read_csv(path, sep="\t", dtype=str).fillna("")
    gt = {}
    for _, row in df.iterrows():
        raw = row.get("matched_entity_ids", "").strip()
        gt[row["source1_entity_id"]] = set(raw.split(",")) - {""} if raw else set()
    return gt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path",   default="models/entity_resolution")
    ap.add_argument("--val-fraction", type=float, default=0.2)
    ap.add_argument("--top-n",        type=int, default=20,
                    help="Number of FP/FN examples to show")
    args = ap.parse_args()

    # --- Load ---
    s1 = load_tsv(TRAIN_S1)
    s2 = load_tsv(TRAIN_S2)
    s3 = load_tsv(TRAIN_S3)
    gt_all = load_gt(TRAIN_GT)
    pool = pd.concat([s2, s3], ignore_index=True)

    pool_idx = {row.entity_id: row for row in pool.itertuples(index=False)}
    s1_idx   = {row.entity_id: row for row in s1.itertuples(index=False)}

    # --- Split ---
    rng = np.random.default_rng(42)
    all_ids = s1["entity_id"].tolist()
    rng.shuffle(all_ids)
    n_val = int(len(all_ids) * args.val_fraction)
    val_ids = set(all_ids[:n_val])
    val_s1 = s1[s1["entity_id"].isin(val_ids)].reset_index(drop=True)
    val_gt = {k: v for k, v in gt_all.items() if k in val_ids}

    # --- Blocking ---
    val_cands = generate_candidates(val_s1, pool, tfidf_name_k=30, tfidf_addr_k=20)

    # --- Load model and run inference ---
    model, threshold = load_model(args.model_path)
    feat_cols = list(getattr(model, "_feature_cols", ALL_FEATURE_NAMES))

    # Build all pair features
    pair_rows = []
    for s1_row in val_s1.itertuples(index=False):
        s1_id = s1_row.entity_id
        for cid in val_cands.get(s1_id, set()):
            cx = pool_idx.get(cid)
            if cx is None:
                continue
            feats = compute_pair_features(
                s1_name_raw=getattr(s1_row, "business_name", ""),
                s1_addr_raw=getattr(s1_row, "business_address", ""),
                s1_country=getattr(s1_row, "country", ""),
                sx_name_raw=getattr(cx, "business_name", ""),
                sx_addr_raw=getattr(cx, "business_address", ""),
                sx_country=getattr(cx, "country", ""),
            )
            feats["s1_id"] = s1_id
            feats["cand_id"] = cid
            feats["true_label"] = int(cid in val_gt.get(s1_id, set()))
            feats["s1_name"] = getattr(s1_row, "business_name", "")
            feats["s1_addr"] = getattr(s1_row, "business_address", "")
            feats["cx_name"] = getattr(cx, "business_name", "")
            feats["cx_addr"] = getattr(cx, "business_address", "")
            pair_rows.append(feats)

    df = pd.DataFrame(pair_rows)
    df_feats = add_relative_features(df)

    X = df_feats[[c for c in feat_cols if c in df_feats.columns]].values.astype(np.float32)
    df["prob"] = model.predict(X)
    df["pred"] = (df["prob"] >= threshold).astype(int)

    # --- Error categories ---
    fp = df[(df["pred"] == 1) & (df["true_label"] == 0)].copy()
    fn = df[(df["pred"] == 0) & (df["true_label"] == 1)].copy()

    # Singleton false merges: S1 entities with no true matches that got a prediction
    singleton_ids = {s1_id for s1_id, m in val_gt.items() if not m}
    singleton_fp = fp[fp["s1_id"].isin(singleton_ids)]

    print("\n" + "="*70)
    print(f"VALIDATION METRICS  (threshold={threshold:.2f})")
    print("="*70)
    predictions = {}
    for _, row in df[df["pred"] == 1].iterrows():
        predictions.setdefault(row["s1_id"], set()).add(row["cand_id"])
    f05 = macro_f05(predictions, val_gt, val_s1["entity_id"].tolist())
    print(f"Macro F_0.5:    {f05:.4f}")
    print(f"False positives: {len(fp)}")
    print(f"False negatives: {len(fn)}")
    print(f"Singleton FP:    {len(singleton_fp)} (most costly!)")

    disp_cols = ["s1_id", "cand_id", "prob", "s1_name", "cx_name",
                 "s1_addr", "cx_addr", "name_jaccard_3g", "addr_jaccard_tok"]
    disp_cols = [c for c in disp_cols if c in df.columns]

    print(f"\n--- TOP {args.top_n} FALSE POSITIVES (wrong merges) ---")
    fp_sorted = fp.sort_values("prob", ascending=False).head(args.top_n)
    for _, row in fp_sorted.iterrows():
        print(f"\n  S1:  {row.get('s1_name','')} | {row.get('s1_addr','')}")
        print(f"  CX:  {row.get('cx_name','')} | {row.get('cx_addr','')}")
        print(f"  prob={row['prob']:.3f}  "
              f"name_3g={row.get('name_jaccard_3g',0):.3f}  "
              f"addr_tok={row.get('addr_jaccard_tok',0):.3f}")

    print(f"\n--- TOP {args.top_n} FALSE NEGATIVES (missed true matches) ---")
    fn_sorted = fn.sort_values("prob", ascending=False).head(args.top_n)
    for _, row in fn_sorted.iterrows():
        print(f"\n  S1:  {row.get('s1_name','')} | {row.get('s1_addr','')}")
        print(f"  CX:  {row.get('cx_name','')} | {row.get('cx_addr','')}")
        print(f"  prob={row['prob']:.3f}  "
              f"name_3g={row.get('name_jaccard_3g',0):.3f}  "
              f"addr_tok={row.get('addr_jaccard_tok',0):.3f}")

    print(f"\n--- SINGLETON FALSE MERGES (most damaging) ---")
    for _, row in singleton_fp.sort_values("prob", ascending=False).head(10).iterrows():
        print(f"\n  S1:  {row.get('s1_name','')} | {row.get('s1_addr','')}")
        print(f"  CX:  {row.get('cx_name','')} | {row.get('cx_addr','')}")
        print(f"  prob={row['prob']:.3f}  "
              f"name_3g={row.get('name_jaccard_3g',0):.3f}  "
              f"addr_tok={row.get('addr_jaccard_tok',0):.3f}")

    # Feature importance
    print("\n--- FEATURE IMPORTANCE (HistGradientBoosting gain proxy) ---")
    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
        names = feat_cols
        fi = sorted(zip(names, importance), key=lambda x: -x[1])
        for fname, imp in fi[:20]:
            print(f"  {fname:30s}: {imp:.4f}")
    else:
        print("  feature_importances_ not available for this estimator")


if __name__ == "__main__":
    main()
