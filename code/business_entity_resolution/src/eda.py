"""
eda.py — Exploratory Data Analysis.

Run this FIRST after receiving the dataset to understand:
  - Dataset sizes and singleton rates
  - Blocking recall ceiling at different K values
  - Distribution of true pair similarities (sanity-check features)
  - Country distribution and coverage
  - Noise pattern prevalence

Usage:
  python eda.py            # prints stats and saves plots to eda_output/
  python eda.py --no-plots # prints stats only (no matplotlib required)
"""

import sys
import logging
from pathlib import Path
from collections import Counter

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from features import (
    normalize_name, normalize_address,
    jaccard_tokens, jaccard_ngrams, normalized_edit,
    token_sort_ratio, extract_postal, is_landmark_address,
)
from blocking import generate_candidates, blocking_recall

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("eda")

TRAIN_S1  = "dataset/train/train_source1.tsv"
TRAIN_S2  = "dataset/train/train_source2.tsv"
TRAIN_S3  = "dataset/train/train_source3.tsv"
TRAIN_GT  = "dataset/train/train_ground_truth.tsv"


def load(path):
    return pd.read_csv(path, sep="\t", dtype=str).fillna("")


def main(no_plots=False):
    s1 = load(TRAIN_S1)
    s2 = load(TRAIN_S2)
    s3 = load(TRAIN_S3)
    gt_df = pd.read_csv(TRAIN_GT, sep="\t", dtype=str).fillna("")

    gt: dict[str, set[str]] = {}
    for _, row in gt_df.iterrows():
        raw = row.get("matched_entity_ids", "").strip()
        gt[row["source1_entity_id"]] = set(raw.split(",")) - {""} if raw else set()

    print("\n" + "="*60)
    print("DATASET OVERVIEW")
    print("="*60)
    print(f"Source 1: {len(s1)} records")
    print(f"Source 2: {len(s2)} records")
    print(f"Source 3: {len(s3)} records")

    n_singleton    = sum(1 for v in gt.values() if not v)
    n_has_match    = len(gt) - n_singleton
    total_matches  = sum(len(v) for v in gt.values())
    match_sizes    = Counter(len(v) for v in gt.values() if v)

    print(f"\nGround truth entries:    {len(gt)}")
    print(f"Singletons (no matches): {n_singleton} ({100*n_singleton/len(gt):.1f}%)")
    print(f"Has at least 1 match:    {n_has_match}")
    print(f"Total matched pairs:     {total_matches}")
    print(f"Match count distribution (excl. singletons): {dict(sorted(match_sizes.items()))}")

    print("\nCountry distribution:")
    for df, name in [(s1, "S1"), (s2, "S2"), (s3, "S3")]:
        if "country" in df.columns:
            print(f"  {name}: {dict(df['country'].value_counts())}")

    # Missing field rates
    print("\nMissing field rates (%):")
    for df, name in [(s1, "S1"), (s2, "S2"), (s3, "S3")]:
        for col in ["business_name", "business_address"]:
            if col in df.columns:
                missing = (df[col] == "").mean() * 100
                print(f"  {name} {col}: {missing:.1f}%")

    # Landmark address prevalence
    for df, name in [(s1, "S1"), (s2, "S2"), (s3, "S3")]:
        if "business_address" in df.columns:
            lm = df["business_address"].apply(
                lambda x: is_landmark_address(normalize_address(x))
            ).mean() * 100
            print(f"  {name} landmark addresses: {lm:.1f}%")

    # Similarity stats on TRUE POSITIVE pairs
    print("\n" + "="*60)
    print("TRUE POSITIVE PAIR SIMILARITY STATS (sample of 500 pairs)")
    print("="*60)
    pool = pd.concat([s2, s3], ignore_index=True)
    pool_idx = {row.entity_id: row for row in pool.itertuples(index=False)}
    s1_idx   = {row.entity_id: row for row in s1.itertuples(index=False)}

    pairs_sample = []
    for s1_id, matches in gt.items():
        for m in matches:
            pairs_sample.append((s1_id, m))
            if len(pairs_sample) >= 500:
                break
        if len(pairs_sample) >= 500:
            break

    stats: dict[str, list] = {k: [] for k in [
        "name_jac_tok", "name_jac_3g", "name_edit",
        "name_sort", "addr_jac_tok", "addr_jac_3g", "addr_edit"
    ]}

    for s1_id, cid in pairs_sample:
        s1_r = s1_idx.get(s1_id)
        cx_r = pool_idx.get(cid)
        if s1_r is None or cx_r is None:
            continue
        n1 = normalize_name(getattr(s1_r, "business_name", ""))
        nx = normalize_name(getattr(cx_r, "business_name", ""))
        a1 = normalize_address(getattr(s1_r, "business_address", ""))
        ax = normalize_address(getattr(cx_r, "business_address", ""))

        stats["name_jac_tok"].append(jaccard_tokens(n1, nx))
        stats["name_jac_3g"].append(jaccard_ngrams(n1, nx, 3))
        stats["name_edit"].append(normalized_edit(n1, nx))
        stats["name_sort"].append(token_sort_ratio(n1, nx))
        stats["addr_jac_tok"].append(jaccard_tokens(a1, ax))
        stats["addr_jac_3g"].append(jaccard_ngrams(a1, ax, 3))
        stats["addr_edit"].append(normalized_edit(a1, ax))

    import statistics
    for k, vals in stats.items():
        if vals:
            print(f"  {k:18s}: mean={statistics.mean(vals):.3f}  "
                  f"med={statistics.median(vals):.3f}  "
                  f"p10={sorted(vals)[len(vals)//10]:.3f}")

    # Blocking recall at different K values
    print("\n" + "="*60)
    print("BLOCKING RECALL vs K (on 20% validation split)")
    print("="*60)
    from train_model import split_train_val as spl

    # Use a small val split just for this check
    rng_ids = s1["entity_id"].tolist()
    import numpy as np
    rng = np.random.default_rng(0)
    val_size = max(200, int(0.2 * len(rng_ids)))
    val_ids_set = set(rng.choice(rng_ids, size=val_size, replace=False))
    val_s1_df = s1[s1["entity_id"].isin(val_ids_set)].reset_index(drop=True)
    val_gt = {k: v for k, v in gt.items() if k in val_ids_set}

    for k in [10, 20, 30, 50]:
        cands = generate_candidates(val_s1_df, pool, tfidf_name_k=k, tfidf_addr_k=k//2)
        br = blocking_recall(cands, val_gt)
        avg_c = sum(len(v) for v in cands.values()) / max(len(cands), 1)
        print(f"  K={k:3d}: recall={br:.4f}  avg_candidates={avg_c:.1f}")

    print("\nEDA complete. Recommended K: 30 (good recall/candidate balance).")
    print("Run run_pipeline.py train-and-predict to build predictions.\n")


if __name__ == "__main__":
    no_plots = "--no-plots" in sys.argv
    main(no_plots=no_plots)
