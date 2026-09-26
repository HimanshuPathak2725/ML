"""
run_pipeline.py — End-to-end pipeline runner.

Usage:
  # Train on full training data, predict on test set:
  python run_pipeline.py train-and-predict

  # Train with validation split (prints F_0.5 before predicting on test):
  python run_pipeline.py train-and-predict --val-fraction 0.2

  # Run only blocking and print recall ceiling on validation split:
  python run_pipeline.py blocking-check --val-fraction 0.2

  # Run inference only using a saved model:
  python run_pipeline.py predict --model-path models/entity_resolution

Pipeline stages:
  1. Load data
  2. Blocking (candidate generation) on train + test pool
  3. [If val split] check blocking recall ceiling
  4. Build training pairs (positives + hard negatives from blocking)
  5. Add candidate-relative features
  6. Train LightGBM classifier
  7. Tune threshold on validation F_0.5
  8. Run inference on test set
  9. Write output/matching_results.tsv and output/candidate_pairs.tsv
  10. Validate output format
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure src is in path when running from inside src/
sys.path.insert(0, str(Path(__file__).parent))

from blocking import generate_candidates, blocking_recall
from train_model import (
    build_training_pairs, add_relative_features,
    train_model, tune_threshold, predict,
    save_model, load_model, macro_f05,
)
from output_utils import (
    write_matching_results, write_candidate_pairs,
    validate_outputs, compute_local_f05,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pipeline")

# ---------------------------------------------------------------------------
# Paths (relative to student_resource/)
# ---------------------------------------------------------------------------

TRAIN_S1   = "dataset/train/train_source1.tsv"
TRAIN_S2   = "dataset/train/train_source2.tsv"
TRAIN_S3   = "dataset/train/train_source3.tsv"
TRAIN_GT   = "dataset/train/train_ground_truth.tsv"
TEST_S1    = "dataset/test/test_source1.tsv"
TEST_S2    = "dataset/test/test_source2.tsv"
TEST_S3    = "dataset/test/test_source3.tsv"
OUTPUT_DIR = "output"
MODEL_PATH = "models/entity_resolution"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_tsv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str).fillna("")
    log.info("Loaded %s: %d rows, columns=%s", path, len(df), list(df.columns))
    return df


def load_ground_truth(path: str) -> dict[str, set[str]]:
    df = pd.read_csv(path, sep="\t", dtype=str).fillna("")
    gt: dict[str, set[str]] = {}
    for _, row in df.iterrows():
        s1_id = row["source1_entity_id"]
        raw = row.get("matched_entity_ids", "").strip()
        gt[s1_id] = set(raw.split(",")) - {""} if raw else set()
    log.info("Ground truth loaded: %d S1 entries, %d total matches",
             len(gt), sum(len(v) for v in gt.values()))
    return gt


def split_train_val(s1_df, gt, val_fraction=0.2, seed=42):
    """Stratified split: singletons and non-singletons split proportionally."""
    rng = np.random.default_rng(seed)
    s1_ids = s1_df["entity_id"].tolist()
    singleton_ids = [i for i in s1_ids if not gt.get(i)]
    matched_ids   = [i for i in s1_ids if gt.get(i)]

    rng.shuffle(singleton_ids)
    rng.shuffle(matched_ids)

    n_sing_val = int(len(singleton_ids) * val_fraction)
    n_match_val = int(len(matched_ids) * val_fraction)

    val_ids = set(singleton_ids[:n_sing_val] + matched_ids[:n_match_val])
    train_ids = set(s1_ids) - val_ids

    train_s1 = s1_df[s1_df["entity_id"].isin(train_ids)].reset_index(drop=True)
    val_s1   = s1_df[s1_df["entity_id"].isin(val_ids)].reset_index(drop=True)
    train_gt = {k: v for k, v in gt.items() if k in train_ids}
    val_gt   = {k: v for k, v in gt.items() if k in val_ids}

    log.info("Train: %d S1 entities (%d singletons). Val: %d S1 (%d singletons).",
             len(train_ids), sum(1 for i in train_ids if not gt.get(i)),
             len(val_ids),   sum(1 for i in val_ids   if not gt.get(i)))
    return train_s1, train_gt, val_s1, val_gt


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_train_and_predict(args):
    # --- Load training data ---
    train_s1 = load_tsv(args.train_s1)
    train_s2 = load_tsv(args.train_s2)
    train_s3 = load_tsv(args.train_s3)
    gt_all   = load_ground_truth(args.train_gt)

    # --- Load test data ---
    test_s1  = load_tsv(args.test_s1)
    test_s2  = load_tsv(args.test_s2)
    test_s3  = load_tsv(args.test_s3)

    # Combined pool for each split
    train_pool = pd.concat([train_s2, train_s3], ignore_index=True)
    test_pool  = pd.concat([test_s2,  test_s3],  ignore_index=True)
    pool_ids_test = set(test_pool["entity_id"].tolist())

    # --- Optional validation split ---
    if args.val_fraction > 0:
        train_s1_fit, train_gt_fit, val_s1, val_gt = split_train_val(
            train_s1, gt_all, val_fraction=args.val_fraction
        )
    else:
        train_s1_fit = train_s1
        train_gt_fit = gt_all
        val_s1 = None
        val_gt = {}

    # --- Stage 1: Blocking on train pool ---
    log.info("=== Stage 1: Blocking (train pool) ===")
    train_cands = generate_candidates(
        train_s1_fit, train_pool,
        tfidf_name_k=args.tfidf_name_k,
        tfidf_addr_k=args.tfidf_addr_k,
    )

    if val_s1 is not None:
        val_cands = generate_candidates(
            val_s1, train_pool,
            tfidf_name_k=args.tfidf_name_k,
            tfidf_addr_k=args.tfidf_addr_k,
        )
        br = blocking_recall(val_cands, val_gt)
        log.info("Blocking recall on val set: %.4f", br)
        if br < 0.90:
            log.warning(
                "Blocking recall is LOW (%.2f%%). Consider increasing "
                "tfidf_name_k or tfidf_addr_k.", br * 100
            )

    # --- Stage 2: Build training pairs ---
    log.info("=== Stage 2: Building training pairs ===")
    train_pairs = build_training_pairs(
        train_s1_fit, train_pool, train_gt_fit, train_cands,
        neg_ratio=args.neg_ratio,
    )
    train_pairs = add_relative_features(train_pairs)
    log.info("Training pair shape: %s", train_pairs.shape)

    val_pairs = None
    if val_s1 is not None:
        val_pairs = build_training_pairs(
            val_s1, train_pool, val_gt, val_cands, neg_ratio=args.neg_ratio,
        )
        val_pairs = add_relative_features(val_pairs)
        log.info("Validation pair shape: %s", val_pairs.shape)

    # --- Stage 3: Train model ---
    log.info("=== Stage 3: Training HistGradientBoosting ===")
    model = train_model(train_pairs, val_pair_df=val_pairs)

    # --- Stage 4: Tune threshold ---
    log.info("=== Stage 4: Threshold selection ===")
    if val_s1 is not None and val_pairs is not None and len(val_pairs) > 0:
        threshold, best_f05 = tune_threshold(
            model, val_pairs,
            val_s1["entity_id"].tolist(),
            val_gt,
        )
        log.info("Val F_0.5 = %.4f at threshold %.2f", best_f05, threshold)
    else:
        threshold = args.default_threshold
        log.info("No validation split. Using default threshold %.2f", threshold)

    # --- Save model ---
    Path(args.model_path).parent.mkdir(parents=True, exist_ok=True)
    save_model(model, threshold, args.model_path)

    # --- Stage 5: Blocking on test pool ---
    log.info("=== Stage 5: Blocking (test pool) ===")
    test_cands = generate_candidates(
        test_s1, test_pool,
        tfidf_name_k=args.tfidf_name_k,
        tfidf_addr_k=args.tfidf_addr_k,
    )
    n_test_with_cands = sum(1 for v in test_cands.values() if v)
    log.info("%d / %d test S1 entities have at least one candidate.",
             n_test_with_cands, len(test_s1))

    # --- Stage 6: Inference on test ---
    log.info("=== Stage 6: Inference on test set ===")
    test_predictions = predict(model, test_s1, test_pool, test_cands, threshold)

    # --- Stage 7: Write output ---
    log.info("=== Stage 7: Writing output ===")
    test_s1_ids = test_s1["entity_id"].tolist()
    write_matching_results(test_s1_ids, test_predictions, args.output_dir)
    write_candidate_pairs(test_s1_ids, test_cands, args.output_dir)

    # --- Stage 8: Validate ---
    log.info("=== Stage 8: Validating output ===")
    errors = validate_outputs(test_s1_ids, test_cands, test_predictions, pool_ids_test)
    if errors:
        log.error("Output validation failed — fix before submitting!")
        sys.exit(1)

    log.info("Pipeline complete. Upload output/matching_results.tsv to the leaderboard.")


def cmd_blocking_check(args):
    """Quick check of blocking recall without training a model."""
    train_s1  = load_tsv(args.train_s1)
    train_s2  = load_tsv(args.train_s2)
    train_s3  = load_tsv(args.train_s3)
    gt_all    = load_ground_truth(args.train_gt)
    train_pool = pd.concat([train_s2, train_s3], ignore_index=True)

    _, _, val_s1, val_gt = split_train_val(
        train_s1, gt_all, val_fraction=max(args.val_fraction, 0.2)
    )
    val_cands = generate_candidates(
        val_s1, train_pool,
        tfidf_name_k=args.tfidf_name_k,
        tfidf_addr_k=args.tfidf_addr_k,
    )
    br = blocking_recall(val_cands, val_gt)
    avg_cands = np.mean([len(v) for v in val_cands.values()])
    log.info("Blocking recall:         %.4f", br)
    log.info("Avg candidates per S1:   %.1f", avg_cands)
    log.info("Max theoretical F_0.5:   governed by blocking recall (%.4f)", br)


def cmd_predict_only(args):
    """Run inference using a saved model (skip training)."""
    test_s1  = load_tsv(args.test_s1)
    test_s2  = load_tsv(args.test_s2)
    test_s3  = load_tsv(args.test_s3)
    test_pool = pd.concat([test_s2, test_s3], ignore_index=True)
    pool_ids_test = set(test_pool["entity_id"].tolist())

    model, threshold = load_model(args.model_path)
    log.info("Loaded model from %s, threshold=%.3f", args.model_path, threshold)

    test_cands = generate_candidates(
        test_s1, test_pool,
        tfidf_name_k=args.tfidf_name_k,
        tfidf_addr_k=args.tfidf_addr_k,
    )
    test_predictions = predict(model, test_s1, test_pool, test_cands, threshold)

    test_s1_ids = test_s1["entity_id"].tolist()
    write_matching_results(test_s1_ids, test_predictions, args.output_dir)
    write_candidate_pairs(test_s1_ids, test_cands, args.output_dir)
    validate_outputs(test_s1_ids, test_cands, test_predictions, pool_ids_test)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(description="Business Entity Resolution Pipeline")
    sub = p.add_subparsers(dest="command")

    # Shared args factory
    def add_shared(sp):
        sp.add_argument("--train-s1",   default=TRAIN_S1)
        sp.add_argument("--train-s2",   default=TRAIN_S2)
        sp.add_argument("--train-s3",   default=TRAIN_S3)
        sp.add_argument("--train-gt",   default=TRAIN_GT)
        sp.add_argument("--test-s1",    default=TEST_S1)
        sp.add_argument("--test-s2",    default=TEST_S2)
        sp.add_argument("--test-s3",    default=TEST_S3)
        sp.add_argument("--output-dir", default=OUTPUT_DIR)
        sp.add_argument("--model-path", default=MODEL_PATH)
        sp.add_argument("--tfidf-name-k", type=int, default=30,
                        help="Top-K from TF-IDF name retrieval per S1 entity")
        sp.add_argument("--tfidf-addr-k", type=int, default=20,
                        help="Top-K from TF-IDF address retrieval per S1 entity")
        sp.add_argument("--val-fraction", type=float, default=0.2,
                        help="Fraction of train data to hold out for validation (0=no val)")
        sp.add_argument("--neg-ratio", type=int, default=5,
                        help="Hard negatives per positive during training pair construction")
        sp.add_argument("--default-threshold", type=float, default=0.5,
                        help="Fallback threshold when no validation split is used")

    tap = sub.add_parser("train-and-predict", help="Full pipeline: train then predict test")
    add_shared(tap)

    bcp = sub.add_parser("blocking-check", help="Check blocking recall only (no model training)")
    add_shared(bcp)

    pp = sub.add_parser("predict", help="Inference only using saved model")
    add_shared(pp)

    return p


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "train-and-predict":
        cmd_train_and_predict(args)
    elif args.command == "blocking-check":
        cmd_blocking_check(args)
    elif args.command == "predict":
        cmd_predict_only(args)
    else:
        parser.print_help()
        sys.exit(1)
