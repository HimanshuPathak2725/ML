"""Scalable end-to-end pipeline using SQLite-backed blocking and sklearn training."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from disk_blocking import build_index, generate_candidates_file, candidates as query_candidates
from output_utils import validate_outputs, write_candidate_pairs, write_matching_results
from train_model import add_relative_features, build_training_pairs, load_model, macro_f05, predict, save_model, train_model, tune_threshold

LOG = logging.getLogger("scalable_pipeline")


def load_tsv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", dtype=str).fillna("")


def load_ground_truth(path: str | Path) -> dict[str, set[str]]:
    df = load_tsv(path)
    gt: dict[str, set[str]] = {}
    for _, row in df.iterrows():
        raw = (row.get("matched_entity_ids") or "").strip()
        gt[row["source1_entity_id"]] = set(raw.split(",")) - {""} if raw else set()
    return gt


def split_train_val(s1_df: pd.DataFrame, gt: dict[str, set[str]], val_fraction: float = 0.2, seed: int = 42):
    rng = np.random.default_rng(seed)
    s1_ids = s1_df["entity_id"].tolist()
    singleton_ids = [sid for sid in s1_ids if not gt.get(sid)]
    matched_ids = [sid for sid in s1_ids if gt.get(sid)]
    rng.shuffle(singleton_ids)
    rng.shuffle(matched_ids)
    n_sing_val = int(len(singleton_ids) * val_fraction)
    n_match_val = int(len(matched_ids) * val_fraction)
    val_ids = set(singleton_ids[:n_sing_val] + matched_ids[:n_match_val])
    train_ids = set(s1_ids) - val_ids
    train_s1 = s1_df[s1_df["entity_id"].isin(train_ids)].reset_index(drop=True)
    val_s1 = s1_df[s1_df["entity_id"].isin(val_ids)].reset_index(drop=True)
    train_gt = {k: v for k, v in gt.items() if k in train_ids}
    val_gt = {k: v for k, v in gt.items() if k in val_ids}
    return train_s1, train_gt, val_s1, val_gt


def make_database(root: Path, database_path: Path, pool_files: list[Path]) -> None:
    database_path.parent.mkdir(parents=True, exist_ok=True)
    build_index(pool_files, database_path)


def build_blocking_candidates(s1_path: Path, pool_files: list[Path], db_path: Path, name_k: int = 30, address_k: int = 20):
    if not db_path.exists():
        build_index(pool_files, db_path)
    candidates: dict[str, set[str]] = {}
    connection = __import__("sqlite3").connect(db_path)
    try:
        for row in load_tsv(s1_path).itertuples(index=False):
            cands = set()
            for key in ["name", "name_stripped", "name_sorted", "name_first", "name_first_two", "postal"]:
                pass
            cands = query_candidates(connection, row.business_name, row.business_address, row.country, name_k=name_k, address_k=address_k)
            candidates[row.entity_id] = cands
    finally:
        connection.close()
    return candidates


def cmd_train_and_predict(args):
    root = Path(args.root)
    train_root = root / "dataset" / "train"
    test_root = root / "dataset" / "test"

    train_s1 = load_tsv(train_root / "train_source1.tsv")
    train_s2 = load_tsv(train_root / "train_source2.tsv")
    train_s3 = load_tsv(train_root / "train_source3.tsv")
    gt = load_ground_truth(train_root / "train_ground_truth.tsv")

    test_s1 = load_tsv(test_root / "test_source1.tsv")
    test_s2 = load_tsv(test_root / "test_source2.tsv")
    test_s3 = load_tsv(test_root / "test_source3.tsv")

    if args.val_fraction > 0:
        train_s1_fit, train_gt_fit, val_s1, val_gt = split_train_val(train_s1, gt, val_fraction=args.val_fraction)
    else:
        train_s1_fit, train_gt_fit, val_s1, val_gt = train_s1, gt, None, {}

    train_pool = pd.concat([train_s2, train_s3], ignore_index=True)
    test_pool = pd.concat([test_s2, test_s3], ignore_index=True)

    db_dir = Path(args.db_dir)
    db_dir.mkdir(parents=True, exist_ok=True)
    train_db = db_dir / "train_index.sqlite"
    test_db = db_dir / "test_index.sqlite"
    make_database(root, train_db, [train_root / "train_source2.tsv", train_root / "train_source3.tsv"])
    make_database(root, test_db, [test_root / "test_source2.tsv", test_root / "test_source3.tsv"])

    train_cands = build_blocking_candidates(train_root / "train_source1.tsv", [train_root / "train_source2.tsv", train_root / "train_source3.tsv"], train_db, args.name_k, args.address_k)
    if val_s1 is not None:
        val_cands = build_blocking_candidates(train_root / "train_source1.tsv", [train_root / "train_source2.tsv", train_root / "train_source3.tsv"], train_db, args.name_k, args.address_k)
        val_for_split = {s1_id: val_cands.get(s1_id, set()) for s1_id in val_s1["entity_id"].tolist()}
        val_can = {sid: val_for_split.get(sid, set()) for sid in val_s1["entity_id"].tolist()}
    else:
        val_cands = {}
        val_can = {}

    train_pairs = build_training_pairs(train_s1_fit, train_pool, train_gt_fit, train_cands, neg_ratio=args.neg_ratio, seed=args.seed)
    train_pairs = add_relative_features(train_pairs)

    if val_s1 is not None:
        val_pairs = build_training_pairs(val_s1, train_pool, val_gt, val_can, neg_ratio=args.neg_ratio, seed=args.seed)
        val_pairs = add_relative_features(val_pairs)
    else:
        val_pairs = None

    model = train_model(train_pairs, val_pair_df=val_pairs)

    if val_s1 is not None and val_pairs is not None and len(val_pairs) > 0:
        threshold, best_f05 = tune_threshold(model, val_pairs, val_s1["entity_id"].tolist(), val_gt)
    else:
        threshold = args.default_threshold
        best_f05 = 0.0

    save_model(model, threshold, str(args.model_path))
    test_cands = build_blocking_candidates(test_root / "test_source1.tsv", [test_root / "test_source2.tsv", test_root / "test_source3.tsv"], test_db, args.name_k, args.address_k)
    predictions = predict(model, test_s1, test_pool, test_cands, threshold)

    write_matching_results(test_s1["entity_id"].tolist(), predictions, args.output_dir)
    write_candidate_pairs(test_s1["entity_id"].tolist(), test_cands, args.output_dir)
    errors = validate_outputs(test_s1["entity_id"].tolist(), test_cands, predictions, set(test_pool["entity_id"]))
    if errors:
        raise SystemExit(f"Output validation failed: {errors[:5]}")

    LOG.info("Completed scalable pipeline; selected threshold=%.3f F_0.5=%.4f", threshold, best_f05)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".", help="student_resource directory")
    parser.add_argument("--db-dir", default="work/sqlite", help="Directory for SQLite indexes")
    parser.add_argument("--model-path", default="models/entity_resolution", help="Model artifact prefix")
    parser.add_argument("--output-dir", default="output", help="Output directory")
    parser.add_argument("--name-k", type=int, default=30)
    parser.add_argument("--address-k", type=int, default=20)
    parser.add_argument("--neg-ratio", type=int, default=5)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--default-threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = build_parser().parse_args()
    cmd_train_and_predict(args)


if __name__ == "__main__":
    main()
