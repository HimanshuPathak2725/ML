"""
output_utils.py — Write matching_results.tsv and candidate_pairs.tsv.

Rules enforced here (mirrors validate_submission.py):
  - Every Source 1 entity has exactly one row (even singletons).
  - matched_entity_ids / candidate_entity_ids are comma-separated, no quotes.
  - No duplicate IDs within a list.
  - No S1- IDs in the lists; no IDs absent from the test pool.
  - Final matches are a subset of candidates (pipeline sanity check).
"""

import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)


def write_tsv(
    s1_ids: list[str],
    id_sets: dict[str, set[str]],
    filepath: str | Path,
    col_name: str,
) -> None:
    """
    Writes a two-column TSV: source1_entity_id | col_name.
    s1_ids determines row order and ensures every S1 entity appears.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for s1_id in s1_ids:
        matched = sorted(id_sets.get(s1_id, set()))  # sorted for determinism
        rows.append({
            "source1_entity_id": s1_id,
            col_name: ",".join(matched),
        })

    df = pd.DataFrame(rows)
    df.to_csv(filepath, sep="\t", index=False)
    log.info("Wrote %d rows to %s", len(rows), filepath)


def write_matching_results(
    s1_ids: list[str],
    predictions: dict[str, set[str]],
    out_dir: str | Path = "output",
) -> None:
    write_tsv(s1_ids, predictions, Path(out_dir) / "matching_results.tsv", "matched_entity_ids")


def write_candidate_pairs(
    s1_ids: list[str],
    candidates: dict[str, set[str]],
    out_dir: str | Path = "output",
) -> None:
    write_tsv(s1_ids, candidates, Path(out_dir) / "candidate_pairs.tsv", "candidate_entity_ids")


def validate_outputs(
    s1_ids: list[str],
    candidates: dict[str, set[str]],
    predictions: dict[str, set[str]],
    pool_ids: set[str],
) -> list[str]:
    """
    Returns list of validation error strings (empty = PASS).
    Mirrors the logic in utils/validate_submission.py.
    """
    errors = []
    s1_id_set = set(s1_ids)

    # 1. Every S1 entity present
    for s1_id in s1_ids:
        if s1_id not in predictions:
            errors.append(f"MISSING S1 entity in predictions: {s1_id}")
        if s1_id not in candidates:
            errors.append(f"MISSING S1 entity in candidates: {s1_id}")

    for s1_id, pred in predictions.items():
        # 2. No S1- IDs in output
        s1_in_pred = [i for i in pred if i.startswith("S1-")]
        if s1_in_pred:
            errors.append(f"S1- IDs in predictions for {s1_id}: {s1_in_pred[:3]}")

        # 3. All IDs exist in pool
        unknown = [i for i in pred if i not in pool_ids]
        if unknown:
            errors.append(f"Unknown IDs in predictions for {s1_id}: {unknown[:3]}")

        # 4. No duplicates (handled by set, but check anyway)
        if len(pred) != len(set(pred)):
            errors.append(f"Duplicate IDs in predictions for {s1_id}")

        # 5. Predictions are subset of candidates
        cands = candidates.get(s1_id, set())
        not_in_cands = pred - cands
        if not_in_cands:
            errors.append(
                f"Matched IDs not in candidates for {s1_id}: "
                f"{list(not_in_cands)[:3]} — pipeline bug!"
            )

    if errors:
        log.error("Validation FAILED with %d errors", len(errors))
        for e in errors[:20]:
            log.error("  %s", e)
    else:
        log.info("Validation PASS — output is safe to submit.")

    return errors


def compute_local_f05(
    predictions: dict[str, set[str]],
    ground_truth: dict[str, set[str]],
    all_s1_ids: list[str],
) -> float:
    """Compute macro-averaged F_0.5 on a labelled split (validation or train)."""
    import numpy as np

    scores = []
    for s1_id in all_s1_ids:
        pred = predictions.get(s1_id, set())
        true = ground_truth.get(s1_id, set())

        if not pred and not true:
            scores.append(1.0)
        elif not pred or not true:
            tp = 0
            prec = 1.0 if not pred else 0.0
            rec  = 0.0
            if not pred and true:
                scores.append(0.0)
            elif pred and not true:
                scores.append(0.0)
        else:
            tp = len(pred & true)
            prec = tp / len(pred)
            rec  = tp / len(true)
            denom = 0.25 * prec + rec
            f05 = (1.25 * prec * rec / denom) if denom > 0 else 0.0
            scores.append(f05)

    return float(np.mean(scores)) if scores else 0.0
