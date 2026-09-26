# Business Entity Resolution — Reproduction Guide

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Place the student_resource/ folder in your working directory so that
#    dataset/train/ and dataset/test/ are accessible relative to it,
#    then run the pipeline from inside student_resource/:
cd student_resource

# 3. EDA (understand the data first — takes ~2 min)
python ../code/business_entity_resolution/src/eda.py

# 4. Full pipeline: train + predict test set (with validation split)
python ../code/business_entity_resolution/src/run_pipeline.py train-and-predict \
    --val-fraction 0.2

# 5. Validate output (should print PASS)
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test

# 6. Upload output/matching_results.tsv to the leaderboard portal
```

## File Structure

```
src/
├── features.py        # Text normalization + all similarity features
├── tfidf_index.py     # Pure-Python TF-IDF retrieval index
├── blocking.py        # Multi-strategy candidate generation
├── train_model.py     # Training pairs, LightGBM model, threshold tuning
├── output_utils.py    # TSV writing and pre-submission validation
├── run_pipeline.py    # End-to-end runner (CLI entry point)
├── eda.py             # Exploratory data analysis
└── error_analysis.py  # FP/FN diagnostics after training
```

## Pipeline Architecture

```
Source 1 ──┐
           ├─→ BLOCKING ──→ candidate_pairs.tsv (stored)
Source 2/3 ┘       ↓
           (S1, candidate) pairs
                   ↓
           FEATURE ENGINEERING
           (name sim, addr sim, postal, relative rank)
                   ↓
           LIGHTGBM CLASSIFIER
                   ↓
           THRESHOLD (tuned on F_0.5, not F1)
                   ↓
           matching_results.tsv
```

## Key Design Decisions

### Why F_0.5 requires a high threshold
F_0.5 penalises false positives 4× more than F_1 per match,
and a false merge on a singleton entity drops that entity's score
from 1.0 to 0.0. The optimal threshold on validation is therefore
consistently higher than 0.5 — typically 0.6–0.8.

### Why union-of-blocking-keys, not intersection
True matches only need to survive **one** blocking key.
Using the union of 8 independent strategies drives blocking recall
above 95% even when any single key misses.

### Why candidate-relative features matter
The model needs to know not just "is this pair similar?" but "is
this the *best* candidate for this S1 entity?" The z-score and rank
features enable abstention: if no candidate is clearly better than
noise, the model assigns scores below threshold → empty prediction
→ correct singleton.

## Hyperparameter Tuning

Adjust via CLI flags:

| Flag | Default | Effect |
|------|---------|--------|
| `--tfidf-name-k` | 30 | Candidates per TF-IDF name query |
| `--tfidf-addr-k` | 20 | Candidates per TF-IDF address query |
| `--neg-ratio` | 5 | Hard negatives per positive |
| `--val-fraction` | 0.2 | Fraction held out for val/threshold tuning |

If blocking recall (printed during training) is below 90%, increase `--tfidf-name-k`.

## After Training: Error Analysis

```bash
python ../code/business_entity_resolution/src/error_analysis.py \
    --model-path models/entity_resolution \
    --val-fraction 0.2
```

This prints the top false positives, false negatives, and singleton
false merges — the three categories that most move F_0.5.

## Environment

Python 3.10+. All core logic (`features.py`, `tfidf_index.py`, `blocking.py`)
uses only the standard library and numpy, so blocking/feature computation
runs without any ML framework installed.

The model is LightGBM (MIT license, <<8B parameters — it has no parameters
in the LLM sense, it's a gradient-boosted tree ensemble).
