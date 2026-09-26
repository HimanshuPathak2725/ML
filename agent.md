---
name: entity-resolution-pipeline
description: "Use when working on the ML Challenge 2026 business entity resolution pipeline, including blocking, matching, F_0.5 optimization, large TSV processing, outputs, or submission validation."
---

# Entity Resolution Pipeline Agent

## Repository context

- Inputs are under `student_resource/dataset/train/` and `student_resource/dataset/test/`.
- Pipeline source is under `code/business_entity_resolution/src/`.
- Run documented commands from `student_resource/` because dataset paths are relative.

## Engineering rules

- Treat the TSVs as multi-million-row data. Prefer chunked reads, compact dtypes, disk-backed indexes, and incremental writes.
- Source 1 is the reference side; Source 2 and Source 3 are match targets.
- Use union blocking for recall, then use the matcher and F_0.5 threshold for precision.
- Treat `country` as an open set. Do not restrict logic to US and India; France occurs in test data.
- Do not use external business lookup or labels.
- Avoid the full Cartesian product and avoid materializing all candidates in memory.
- Keep final matches a subset of `candidate_pairs.tsv`.

## Validation workflow

1. Compile with `python3 -m py_compile code/business_entity_resolution/src/*.py`.
2. Run synthetic tests for normalization, blocking recall, multilingual data, singleton abstention, and output validation.
3. Profile real TSVs with streaming reads.
4. Run sampled real-data inference.
5. Validate final files with `student_resource/utils/validate_submission.py`.

## Metric priorities

- Optimize macro F_0.5 directly.
- False merges cost more than missed links.
- Correct empty predictions for true singletons matter.
- Tune thresholds on held-out Source 1 data, never on the test set.

## File ownership

- `blocking.py`: candidate generation.
- `tfidf_index.py`: retrieval implementation.
- `features.py`: pairwise features.
- `train_model.py`: pair sampling, LightGBM, threshold tuning.
- `run_pipeline.py`: orchestration and CLI.
- `output_utils.py`: output formatting and validation.