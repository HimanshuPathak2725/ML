## Plan: Scale The Entity Resolution Pipeline

The current pipeline has the right stages, but pandas plus pure-Python TF-IDF is terminated while loading the multi-million-row dataset. Preserve the blocking, feature, LightGBM, and output contracts while replacing unbounded in-memory work with chunked and disk-backed processing.

### Steps

1. Profile every TSV with streaming reads: row counts, field lengths, country distribution, missingness, and ground-truth match counts. **Completed:** the profiler reports all six source files and the ground-truth cardinality distribution.
2. Add disk-backed preparation for normalized Source 2 and Source 3 fields plus exact-key indexes. **In progress:** `disk_blocking.py` builds SQLite records and exact-key indexes incrementally.
3. Replace the pure-Python TF-IDF indexes with bounded retrieval while retaining union blocking across names, addresses, postal codes, and exact token keys. **In progress:** SQLite FTS5 trigram retrieval is available and candidate TSV generation is streaming.
4. Process Source 1 in batches and write candidate pairs incrementally.
5. Generate features and hard negatives in bounded batches; preserve candidate-relative rank and score features.
6. Train LightGBM on controlled samples and tune the threshold directly for macro F_0.5 on a held-out Source 1 split.
7. Run chunked test inference and write validated matching and candidate output files incrementally.
8. Update reproduction documentation and the methodology template.

### Verification

- Compile all source modules.
- Run synthetic tests for blocking recall, multilingual and France records, singleton abstention, threshold tuning, and output subset validation.
- Run the streaming profiler and sampled real-data pipeline.
- Run the full pipeline with bounded memory and validate every test Source 1 ID.
- Compare blocking recall and macro F_0.5 with the current baseline.

### Decisions

- Source 1 remains the reference side; Source 2 and Source 3 are match targets.
- `country` is an open-set feature; France must work without hard-coded country lists.
- Optimize macro F_0.5, prioritizing precision and correct singleton predictions.
- Do not use external business lookup or labels.
- Every final match must come from the last candidate set sent to the model.

### Scope

Included: scalable data preparation, blocking, feature generation, training, inference, outputs, documentation, and focused tests.

Excluded: unrelated cleanup, leaderboard submission, external enrichment, and speculative deep-learning models before the scalable baseline works.