"""
tfidf_index.py — Lightweight TF-IDF + cosine retrieval index.

Used during blocking to retrieve the top-K most similar records from
Source 2/3 for each Source 1 query without external dependencies.

Design: pure stdlib + Python lists; no scipy/sklearn required, so the
blocking stage runs with zero extra installs if needed. The index is
built once (fit), then queried many times (query_topk).
"""

import math
import heapq
from collections import Counter
from typing import Callable


class TFIDFIndex:
    """
    Sparse TF-IDF index over a corpus of strings.

    Usage:
        idx = TFIDFIndex(tokenizer=char_ngrams_3)
        idx.fit(records)          # list of (id, text)
        hits = idx.query_topk(query_text, k=20)  # → [(score, id), ...]
    """

    def __init__(self, tokenizer: Callable[[str], list[str]] = None):
        if tokenizer is None:
            # Default: character 3-grams — handles typos, morphological variants.
            def _default(text: str) -> list[str]:
                padded = f"  {text}  "
                return [padded[i:i+3] for i in range(len(padded) - 2)]
            tokenizer = _default
        self.tokenizer = tokenizer
        self._idf: dict[str, float] = {}
        self._docs: list[tuple[str, dict[str, float]]] = []  # (id, tfidf_vec)
        self._norms: list[float] = []

    def fit(self, records: list[tuple[str, str]]) -> None:
        """
        records: list of (record_id, normalized_text) pairs.
        Builds TF-IDF vectors and the IDF table.
        """
        N = len(records)
        # --- Document frequency ---
        df: Counter = Counter()
        tf_raw: list[tuple[str, Counter]] = []
        for rid, text in records:
            toks = self.tokenizer(text)
            tc = Counter(toks)
            for t in tc:
                df[t] += 1
            tf_raw.append((rid, tc))

        # --- IDF (smooth: log((N+1)/(df+1)) + 1) ---
        self._idf = {t: math.log((N + 1) / (cnt + 1)) + 1 for t, cnt in df.items()}

        # --- Build TF-IDF vectors and norms ---
        self._docs = []
        self._norms = []
        for rid, tc in tf_raw:
            total = sum(tc.values())
            vec: dict[str, float] = {}
            for t, cnt in tc.items():
                tf = cnt / total
                vec[t] = tf * self._idf.get(t, 0.0)
            norm = math.sqrt(sum(v*v for v in vec.values()))
            self._docs.append((rid, vec))
            self._norms.append(norm if norm > 0 else 1.0)

        # --- Inverted index for fast retrieval ---
        self._inverted: dict[str, list[int]] = {}
        for doc_idx, (rid, vec) in enumerate(self._docs):
            for t in vec:
                if t not in self._inverted:
                    self._inverted[t] = []
                self._inverted[t].append(doc_idx)

    def query_topk(self, text: str, k: int = 30) -> list[tuple[float, str]]:
        """
        Returns up to k (score, record_id) pairs sorted descending by cosine similarity.
        """
        if not self._docs:
            return []

        toks = self.tokenizer(text)
        tc = Counter(toks)
        total = sum(tc.values())
        if total == 0:
            return []

        q_vec: dict[str, float] = {}
        for t, cnt in tc.items():
            idf = self._idf.get(t, 0.0)
            q_vec[t] = (cnt / total) * idf

        q_norm = math.sqrt(sum(v*v for v in q_vec.values()))
        if q_norm == 0:
            return []

        # Accumulate dot products over candidate docs via inverted index
        scores: dict[int, float] = {}
        for t, qv in q_vec.items():
            for doc_idx in self._inverted.get(t, []):
                dv = self._docs[doc_idx][1].get(t, 0.0)
                scores[doc_idx] = scores.get(doc_idx, 0.0) + qv * dv

        # Normalize to cosine
        results = []
        for doc_idx, dot in scores.items():
            cos = dot / (q_norm * self._norms[doc_idx])
            results.append((cos, self._docs[doc_idx][0]))

        return heapq.nlargest(k, results)


# ---------------------------------------------------------------------------
# Token-based index (word tokens, for name blocking)
# ---------------------------------------------------------------------------

class TokenIndex(TFIDFIndex):
    """Word-token TF-IDF index — faster than n-gram for long texts."""

    def __init__(self):
        def _tok(text: str) -> list[str]:
            return text.split()
        super().__init__(tokenizer=_tok)
