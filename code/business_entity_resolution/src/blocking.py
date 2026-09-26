"""
blocking.py — Multi-strategy candidate generation (blocking).

Goal: for each Source 1 entity, produce a small set of candidate Source 2/3
records that *includes* all true matches, while keeping candidate set size
manageable (reduces n_comparisons for the feature+model stage).

Strategy: UNION of multiple blocking keys. A true pair only needs to survive
ONE blocking key, so we can be aggressive on individual keys without hurting recall.

Blocking keys implemented:
  1. TF-IDF name (3-gram)    — handles typos, abbreviations, word order
  2. TF-IDF name (token)     — handles clean-ish matches fast
  3. TF-IDF address (3-gram) — colocation signal when name is ambiguous
  4. Sorted-token name key   — exact hash match after sorting tokens (free, 100% recall on sorted-match pairs)
  5. Suffix-stripped name key
  6. Postal code exact match — high precision, catches geographically anchored pairs
  7. First-token name key    — catches cases where first word is distinctive

All keys are country-agnostic (postcodes are extracted with the generic
fallback when country is unknown, e.g., France).
"""

import re
import logging
from collections import defaultdict
from typing import Iterable

import pandas as pd

from features import normalize_name, normalize_address, extract_postal, tokens
from tfidf_index import TFIDFIndex, TokenIndex

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sorted_token_key(name: str) -> str:
    """Canonical form: sorted tokens joined — handles word-order transpositions."""
    return " ".join(sorted(tokens(name)))


def _first_token(name: str) -> str:
    toks = tokens(name)
    return toks[0] if toks else ""


def _first_two_tokens(name: str) -> str:
    toks = tokens(name)
    return " ".join(toks[:2]) if len(toks) >= 2 else " ".join(toks)


# ---------------------------------------------------------------------------
# Blocker class
# ---------------------------------------------------------------------------

class Blocker:
    """
    Fits on Source 2 + Source 3 records, then for each Source 1 record
    returns a set of candidate IDs (union of all blocking strategies).
    """

    def __init__(
        self,
        tfidf_name_k: int = 30,
        tfidf_addr_k: int = 20,
        min_first_token_len: int = 4,
    ):
        self.tfidf_name_k = tfidf_name_k
        self.tfidf_addr_k = tfidf_addr_k
        self.min_first_token_len = min_first_token_len

        # TF-IDF indices
        self._name_3g_idx = TFIDFIndex()    # character 3-gram name
        self._name_tok_idx = TokenIndex()   # word-token name
        self._name_3g_stripped_idx = TFIDFIndex()  # suffix-stripped
        self._addr_3g_idx = TFIDFIndex()    # character 3-gram address

        # Exact-match inverted indices (hash → list of IDs)
        self._sorted_tok_inv: dict[str, list[str]] = defaultdict(list)
        self._first_tok_inv: dict[str, list[str]] = defaultdict(list)
        self._first_two_inv: dict[str, list[str]] = defaultdict(list)
        self._postal_inv: dict[str, list[str]] = defaultdict(list)

        # Store raw records for address-only fallback
        self._records: pd.DataFrame | None = None

    def fit(self, df: pd.DataFrame) -> None:
        """
        df must have columns: entity_id, business_name, business_address, country.
        This is the combined Source 2 + Source 3 pool.
        """
        self._records = df.copy()

        ids = df["entity_id"].tolist()
        names_raw = df["business_name"].fillna("").tolist()
        addrs_raw = df["business_address"].fillna("").tolist()
        countries = df["country"].fillna("").tolist()

        norm_names   = [normalize_name(n) for n in names_raw]
        norm_stripped = [normalize_name(n, strip_suffix=True) for n in names_raw]
        norm_addrs   = [normalize_address(a) for a in addrs_raw]

        # --- TF-IDF indices ---
        log.info("Building TF-IDF name index (3-gram)…")
        self._name_3g_idx.fit(list(zip(ids, norm_names)))
        log.info("Building TF-IDF name index (token)…")
        self._name_tok_idx.fit(list(zip(ids, norm_names)))
        log.info("Building TF-IDF name index (stripped 3-gram)…")
        self._name_3g_stripped_idx.fit(list(zip(ids, norm_stripped)))
        log.info("Building TF-IDF address index (3-gram)…")
        self._addr_3g_idx.fit(list(zip(ids, norm_addrs)))

        # --- Exact-match inverted indices ---
        for eid, nn, na, country in zip(ids, norm_names, norm_addrs, countries):
            st_key = _sorted_token_key(nn)
            if st_key:
                self._sorted_tok_inv[st_key].append(eid)

            ft = _first_token(nn)
            if len(ft) >= self.min_first_token_len:
                self._first_tok_inv[ft].append(eid)

            f2t = _first_two_tokens(nn)
            if f2t:
                self._first_two_inv[f2t].append(eid)

            postal = extract_postal(na, country)
            if postal:
                self._postal_inv[postal].append(eid)

        log.info(
            "Blocker fitted on %d records. "
            "Postal keys: %d, sorted-token keys: %d",
            len(ids), len(self._postal_inv), len(self._sorted_tok_inv),
        )

    def get_candidates(self, entity_id: str, name_raw: str, addr_raw: str, country: str) -> set[str]:
        """
        Returns the set of candidate IDs from the pool for a single S1 record.
        """
        candidates: set[str] = set()

        nn = normalize_name(name_raw)
        ns = normalize_name(name_raw, strip_suffix=True)
        na = normalize_address(addr_raw)

        # 1. TF-IDF name (3-gram)
        for _, cid in self._name_3g_idx.query_topk(nn, k=self.tfidf_name_k):
            candidates.add(cid)

        # 2. TF-IDF name (token)
        for _, cid in self._name_tok_idx.query_topk(nn, k=self.tfidf_name_k):
            candidates.add(cid)

        # 3. TF-IDF suffix-stripped name (3-gram)
        for _, cid in self._name_3g_stripped_idx.query_topk(ns, k=self.tfidf_name_k):
            candidates.add(cid)

        # 4. TF-IDF address (3-gram)
        if na:
            for _, cid in self._addr_3g_idx.query_topk(na, k=self.tfidf_addr_k):
                candidates.add(cid)

        # 5. Sorted-token exact match
        st_key = _sorted_token_key(nn)
        if st_key:
            candidates.update(self._sorted_tok_inv.get(st_key, []))

        # 6. First-two-token exact match
        f2t = _first_two_tokens(nn)
        if f2t:
            candidates.update(self._first_two_inv.get(f2t, []))

        # 7. First-token exact match (only for long distinctive tokens)
        ft = _first_token(nn)
        if len(ft) >= self.min_first_token_len:
            hits = self._first_tok_inv.get(ft, [])
            # Limit to avoid pulling every "Global X" / "National Y" entity
            if len(hits) <= 200:
                candidates.update(hits)

        # 8. Postal code exact match
        postal = extract_postal(na, country)
        if postal:
            candidates.update(self._postal_inv.get(postal, []))

        # Remove self if somehow present (shouldn't happen across sources)
        candidates.discard(entity_id)

        return candidates


# ---------------------------------------------------------------------------
# Batch candidate generation
# ---------------------------------------------------------------------------

def generate_candidates(
    s1_df: pd.DataFrame,
    pool_df: pd.DataFrame,
    tfidf_name_k: int = 30,
    tfidf_addr_k: int = 20,
) -> dict[str, set[str]]:
    """
    Returns dict: s1_entity_id → set of candidate IDs from pool_df.

    pool_df = combined s2 + s3 records.
    """
    blocker = Blocker(tfidf_name_k=tfidf_name_k, tfidf_addr_k=tfidf_addr_k)
    blocker.fit(pool_df)

    result: dict[str, set[str]] = {}
    n = len(s1_df)
    for i, row in enumerate(s1_df.itertuples(index=False)):
        if i % 500 == 0:
            log.info("Blocking: %d / %d", i, n)
        cands = blocker.get_candidates(
            entity_id=row.entity_id,
            name_raw=getattr(row, "business_name", "") or "",
            addr_raw=getattr(row, "business_address", "") or "",
            country=getattr(row, "country", "") or "",
        )
        result[row.entity_id] = cands

    total_candidates = sum(len(v) for v in result.values())
    log.info(
        "Blocking done. Total candidates: %d, avg per S1: %.1f",
        total_candidates, total_candidates / max(len(result), 1),
    )
    return result


def blocking_recall(candidates: dict[str, set[str]], ground_truth: dict[str, set[str]]) -> float:
    """
    Fraction of true pairs that appear in the candidate set.
    Call on validation split to verify recall ceiling before training model.
    """
    hits = total = 0
    for s1_id, true_matches in ground_truth.items():
        if not true_matches:
            continue
        cands = candidates.get(s1_id, set())
        for m in true_matches:
            total += 1
            if m in cands:
                hits += 1
    return hits / total if total > 0 else 1.0
