"""
features.py — Text normalization, similarity features, and candidate scoring.

Design principles:
- Every function is stateless and handles None/NaN safely.
- Country-agnostic by default; country-specific logic is additive,
  never a hard gate, so France (unseen in training) does not break anything.
- Legal-suffix normalization is a separate feature track from raw similarity
  so the model can learn how much weight to put on suffix-stripped vs raw match.
"""

import re
import unicodedata
from typing import Optional

# ---------------------------------------------------------------------------
# Legal / organisational suffix normalization
# ---------------------------------------------------------------------------

# Covers common US, India, and France forms (and their abbreviations).
# Sorted longest-first so greedier patterns match first.
_LEGAL_SUFFIXES = sorted([
    # English / generic
    "private limited", "pvt ltd", "pvt. ltd.", "pvt. ltd", "pvt ltd.",
    "public limited company", "plc",
    "limited liability company", "llc", "l.l.c.",
    "limited liability partnership", "llp", "l.l.p.",
    "incorporated", "inc.", "inc",
    "corporation", "corp.", "corp",
    "limited", "ltd.", "ltd",
    "company", "co.", "co",
    "private", "pvt.", "pvt",
    "enterprises", "enterprise",
    "associates", "associate",
    "brothers", "bros.", "bros",
    "industries", "industry",
    "international", "intl.", "intl",
    "group", "grp.",
    "services", "service",
    "solutions", "solution",
    "technologies", "technology", "tech",
    "systems", "system",
    "trading", "traders", "trader",
    "distributors", "distributor",
    "agencies", "agency",
    "holdings", "holding",
    # India-specific
    "pvt. limited",
    # France-specific
    "société anonyme", "s.a.", "sa",
    "société à responsabilité limitée", "sarl", "s.a.r.l.",
    "société par actions simplifiée", "sas", "s.a.s.",
    "société en nom collectif", "snc",
    "entreprise individuelle", "ei",
    "auto-entrepreneur",
    "établissements", "etab.",
    # More generic
    "& co", "and co",
], key=len, reverse=True)

_SUFFIX_RE = re.compile(
    r"\b(" + "|".join(re.escape(s) for s in _LEGAL_SUFFIXES) + r")\.?\s*$",
    re.IGNORECASE
)

# Ampersand/and normalization
_AND_RE = re.compile(r'\b&\b|\band\b', re.IGNORECASE)

# Street abbreviation map (US + India + France)
_ADDR_ABBR = {
    r'\brd\b': 'road', r'\bst\b': 'street', r'\bave\b': 'avenue',
    r'\bavenue\b': 'avenue', r'\blane\b': 'lane', r'\bln\b': 'lane',
    r'\bblvd\b': 'boulevard', r'\bdr\b': 'drive', r'\bct\b': 'court',
    r'\bpl\b': 'place', r'\bsq\b': 'square', r'\bnr\b': 'near',
    r'\bno\b': 'number', r'\bno\.\b': 'number', r'\bopp\b': 'opposite',
    r'\bopp\.\b': 'opposite', r'\bbldg\b': 'building', r'\bbldg\.\b': 'building',
    r'\bappt?\b': 'apartment', r'\bapt\b': 'apartment',
    r'\bflr\b': 'floor', r'\bfl\b': 'floor',
    r'\bdist\b': 'district', r'\bph\b': 'phase',
    r'\bsec\b': 'sector', r'\bsect\b': 'sector',
    # France
    r'\brue\b': 'rue', r'\bav\b': 'avenue', r'\bbd\b': 'boulevard',
    r'\bimpasse\b': 'impasse', r'\bplace\b': 'place',
}
_ADDR_ABBR_RE = {re.compile(k, re.IGNORECASE): v for k, v in _ADDR_ABBR.items()}

# Punctuation to strip (keep hyphens inside words, strip leading/trailing)
_MULTI_SPACE_RE = re.compile(r"\s+")


# ---------------------------------------------------------------------------
# Core normalization
# ---------------------------------------------------------------------------

def normalize_unicode(text: str) -> str:
    has_nonlatin = any(
        unicodedata.category(ch)[0] == 'L'
        and not unicodedata.name(ch, '').startswith('LATIN')
        for ch in text if ch.strip()
    )
    if has_nonlatin:
        result = []
        for ch in text:
            cat = unicodedata.category(ch)
            if cat[0] in ('L', 'M', 'N') or ch in (' ', '-'):
                result.append(ch.lower())
        return ''.join(result)
    else:
        nfkd = unicodedata.normalize('NFKD', text)
        result = []
        for ch in nfkd:
            cat = unicodedata.category(ch)
            if cat == 'Mn':
                continue
            elif cat[0] in ('L', 'N') or ch in (' ', '-'):
                result.append(ch.lower())
        return ''.join(result)


def strip_punctuation(text: str) -> str:
    """Remove punctuation while preserving Unicode letters and script marks."""
    return "".join(
        character for character in text
        if character == "-"
        or character.isspace()
        or unicodedata.category(character)[0] in {"L", "M", "N"}
    )


def normalize_name(raw: Optional[str], *, strip_suffix: bool = False) -> str:
    """
    Canonical business name normalization.
    strip_suffix=True produces the suffix-stripped variant used as a
    separate feature track, not as a replacement.
    """
    if not raw or (isinstance(raw, float)):
        return ""
    text = str(raw)
    text = normalize_unicode(text)
    text = _AND_RE.sub(" and ", text)
    text = strip_punctuation(text)
    if strip_suffix:
        text = _SUFFIX_RE.sub("", text)
    text = _MULTI_SPACE_RE.sub(" ", text).strip()
    return text


def normalize_address(raw: Optional[str]) -> str:
    """Canonical address normalization — abbreviation expansion, unicode fold."""
    if not raw or (isinstance(raw, float)):
        return ""
    text = str(raw)
    text = normalize_unicode(text)
    for pattern, replacement in _ADDR_ABBR_RE.items():
        text = pattern.sub(replacement, text)
    text = strip_punctuation(text)
    text = _MULTI_SPACE_RE.sub(" ", text).strip()
    return text


# ---------------------------------------------------------------------------
# Tokenization helpers
# ---------------------------------------------------------------------------

def tokens(text: str) -> list[str]:
    return [t for t in text.split() if t]


def char_ngrams(text: str, n: int = 3) -> list[str]:
    padded = f"  {text}  "  # boundary pads
    return [padded[i:i+n] for i in range(len(padded) - n + 1)]


# ---------------------------------------------------------------------------
# Similarity measures
# ---------------------------------------------------------------------------

def jaccard_tokens(a: str, b: str) -> float:
    ta, tb = set(tokens(a)), set(tokens(b))
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def jaccard_ngrams(a: str, b: str, n: int = 3) -> float:
    na, nb = set(char_ngrams(a, n)), set(char_ngrams(b, n))
    if not na and not nb:
        return 1.0
    if not na or not nb:
        return 0.0
    return len(na & nb) / len(na | nb)


def token_sort_ratio(a: str, b: str) -> float:
    """Sort tokens then compute character-level Jaccard — handles word reordering."""
    sa = " ".join(sorted(tokens(a)))
    sb = " ".join(sorted(tokens(b)))
    return jaccard_ngrams(sa, sb, n=3)


def levenshtein(a: str, b: str) -> int:
    """Standard DP Levenshtein — O(|a|·|b|), capped at 300 chars for speed."""
    a, b = a[:300], b[:300]
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    la, lb = len(a), len(b)
    prev = list(range(lb + 1))
    for i, ca in enumerate(a):
        curr = [i + 1] + [0] * lb
        for j, cb in enumerate(b):
            curr[j+1] = min(
                prev[j+1] + 1,
                curr[j] + 1,
                prev[j] + (0 if ca == cb else 1)
            )
        prev = curr
    return prev[lb]


def normalized_edit(a: str, b: str) -> float:
    """Normalized edit similarity: 1 - (edit_dist / max_len)."""
    if not a and not b:
        return 1.0
    maxlen = max(len(a), len(b))
    if maxlen == 0:
        return 1.0
    return 1.0 - levenshtein(a, b) / maxlen


def token_overlap_ratio(a: str, b: str) -> float:
    """|intersection| / min(|A|, |B|) — precision-like, less punishing than Jaccard."""
    ta, tb = set(tokens(a)), set(tokens(b))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / min(len(ta), len(tb))


def tfidf_cosine(a_vec: dict, b_vec: dict) -> float:
    """
    Cosine similarity from pre-built TF-IDF token→weight dicts.
    Caller builds per-record vectors; this function just computes the score.
    """
    common = set(a_vec) & set(b_vec)
    if not common:
        return 0.0
    dot = sum(a_vec[t] * b_vec[t] for t in common)
    mag_a = sum(v*v for v in a_vec.values()) ** 0.5
    mag_b = sum(v*v for v in b_vec.values()) ** 0.5
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


# ---------------------------------------------------------------------------
# Address component extraction (language-agnostic best-effort)
# ---------------------------------------------------------------------------

_PIN_INDIA_RE = re.compile(r'\b([1-9][0-9]{5})\b')        # 6-digit Indian PIN
_ZIP_US_RE = re.compile(r'\b([0-9]{5})(?:-[0-9]{4})?\b')  # US ZIP / ZIP+4
_ZIP_FRANCE_RE = re.compile(r'\b([0-9]{5})\b')             # French postal code

def extract_postal(address: str, country: str) -> Optional[str]:
    c = (country or "").strip().lower()
    if c == "india":
        m = _PIN_INDIA_RE.search(address)
    elif c == "us":
        m = _ZIP_US_RE.search(address)
    elif c == "france":
        m = _ZIP_FRANCE_RE.search(address)
    else:
        # Fallback: try any 5-or-6-digit code
        m = re.search(r'\b([0-9]{5,6})\b', address)
    return m.group(1) if m else None


def extract_city_tokens(address: str) -> list[str]:
    """
    Heuristic: city-like tokens are alphabetic tokens longer than 3 chars
    that appear in the latter half of the address string.
    Returns sorted list for set-based comparison.
    """
    toks = [t for t in address.split() if t.isalpha() and len(t) > 3]
    half = toks[len(toks)//2:]  # bias toward latter half (city/state position)
    return sorted(set(half))


def is_landmark_address(address: str) -> bool:
    """True if address is a landmark-style reference (Near X, Opposite Y, etc.)"""
    keywords = {"near", "opposite", "opp", "behind", "beside", "adj", "adjacent",
                "next to", "landmark", "main road", "highway"}
    low = address.lower()
    return any(kw in low for kw in keywords)


# ---------------------------------------------------------------------------
# Full pairwise feature vector
# ---------------------------------------------------------------------------

def compute_pair_features(
    s1_name_raw: str, s1_addr_raw: str, s1_country: str,
    sx_name_raw: str, sx_addr_raw: str, sx_country: str,
) -> dict:
    """
    Returns a flat dict of named float features for one S1 vs S2/S3 candidate pair.
    All features are bounded [0,1] or binary {0,1} for easy consumption by LGBM/XGB.
    """
    # --- Normalize names ---
    n1 = normalize_name(s1_name_raw)
    nx = normalize_name(sx_name_raw)
    n1s = normalize_name(s1_name_raw, strip_suffix=True)
    nxs = normalize_name(sx_name_raw, strip_suffix=True)

    # --- Normalize addresses ---
    a1 = normalize_address(s1_addr_raw)
    ax = normalize_address(sx_addr_raw)

    # --- Name features ---
    f = {}
    f["name_jaccard_tok"]     = jaccard_tokens(n1, nx)
    f["name_jaccard_3g"]      = jaccard_ngrams(n1, nx, 3)
    f["name_jaccard_2g"]      = jaccard_ngrams(n1, nx, 2)
    f["name_edit"]            = normalized_edit(n1, nx)
    f["name_token_sort"]      = token_sort_ratio(n1, nx)
    f["name_overlap"]         = token_overlap_ratio(n1, nx)
    # Suffix-stripped variants — separate "track"
    f["names_jaccard_tok"]    = jaccard_tokens(n1s, nxs)
    f["names_jaccard_3g"]     = jaccard_ngrams(n1s, nxs, 3)
    f["names_edit"]           = normalized_edit(n1s, nxs)
    f["names_token_sort"]     = token_sort_ratio(n1s, nxs)
    # Suffix-stripping delta (how much the suffix was obscuring a name match)
    f["name_suffix_delta"]    = max(0.0, f["names_jaccard_3g"] - f["name_jaccard_3g"])

    # --- Address features ---
    f["addr_jaccard_tok"]     = jaccard_tokens(a1, ax)
    f["addr_jaccard_3g"]      = jaccard_ngrams(a1, ax, 3)
    f["addr_edit"]            = normalized_edit(a1, ax)
    f["addr_overlap"]         = token_overlap_ratio(a1, ax)

    # --- Postal code agreement ---
    p1 = extract_postal(a1, s1_country)
    px = extract_postal(ax, sx_country)
    if p1 is not None and px is not None:
        f["postal_match"]     = 1.0 if p1 == px else 0.0
        f["postal_prefix_match"] = 1.0 if p1[:3] == px[:3] else 0.0
        f["postal_known"]     = 1.0
    else:
        f["postal_match"]     = 0.0
        f["postal_prefix_match"] = 0.0
        f["postal_known"]     = 0.0

    # --- City token overlap ---
    c1 = extract_city_tokens(a1)
    cx = extract_city_tokens(ax)
    if c1 and cx:
        common_city = set(c1) & set(cx)
        f["city_overlap"]     = len(common_city) / min(len(c1), len(cx))
    else:
        f["city_overlap"]     = 0.0

    # --- Landmark flag (lowers address-signal reliability) ---
    f["s1_landmark"]          = float(is_landmark_address(a1))
    f["sx_landmark"]          = float(is_landmark_address(ax))

    # --- Missingness flags ---
    f["s1_addr_missing"]      = float(not a1)
    f["sx_addr_missing"]      = float(not ax)
    f["s1_name_missing"]      = float(not n1)
    f["sx_name_missing"]      = float(not nx)

    # --- Country agreement ---
    c1_raw = (s1_country or "").strip().lower()
    cx_raw = (sx_country or "").strip().lower()
    f["country_match"]        = float(c1_raw == cx_raw)

    # --- Combined signals ---
    # Precision-friendly combined: name AND address both moderate → strong positive
    name_score = max(f["name_jaccard_3g"], f["names_jaccard_3g"])
    addr_score = f["addr_jaccard_tok"] if not (f["s1_addr_missing"] or f["sx_addr_missing"]) else 0.5
    f["name_x_addr"]          = name_score * addr_score
    f["name_safe"]            = name_score  # convenience alias

    return f


FEATURE_NAMES = list(compute_pair_features("a", "a", "", "a", "a", "").keys())
