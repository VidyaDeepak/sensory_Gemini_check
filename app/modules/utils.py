"""Shared text-matching helpers used by every analysis module and the router."""
from __future__ import annotations

import difflib
import re
from typing import Optional, Sequence


def _norm(s: str) -> str:
    s = re.sub(r"[_\-]+", " ", s)
    s = re.sub(r"[^a-z0-9 ]+", " ", s.lower())
    return re.sub(r"\s+", " ", s).strip()


def best_match(text: str, candidates: Sequence[str], threshold: float = 0.55) -> Optional[str]:
    """Fuzzy-match `text` against a list of candidate strings (e.g. benefit
    names, product names, platform names). Tries, in order:
      1. exact substring match (case/punctuation-insensitive), either direction
      2. whole-string similarity ratio
      3. sliding-window similarity ratio (candidate embedded in a longer question)
    Returns the best-scoring candidate, or None if nothing clears `threshold`.
    """
    if not text or not candidates:
        return None

    text_n = _norm(text)

    # 1) direct substring, either direction (handles both "the full candidate
    #    name appears in the question" and "a short question fragment like
    #    'shine' is itself a word inside a longer candidate name")
    substring_hits = []
    for c in candidates:
        c_n = _norm(c)
        if c_n in text_n:
            return c
        if c_n and any(word == c_n or word in c_n.split() for word in text_n.split() if len(word) >= 5):
            substring_hits.append(c)
    if len(substring_hits) == 1:
        return substring_hits[0]

    # 2) whole-string fuzzy ratio (skip very short candidates — they inflate
    #    ratios against arbitrary words and cause false positives)
    best, best_score = None, 0.0
    for c in candidates:
        if len(_norm(c)) < 4:
            continue
        score = difflib.SequenceMatcher(None, _norm(c), text_n).ratio()
        if score > best_score:
            best, best_score = c, score

    # 3) sliding window (helps short candidate names inside a longer question)
    words = text_n.split()
    for c in candidates:
        if len(_norm(c)) < 4:
            continue
        c_words = _norm(c).split()
        window = len(c_words)
        if window == 0 or window > len(words):
            continue
        for i in range(len(words) - window + 1):
            chunk = " ".join(words[i : i + window])
            score = difflib.SequenceMatcher(None, _norm(c), chunk).ratio()
            if score > best_score:
                best, best_score = c, score

    return _guarded(text_n, best, best_score, threshold)


def _guarded(text_n: str, best: Optional[str], best_score: float, threshold: float) -> Optional[str]:
    """Reject a fuzzy (non-exact-substring) match that has no real word
    overlap with the query at all — pure character-shape similarity between
    a short query and a long, unrelated candidate can clear a 0.5-0.6 ratio
    threshold by coincidence (e.g. "Anessa" fuzzy-matching "nivea soft" on
    two short, vowel-heavy words with no shared word whatsoever).

    The overlap word itself must be >=5 chars, matching strategy 1's own
    convention — a 4-char word like "soft" is common enough in product
    names (e.g. "Lux Soft Touch" vs "nivea soft") to produce the exact
    same class of false positive this guard exists to prevent. Exact
    substring hits already returned earlier and never reach this check."""
    if best is None or best_score < threshold:
        return None
    text_words = {w for w in text_n.split() if len(w) >= 5}
    if not text_words:
        return best
    best_words = set(_norm(best).split())
    return best if (text_words & best_words) else None


def exact_match(text: str, candidates: Sequence[str]) -> Optional[str]:
    """Strategy-1-only substring match (no fuzzy fallback) — used when a
    caller needs certainty that a candidate was explicitly named, not
    guessed via similarity ratio (e.g. disambiguating which module's
    entity list a "compare A and B" question's two halves belong to)."""
    if not text or not candidates:
        return None
    text_n = _norm(text)
    for c in candidates:
        c_n = _norm(c)
        if c_n and c_n in text_n:
            return c
    return None


def any_keyword(text: str, keywords: Sequence[str]) -> bool:
    text_l = text.lower()
    return any(kw in text_l for kw in keywords)


# --- Benefit-stage categorisation --------------------------------------
# Both sst_attribute_rank.csv (via `attr_group`) and sst_driver_rank.csv
# (via the descriptive suffix baked into `driver`) use ten raw labels that
# roll up into four human "benefit stage" buckets. Centralising the mapping
# here lets every module that touches test-stage data group/filter the same
# way, instead of each module flattening everything into one bag.
BENEFIT_STAGE_ORDER = ["Skin Feel", "Fingertip", "Skin Look", "Fragrance"]

_STAGE_CATEGORY_RULES = [
    ("Fingertip", ["fingertip"]),
    ("Fragrance", ["fragrance"]),
    ("Skin Feel", ["skin feel"]),
    ("Skin Look", ["skin look", "appearance"]),
]


def stage_category(text: str) -> str:
    """Map a raw attr_group / driver label (e.g. 'CLT Fragrance IMD') to one
    of BENEFIT_STAGE_ORDER, or 'Other' if it doesn't match any rule."""
    t = (text or "").lower()
    for cat, keywords in _STAGE_CATEGORY_RULES:
        if any(kw in t for kw in keywords):
            return cat
    return "Other"


def match_stage_category(question: str) -> Optional[str]:
    """If the question explicitly names one of the four benefit-stage
    categories (Skin Feel / Fingertip / Skin Look / Fragrance), return it."""
    return best_match(question, BENEFIT_STAGE_ORDER, threshold=0.6)


_BREAKDOWN_PHRASES = [
    "by benefit stage", "by stage", "by benefit", "benefit stage",
    "breakdown by stage", "break down by stage", "per stage", "by category",
]


def wants_benefit_stage_breakdown(question: str) -> bool:
    """True if the question asks for a benefit-stage breakdown without
    naming one specific stage/category (e.g. "top attributes ... by
    benefit stage" rather than "... for the Fragrance stage")."""
    q = question.lower()
    return any(p in q for p in _BREAKDOWN_PHRASES)


CLT_CODE_RE = re.compile(r"(?i)\bCLT\s+[A-Z0-9-]+\b")


def extract_clt_code(text: str) -> Optional[str]:
    """Pull a literal test-wave code like 'CLT B7-8' out of a longer string
    (e.g. a driver label 'CLT B7-8 Skin feel IU')."""
    m = CLT_CODE_RE.match(text.strip()) or CLT_CODE_RE.search(text)
    return m.group(0) if m else None


_CONTINUATION_RE = re.compile(r"(?i)^\s*(and\b|what about\b|how about\b|also\b|what's\b|whats\b|why\b)")


def looks_like_continuation(question: str) -> bool:
    """True for short follow-ups that lean on an earlier turn for context
    ("and Skin Look?", "what about the Fragrance stage?") rather than
    naming their subject outright."""
    return bool(_CONTINUATION_RE.match(question.strip()))


def text_contains(question: str, value: Optional[str]) -> bool:
    """Whether `value` (e.g. a resolved benefit/product name) is literally
    present in `question` — used to tell a confident, explicit match from
    one that only came from fuzzy/sliding-window guesswork."""
    if not value:
        return False
    return _norm(str(value)) in _norm(question)


def snake_to_title(s: str) -> str:
    return s.replace("_", " ").strip().title()


COMPARE_SPLIT_RE = re.compile(r"(?i)\s+(?:and|vs\.?|versus|against|from|with)\s+")
COMPARE_TRIGGER_RE = re.compile(
    r"(?i)\bcompare\b|\bvs\.?\b|\bversus\b|\bseparates?\b|\bdiffers?\b|"
    r"\bdifference between\b|\bdistinguish(?:es)?\b|\bsimilar\b|\balike\b"
)

# Lead-in phrasings stripped before splitting "X and Y" out of a compare-style
# question, so e.g. "which attributes separate A from B" and "why are A and
# B similar" split cleanly into (A, B) instead of routing on the whole
# sentence (which then risks a spurious fuzzy match on unrelated text).
_COMPARE_LEADIN_PATTERNS = [
    r"^\s*compare\s+",
    r"^\s*(?:which|what)\s+attributes?\s+(?:separate|distinguish|differentiate)\s+",
    r"^\s*(?:what'?s|what\s+is)\s+the\s+difference\s+between\s+",
    r"^\s*why\s+(?:are|is)\s+",
    r"^\s*how\s+(?:do|does)\s+",
]
_COMPARE_TRAILING_RE = re.compile(r"(?i)\s+(?:similar|alike|different|distinguishable)\s*[?.]*\s*$")


def is_compare_query(question: str) -> bool:
    return bool(COMPARE_TRIGGER_RE.search(question))


def split_compare_halves(question: str) -> Optional[tuple[str, str]]:
    """"Compare A and B" / "A vs B" / "A versus B" / "attributes separate A
    from B" / "why are A and B similar" -> (A, B). Returns None if the
    question doesn't clearly split into two comparable halves."""
    q = question.strip(" ?.")
    for pat in _COMPARE_LEADIN_PATTERNS:
        q = re.sub(pat, "", q, flags=re.I)
    parts = COMPARE_SPLIT_RE.split(q, maxsplit=1)
    if len(parts) == 2 and parts[0].strip() and parts[1].strip():
        a = parts[0].strip(" ?.")
        b = _COMPARE_TRAILING_RE.sub("", parts[1]).strip(" ?.")
        return a, b
    return None


# Chart types a user can explicitly request in their question, and the
# phrases that signal each one. Only types that share the same underlying
# data shape as "categorical" (labels + values) are swappable — scatter and
# radar have structurally different data and aren't included here.
CHART_TYPE_KEYWORDS = {
    "bar": ["bar chart", "bar graph", "column chart", " as a bar", " as bar"],
    "line": ["line chart", "line graph", "trend chart", " as a line", " as line"],
    "pie": ["pie chart", " as a pie", " as pie"],
    "doughnut": ["doughnut chart", "donut chart", "donut", "doughnut"],
}


def detect_requested_chart_type(question: str) -> Optional[str]:
    """If the question explicitly names a chart type (e.g. "...as a pie
    chart", "line graph of..."), return it. Otherwise None — the module's
    default chart type is used."""
    q = question.lower()
    for chart_type, phrases in CHART_TYPE_KEYWORDS.items():
        if any(p in q for p in phrases):
            return chart_type
    return None


TABLE_ONLY_PHRASES = [
    "table only", "just a table", "just the table", "as a table", "in table form",
    "no chart", "without a chart", "skip the chart", "text only", "no visual",
]


def wants_table_only(question: str) -> bool:
    """Explicit request to skip the chart and show only the table."""
    q = question.lower()
    return any(p in q for p in TABLE_ONLY_PHRASES)


EXEC_SUMMARY_PHRASES = [
    "executive summary", "exec summary", "summary for leadership", "management summary",
    "business summary", "tl;dr", "tldr", "give me a summary",
]


def wants_executive_summary(question: str) -> bool:
    q = question.lower()
    return any(p in q for p in EXEC_SUMMARY_PHRASES)
