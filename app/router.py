"""
Router: decides which of the 6 analysis modules should answer a question.

Deterministic, keyword + fuzzy-match based (no LLM required for routing —
keeps it fast, free, and debuggable). Order matters: more specific/ambiguous
signals are checked first.
"""
from __future__ import annotations

import re
from typing import Optional

from .context import context_hint_text
from .modules import bh_blindhut, pc_perceptual, rw_relweight, sd_driver, spider_profile, sr_attribute
from .modules.utils import (
    BENEFIT_STAGE_ORDER,
    any_keyword,
    best_match,
    exact_match,
    is_compare_query,
    looks_like_continuation,
    split_compare_halves,
    text_contains,
)

RW_KEYWORDS = [
    "hair", "shampoo", "hairfall", "conditioner", "scalp", "rwa",
    "weakest driver", "strongest driver", "strongest rwa", "weakest rwa",
    "consistent across studies", "summarize key rwa", "rwa learnings", "rwa driver",
]
SPIDER_KEYWORDS = ["profile", "radar", "spider", "gap", "distinguish", "differentiate", "separate", "unique", "improvement relative", "fingerprint", "sensory profile of"]
PC_KEYWORDS = [
    "perceptual", "map", "position", "pca", "coordinate", "nearest neighbour", "nearest neighbor",
    "closest together", "positioned closest", "most different", "furthest apart", "farthest apart",
    "cluster", "whitespace", "white space", "sensory map", "most similar", "similar to",
]
BH_KEYWORDS = [
    "score", "highest scoring", "best product", "which product", "top product", "compare product",
    "products ranked", "ranked for", "predicted performer", "predicted to perform", "hut score",
    "best deliver", "products best deliver",
]

DRIVER_STEM_RE = re.compile(r"(?i)\bdriv\w*\b")
DRIVER_PHRASES = ["influenced by", "influenced most by", "contributes most", "contribute most"]

MODULES = {
    "sr_attribute": sr_attribute,
    "sd_driver": sd_driver,
    "bh_blindhut": bh_blindhut,
    "pc_perceptual": pc_perceptual,
    "rw_relweight": rw_relweight,
    "spider_profile": spider_profile,
}

# Which result field(s) hold the "headline" entity for each module — used to
# tell whether a continuation-style follow-up's answer was explicitly named
# in the question, or only landed there via a fuzzy-match guess.
_PRIMARY_ENTITY_KEYS = {
    "sr_attribute": ("benefit",),
    "sd_driver": ("benefit",),
    "bh_blindhut": ("benefit", "product"),
    "pc_perceptual": ("category", "product"),
    "rw_relweight": ("platform",),
    "spider_profile": ("product",),
}


def preload():
    """Preload datasets for all registered modules to eliminate cold-start latency."""
    for mod in MODULES.values():
        if hasattr(mod, "preload"):
            mod.preload()


# Modules whose compare() logic matches against a specific candidate list —
# used to disambiguate "compare A and B" style questions by checking which
# module's own entity list both halves actually belong to.
COMPARE_CANDIDATES = [
    ("sr_attribute", sr_attribute, lambda: sr_attribute.benefits()),
    ("sd_driver", sd_driver, lambda: sd_driver.benefits()),
    ("bh_blindhut", bh_blindhut, lambda: bh_blindhut.products()),
    ("spider_profile", spider_profile, lambda: spider_profile.products() + spider_profile.brands()),
    ("pc_perceptual", pc_perceptual, lambda: pc_perceptual.products()),
]


def _route_compare(question: str) -> str | None:
    """For "compare A and B" questions, find the module whose own candidate
    list both halves actually match — far more precise than fuzzy-matching
    the whole raw question against every module's product list.

    Two passes: exact-substring first (immune to two loose fuzzy matches
    spuriously collapsing onto the same wrong candidate — a real bug this
    guards against), fuzzy only as a fallback, and either way the two
    halves must resolve to two *different* candidates."""
    halves = split_compare_halves(question)
    if not halves:
        return None
    a, b = halves

    for name, _module, get_candidates in COMPARE_CANDIDATES:
        candidates = get_candidates()
        ea, eb = exact_match(a, candidates), exact_match(b, candidates)
        if ea and eb and ea != eb:
            return name

    for name, _module, get_candidates in COMPARE_CANDIDATES:
        candidates = get_candidates()
        ma, mb = best_match(a, candidates, threshold=0.55), best_match(b, candidates, threshold=0.55)
        if ma and mb and ma != mb:
            return name
    return None


def route(question: str) -> str:
    """Return the module name best suited to answer `question`."""
    q = question.lower()

    # 1. Hair care platform (RW) — distinct category, check first
    if any_keyword(q, RW_KEYWORDS) or best_match(question, rw_relweight.platforms(), threshold=0.72):
        return "rw_relweight"

    # 2. Explicit spider/radar/profile/gap language
    if any_keyword(q, SPIDER_KEYWORDS):
        return "spider_profile"

    # 3. Explicit perceptual-map language, or a matched PC category
    if any_keyword(q, PC_KEYWORDS):
        return "pc_perceptual"
    if best_match(question, pc_perceptual.categories(), threshold=0.75):
        return "pc_perceptual"

    # 4. Compare queries — disambiguate by which module's entity list both
    #    halves actually belong to, rather than fuzzy-matching the raw
    #    question (which produces false positives on long product names).
    if is_compare_query(question):
        compare_module = _route_compare(question)
        if compare_module:
            return compare_module

    # 5. "attributes" is a strong, specific signal for sr_attribute even
    #    when the question also contains a driver-stem word ("...drive...").
    if "attribute" in q:
        return "sr_attribute"

    # 6. Bare driver/drives/driven language (no "attribute") -> SD
    if DRIVER_STEM_RE.search(q) or any(p in q for p in DRIVER_PHRASES):
        return "sd_driver"

    # 7. Blind-hut product scoring language
    if any_keyword(q, BH_KEYWORDS):
        return "bh_blindhut"

    # 8. If the question names a spider product/brand specifically (profile-style ask)
    if best_match(question, spider_profile.products(), threshold=0.6) or best_match(question, spider_profile.brands(), threshold=0.65):
        return "spider_profile"

    # 9. If it names a BH product specifically
    if best_match(question, bh_blindhut.products(), threshold=0.6):
        return "bh_blindhut"

    # 10. Default: attribute-level driver analysis (the most common ask)
    return "sr_attribute"


def _first_other(items: list[str], *exclude) -> str | None:
    """First item in `items` that isn't one of `exclude` (order-preserving)."""
    ex = {e for e in exclude if e}
    return next((i for i in items if i not in ex), None)


def _followups(module_name: str, result: dict) -> list[str]:
    """Build 2-4 natural next-question suggestions from the shape of the
    answer just given, using each module's own known-entity lists so
    suggestions are always answerable, not just plausible-looking."""
    module = MODULES.get(module_name)
    out: list[str] = []
    try:
        if module_name in ("sr_attribute", "sd_driver"):
            noun = "attributes" if module_name == "sr_attribute" else "drivers"
            benefit = result.get("benefit")
            all_benefits = module.benefits()

            if result.get("compare_benefit"):
                b2 = result["compare_benefit"]
                nxt = _first_other(all_benefits, benefit, b2)
                out.append(f"Show top {noun} for {benefit} by benefit stage")
                if nxt:
                    out.append(f"Compare {b2} and {nxt}")
            elif result.get("breakdown"):
                cat = BENEFIT_STAGE_ORDER[0]
                out.append(f"Top {noun} for {benefit} in the {cat} stage")
                nxt = _first_other(all_benefits, benefit)
                if nxt:
                    out.append(f"Compare {benefit} and {nxt}")
            elif result.get("category"):
                out.append(f"Show top {noun} for {benefit} by benefit stage")
                nxt_cat = _first_other(BENEFIT_STAGE_ORDER, result["category"])
                if nxt_cat:
                    out.append(f"What about the {nxt_cat} stage for {benefit}?")
            else:
                out.append(f"Show top {noun} for {benefit} by benefit stage")
                nxt = _first_other(all_benefits, benefit)
                if nxt:
                    out.append(f"Compare {benefit} and {nxt}")
            nxt2 = _first_other(all_benefits, benefit, result.get("compare_benefit"))
            if nxt2:
                verb = "attributes drive" if module_name == "sr_attribute" else "drives"
                out.append(f"What {verb} {nxt2}?")

        elif module_name == "bh_blindhut":
            if result.get("compare_product"):
                p2 = result["compare_product"]
                nxt = _first_other(module.products(), result.get("product"), p2)
                if nxt:
                    out.append(f"Compare {p2} and {nxt}")
            elif result.get("product"):
                p = result["product"]
                b = _first_other(module.benefits())
                if b:
                    out.append(f"Which products score highest on {b}?")
                nxt = _first_other(module.products(), p)
                if nxt:
                    out.append(f"Compare {p} and {nxt}")
            elif result.get("benefit"):
                nxt_b = _first_other(module.benefits(), result["benefit"])
                if nxt_b:
                    out.append(f"Which products score highest on {nxt_b}?")
                top_product = result["table"][0]["Product"] if result.get("table") else None
                if top_product:
                    out.append(f"How does {top_product} score across benefits?")

        elif module_name == "pc_perceptual":
            cat = result.get("category")
            title_l = result.get("title", "").lower()
            if cat:
                if "cluster" not in title_l:
                    out.append(f"Show product clusters for {cat}")
                if "whitespace" not in title_l:
                    out.append(f"What's the whitespace in {cat}?")
                nxt_cat = _first_other(module.categories(), cat)
                if nxt_cat:
                    out.append(f"Show the perceptual map for {nxt_cat}")

        elif module_name == "rw_relweight":
            plat = result.get("platform")
            if plat:
                out.append(f"What are the weakest RWA drivers for {plat}?")
                out.append(f"Summarize key RWA learnings for {plat}")

        elif module_name == "spider_profile":
            prod = result.get("product")
            if result.get("compare_product"):
                p2 = result["compare_product"]
                nxt = _first_other(module.products(), prod, p2)
                if nxt:
                    out.append(f"Compare {p2} and {nxt}")
            elif prod:
                out.append(f"What makes {prod} unique?")
                nxt = _first_other(module.products(), prod)
                if nxt:
                    out.append(f"Compare {prod} and {nxt}")
    except Exception:
        # Suggestions are a nice-to-have; never let them break the real answer.
        pass

    seen, deduped = set(), []
    for s in out:
        if s and s not in seen:
            seen.add(s)
            deduped.append(s)
    return deduped[:4]


def followups(module_name: str, result: dict) -> list[str]:
    """Public wrapper around _followups() so other entry points (e.g. the
    LLM router) can reuse the same suggestion logic."""
    return _followups(module_name, result)


_ENTITY_LISTS = {
    "sr_attribute": lambda: sr_attribute.benefits(),
    "sd_driver": lambda: sd_driver.benefits(),
    "bh_blindhut": lambda: bh_blindhut.products() + bh_blindhut.benefits(),
    "pc_perceptual": lambda: pc_perceptual.products() + pc_perceptual.categories(),
    "rw_relweight": lambda: rw_relweight.platforms(),
    "spider_profile": lambda: spider_profile.products() + spider_profile.brands(),
}


def _has_explicit_entity(module_name: str, question: str) -> bool:
    """True only if one of module_name's own known entities is literally,
    exactly present in the question — not a fuzzy/partial-word guess. Used
    to gate fallback routing so a module we didn't select can't hijack an
    answer via a loose ratio match on an unrelated dataset (e.g. matching
    "Dove Original" to an unrelated product purely because both contain
    the word "original")."""
    get_list = _ENTITY_LISTS.get(module_name)
    if not get_list:
        return False
    q_n = question.lower()
    return any(len(str(e)) >= 4 and str(e).lower() in q_n for e in get_list())


def answer(question: str, top_n: int = 8) -> dict:
    module_name = route(question)
    module = MODULES[module_name]
    result = module.answer(question, top_n=top_n)

    if not result.get("ok"):
        # The routed module couldn't answer. Only try another module if the
        # question literally, explicitly names one of *that* module's own
        # entities — never on a loose fuzzy-match basis. A wrong "I don't
        # know" from the right module beats a confident answer from the
        # wrong dataset.
        for fallback_name in ["spider_profile", "bh_blindhut", "pc_perceptual", "sd_driver", "sr_attribute", "rw_relweight"]:
            if fallback_name == module_name:
                continue
            if not _has_explicit_entity(fallback_name, question):
                continue
            fb = MODULES[fallback_name].answer(question, top_n=top_n)
            if fb.get("ok"):
                fb["suggestions"] = _followups(fallback_name, fb)
                return fb
        return result

    result["suggestions"] = _followups(module_name, result)
    return result


def answer_pinned(question: str, module_name: str, top_n: int = 8, context: Optional[dict] = None) -> dict:
    """Answer using ONLY the named module — no cross-dataset routing, no
    fallback to any other module, ever. This is what a per-CSV "tab"
    should call: once the person has told us which specialist they want,
    there's no ambiguity left to resolve, which makes this the most
    accurate mode available — it can't be hijacked by another dataset's
    fuzzy match the way general routing sometimes can be."""
    module = MODULES.get(module_name)
    if module is None:
        return {"ok": False, "error": "unknown_module"}

    result = module.answer(question, top_n=top_n)
    if result.get("ok"):
        result["suggestions"] = _followups(module_name, result)
        return result

    if not context or context.get("module") != module_name:
        return result
    hint = context_hint_text(context)
    if not hint:
        return result

    fb = module.answer(f"{question} {hint}".strip(), top_n=top_n)
    if fb.get("ok"):
        fb["suggestions"] = _followups(module_name, fb)
        fb["used_context"] = True
        fb["context_hint"] = hint
        return fb
    return result


def answer_with_context(question: str, top_n: int = 8, context: Optional[dict] = None) -> dict:
    """Like answer(), but if the question can't be resolved on its own — or
    only resolved via a shaky fuzzy guess on a "and X?" / "what about X?"
    style follow-up — retry it as a continuation of the session's previous
    turn, augmented with the entities we already know
    (benefit/product/platform/category).

    Tries two candidate modules, in order: whichever module the question's
    *own* keywords route to (e.g. "which attributes separate them?" still
    correctly targets spider_profile even mid a pc_perceptual conversation,
    because "separate" is a spider_profile signal), then the previous
    turn's module as a second attempt. This is what lets "what about the
    Fragrance stage?" work after a Natural Product question, and "which
    attributes separate them?" correctly switch modules mid-conversation
    instead of being trapped in whatever module the last turn used."""
    result = answer(question, top_n=top_n)

    shaky_continuation = (
        bool(context)
        and looks_like_continuation(question)
        and not any(text_contains(question, result.get(k)) for k in _PRIMARY_ENTITY_KEYS.get(result.get("module"), ()))
    )

    if result.get("ok") and not shaky_continuation:
        return result
    if not context:
        return result

    hint = context_hint_text(context)
    if not hint:
        return result

    candidate_modules = []
    routed_module = result.get("module")
    if routed_module in MODULES:
        candidate_modules.append(routed_module)
    prev_module = context.get("module")
    if prev_module in MODULES and prev_module not in candidate_modules:
        candidate_modules.append(prev_module)

    augmented = f"{question} {hint}".strip()
    for name in candidate_modules:
        fb = MODULES[name].answer(augmented, top_n=top_n)
        if fb.get("ok"):
            fb["suggestions"] = _followups(name, fb)
            fb["used_context"] = True
            fb["context_hint"] = hint
            return fb

    return result
