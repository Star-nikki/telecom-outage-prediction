"""
Orchestrator LLM -- same graph structure as orchestrator_v3.py, but the Root
Cause and Recommendation steps get LLM involvement WITH validation, not raw
trust. Rationale (after two rounds of prompt-only fixes still producing
unstable categories, invented trend numbers, invented tower IDs, and
occasional unrelated code/math generation): prompt instructions alone cannot
guarantee a small local model's output. So:

  ROOT CAUSE: category + baseline reasoning are computed DETERMINISTICALLY in
  Python (the same multi-signal scorer used in orchestrator_v3's rule-based
  path -- via _rule_based_root_cause). The LLM is asked ONLY to rephrase that
  baseline into more natural prose. Its rephrasing is validated (no invented
  numbers, no invented timing, no invented tower/category, no code) before
  use -- if it fails validation for any reason, the deterministic baseline
  text is used verbatim instead. This makes category and confidence fully
  stable (same input -> same output, always), while still getting LLM-quality
  prose whenever the model behaves.

  RECOMMENDATION: raw LLM output is parsed for clean numbered lines only;
  anything code-like, oversized, or mentioning a different tower ID is
  discarded; if nothing valid survives, a deterministic knowledge-base-
  grounded fallback (matching the same conditions as the SOPs) is used.

WHERE TO SAVE: agents/orchestrator_llm.py (overwrite)

USAGE:
    python -m agents.orchestrator_llm --tower T0000 --hours-before 3 --provider ollama
"""

import argparse
import os
import re
import sys

import pandas as pd
from langgraph.graph import StateGraph, END

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # agents/ itself, for
                                                                  # sibling imports below
from config import PREDICTION_HORIZONS  # noqa: E402
import decision_agent as da  # noqa: E402

from orchestrator_v3 import (  # noqa: E402
    AgentState, monitoring_agent, decision_agent_node, route_after_decision,
    knowledge_agent, historical_events_agent, report_agent, finish_agent, compute_priority,
    pick_demo_row, list_events_for_tower, _rule_based_root_cause,
    FAILURE_CATEGORIES, DATA_PATH,
)
from llm_providers import get_provider  # noqa: E402

DEBUG_PRINT_RAW = True

# ---------------------------------------------------------------------------
# Shared validation helpers
# ---------------------------------------------------------------------------
_TOWER_ID_PATTERN = re.compile(r"\bT\d{1,5}\b")
_CODE_PATTERN = re.compile(r"[a-zA-Z_]\w*\s*=\s*\[|def\s+\w+\(|import\s+\w+|```")
_NUM_PATTERN = re.compile(r"\d+\.?\d*")
_CONTINUATION_PHRASES = ["you're a", "you are a", "operations engineer", "telecom network",
                          "site s", "san marino", "battery 100%", "battery:"]
# Catches corrupted tokens like "3t0h24d" (digit-letter-digit mixed in one
# word) -- a real failure mode seen from small local models mid-generation.
_GARBLED_TOKEN = re.compile(r"\b\w*\d[a-zA-Z]+\d[a-zA-Z0-9]*\b")
_SHORT_SOP_CITATION = re.compile(r"\s*\[SOP-[A-Z0-9-]+\]\s*$", re.IGNORECASE)
_LONG_BRACKET_TAIL = re.compile(r"\s*\[[^\]]{15,}\]\s*$")


def _clean_citation_tail(text: str) -> str:
    """Keeps short '[SOP-XXX]' style citations as-is, but strips longer
    bracketed content (the model restating/duplicating SOP body text) --
    that duplication was tripping the length cap and merged-actions
    heuristic on otherwise perfectly valid, condition-preserving lines."""
    if _SHORT_SOP_CITATION.search(text):
        return text
    return _LONG_BRACKET_TAIL.sub("", text).strip()


def _looks_like_merged_actions(text: str) -> bool:
    """Heuristic: a long line joining ' and ' with a second conditional
    clause (comma or a second 'if') usually means two distinct SOP actions
    got fused into one numbered item instead of staying separate."""
    if len(text) > 120 and " and " in text.lower():
        clause_count = text.lower().count(",") + text.lower().count(" if ")
        if clause_count >= 1:
            return True
    return False


_ACTION_VERBS = (
    "dispatch", "inspect", "replace", "activate", "flag", "schedule",
    "cross-reference", "monitor", "check", "verify", "confirm", "notify",
    "deploy", "review", "log", "plan",
)
_ACTION_VERB_PATTERN = re.compile(
    r"\b(" + "|".join(v.replace("-", r"[\s-]") for v in _ACTION_VERBS) + r")\w*\b", re.IGNORECASE)

# Strips a leading section-title / label fragment like "Battery and grid
# voltage SOP:", "Standard procedure:", "Recommended action:", "General
# incident response guideline:" -- these describe the SOP entry itself, not
# an instruction, and would otherwise get picked up as bogus "action" text.
_TITLE_PREFIX_PATTERN = re.compile(
    r"^[^:]{0,70}\b(?:sop|procedure|guideline|recommended actions?)\b[^:]{0,10}:\s*", re.IGNORECASE)

# "No immediate dispatch needed unless X" / "No dispatch needed unless X"
# -> a genuine (negative) instruction that the old code never picked up at
# all, since it has no "if".
_NO_ACTION_NEEDED_PATTERN = re.compile(
    r"^no\s+(?:immediate\s+)?([a-z][a-z\s]{0,30}?)\s+(?:is\s+)?needed\s+unless\s+(.+)$", re.IGNORECASE)


def _split_and_actions(clause: str) -> list:
    """Splits a clause on ' and ' only where the following word is itself a
    known action verb -- so 'dispatch X and replace Y' becomes two separate
    actions, while an incidental ' and ' inside descriptive text (e.g.
    'packet loss and latency') is left untouched."""
    verb_alt = "|".join(v.replace("-", r"[\s-]") for v in _ACTION_VERBS)
    parts = re.split(r"\s+and\s+(?=(?:" + verb_alt + r")\b)", clause, flags=re.IGNORECASE)
    return [p.strip() for p in parts if p.strip()]


def _extract_sop_conditions(knowledge_context: list) -> list:
    """Extracts actionable instructions from the retrieved SOP text -- both
    conditional ("do X if Y") and unconditional ("do X") -- as action/
    condition pairs used to validate LLM output and, if needed,
    deterministically fill in missing recommendation lines.

    An earlier version treated "everything before the word 'if'" as the
    action. That breaks on real SOP prose in several ways: it can pick up a
    section title ("Battery and grid voltage SOP:"), a diagnostic clause
    with no instruction in it ("...this indicates imminent backup power
    exhaustion"), a bare status statement ("Priority escalates to urgent"),
    or a mid-sentence fragment starting right after a comma ("but check for
    physical antenna/feeder line damage"). It also only ever looked at
    clauses containing "if", so genuinely unconditional actions (e.g.
    "dispatch a field technician... and replace the backup battery") were
    silently dropped from fallback coverage -- not malformed, just missing.

    This version instead anchors on a small, curated action-verb vocabulary
    (dispatch/inspect/replace/activate/flag/schedule/monitor/check/etc): a
    clause only becomes a candidate action if an action verb is found in it
    at all (after stripping a leading section-title/label). Everything
    before that verb (descriptive lead-in, "but", title text) is discarded;
    everything from the verb onward -- up to the first "if"/";"/" -- "/
    " but " boundary, or the clause's end -- becomes the action. A clause
    that never reaches an action verb (a title, a diagnostic sentence, a
    status statement like "Priority escalates...") is skipped rather than
    guessed at.
    """
    conditions = []
    for doc in knowledge_context or []:
        text = doc.get("text", "")
        for sentence in re.split(r"\.\s+", text):
            sentence = _TITLE_PREFIX_PATTERN.sub("", sentence.strip()).strip().rstrip(".")
            if not sentence:
                continue

            no_action_match = _NO_ACTION_NEEDED_PATTERN.match(sentence)
            if no_action_match:
                action_noun, condition_tail = no_action_match.groups()
                conditions.append({
                    "action_key": f"do not {action_noun.strip()}",
                    "condition": f"unless {condition_tail.strip().rstrip(' ,;.')}",
                })
                continue

            for clause in _split_and_actions(sentence):
                verb_match = _ACTION_VERB_PATTERN.search(clause)
                if not verb_match:
                    continue  # no instruction in this clause -- descriptive/diagnostic text, skip
                action_clause = clause[verb_match.start():]

                if_match = re.search(r"\bif\b", action_clause, re.IGNORECASE)
                boundary = if_match.start() if if_match else len(action_clause)
                for stop_phrase in (" -- ", " but "):
                    idx = action_clause.lower().find(stop_phrase)
                    if idx != -1:
                        boundary = min(boundary, idx)

                action_part = action_clause[:boundary].strip().rstrip(" ,;.")
                if not action_part:
                    continue
                if if_match and if_match.start() == boundary:
                    condition_part = "if " + action_clause[if_match.end():].strip().rstrip(" ,;.")
                else:
                    condition_part = ""
                conditions.append({"action_key": action_part.lower(), "condition": condition_part.lower()})
    return conditions


def _sop_action_covered(cond: dict, items: list) -> bool:
    """True if some already-validated item sufficiently overlaps with this
    SOP action's keywords. Since items only ever contain text that already
    passed _violates_sop_condition, keyword overlap alone is enough to know
    the condition is correct too -- no need to re-check the condition here."""
    action_words = [w for w in re.findall(r"[a-z]+", cond["action_key"]) if len(w) > 4]
    if not action_words:
        return True
    for item in items:
        lower = item.lower()
        hits = sum(1 for w in action_words if w in lower)
        if hits / len(action_words) >= 0.5:
            return True
    return False


def _build_deterministic_action_text(cond: dict) -> str:
    """Builds a guaranteed-grounded recommendation line directly from the
    SOP's own action+condition text -- used to fill in any SOP action the
    LLM failed to produce validly, so nothing goes missing. Produces a
    single clean, complete sentence whether the action is conditional
    ("Replace the battery if X.") or unconditional ("Replace the battery.")
    -- previously this always appended the condition text even when it was
    empty, which is now handled explicitly rather than left to string
    concatenation."""
    action_text = cond["action_key"].strip().rstrip(".")
    if action_text:
        action_text = action_text[0].upper() + action_text[1:]
    if cond["condition"]:
        return f"{action_text} {cond['condition']}."
    return f"{action_text}."


def _violates_sop_condition(text: str, sop_conditions: list) -> bool:
    """Rejects a recommendation line that appears to reference an SOP
    action which has a specific condition, but either drops the condition
    entirely or restates it with different wording/numbers than the SOP's
    exact text. Preventing paraphrase here matters -- a condition like
    "if congestion recurs more than twice in 30 days" silently rewritten as
    "within a short window" changes the actual operational meaning."""
    lower = text.lower()
    for cond in sop_conditions:
        action_words = [w for w in re.findall(r"[a-z]+", cond["action_key"]) if len(w) > 4]
        if not action_words:
            continue
        keyword_hits = sum(1 for w in action_words if w in lower)
        if keyword_hits / len(action_words) >= 0.5:
            if cond["condition"] not in lower:
                return True
    return False

_FORBIDDEN_ROOT_CAUSE_SNIPPETS = [
    "hour", "minute", "snapshot", "lead time", "before the outage", "window",
    "timestamp", "san marino", "site s",
]

# Phrases claiming utilization is currently HIGH -- forbidden unless the
# actual level is genuinely high. Split from the trend-claim list below
# because a moderate (not low, not high) value like 56% is just as
# misrepresented by "high utilization" as a low one is, regardless of
# which direction it's currently trending.
_HIGH_UTIL_LEVEL_PHRASES = [
    "high utilization", "heavy traffic", "traffic overload", "overloaded",
    "network overload", "high traffic volume", "capacity exceeded",
    "congested due to volume", "heavy load", "high demand",
]

# Phrases claiming utilization is currently RISING/growing -- forbidden
# unless the actual short-term trend is genuinely upward (a real case: LLM
# said "traffic jam building up" when utilization=25% and 24h trend=-0.7pts).
_RISING_UTIL_TREND_PHRASES = [
    "traffic jam", "building up", "increase in demand", "rising traffic",
    "climbing", "surge", "spike in traffic", "growing demand", "jam building",
    "traffic spike",
]

# congestion_collapse is structurally a fast event in this simulator (short
# lead time) -- describing it as a slow multi-day buildup is exactly the
# same category of error as claiming a low reading is high, just phrased as
# a false trend instead of a false level.
_FALSE_GRADUAL_PHRASES = [
    "gradual creep", "slow build", "slowly building", "over several days",
    "long-term trend", "slowly accumulating", "gradual increase over",
    "built up over time", "steadily over days",
]

# capacity_creep is structurally slow (24-72h lead time) -- describing it as
# sudden/instantaneous is the mirror-image error.
_FALSE_ACUTE_PHRASES = [
    "sudden spike", "abruptly", "instantaneous", "rapid onset",
    "out of nowhere", "spiked suddenly", "without warning", "all at once",
]

# Generic connector/report vocabulary that's always safe to use regardless
# of what's in the evidence/baseline -- used by the grounding check below so
# ordinary paraphrase words aren't mistaken for invented claims.
_SAFE_REPORT_VOCAB = {
    "this", "that", "with", "from", "into", "over", "than", "then", "have",
    "been", "were", "which", "while", "also", "here", "there", "these",
    "those", "such", "more", "most", "less", "least", "very", "some",
    "consistent", "indicates", "indicating", "indicate", "suggests",
    "suggesting", "suggest", "reflects", "reflecting", "reflect", "pattern",
    "evidence", "shows", "showing", "show", "observed", "observation",
    "rather", "simple", "explain", "explains", "explained", "explanation",
    "network", "tower", "outage", "root", "cause", "caused", "causing",
    "category", "issue", "problem", "condition", "level", "levels",
    "elevated", "degraded", "degradation", "signal", "signals", "packet",
    "loss", "latency", "utilization", "trend", "trends", "sustained",
    "gradual", "gradually", "acute", "onset", "baseline", "creep",
    "congestion", "collapse", "capacity", "hardware", "power", "weather",
    "storm", "rainfall", "battery", "voltage", "combined", "alongside",
    "still", "already", "close", "within", "without", "before", "after",
    "flat", "declining", "decreasing", "increasing", "falling", "dropping",
    # generic connector/reporting words safe regardless of source content
    "since", "throughout", "stayed", "staying", "remained", "remaining",
    "despite", "instead", "resulting", "resulted", "associated", "matches",
    "matching", "pointing", "points", "direct", "directly", "primarily",
    "mainly", "largely", "overall", "during", "across", "specifically",
    "particularly", "notably", "clear", "clearly", "strongly", "strong",
    "driven", "drive", "spike", "spiked", "spiking", "reflecting", "report",
    "operations", "attributed", "attribute", "attributable", "emerged",
    "emerging", "appears", "appear", "appearing", "behavior", "typical",
    "typically", "characteristic", "scenario", "situation", "circumstance",
    "single", "period", "recent", "recently", "period", "reading",
    "readings", "value", "values", "measurement", "measurements", "point",
    "rapid", "rapidly", "quick", "quickly", "escalated",
    "escalating", "escalation", "worsened", "worsening", "worse",
    # generic magnitude/comparison descriptors -- safe regardless of source
    # content since they're common English adjectives, not domain claims
    "heavy", "volume", "demand", "traffic", "light", "minimal", "moderate",
    "significant", "significantly", "substantial", "fresh", "freshly",
    "high", "higher", "highest", "growth", "growing", "earlier", "later",
    "began", "begun", "start", "started", "building", "builds", "expand",
    "expanding", "expansion", "persistent", "persisting", "persist",
    "confirmed", "confirm", "confirming", "physical", "damage", "damaged",
    "both", "either", "neither", "each", "every", "all", "same", "similar",
    "along", "alongside", "present", "presence", "absent", "absence",
}

KNOWLEDGE_FALLBACK_RECOMMENDATIONS = {
    "hardware_power_failure": [
        "Inspect the power supply unit and backup battery.",
        "Replace the backup battery within 12 hours if degradation is confirmed (per SOP-PWR-01).",
    ],
    "congestion_collapse": [
        "Activate traffic load-balancing to neighboring towers if available (per SOP-CONG-01).",
        "Flag the site for capacity upgrade evaluation if congestion recurs more than twice in 30 days.",
    ],
    "weather_induced_degradation": [
        "Monitor signal recovery once the storm passes; no dispatch needed unless degradation persists beyond 2 hours after clearing.",
        "Inspect antenna/feeder lines for physical damage if degradation does not resolve post-storm.",
    ],
    "capacity_creep": [
        "Schedule a capacity planning review rather than an emergency dispatch.",
        "Cross-reference the mobility events log before attributing the trend to a fault (per SOP-GEN-01).",
    ],
}


def _forbidden_phrase_present(lower_text: str, phrases: list) -> bool:
    """True if any forbidden phrase appears WITHOUT a nearby negation cue
    just before it. Plain substring matching can't tell "heavy traffic
    caused this" (a false claim) apart from "rather than heavy traffic"
    (correctly ruling it out) -- the second is exactly the kind of
    faithful, evidence-consistent phrasing we want to allow, so a phrase
    match preceded closely by a negation cue is not treated as a
    violation."""
    negation_cues = ("not ", "n't", "rather than", "without", "no ",
                      "instead of", "isn't", "wasn't", "unlike", "far from")
    for phrase in phrases:
        idx = lower_text.find(phrase)
        if idx == -1:
            continue
        window = lower_text[max(0, idx - 30):idx]
        if any(cue in window for cue in negation_cues):
            continue  # negated -- this is a correct, evidence-consistent statement
        return True
    return False


def _is_grammatically_valid(text: str) -> bool:
    """Cheap, dictionary-free sanity checks for the failure mode seen with
    small/local models: truncated or corrupted generation (e.g. "The network
    is experienpending..."). None of this requires a dictionary -- it just
    checks that the output looks like a complete, well-formed sentence
    rather than a cut-off or garbled one."""
    stripped = text.strip()
    if not stripped:
        return False
    # Truncated mid-generation: trailing ellipsis, or no terminal punctuation.
    if stripped.endswith("...") or stripped.endswith(".."):
        return False
    if stripped[-1] not in ".!?":
        return False
    words = re.findall(r"[A-Za-z']+", stripped)
    if len(words) < 5 or len(words) > 90:
        return False
    # Any single "word" longer than this is almost certainly two words
    # fused together by a corrupted generation, not real English prose.
    if any(len(w) > 20 for w in words):
        return False
    # Stutter/repeat artifact ("the the", "is is").
    lowered_words = [w.lower() for w in words]
    if any(a == b for a, b in zip(lowered_words, lowered_words[1:])):
        return False
    if _GARBLED_TOKEN.search(text):
        return False
    return True


def _has_ungrounded_claims(text: str, evidence_block: str, baseline: str, category: str) -> bool:
    """Flags reasoning that introduces substantive content not traceable to
    the evidence/baseline it was supposed to be paraphrasing -- i.e. the LLM
    interpreting/inventing rather than faithfully restating. Content words
    (len>=4) are compared against the source text plus a fixed allowlist of
    generic report vocabulary, using a light prefix-stem match (5 chars) so
    ordinary morphological variants (increase/increases/increasing) aren't
    mistaken for new claims.

    Threshold is zero-tolerance (any novel word rejects): a real observed
    failure produced "...never seen outside hardware_power diagonal" --
    "diagonal" is a single fabricated word, and a >5-word budget let it
    straight through. Priority 2's requirement is that reasoning be grounded
    ENTIRELY in the evidence/baseline, not "mostly" -- one hallucinated word
    is exactly the failure mode this check exists to catch, so it can't have
    a multi-word allowance. This means legitimate rephrasing now depends
    more on _SAFE_REPORT_VOCAB being reasonably generous; the tradeoff is
    intentional -- rejecting a decent paraphrase just falls back to the
    (always-grounded) deterministic baseline, which is the designed-for
    outcome, not a failure."""
    source_words = re.findall(r"[a-z']{3,}", (evidence_block + " " + baseline + " " + category).lower())
    source_stems = {w[:5] for w in source_words}
    safe_stems = {w[:5] for w in _SAFE_REPORT_VOCAB} | _SAFE_REPORT_VOCAB
    text_words = re.findall(r"[a-z']{4,}", text.lower())
    novel = [w for w in text_words if w[:5] not in source_stems and w[:5] not in safe_stems]
    return len(novel) > 0


def _root_cause_text_is_safe(text: str, evidence_block: str, baseline: str, category: str, row: dict) -> bool:
    """Rejects the LLM's rephrasing if it invents numbers, mentions timing,
    mentions a different category, mentions a tower ID, looks like code,
    isn't a well-formed/complete sentence, introduces claims not grounded in
    the evidence/baseline, or contradicts the actual telemetry -- e.g.
    describing low/flat/declining utilization as high or rising (or the
    reverse), or describing a structurally-fast category (congestion_collapse)
    as a gradual buildup, or a structurally-slow one (capacity_creep) as
    sudden."""
    if not text or len(text) < 10 or len(text) > 500:
        return False
    if not _is_grammatically_valid(text):
        return False
    lower = text.lower()
    if any(bad in lower for bad in _FORBIDDEN_ROOT_CAUSE_SNIPPETS):
        return False
    if _TOWER_ID_PATTERN.search(text):
        return False
    if _CODE_PATTERN.search(text):
        return False
    other_categories = [c for c in FAILURE_CATEGORIES if c != category]
    if any(c in lower or c.replace("_", " ") in lower for c in other_categories):
        return False

    # Level/trend-direction consistency: driven entirely by this row's own
    # telemetry, not by category or tower ID, so it generalizes to any tower.
    util = row.get("utilization_pct", 0)
    util_trend_1h = row.get("utilization_pct_delta_1h") or 0
    util_trend_24h = row.get("utilization_pct_delta_24h") or 0
    if util <= 70:  # not actually high -- "high utilization"/"overloaded" etc would misstate the level
        if _forbidden_phrase_present(lower, _HIGH_UTIL_LEVEL_PHRASES):
            return False
    if util_trend_1h <= 2 and util_trend_24h <= 2:  # not actually rising
        if _forbidden_phrase_present(lower, _RISING_UTIL_TREND_PHRASES):
            return False

    # Category-shape consistency: congestion_collapse is fast, capacity_creep
    # is slow -- these are structural properties of the category itself, not
    # something that varies by tower, so the check applies uniformly.
    if category == "congestion_collapse" and _forbidden_phrase_present(lower, _FALSE_GRADUAL_PHRASES):
        return False
    if category == "capacity_creep" and _forbidden_phrase_present(lower, _FALSE_ACUTE_PHRASES):
        return False

    if _has_ungrounded_claims(text, evidence_block, baseline, category):
        return False

    allowed_source = evidence_block + " " + baseline
    for num in _NUM_PATTERN.findall(text):
        if num not in allowed_source:
            return False
    return True


def _category_action_keywords() -> dict:
    """Builds a per-category vocabulary of distinctive action words directly
    from KNOWLEDGE_FALLBACK_RECOMMENDATIONS -- the same text the code already
    trusts as grounded, category-correct fallback content -- rather than a
    separately maintained list that could drift out of sync with it."""
    keywords = {}
    for cat, lines in KNOWLEDGE_FALLBACK_RECOMMENDATIONS.items():
        words = set()
        for line in lines:
            words |= {w for w in re.findall(r"[a-z]+", line.lower()) if len(w) >= 6}
        keywords[cat] = words
    return keywords


_CATEGORY_ACTION_KEYWORDS = _category_action_keywords()

# Generic operational/report words safe to use in a recommendation line
# regardless of whether they appear verbatim in the retrieved SOP text --
# mirrors _SAFE_REPORT_VOCAB's role for root-cause reasoning, but scoped to
# recommendation phrasing (which legitimately needs to state times/units
# the way an SOP typically already does: "within X hours", "the site").
_REC_SAFE_VOCAB = {
    "this", "that", "with", "from", "into", "over", "than", "then", "have",
    "been", "were", "which", "while", "also", "here", "there", "these",
    "those", "such", "more", "most", "less", "least", "very", "some",
    "action", "actions", "recommend", "recommended", "procedure",
    "standard", "field", "team", "unit", "units", "site", "tower",
    "network", "issue", "condition", "conditions", "level", "levels",
    "within", "without", "before", "after", "until", "once", "since",
    "current", "currently", "immediate", "immediately", "further",
    "additional", "necessary", "required", "needed", "confirmed", "confirm",
    "confirming", "persist", "persists", "persisting", "persistent",
    "resolve", "resolved", "resolving", "clear", "clears", "cleared",
    "monitor", "monitoring", "monitored", "quality", "remain", "remains",
    "remaining", "expect", "expects", "expected", "wait", "waiting",
}


def _recommendation_is_grounded(text: str, sop_text: str) -> bool:
    """True if the line's substantive content is traceable to the retrieved
    SOP text as a whole. _violates_sop_condition (above) only fires once a
    line already overlaps a KNOWN action's keywords by >=50% -- it can't
    catch a line that invents an entirely new action, threshold, or
    condition with no such overlap to check against (e.g. "consider
    installing protective measures" or "prioritize repair to restore
    service" -- neither shares enough keywords with any real SOP action to
    trip that check, but neither appears anywhere in the SOP either). This
    check is content-source-driven rather than category-driven: it works
    off whatever knowledge_context was actually retrieved for this event, so
    it applies uniformly to every category without hardcoding category- or
    tower-specific phrases."""
    source_words = re.findall(r"[a-z']{3,}", sop_text.lower())
    source_stems = {w[:5] for w in source_words}
    safe_stems = {w[:5] for w in _REC_SAFE_VOCAB} | _REC_SAFE_VOCAB
    text_words = re.findall(r"[a-z']{4,}", text.lower())
    novel = [w for w in text_words if w[:5] not in source_stems and w[:5] not in safe_stems]
    return len(novel) <= 2


def _mentions_other_category(text: str, category: str) -> bool:
    """Rejects a recommendation line that names a different failure category
    outright, or whose action vocabulary matches another category's SOP
    actions more than the current one's -- e.g. an "inspect antenna/feeder"
    (weather) line leaking into a capacity_creep recommendation. Uses
    keywords unique to the OTHER category (not shared with the current one)
    and requires 2+ hits, so incidental overlap on a generic word doesn't
    cause a false reject."""
    lower = text.lower()
    if any(c in lower or c.replace("_", " ") in lower
           for c in FAILURE_CATEGORIES if c != category):
        return True
    own_words = _CATEGORY_ACTION_KEYWORDS.get(category, set())
    text_words = set(re.findall(r"[a-z]+", lower))
    for other_cat, other_words in _CATEGORY_ACTION_KEYWORDS.items():
        if other_cat == category:
            continue
        signal_words = other_words - own_words
        if len(text_words & signal_words) >= 2:
            return True
    return False


def _extract_valid_recommendation_lines(raw: str, tower_id: str, sop_conditions: list,
                                          category: str = None, sop_text: str = "",
                                          max_items: int = 4) -> list:
    """Parses only clean numbered lines; discards anything code-like,
    oversized, mentioning a wrong tower ID, a restarted/duplicated
    numbering sequence, mentioning/matching another failure category
    (see _mentions_other_category), one that drops/paraphrases an SOP's
    stated condition (see _violates_sop_condition), OR one whose content
    isn't traceable to the retrieved SOP text at all (see
    _recommendation_is_grounded)."""
    if not raw:
        return []
    numbered_line = re.compile(r"^\s*(\d+)[.)]\s+(.+)$")
    items, seen_numbers = [], set()
    for line in raw.strip().split("\n"):
        m = numbered_line.match(line.strip())
        if not m:
            continue
        num, text = m.group(1), m.group(2).strip()
        if num in seen_numbers:
            break  # duplicate/restarted numbering -- stop, likely a new unrelated block began
        seen_numbers.add(num)
        text = _clean_citation_tail(text)  # strip bloated citation tails before judging length/content
        if len(text) > 180 or len(text) < 5:
            continue
        if _CODE_PATTERN.search(text):
            continue
        if any(p in text.lower() for p in _CONTINUATION_PHRASES):
            continue
        if _GARBLED_TOKEN.search(text):
            continue  # corrupted mixed digit/letter token (e.g. "3t0h24d")
        if _looks_like_merged_actions(text):
            continue  # two distinct SOP actions fused into one line -- reject, want them separate
        if _violates_sop_condition(text, sop_conditions):
            continue  # drops or paraphrases an SOP's stated condition -- reject
        if category and _mentions_other_category(text, category):
            continue  # names a different category, or matches its SOP-action vocabulary -- reject
        if sop_text and not _recommendation_is_grounded(text, sop_text):
            continue  # introduces content (action/threshold/condition/equipment) not traceable to the retrieved SOP -- reject
        wrong_tower = _TOWER_ID_PATTERN.findall(text)
        if wrong_tower and tower_id not in wrong_tower:
            continue  # mentions a tower ID that isn't the actual one -- discard
        items.append(text)
        if len(items) >= max_items:
            break
    return items


# ---------------------------------------------------------------------------
# Root Cause Agent -- deterministic category+baseline, validated LLM polish
# ---------------------------------------------------------------------------
def _util_direction_clause(row: dict) -> str:
    """Builds a short, factual, always-true clause stating utilization's
    actual level and short-term direction from this row's own telemetry.
    Handed to the LLM as an explicit anchor so it has no reason to guess
    (and invent) the direction -- generalizes to any tower/row since it's
    computed from the row, not hardcoded."""
    util = row.get("utilization_pct", 0)
    trend_1h = row.get("utilization_pct_delta_1h") or 0
    level = "low" if util <= 40 else ("moderate" if util <= 70 else "high")
    if trend_1h > 2:
        direction = "rising"
    elif trend_1h < -2:
        direction = "falling"
    else:
        direction = "flat"
    return f"Utilization is currently {level} ({util:.0f}%) and {direction} in the short term."


def root_cause_agent_llm(state: AgentState) -> dict:
    evidence = state["evidence"]
    evidence_block = "\n".join(f"- {k}: {v}" for k, v in evidence.items())

    # Category and a grounded baseline explanation are computed
    # DETERMINISTICALLY -- this is what makes category/confidence stable
    # run to run, since it no longer depends on the LLM at all.
    category, baseline_reasoning = _rule_based_root_cause(state["row"])
    util_clause = _util_direction_clause(state["row"])

    prompt = f"""Evidence:
{evidence_block}

The determined root cause category is: {category}
Baseline explanation: {baseline_reasoning}
Fact you must not contradict: {util_clause}

Rewrite the baseline explanation in 1-2 natural, grammatically complete sentences for an operations report.
Rules:
- Faithfully restate the Baseline explanation above -- do not add any interpretation, mechanism, or cause that isn't already stated there.
- Do not introduce any number, percentage, or measurement not already present in the Evidence or Baseline explanation above.
- Do not describe utilization as high, heavy, or rising if the Fact above says it is low/moderate and flat/falling, and do not describe it as low or falling if the Fact says it is high and rising.
- Do not mention time, timestamps, hours, minutes, "before the outage", "snapshot", or any temporal reference.
- Do not mention any category other than {category}.
- Do not generate code, mathematics, tower IDs, site names, or anything unrelated.
- End with a single period. Do not truncate mid-sentence or use "...".
- Output ONLY the rewritten explanation text -- no labels, no lists, no extra commentary."""

    provider = get_provider(state["provider_name"])
    raw = (provider.generate(prompt) or "").strip()

    if DEBUG_PRINT_RAW:
        print(f"\n===== RAW ROOT-CAUSE REPHRASE (category fixed at '{category}') =====")
        print(raw)
        print("=====================================================\n")

    if _root_cause_text_is_safe(raw, evidence_block, baseline_reasoning, category, state["row"]):
        reasoning = raw
    else:
        if DEBUG_PRINT_RAW:
            print("REJECTED LLM rephrasing (failed grounding/grammar/consistency check) -- "
                  "using deterministic baseline instead.\n")
        reasoning = baseline_reasoning

    return {"root_cause_category": category, "root_cause_reasoning": reasoning}


# ---------------------------------------------------------------------------
# Recommendation Agent -- strict parsing + validation + grounded fallback
# ---------------------------------------------------------------------------
def recommendation_agent_llm(state: AgentState) -> dict:
    row = state["row"]
    tower_id = state["tower_id"]
    knowledge_text = "\n".join(f"- [{d['id']}] {d['text']}" for d in (state["knowledge_context"] or []))
    historical_text = (state.get("historical_context") or {}).get(
        "summary", "No historical event data available.")

    prompt = f"""You are a telecom network operations engineer.

Root cause: {state['root_cause_category']} -- {state['root_cause_reasoning']}

Relevant maintenance knowledge:
{knowledge_text}

Historical pattern at this tower (background context only -- do not cite this as an SOP
and do not turn it into a numbered action item):
{historical_text}

Generate 2-4 concise, actionable recommendations for the operations team.
Return ONLY a numbered list, nothing else.

Each item must:
- Be exactly ONE concise operational action, directly supported by the knowledge base above.
- NEVER combine two actions into one item using "and" -- if an SOP describes two actions, put them on two separate numbered lines.
- Preserve any condition stated in the knowledge base exactly (e.g. "if available", "if it recurs more than twice in 30 days") -- copy numeric/time conditions verbatim, do not paraphrase or reword them.
- Contain no telemetry values, timestamps, tower IDs, confidence numbers, code, or math.
- Never state a failure mode (e.g. "grid failure") unless it is the stated root cause above."""

    provider = get_provider(state["provider_name"])
    raw = (provider.generate(prompt) or "").strip()

    if DEBUG_PRINT_RAW:
        print("\n===== RAW RECOMMENDATION OUTPUT (attempt 1) =====")
        print(repr(raw))
        print("======================================\n")

    sop_conditions = _extract_sop_conditions(state["knowledge_context"])
    category = state["root_cause_category"]
    items = _extract_valid_recommendation_lines(raw, tower_id, sop_conditions,
                                                  category=category, sop_text=knowledge_text)

    if not items:
        # one retry before giving up on the LLM entirely
        raw_retry = (provider.generate(prompt) or "").strip()
        if DEBUG_PRINT_RAW:
            print("\n===== RAW RECOMMENDATION OUTPUT (attempt 2 / retry) =====")
            print(repr(raw_retry))
            print("======================================\n")
        items = _extract_valid_recommendation_lines(raw_retry, tower_id, sop_conditions,
                                                      category=category, sop_text=knowledge_text)

    # Fill gaps: for any SOP action neither attempt produced validly, add a
    # deterministic line built directly from the SOP text (guaranteed to
    # preserve the exact condition) -- this replaces "all or nothing"
    # fallback with "cover what's missing, keep what's already valid."
    for cond in sop_conditions:
        if not _sop_action_covered(cond, items):
            if DEBUG_PRINT_RAW:
                print(f"SOP action not validly produced by LLM -- filling deterministically: {cond['action_key']}\n")
            items.append(_build_deterministic_action_text(cond))

    if not items:
        if DEBUG_PRINT_RAW:
            print("No valid recommendation lines and no SOP conditions to fall back on -- using generic fallback.\n")
        items = KNOWLEDGE_FALLBACK_RECOMMENDATIONS.get(
            state["root_cause_category"],
            ["Review root cause and knowledge base references above for manual action."],
        )

    items = items[:4]  # respect the 2-4 item cap even after gap-filling
    recommendation = "\n".join(f"{i+1}. {text}" for i, text in enumerate(items))
    return {"recommendation": recommendation}


# ---------------------------------------------------------------------------
def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("monitoring", monitoring_agent)
    graph.add_node("decision", decision_agent_node)
    graph.add_node("root_cause", root_cause_agent_llm)
    graph.add_node("knowledge", knowledge_agent)
    graph.add_node("historical_events", historical_events_agent)
    graph.add_node("recommendation", recommendation_agent_llm)
    graph.add_node("report", report_agent)
    graph.add_node("finish", finish_agent)

    graph.set_entry_point("monitoring")
    graph.add_edge("monitoring", "decision")

    graph.add_conditional_edges(
        "decision", route_after_decision,
        {"root_cause": "root_cause", "finish": "finish"},
    )

    # Agentic RAG, same parallel fan-out/fan-in as orchestrator_v3.py:
    # Knowledge Agent and Historical Events Agent both run off Root Cause,
    # in parallel, then feed into Recommendation.
    graph.add_edge("root_cause", "knowledge")
    graph.add_edge("root_cause", "historical_events")
    graph.add_edge("knowledge", "recommendation")
    graph.add_edge("historical_events", "recommendation")
    graph.add_edge("recommendation", "report")
    graph.add_edge("report", END)
    graph.add_edge("finish", END)

    return graph.compile()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tower", default=None)
    parser.add_argument("--backend", default="lightgbm", choices=["lightgbm", "xgboost", "random_forest"])
    parser.add_argument("--provider", default="ollama",
                         choices=["phi4_local", "anthropic", "openai", "ollama"])
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--primary-horizon", default="3h",
                         choices=[label for label, _ in PREDICTION_HORIZONS])
    parser.add_argument("--hours-before", type=float, default=3.0)
    parser.add_argument("--failure-type", default=None, choices=FAILURE_CATEGORIES,
                         help="Force selection of a specific event type for this tower "
                              "(reuses pick_demo_row's existing failure_type_filter) -- "
                              "use this to test hardware_power_failure / "
                              "weather_induced_degradation / congestion_collapse / "
                              "capacity_creep specifically instead of getting whichever "
                              "event happens to be first for the tower.")
    parser.add_argument("--list-events", action="store_true")
    args = parser.parse_args()

    if args.list_events:
        list_events_for_tower(args.tower)
        return

    print(f"Loading {DATA_PATH} ...")
    df = pd.read_parquet(DATA_PATH)
    tower_id = args.tower or df["tower_id"].iloc[0]

    tdf = df[df["tower_id"] == tower_id].sort_values("timestamp")
    tdf = da.add_trend_features(tdf)
    row_series = pick_demo_row(tdf, tower_id, hours_before_outage=args.hours_before,
                                failure_type_filter=args.failure_type)
    row = row_series.to_dict()
    row["timestamp"] = str(row["timestamp"])

    initial_state: AgentState = {
        "tower_id": tower_id,
        "row": row,
        "evidence": {},
        "predictions": {},
        "primary_probability": 0.0,
        "primary_confidence": 0.0,
        "threshold": args.threshold,
        "root_cause_category": None,
        "root_cause_reasoning": None,
        "knowledge_context": None,
        "recommendation": None,
        "final_report": None,
        "provider_name": args.provider,
        "backend": args.backend,
        "primary_horizon": args.primary_horizon,
    }

    print(f"Running LLM-based orchestration for tower {tower_id} "
          f"(backend={args.backend}, provider={args.provider}, threshold={args.threshold})...\n")
    app = build_graph()
    final_state = app.invoke(initial_state, config={"recursion_limit": 50})

    print("=" * 70)
    print("FINAL REPORT")
    print("=" * 70)
    print(final_state["final_report"])


if __name__ == "__main__":
    main()
