"""
Advanced RAG pipeline -- upgrades the simple TF-IDF-only retrieval in
knowledge_base.py with four techniques:

  1. HYBRID RETRIEVAL: combines BM25 (precise keyword/term-frequency
     matching) with TF-IDF cosine similarity (softer term-overlap matching)
     as two independent scoring signals, then blends them. This is a real
     hybrid-retrieval pattern, just without a heavy embedding model --
     deliberate, given this needs to run on an 8GB CPU-only machine that's
     already running Ollama. Swap the TF-IDF leg for sentence-transformer
     embeddings later if you get access to more RAM/GPU; the retrieve()
     interface below won't need to change.

  2. CONTEXTUAL RAG: the retrieval query is built from live state (root
     cause category, key telemetry values, tower context), not just a bare
     category name -- so retrieval reflects the actual situation, not a
     generic keyword.

  3. PARENT-CHILD RETRIEVAL: each SOP is split into sentence-level "child"
     chunks for matching (finer-grained relevance), but the FULL parent SOP
     text is what's returned to the Recommendation Agent -- so it always
     sees complete operational context, never an isolated fragment.

  4. RERANKING: candidates are rescored using a weighted combination of
     BM25 + TF-IDF + contextual keyword overlap (e.g. boosting SOPs whose
     text matches specific telemetry terms mentioned in the query) -- a
     lightweight stand-in for a cross-encoder reranker, same reasoning as
     above (no heavy model download required).

Install: pip install rank_bm25 scikit-learn --break-system-packages
"""

import re
import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# PARENT documents (same SOP corpus as knowledge_base.py)
# ---------------------------------------------------------------------------
PARENT_DOCUMENTS = [
    {
        "id": "SOP-PWR-01",
        "failure_type": "hardware_power_failure",
        "text": "Battery and grid voltage SOP: if battery level falls below 50% while grid "
                "voltage is below 210V, this indicates imminent backup power exhaustion. "
                "Standard procedure: dispatch a field technician to inspect the power supply "
                "unit and replace the backup battery within 12 hours of alert. Priority escalates "
                "to urgent if battery falls below 20%.",
    },
    {
        "id": "SOP-PWR-02",
        "failure_type": "hardware_power_failure",
        "text": "Historical maintenance data shows most hardware power failures at macro towers "
                "are caused by battery cell degradation after 18-24 months of service, or grid "
                "supply faults during voltage sag events. Recommended action: check battery "
                "manufacture date against the 24-month replacement cycle before dispatching.",
    },
    {
        "id": "SOP-CONG-01",
        "failure_type": "congestion_collapse",
        "text": "Congestion SOP: utilization sustained above 80% with rising packet loss and "
                "latency indicates capacity exhaustion. Recommended action: activate traffic "
                "load-balancing to neighboring towers if available, and flag the site for capacity "
                "upgrade evaluation if congestion recurs more than twice in 30 days.",
    },
    {
        "id": "SOP-WTHR-01",
        "failure_type": "weather_induced_degradation",
        "text": "Weather-induced degradation SOP: signal strength (RSRP) dropping in correlation "
                "with active storm/rainfall is typically self-resolving once the storm passes, but "
                "check for physical antenna/feeder line damage if degradation persists more than "
                "2 hours after the storm clears. No immediate dispatch needed unless persistence "
                "is confirmed.",
    },
    {
        "id": "SOP-CAP-01",
        "failure_type": "capacity_creep",
        "text": "Slow capacity creep SOP: gradual utilization increase over 24-72 hours without an "
                "identifiable single-event trigger indicates organic demand growth exceeding site "
                "capacity. Recommended action: schedule a capacity planning review rather than an "
                "emergency dispatch; this is a planning issue, not a hardware fault.",
    },
    {
        "id": "SOP-GEN-01",
        "failure_type": "general",
        "text": "General incident response guideline: always cross-reference the mobility events "
                "log before attributing a traffic spike to a fault -- named events (festivals, "
                "matches, concerts) can produce load patterns that superficially resemble "
                "congestion collapse but require no maintenance action.",
    },
]

# ---------------------------------------------------------------------------
# Build PARENT-CHILD index: split each parent into sentence-level children
# ---------------------------------------------------------------------------
def _split_into_sentences(text: str) -> list:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 10]


_child_chunks = []       # list of chunk texts, for indexing
_child_to_parent = []    # parallel list: parent doc dict for each chunk

for parent in PARENT_DOCUMENTS:
    for sentence in _split_into_sentences(parent["text"]):
        _child_chunks.append(sentence)
        _child_to_parent.append(parent)

_tokenized_chunks = [re.findall(r"[a-z0-9]+", c.lower()) for c in _child_chunks]
_bm25_index = BM25Okapi(_tokenized_chunks)

_tfidf_vectorizer = TfidfVectorizer(stop_words="english")
_tfidf_matrix = _tfidf_vectorizer.fit_transform(_child_chunks)


def _normalize(scores: np.ndarray) -> np.ndarray:
    """Min-max normalize to [0, 1] so BM25 and TF-IDF scores (different
    scales) can be combined meaningfully."""
    lo, hi = scores.min(), scores.max()
    if hi - lo < 1e-9:
        return np.zeros_like(scores)
    return (scores - lo) / (hi - lo)


# ---------------------------------------------------------------------------
# CONTEXTUAL RAG: build a query from live state, not just a category name
# ---------------------------------------------------------------------------
def build_contextual_query(root_cause_category: str, root_cause_reasoning: str,
                            row: dict = None) -> str:
    """Enriches the retrieval query with actual telemetry terms (not just
    the category label), so retrieval reflects the specific situation."""
    parts = [root_cause_category.replace("_", " "), root_cause_reasoning]
    if row:
        if row.get("battery_level_pct", 100) < 50:
            parts.append("low battery voltage power supply")
        if row.get("utilization_pct", 0) > 70:
            parts.append("high utilization capacity exhaustion")
        if row.get("packet_loss_pct", 0) > 15:
            parts.append("packet loss")
        if row.get("storm_flag", 0) == 1:
            parts.append("storm weather signal degradation")
        # Only add "gradual creep" keywords when the root cause category
        # ITSELF is capacity_creep -- the Root Cause Agent has already run
        # its onset-acuity check (acute-recent-onset vs sustained-creep) by
        # this point, so re-deriving "gradual" from a raw 24h trend number
        # here would contradict that decision and pull in a mismatched SOP
        # (e.g. injecting capacity-creep language for a diagnosed acute
        # congestion_collapse case just because utilization also happened
        # to trend up over 24h).
        if root_cause_category == "capacity_creep":
            util_trend_24h = row.get("utilization_pct_delta_24h") or 0
            if util_trend_24h > 10:
                parts.append("sustained gradual utilization growth capacity planning")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# HYBRID RETRIEVAL + RERANKING, returning full PARENT documents
# ---------------------------------------------------------------------------
def retrieve(query: str, top_k: int = 2, failure_type: str = None) -> list:
    """
    Runs BM25 + TF-IDF over child chunks, blends scores (hybrid), applies a
    contextual-keyword reranking pass, then maps back to PARENT documents
    (deduplicated, highest-scoring child per parent wins) and returns the
    top_k full parent documents.
    """
    query_tokens = re.findall(r"[a-z0-9]+", query.lower())

    bm25_scores = _bm25_index.get_scores(query_tokens)
    query_vec = _tfidf_vectorizer.transform([query])
    tfidf_scores = cosine_similarity(query_vec, _tfidf_matrix)[0]

    bm25_norm = _normalize(np.array(bm25_scores))
    tfidf_norm = _normalize(np.array(tfidf_scores))
    hybrid_scores = 0.5 * bm25_norm + 0.5 * tfidf_norm

    # RERANKING pass: boost chunks whose parent's failure_type matches, and
    # boost chunks with direct keyword overlap with the query terms beyond
    # what BM25/TF-IDF already captured (a lightweight stand-in for a
    # cross-encoder rerank).
    reranked_scores = hybrid_scores.copy()
    for i, chunk_tokens in enumerate(_tokenized_chunks):
        parent = _child_to_parent[i]
        if failure_type and parent["failure_type"] in (failure_type, "general"):
            reranked_scores[i] += 0.15
        keyword_overlap = len(set(chunk_tokens) & set(query_tokens)) / max(len(query_tokens), 1)
        reranked_scores[i] += 0.1 * keyword_overlap

    # Map child scores back to parents (highest-scoring child wins per parent)
    parent_best_score = {}
    for i, score in enumerate(reranked_scores):
        parent_id = _child_to_parent[i]["id"]
        if parent_id not in parent_best_score or score > parent_best_score[parent_id]:
            parent_best_score[parent_id] = score

    ranked_parent_ids = sorted(parent_best_score, key=lambda pid: -parent_best_score[pid])[:top_k]

    results = []
    for pid in ranked_parent_ids:
        parent = next(p for p in PARENT_DOCUMENTS if p["id"] == pid)
        results.append({**parent, "relevance_score": round(float(parent_best_score[pid]), 3)})
    return results


if __name__ == "__main__":
    query = build_contextual_query(
        "hardware_power_failure",
        "Battery/voltage evidence directly indicates power supply degradation.",
        row={"battery_level_pct": 19, "grid_voltage": 147, "utilization_pct": 36},
    )
    print(f"Contextual query: {query}\n")
    for r in retrieve(query, top_k=2, failure_type="hardware_power_failure"):
        print(f"[{r['id']}] (score={r['relevance_score']}) {r['text'][:100]}...")
