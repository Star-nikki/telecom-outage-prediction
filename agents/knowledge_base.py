"""
A small local knowledge base standing in for telecom maintenance manuals /
SOPs / prior incident write-ups, plus lightweight retrieval (TF-IDF cosine
similarity -- no vector DB install needed, works with what's already in
scikit-learn).

This is genuinely RAG in miniature: retrieve() pulls the most relevant
documents for a query, and the Knowledge Agent in orchestrator_v2.py passes
those retrieved snippets to the Recommendation Agent, which cites them
rather than inventing advice from nothing.

To extend this into something closer to a real deployment: replace
DOCUMENTS below with your actual maintenance manual / SOP text (split into
paragraph-sized chunks), and swap TfidfVectorizer for a proper embedding
model + vector store (e.g. sentence-transformers + FAISS/Chroma) if you
want semantic (not just keyword) retrieval. The retrieve() interface below
would stay the same either way, so the rest of the pipeline doesn't need
to change.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DOCUMENTS = [
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

_vectorizer = None
_doc_vectors = None


def _ensure_index():
    global _vectorizer, _doc_vectors
    if _vectorizer is None:
        _vectorizer = TfidfVectorizer(stop_words="english")
        _doc_vectors = _vectorizer.fit_transform([d["text"] for d in DOCUMENTS])


def retrieve(query: str, top_k: int = 2, failure_type: str = None):
    """
    Returns the top_k most relevant documents for the query. If failure_type
    is given, documents matching that category (plus 'general') are
    prioritized -- this mimics filtered retrieval in a real RAG system.
    """
    _ensure_index()
    query_vec = _vectorizer.transform([query])
    scores = cosine_similarity(query_vec, _doc_vectors)[0]

    candidates = list(enumerate(scores))
    if failure_type:
        candidates = [
            (i, s + (0.5 if DOCUMENTS[i]["failure_type"] in (failure_type, "general") else 0))
            for i, s in candidates
        ]
    candidates.sort(key=lambda x: x[1], reverse=True)

    results = []
    for idx, score in candidates[:top_k]:
        results.append({**DOCUMENTS[idx], "relevance_score": round(float(score), 3)})
    return results


if __name__ == "__main__":
    for r in retrieve("battery draining and voltage low", top_k=2, failure_type="hardware_power_failure"):
        print(f"[{r['id']}] (score={r['relevance_score']}) {r['text'][:100]}...")
