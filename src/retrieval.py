import os
import re
from langchain.tools import tool
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "knowledge_base.txt")
THRESHOLD: float = 0.03
TOP_K: int = 3
MAX_CHUNKS_PER_SECTION: int = 2


def load_and_chunk(filepath: str = DATA_PATH) -> list[str]:
    """Load knowledge_base.txt and split into chunks by blank line."""
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()
    chunks = [c.strip() for c in text.split("\n\n") if c.strip()]
    return chunks


def preprocess_text(text: str) -> str:
    """Normalize text for better sparse retrieval (TF-IDF)."""
    # Lowercase and remove punctuation
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text


def _make_chunk_id(section: str, index: int) -> str:
    """Generate a stable slug-based chunk ID from section header and position."""
    slug = re.sub(r"\W+", "_", section.lower()).strip("_")
    return f"{slug}_{index:02d}"


def retrieve(
    query: str,
    top_k: int = TOP_K,
    threshold: float = THRESHOLD,
    max_per_section: int = MAX_CHUNKS_PER_SECTION,
    chunks: list[str] | None = None,
) -> list[dict]:
    """
    Search chunks using TF-IDF + cosine similarity.

    Returns up to top_k dicts with keys: chunk_id, text, section, score.
    Applies threshold filter and section diversity filter before returning.
    Accepts pre-loaded chunks to avoid redundant file I/O.
    """
    if chunks is None:
        chunks = load_and_chunk()

    processed_chunks = [preprocess_text(c) for c in chunks]
    processed_query = preprocess_text(query)
    
    vectorizer = TfidfVectorizer(
        stop_words="english",
        sublinear_tf=True,
    )
    tfidf_matrix = vectorizer.fit_transform(processed_chunks + [processed_query])

    query_vec = tfidf_matrix[-1]
    chunk_vecs = tfidf_matrix[:-1]
    scores = cosine_similarity(query_vec, chunk_vecs).flatten()

    # Fetch more candidates than top_k to allow for diversity filtering
    candidate_count = min(top_k * 3, len(chunks))
    candidate_indices = scores.argsort()[-candidate_count:][::-1]

    # Apply threshold + diversity filter
    seen_sections: dict[str, int] = {}
    results = []
    for i in candidate_indices:
        if scores[i] <= threshold:
            continue
        section = chunks[i].split("\n")[0].strip()
        if seen_sections.get(section, 0) >= max_per_section:
            continue
        seen_sections[section] = seen_sections.get(section, 0) + 1
        results.append({
            "chunk_id": _make_chunk_id(section, i),
            "text": chunks[i],
            "section": section,
            "score": round(float(scores[i]), 3),
        })
        if len(results) == top_k:
            break

    return results


@tool
def retrieve_from_knowledge_base(query: str) -> str:
    """Search knowledge_base.txt and return relevant text snippets for a given query."""
    chunks = load_and_chunk()
    results = retrieve(query, top_k=TOP_K, threshold=THRESHOLD, chunks=chunks)

    print(f"\n{'─' * 55}")
    print(f"[Data Retriever] Query    : {query}")
    print(f"[Data Retriever] Indexed  : {len(chunks)} chunks | top_k={TOP_K} | threshold={THRESHOLD}")
    if not results:
        print("[Data Retriever] Retrieved: (none above threshold)")
    else:
        for r in results:
            print(f"[Data Retriever] Retrieved: '{r['section']}' (score={r['score']})")
    print(f"{'─' * 55}")

    if not results:
        return "No relevant information found in the knowledge base."

    output = []
    for i, r in enumerate(results, 1):
        output.append(f"[Snippet {i}]\n{r['text']}")

    return "\n\n".join(output)


if __name__ == "__main__":
    # Standalone test — run with: python -m retrieval
    test_queries = [
        "What is the policy on international travel?",
        "remote work options",
        "What do I need before going abroad?",
        "cryptocurrency investment policy",
    ]
    chunks = load_and_chunk()
    print(f"Chunks indexed: {len(chunks)}\n")

    for q in test_queries:
        print(f"Query: {q}")
        results = retrieve(q)
        if not results:
            print("  → No chunks above threshold")
        else:
            for r in results:
                print(f"  → '{r['section']}' (score={r['score']})")
        print()
