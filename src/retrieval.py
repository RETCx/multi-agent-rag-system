import os
import re
from langchain.tools import tool
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "knowledge_base.txt")


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


def retrieve(query: str, top_k: int = 3, threshold: float = 0.03, chunks: list[str] | None = None) -> list[dict]:
    """
    Search chunks using TF-IDF + cosine similarity.

    Returns a list of dicts with keys: text, section, score.
    Only chunks with score > threshold are returned.
    Accepts pre-loaded chunks to avoid redundant file I/O.
    """
    if chunks is None:
        chunks = load_and_chunk()

    processed_chunks = [preprocess_text(c) for c in chunks]
    processed_query = preprocess_text(query)

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(processed_chunks + [processed_query])

    query_vec = tfidf_matrix[-1]
    chunk_vecs = tfidf_matrix[:-1]
    scores = cosine_similarity(query_vec, chunk_vecs).flatten()

    top_indices = scores.argsort()[-top_k:][::-1]

    results = []
    for i in top_indices:
        if scores[i] > threshold:
            section = chunks[i].split("\n")[0].strip()
            results.append({
                "text": chunks[i],
                "section": section,
                "score": round(float(scores[i]), 3),
            })

    return results


@tool
def retrieve_from_knowledge_base(query: str) -> str:
    """Search knowledge_base.txt and return relevant text snippets for a given query."""
    chunks = load_and_chunk()
    results = retrieve(query, chunks=chunks)  

    # retrieval pipeline 
    print(f"\n{'─' * 55}")
    print(f"[Data Retriever] Query    : {query}")
    print(f"[Data Retriever] Indexed  : {len(chunks)} chunks | top_k=3 | threshold=0.03")
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
