import os
import re
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from queries import TEST_CASES  # shared with main.py

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "knowledge_base.txt")

THRESHOLD = 0.03
TOP_K = 3

# ─── Chunking strategies ──────────────────────────────────────────────────────

def chunk_v1(text: str) -> list[str]:
    """V1: Whole document as one chunk."""
    return [text.strip()]

def chunk_v2(text: str, size: int = 500) -> list[str]:
    """V2: Fixed-size character split."""
    chunks = []
    for i in range(0, len(text), size):
        chunk = text[i:i + size].strip()
        if chunk:
            chunks.append(chunk)
    return chunks

def chunk_v3(text: str) -> list[str]:
    """V3: Section-aware split on blank lines."""
    return [c.strip() for c in text.split("\n\n") if c.strip()]

# ─── Preprocessing ────────────────────────────────────────────────────────────

def preprocess_text(text: str) -> str:
    """Normalize text for better sparse retrieval."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

# ─── Scoring strategies ───────────────────────────────────────────────────────

def score_keyword(query: str, chunks: list[str]) -> list[float]:
    """Option A: Keyword overlap count (normalized)."""
    query_words = set(query.lower().split())
    scores = []
    for chunk in chunks:
        chunk_words = set(chunk.lower().split())
        overlap = len(query_words & chunk_words)
        scores.append(overlap / max(len(query_words), 1))
    return scores

def score_tfidf(query: str, chunks: list[str]) -> list[float]:
    """Option B: TF-IDF + cosine similarity."""
    processed_chunks = [preprocess_text(c) for c in chunks]
    processed_query = preprocess_text(query)
    
    # sublinear_tf=True matches production retrieval.py configuration
    vectorizer = TfidfVectorizer(stop_words="english", sublinear_tf=True)
    try:
        matrix = vectorizer.fit_transform(processed_chunks + [processed_query])
    except ValueError:
        return [0.0] * len(chunks)
        
    query_vec = matrix[-1]
    chunk_vecs = matrix[:-1]
    return cosine_similarity(query_vec, chunk_vecs).flatten().tolist()

# ─── Evaluate one permutation ─────────────────────────────────────────────────

def evaluate(chunks: list[str], score_fn, test_cases: list[dict]) -> dict:
    results = {}
    for tc in test_cases:
        query = tc["query"]
        scores = score_fn(query, chunks)

        # get top-k above threshold
        indexed = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        top = [(chunks[i], round(s, 3)) for i, s in indexed[:TOP_K] if s > THRESHOLD]

        if not top:
            # Nothing retrieved — correct if query was expected to return empty
            hit = not tc["expect_result"]
            top_section = "(none)"
            top_score = 0.0
        else:
            top_chunk, top_score = top[0]
            top_section = top_chunk.split("\n")[0].strip()[:40]

            if tc["expect_section"] is None:
                hit = False
            else:
                all_sections = [chunk.split("\n")[0].strip() for chunk, _ in top]
                hit = any(
                    tc["expect_section"].lower() in s.lower()
                    for s in all_sections
                )

        results[tc["label"]] = {
            "hit": hit,
            "top_section": top_section,
            "top_score": top_score,
            "returned_count": len(top),
            "top_all": [(chunk.split("\n")[0].strip()[:35], score) for chunk, score in top],
        }
    return results

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        text = f.read()

    chunkers = {
        "V1 Whole Doc": chunk_v1,
        "V2 Fixed 500": chunk_v2,
        "V3 Section \\n\\n": chunk_v3,
    }
    scorers = {
        "A Keyword": score_keyword,
        "B TF-IDF": score_tfidf,
    }

    permutations = []
    for c_name, c_fn in chunkers.items():
        chunks = c_fn(text)
        for s_name, s_fn in scorers.items():
            results = evaluate(chunks, s_fn, TEST_CASES)
            hits = sum(1 for r in results.values() if r["hit"])
            permutations.append({
                "label": f"{c_name} + {s_name}",
                "chunks": len(chunks),
                "results": results,
                "total_hits": hits,
            })

    # ─── Summary table ────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("RETRIEVAL EXPERIMENT RESULTS")
    print("=" * 80)

    labels = [tc["label"] for tc in TEST_CASES]
    header = f"{'Permutation':<30} {'Chunks':>6}  " + "  ".join(f"{l:<14}" for l in labels) + "  Score"
    print(header)
    print("-" * 80)

    for p in permutations:
        row = f"{p['label']:<30} {p['chunks']:>6}  "
        for label in labels:
            r = p["results"][label]
            mark = "PASS" if r["hit"] else "FAIL"
            row += f"  {mark} {r['top_score']:.3f}      "
        row += f"  {p['total_hits']}/{len(TEST_CASES)}"
        print(row)

    print("=" * 80)

    # ─── Detail view ──────────────────────────────────────────────────────────
    print("\nDETAIL:\n")
    for p in permutations:
        print(f"  [{p['label']}]")
        for label, r in p["results"].items():
            mark = "PASS" if r["hit"] else "FAIL"
            others = ""
            if len(r["top_all"]) > 1:
                others = "  also: " + ", ".join(f"'{s}'({sc})" for s, sc in r["top_all"][1:])
            print(f"    {mark} {label:<18} '{r['top_section']}'  score={r['top_score']:.3f}{others}")
        print()

    # ─── Winner + Recall@K ────────────────────────────────────────────────────
    winner = max(permutations, key=lambda p: p["total_hits"])
    recall = winner["total_hits"] / len(TEST_CASES) * 100
    print(f"WINNER : {winner['label']}")
    print(f"Recall@{TOP_K}: {winner['total_hits']}/{len(TEST_CASES)} ({recall:.0f}%)")

if __name__ == "__main__":
    main()
