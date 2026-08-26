import pytest
from src.retrieval import load_and_chunk, retrieve


class TestLoadAndChunk:
    def test_returns_list(self):
        chunks = load_and_chunk()
        assert isinstance(chunks, list)

    def test_chunks_not_empty(self):
        chunks = load_and_chunk()
        assert len(chunks) > 0

    def test_no_empty_chunks(self):
        chunks = load_and_chunk()
        for c in chunks:
            assert c.strip() != ""


class TestRetrieve:
    def test_returns_list(self):
        results = retrieve("international travel policy")
        assert isinstance(results, list)

    def test_result_has_required_keys(self):
        results = retrieve("international travel policy")
        assert len(results) > 0
        for r in results:
            assert "text" in r
            assert "section" in r
            assert "score" in r

    def test_score_is_float(self):
        results = retrieve("international travel policy")
        for r in results:
            assert isinstance(r["score"], float)

    def test_direct_query_returns_correct_section(self):
        results = retrieve("international travel policy")
        assert len(results) > 0
        top = results[0]
        assert "International Travel" in top["section"] or "international" in top["text"].lower()

    def test_out_of_scope_returns_empty(self):
        results = retrieve("quantum computing protein folding")
        for r in results:
            assert r["score"] <= 0.5  

    def test_top_k_respected(self):
        results = retrieve("travel", top_k=2)
        assert len(results) <= 2

    def test_threshold_filters_results(self):
        # With a very high threshold, should return fewer results
        results_low = retrieve("travel", threshold=0.01)
        results_high = retrieve("travel", threshold=0.99)
        assert len(results_low) >= len(results_high)
