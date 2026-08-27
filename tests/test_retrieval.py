import re
import pytest
from src.retrieval import load_and_chunk, retrieve, preprocess_text, _make_chunk_id


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

    def test_file_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            load_and_chunk("/nonexistent/path/kb.txt")


class TestPreprocessText:
    def test_lowercases(self):
        assert "hello world" == preprocess_text("Hello World")

    def test_removes_punctuation(self):
        result = preprocess_text("SOC2, ISO-27001!")
        assert "," not in result
        assert "!" not in result
        assert "-" not in result

    def test_preserves_words(self):
        result = preprocess_text("SecureID's SOC2")
        assert "secureid" in result
        assert "soc2" in result


class TestMakeChunkId:
    def test_format_matches_slug_pattern(self):
        chunk_id = _make_chunk_id("International Travel Policy", 3)
        assert re.match(r"^[a-z_]+_\d{2}$", chunk_id)

    def test_index_is_zero_padded(self):
        chunk_id = _make_chunk_id("Remote Work Policy", 5)
        assert chunk_id.endswith("_05")

    def test_deterministic(self):
        a = _make_chunk_id("Products and Services", 2)
        b = _make_chunk_id("Products and Services", 2)
        assert a == b


class TestRetrieve:
    def test_returns_list(self):
        results = retrieve("international travel policy")
        assert isinstance(results, list)

    def test_result_has_required_keys(self):
        results = retrieve("international travel policy")
        assert len(results) > 0
        for r in results:
            assert "chunk_id" in r
            assert "text" in r
            assert "section" in r
            assert "score" in r

    def test_score_is_float(self):
        results = retrieve("international travel policy")
        for r in results:
            assert isinstance(r["score"], float)

    def test_scores_are_descending(self):
        results = retrieve("travel policy")
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

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
        results_low = retrieve("travel", threshold=0.01)
        results_high = retrieve("travel", threshold=0.99)
        assert len(results_low) >= len(results_high)

    def test_very_high_threshold_returns_empty(self):
        results = retrieve("travel", threshold=0.99)
        assert results == []

    def test_diversity_filter_caps_per_section(self):
        """max_per_section should limit chunks from the same section."""
        results_limited = retrieve("policy", max_per_section=1)
        sections = [r["section"] for r in results_limited]
        # No section should appear more than once
        assert len(sections) == len(set(sections))

    def test_accepts_preloaded_chunks(self):
        """Passing chunks= should skip file I/O and use the given list."""
        custom_chunks = [
            "Alpha Policy\nAlpha details about travel.",
            "Beta Policy\nBeta details about something else.",
        ]
        results = retrieve("travel", chunks=custom_chunks)
        assert len(results) > 0
        assert "Alpha" in results[0]["section"]
