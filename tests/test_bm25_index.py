"""
Unit tests for retrieval/bm25_index.py.

Covers:
- Tokenisation helper
- build / search round-trip
- Empty-corpus edge case
- Empty-query edge case
- save / load persistence round-trip (Property 9: identical rank order)
- Search on an un-built index raises RuntimeError
- Save on an un-built index raises RuntimeError
- Load from a missing path raises FileNotFoundError
"""

from __future__ import annotations

import pickle
from pathlib import Path
from uuid import uuid4

import pytest

from ingestion.models import Chunk
from retrieval.bm25_index import BM25Index, _tokenize


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chunk(text: str, idx: int = 0, strategy: str = "fixed") -> Chunk:
    return Chunk(
        chunk_id=str(uuid4()),
        doc_id="doc-1",
        chunk_index=idx,
        text=text,
        char_start=0,
        char_end=len(text),
        strategy=strategy,  # type: ignore[arg-type]
    )


CORPUS = [
    _make_chunk("Python is a high-level programming language.", 0),
    _make_chunk("BM25 is a bag-of-words retrieval function.", 1),
    _make_chunk("FastAPI makes building APIs easy with Python.", 2),
    _make_chunk("Qdrant is a vector database for similarity search.", 3),
    _make_chunk("Retrieval-Augmented Generation combines retrieval with generation.", 4),
]


# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------


def test_tokenize_splits_on_whitespace_and_punctuation():
    tokens = _tokenize("Hello, world! This is a test.")
    assert "hello" in tokens
    assert "world" in tokens
    assert "test" in tokens
    # Punctuation should not appear as standalone tokens
    for t in tokens:
        assert t.isalnum(), f"Non-alphanumeric token found: {t!r}"


def test_tokenize_returns_lowercase():
    tokens = _tokenize("Python FastAPI BM25")
    assert tokens == ["python", "fastapi", "bm25"]


def test_tokenize_empty_string_returns_empty_list():
    assert _tokenize("") == []


def test_tokenize_only_punctuation_returns_empty_list():
    assert _tokenize("!!! --- ...") == []


# ---------------------------------------------------------------------------
# build / search
# ---------------------------------------------------------------------------


def test_build_and_search_returns_ranked_results():
    index = BM25Index()
    index.build(CORPUS)

    results = index.search("Python programming language", top_k=3)

    assert len(results) <= 3
    # The chunk about Python being a high-level language should be in top results
    top_chunk_texts = [r.chunk.text for r in results]
    assert any("Python" in t for t in top_chunk_texts)


def test_search_results_are_sorted_descending_by_score():
    index = BM25Index()
    index.build(CORPUS)

    results = index.search("retrieval BM25", top_k=5)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_search_ranks_are_one_based_and_sequential():
    index = BM25Index()
    index.build(CORPUS)

    results = index.search("Python", top_k=5)
    for expected_rank, result in enumerate(results, start=1):
        assert result.rank == expected_rank


def test_search_sparse_score_mirrors_score():
    index = BM25Index()
    index.build(CORPUS)

    results = index.search("vector database", top_k=3)
    for r in results:
        assert r.sparse_score == r.score


def test_search_top_k_limits_result_count():
    index = BM25Index()
    index.build(CORPUS)

    for k in (1, 2, 3, len(CORPUS)):
        results = index.search("Python retrieval", top_k=k)
        assert len(results) <= k


def test_search_never_exceeds_corpus_size():
    index = BM25Index()
    index.build(CORPUS)

    results = index.search("anything", top_k=1000)
    assert len(results) <= len(CORPUS)


def test_search_returns_correct_chunk_objects():
    """Chunks returned must be the originals, matched by chunk_id."""
    corpus_ids = {c.chunk_id for c in CORPUS}
    index = BM25Index()
    index.build(CORPUS)

    results = index.search("retrieval generation", top_k=len(CORPUS))
    result_ids = {r.chunk.chunk_id for r in results}
    # All result IDs must come from the corpus
    assert result_ids.issubset(corpus_ids)


# ---------------------------------------------------------------------------
# Empty corpus edge case
# ---------------------------------------------------------------------------


def test_build_empty_corpus_does_not_raise():
    index = BM25Index()
    index.build([])  # Should not raise


def test_search_empty_corpus_returns_empty_list():
    index = BM25Index()
    index.build([])

    results = index.search("anything", top_k=5)
    assert results == []


def test_len_reflects_corpus_size():
    index = BM25Index()
    assert len(index) == 0

    index.build(CORPUS)
    assert len(index) == len(CORPUS)


def test_is_built_false_before_build():
    index = BM25Index()
    assert not index.is_built


def test_is_built_true_after_build():
    index = BM25Index()
    index.build(CORPUS)
    assert index.is_built


# ---------------------------------------------------------------------------
# Empty query edge case
# ---------------------------------------------------------------------------


def test_search_empty_query_returns_empty_list():
    index = BM25Index()
    index.build(CORPUS)

    results = index.search("", top_k=5)
    assert results == []


def test_search_punctuation_only_query_returns_empty_list():
    index = BM25Index()
    index.build(CORPUS)

    results = index.search("!!! ???", top_k=5)
    assert results == []


# ---------------------------------------------------------------------------
# Error cases before build/load
# ---------------------------------------------------------------------------


def test_search_before_build_raises_runtime_error():
    index = BM25Index()
    with pytest.raises(RuntimeError, match="not been built or loaded"):
        index.search("query")


def test_save_before_build_raises_runtime_error(tmp_path):
    index = BM25Index()
    with pytest.raises(RuntimeError, match="uninitialised"):
        index.save(tmp_path / "index.pkl")


def test_load_missing_file_raises_file_not_found(tmp_path):
    index = BM25Index()
    with pytest.raises(FileNotFoundError):
        index.load(tmp_path / "nonexistent.pkl")


def test_load_invalid_pickle_raises_value_error(tmp_path):
    bad_file = tmp_path / "bad.pkl"
    with bad_file.open("wb") as fh:
        pickle.dump({"wrong_key": "data"}, fh)

    index = BM25Index()
    with pytest.raises(ValueError, match="Invalid BM25 index file"):
        index.load(bad_file)


# ---------------------------------------------------------------------------
# Persistence round-trip (validates Property 9: same rank order after reload)
# ---------------------------------------------------------------------------


def test_save_and_load_round_trip(tmp_path):
    """Saving and loading must produce identical search results (Property 9)."""
    index = BM25Index()
    index.build(CORPUS)

    pkl_path = tmp_path / "bm25_index.pkl"
    index.save(pkl_path)
    assert pkl_path.exists()

    loaded = BM25Index()
    loaded.load(pkl_path)

    query = "Python retrieval language"
    original_results = index.search(query, top_k=len(CORPUS))
    loaded_results = loaded.search(query, top_k=len(CORPUS))

    assert len(original_results) == len(loaded_results)
    for orig, reloaded in zip(original_results, loaded_results):
        assert orig.chunk.chunk_id == reloaded.chunk.chunk_id
        assert orig.rank == reloaded.rank
        assert abs(orig.score - reloaded.score) < 1e-9


def test_save_creates_parent_directories(tmp_path):
    """save() must create intermediate directories automatically."""
    index = BM25Index()
    index.build(CORPUS)

    deep_path = tmp_path / "a" / "b" / "c" / "index.pkl"
    index.save(deep_path)
    assert deep_path.exists()


def test_loaded_index_is_built(tmp_path):
    index = BM25Index()
    index.build(CORPUS)
    pkl_path = tmp_path / "index.pkl"
    index.save(pkl_path)

    fresh = BM25Index()
    assert not fresh.is_built
    fresh.load(pkl_path)
    assert fresh.is_built
    assert len(fresh) == len(CORPUS)


def test_rebuild_replaces_existing_index():
    """Calling build() a second time must replace the old index."""
    index = BM25Index()
    small_corpus = CORPUS[:2]
    index.build(small_corpus)
    assert len(index) == 2

    index.build(CORPUS)
    assert len(index) == len(CORPUS)
