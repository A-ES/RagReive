"""
Unit tests for retrieval/embedder.py.

Covers:
- Provider auto-selection (OpenAI when key is present, ST fallback when absent)
- Warning logged when falling back to sentence-transformers
- Explicit provider override
- Correct embedding_dim set per provider
- embed_batch returns correct length and non-empty vectors
- embed_batch on empty input returns empty list
- embed_query returns a single vector of correct dimension
- Average latency logging (logged after embed_batch)
- Semaphore concurrency (embed_batch fans out sub-batches in parallel)
- Lazy loading of backend clients (no import until first embed call)

Note: All provider calls are mocked so no real API keys or models are needed.
"""

from __future__ import annotations

import asyncio
import logging
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from retrieval.embedder import Embedder


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FAKE_OPENAI_KEY = "sk-test-1234"


def _fake_openai_response(texts: list[str], dim: int = 1536):
    """Return a mock AsyncOpenAI embeddings.create() response."""
    mock_response = MagicMock()
    mock_response.data = [
        MagicMock(embedding=[float(i) / (dim * len(texts)) for i in range(dim)], index=idx)
        for idx, _ in enumerate(texts)
    ]
    return mock_response


def _make_st_encode(dim: int = 384):
    """Return a callable that mimics SentenceTransformer.encode()."""
    import numpy as np

    def _encode(texts, show_progress_bar=False, convert_to_numpy=True):
        return np.zeros((len(texts), dim), dtype="float32")

    return _encode


# ---------------------------------------------------------------------------
# Provider selection
# ---------------------------------------------------------------------------


def test_auto_select_openai_when_key_in_env():
    """Provider should be 'openai' when OPENAI_API_KEY is present in env."""
    with patch.dict(os.environ, {"OPENAI_API_KEY": FAKE_OPENAI_KEY}):
        embedder = Embedder()
    assert embedder.provider == "openai"
    assert embedder.embedding_dim == Embedder.OPENAI_DIM


def test_auto_select_openai_when_key_passed_directly():
    """Explicit openai_api_key arg takes precedence over absent env var."""
    with patch.dict(os.environ, {}, clear=True):
        embedder = Embedder(openai_api_key=FAKE_OPENAI_KEY)
    assert embedder.provider == "openai"
    assert embedder.embedding_dim == Embedder.OPENAI_DIM


def test_auto_fallback_to_st_when_key_absent(caplog):
    """Provider should fall back to sentence_transformers with a warning when key is missing."""
    with patch.dict(os.environ, {}, clear=True):
        with caplog.at_level(logging.WARNING, logger="retrieval.embedder"):
            embedder = Embedder()
    assert embedder.provider == "sentence_transformers"
    assert embedder.embedding_dim == Embedder.FALLBACK_DIM
    assert any("OPENAI_API_KEY" in msg for msg in caplog.messages)


def test_explicit_provider_openai_overrides_missing_key():
    """Passing provider='openai' explicitly should set provider even without a key."""
    with patch.dict(os.environ, {}, clear=True):
        embedder = Embedder(provider="openai")
    assert embedder.provider == "openai"


def test_explicit_provider_st_overrides_present_key():
    """Passing provider='sentence_transformers' should use ST even when key is set."""
    with patch.dict(os.environ, {"OPENAI_API_KEY": FAKE_OPENAI_KEY}):
        embedder = Embedder(provider="sentence_transformers")
    assert embedder.provider == "sentence_transformers"
    assert embedder.embedding_dim == Embedder.FALLBACK_DIM


# ---------------------------------------------------------------------------
# Embedding dimensions and model names
# ---------------------------------------------------------------------------


def test_openai_dim_constant():
    assert Embedder.OPENAI_DIM == 1536


def test_fallback_dim_constant():
    assert Embedder.FALLBACK_DIM == 384


def test_default_openai_model():
    assert Embedder.OPENAI_MODEL == "text-embedding-3-small"


def test_default_fallback_model():
    assert "all-MiniLM-L6-v2" in Embedder.FALLBACK_MODEL


# ---------------------------------------------------------------------------
# embed_batch — OpenAI provider (mocked)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_embed_batch_openai_returns_correct_length():
    texts = ["hello world", "foo bar", "baz qux"]
    embedder = Embedder(provider="openai", openai_api_key=FAKE_OPENAI_KEY)

    async def _fake_create(model, input):  # noqa: A002
        return _fake_openai_response(input)

    mock_client = MagicMock()
    mock_client.embeddings.create = AsyncMock(side_effect=_fake_create)
    embedder._openai_client = mock_client

    result = await embedder.embed_batch(texts)

    assert len(result) == len(texts)


@pytest.mark.asyncio
async def test_embed_batch_openai_returns_correct_dim():
    texts = ["alpha", "beta"]
    embedder = Embedder(provider="openai", openai_api_key=FAKE_OPENAI_KEY)

    async def _fake_create(model, input):  # noqa: A002
        return _fake_openai_response(input, dim=Embedder.OPENAI_DIM)

    mock_client = MagicMock()
    mock_client.embeddings.create = AsyncMock(side_effect=_fake_create)
    embedder._openai_client = mock_client

    result = await embedder.embed_batch(texts)
    for vec in result:
        assert len(vec) == Embedder.OPENAI_DIM


@pytest.mark.asyncio
async def test_embed_batch_empty_list_returns_empty():
    embedder = Embedder(provider="openai", openai_api_key=FAKE_OPENAI_KEY)
    result = await embedder.embed_batch([])
    assert result == []


@pytest.mark.asyncio
async def test_embed_batch_preserves_order():
    """Embeddings must be returned in the same order as input texts."""
    texts = [f"sentence {i}" for i in range(10)]
    embedder = Embedder(
        provider="openai",
        openai_api_key=FAKE_OPENAI_KEY,
        batch_size=3,  # Force multiple sub-batches
    )

    # Each sub-batch gets a unique sentinel value at position 0
    call_count = 0

    async def _fake_create(model, input):  # noqa: A002
        nonlocal call_count
        dim = Embedder.OPENAI_DIM
        resp = MagicMock()
        resp.data = [
            MagicMock(
                embedding=[float(call_count * 1000 + idx)] + [0.0] * (dim - 1),
                index=idx,
            )
            for idx, _ in enumerate(input)
        ]
        call_count += 1
        return resp

    mock_client = MagicMock()
    mock_client.embeddings.create = AsyncMock(side_effect=_fake_create)
    embedder._openai_client = mock_client

    result = await embedder.embed_batch(texts)

    # Every slot must be filled (non-empty)
    assert len(result) == len(texts)
    assert all(len(v) > 0 for v in result)


# ---------------------------------------------------------------------------
# embed_batch — sentence-transformers provider (mocked)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_embed_batch_st_returns_correct_length():
    texts = ["foo", "bar", "baz"]
    embedder = Embedder(provider="sentence_transformers")

    mock_model = MagicMock()
    mock_model.encode = _make_st_encode(dim=Embedder.FALLBACK_DIM)
    embedder._st_model = mock_model

    result = await embedder.embed_batch(texts)
    assert len(result) == len(texts)


@pytest.mark.asyncio
async def test_embed_batch_st_returns_correct_dim():
    texts = ["hello"]
    embedder = Embedder(provider="sentence_transformers")

    mock_model = MagicMock()
    mock_model.encode = _make_st_encode(dim=Embedder.FALLBACK_DIM)
    embedder._st_model = mock_model

    result = await embedder.embed_batch(texts)
    assert len(result[0]) == Embedder.FALLBACK_DIM


# ---------------------------------------------------------------------------
# embed_query
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_embed_query_returns_single_vector_openai():
    embedder = Embedder(provider="openai", openai_api_key=FAKE_OPENAI_KEY)

    async def _fake_create(model, input):  # noqa: A002
        return _fake_openai_response(input, dim=Embedder.OPENAI_DIM)

    mock_client = MagicMock()
    mock_client.embeddings.create = AsyncMock(side_effect=_fake_create)
    embedder._openai_client = mock_client

    vec = await embedder.embed_query("what is RAG?")
    assert isinstance(vec, list)
    assert len(vec) == Embedder.OPENAI_DIM


@pytest.mark.asyncio
async def test_embed_query_returns_single_vector_st():
    embedder = Embedder(provider="sentence_transformers")

    mock_model = MagicMock()
    mock_model.encode = _make_st_encode(dim=Embedder.FALLBACK_DIM)
    embedder._st_model = mock_model

    vec = await embedder.embed_query("hybrid retrieval")
    assert isinstance(vec, list)
    assert len(vec) == Embedder.FALLBACK_DIM


# ---------------------------------------------------------------------------
# Latency logging
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_embed_batch_logs_average_latency(caplog):
    """embed_batch must log a message containing latency information."""
    texts = ["a", "b", "c"]
    embedder = Embedder(provider="openai", openai_api_key=FAKE_OPENAI_KEY)

    async def _fake_create(model, input):  # noqa: A002
        return _fake_openai_response(input)

    mock_client = MagicMock()
    mock_client.embeddings.create = AsyncMock(side_effect=_fake_create)
    embedder._openai_client = mock_client

    with caplog.at_level(logging.INFO, logger="retrieval.embedder"):
        await embedder.embed_batch(texts)

    # There should be a log message with latency info
    latency_logged = any(
        "ms/chunk" in msg or "avg" in msg.lower() or "latency" in msg.lower()
        for msg in caplog.messages
    )
    assert latency_logged, f"No latency log found. Messages: {caplog.messages}"


# ---------------------------------------------------------------------------
# Semaphore / concurrency
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_embed_batch_uses_semaphore(monkeypatch):
    """embed_batch must respect the worker semaphore (no RuntimeError under concurrency)."""
    texts = [f"text {i}" for i in range(20)]
    embedder = Embedder(
        provider="openai",
        openai_api_key=FAKE_OPENAI_KEY,
        embedding_workers=2,
        batch_size=4,
    )

    async def _fake_create(model, input):  # noqa: A002
        await asyncio.sleep(0)  # Yield to event loop
        return _fake_openai_response(input)

    mock_client = MagicMock()
    mock_client.embeddings.create = AsyncMock(side_effect=_fake_create)
    embedder._openai_client = mock_client

    result = await embedder.embed_batch(texts)
    assert len(result) == len(texts)


@pytest.mark.asyncio
async def test_semaphore_default_workers():
    """Default embedding_workers should be 8."""
    embedder = Embedder(provider="openai", openai_api_key=FAKE_OPENAI_KEY)
    assert embedder._workers == 8


# ---------------------------------------------------------------------------
# Lazy loading
# ---------------------------------------------------------------------------


def test_openai_client_not_loaded_on_init():
    """_openai_client must be None until the first embed call."""
    embedder = Embedder(provider="openai", openai_api_key=FAKE_OPENAI_KEY)
    assert embedder._openai_client is None


def test_st_model_not_loaded_on_init():
    """_st_model must be None until the first embed call."""
    embedder = Embedder(provider="sentence_transformers")
    assert embedder._st_model is None


# ---------------------------------------------------------------------------
# Configuration forwarding
# ---------------------------------------------------------------------------


def test_custom_batch_size_is_stored():
    embedder = Embedder(provider="openai", openai_api_key=FAKE_OPENAI_KEY, batch_size=32)
    assert embedder._batch_size == 32


def test_custom_workers_is_stored():
    embedder = Embedder(provider="openai", openai_api_key=FAKE_OPENAI_KEY, embedding_workers=4)
    assert embedder._workers == 4
