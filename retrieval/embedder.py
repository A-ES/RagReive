"""
Embedder: Dense text embedding with OpenAI and sentence-transformers fallback.

Provider selection:
- If OPENAI_API_KEY is set, uses OpenAI `text-embedding-3-small` (dim 1536).
- If absent, logs a warning and falls back to `sentence-transformers/all-MiniLM-L6-v2` (dim 384).

Concurrency:
- embed_batch uses asyncio.gather + asyncio.Semaphore to parallelize API calls
  across configurable worker slots (default 8).
- Logs average embedding latency per chunk after each batch.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Literal

logger = logging.getLogger(__name__)


class Embedder:
    """Produces dense embeddings for text using OpenAI or sentence-transformers."""

    OPENAI_MODEL = "text-embedding-3-small"
    OPENAI_DIM = 1536

    FALLBACK_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    FALLBACK_DIM = 384

    def __init__(
        self,
        provider: Literal["openai", "sentence_transformers"] | None = None,
        *,
        openai_api_key: str | None = None,
        openai_model: str = OPENAI_MODEL,
        fallback_model: str = FALLBACK_MODEL,
        embedding_workers: int = 8,
        batch_size: int = 64,
    ) -> None:
        """
        Initialise the embedder.

        If *provider* is None (default), provider is auto-selected:
          - "openai" when OPENAI_API_KEY is available in env or passed explicitly.
          - "sentence_transformers" otherwise (logs a warning).

        Args:
            provider: Force a specific provider, or None for auto-select.
            openai_api_key: Override the OPENAI_API_KEY env variable.
            openai_model: OpenAI embedding model name.
            fallback_model: HuggingFace sentence-transformers model name.
            embedding_workers: Max concurrent embedding requests (semaphore size).
            batch_size: Number of texts per sub-batch sent to the provider.
        """
        self._openai_model = openai_model
        self._fallback_model = fallback_model
        self._workers = embedding_workers
        self._batch_size = batch_size
        self._semaphore: asyncio.Semaphore | None = None  # created lazily per event loop

        # Resolve API key
        resolved_key = openai_api_key or os.environ.get("OPENAI_API_KEY")

        # Determine provider
        if provider is not None:
            self.provider: Literal["openai", "sentence_transformers"] = provider
        elif resolved_key:
            self.provider = "openai"
        else:
            logger.warning(
                "OPENAI_API_KEY is not set. Falling back to local sentence-transformers "
                "model '%s' (dim %d). Set OPENAI_API_KEY to use OpenAI embeddings.",
                self._fallback_model,
                self.FALLBACK_DIM,
            )
            self.provider = "sentence_transformers"

        # Lazy-load heavy clients
        self._openai_client = None
        self._st_model = None

        if self.provider == "openai":
            self._resolved_key = resolved_key
            self.embedding_dim = self.OPENAI_DIM
        else:
            self._resolved_key = None
            self.embedding_dim = self.FALLBACK_DIM

        logger.info(
            "Embedder initialised: provider=%s, model=%s, dim=%d, workers=%d",
            self.provider,
            self._openai_model if self.provider == "openai" else self._fallback_model,
            self.embedding_dim,
            self._workers,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_semaphore(self) -> asyncio.Semaphore:
        """Return (or create) the semaphore bound to the current event loop."""
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self._workers)
        return self._semaphore

    def _get_openai_client(self):
        """Lazy-load the AsyncOpenAI client."""
        if self._openai_client is None:
            from openai import AsyncOpenAI  # noqa: PLC0415

            self._openai_client = AsyncOpenAI(api_key=self._resolved_key)
        return self._openai_client

    def _get_st_model(self):
        """Lazy-load the SentenceTransformer model (CPU-friendly)."""
        if self._st_model is None:
            from sentence_transformers import SentenceTransformer  # noqa: PLC0415

            logger.info("Loading SentenceTransformer model '%s'…", self._fallback_model)
            self._st_model = SentenceTransformer(self._fallback_model)
            logger.info("SentenceTransformer model loaded.")
        return self._st_model

    # ------------------------------------------------------------------
    # Provider-specific sub-batch calls
    # ------------------------------------------------------------------

    async def _embed_subbatch_openai(self, texts: list[str]) -> list[list[float]]:
        """Call the OpenAI embeddings API for one sub-batch."""
        client = self._get_openai_client()
        response = await client.embeddings.create(
            model=self._openai_model,
            input=texts,
        )
        # API returns items ordered by index
        return [item.embedding for item in sorted(response.data, key=lambda x: x.index)]

    async def _embed_subbatch_st(self, texts: list[str]) -> list[list[float]]:
        """Run sentence-transformers inference in a thread pool to keep the event loop free."""
        model = self._get_st_model()
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None,
            lambda: model.encode(texts, show_progress_bar=False, convert_to_numpy=True),
        )
        return [emb.tolist() for emb in embeddings]

    async def _embed_subbatch(self, texts: list[str]) -> list[list[float]]:
        """Dispatch a sub-batch to the active provider."""
        if self.provider == "openai":
            return await self._embed_subbatch_openai(texts)
        return await self._embed_subbatch_st(texts)

    # ------------------------------------------------------------------
    # Public async API
    # ------------------------------------------------------------------

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a list of texts in parallel, respecting the semaphore limit.

        Splits *texts* into sub-batches of *batch_size* and fans them out with
        asyncio.gather.  Logs average embedding latency per chunk on completion.

        Args:
            texts: Non-empty list of strings to embed.

        Returns:
            List of embedding vectors, same length and order as *texts*.
        """
        if not texts:
            return []

        sem = self._get_semaphore()

        # Split into sub-batches
        sub_batches: list[tuple[int, list[str]]] = []
        for start in range(0, len(texts), self._batch_size):
            sub_batches.append((start, texts[start : start + self._batch_size]))

        async def _bounded(start: int, batch: list[str]) -> tuple[int, list[list[float]]]:
            async with sem:
                result = await self._embed_subbatch(batch)
                return start, result

        t0 = time.perf_counter()
        results = await asyncio.gather(*[_bounded(s, b) for s, b in sub_batches])
        elapsed = time.perf_counter() - t0

        # Re-assemble in original order
        output: list[list[float]] = [[] for _ in texts]
        for start, embeddings in results:
            for i, emb in enumerate(embeddings):
                output[start + i] = emb

        avg_latency_ms = (elapsed / len(texts)) * 1000
        logger.info(
            "embed_batch: embedded %d texts in %.3fs (avg %.2f ms/chunk), provider=%s",
            len(texts),
            elapsed,
            avg_latency_ms,
            self.provider,
        )

        return output

    async def embed_query(self, query: str) -> list[float]:
        """
        Embed a single query string for retrieval.

        Uses the same provider as embed_batch but skips the semaphore overhead
        since it is a single, low-latency call.

        Args:
            query: The search query to embed.

        Returns:
            A single embedding vector.
        """
        t0 = time.perf_counter()
        embeddings = await self._embed_subbatch([query])
        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.debug(
            "embed_query: embedded query in %.2f ms, provider=%s",
            elapsed_ms,
            self.provider,
        )
        return embeddings[0]
