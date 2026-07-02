"""Dense retrieval wrapper around the vector store."""

from __future__ import annotations

import logging
import time

from ingestion.models import ScoredChunk
from retrieval.vector_store import VectorStoreClient

logger = logging.getLogger(__name__)


class DenseRetriever:
    def __init__(self, vector_store: VectorStoreClient) -> None:
        self.vector_store = vector_store

    async def search(self, query_vector: list[float], top_k: int = 10) -> list[ScoredChunk]:
        start = time.perf_counter()
        results = await self.vector_store.search(query_vector, top_k=top_k)
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info("Dense retrieval returned %d result(s) in %.2f ms.", len(results), elapsed_ms)
        return results
