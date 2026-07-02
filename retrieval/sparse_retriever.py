"""Sparse BM25 retrieval wrapper."""

from __future__ import annotations

import logging
import time

from ingestion.models import ScoredChunk
from retrieval.bm25_index import BM25Index

logger = logging.getLogger(__name__)


class SparseRetriever:
    def __init__(self, bm25_index: BM25Index) -> None:
        self.bm25_index = bm25_index

    async def search(self, query: str, top_k: int = 10) -> list[ScoredChunk]:
        start = time.perf_counter()
        results = self.bm25_index.search(query, top_k=top_k)
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info("Sparse retrieval returned %d result(s) in %.2f ms.", len(results), elapsed_ms)
        return results
