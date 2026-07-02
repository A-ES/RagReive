"""
Qdrant vector store client wrapper.

Provides async methods for upserting chunks with their embeddings, performing
ANN search, and listing document metadata aggregated from the payload store.

Collection schema (each Qdrant point):
  - id:      chunk UUID (str)
  - vector:  dense embedding (list[float])
  - payload: {doc_id, filename, format, chunk_index, char_start, char_end,
              strategy, text, ingested_at}
"""

from __future__ import annotations

import logging
from datetime import datetime

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qmodels

from ingestion.models import Chunk, DocumentMeta, ScoredChunk

logger = logging.getLogger(__name__)

# Embedding dimensions per provider
_DIM_OPENAI = 1536
_DIM_SENTENCE_TRANSFORMERS = 384


class VectorStoreClient:
    """Async wrapper around the Qdrant HTTP/gRPC client.

    Args:
        host: Qdrant server hostname.
        port: Qdrant server port (default 6333).
        collection_name: Name of the Qdrant collection to use.
        vector_dim: Expected embedding dimension. Pass 1536 for OpenAI
            ``text-embedding-3-small`` or 384 for ``all-MiniLM-L6-v2``.
            The collection is created with this dimension on first use.
    """

    def __init__(
        self,
        host: str,
        port: int,
        collection_name: str,
        vector_dim: int = _DIM_OPENAI,
    ) -> None:
        self._client = AsyncQdrantClient(host=host, port=port)
        self._collection = collection_name
        self._vector_dim = vector_dim
        self._collection_ready = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _ensure_collection(self) -> None:
        """Create the collection if it does not already exist."""
        if self._collection_ready:
            return

        existing = {
            c.name
            for c in (await self._client.get_collections()).collections
        }

        if self._collection not in existing:
            await self._client.create_collection(
                collection_name=self._collection,
                vectors_config=qmodels.VectorParams(
                    size=self._vector_dim,
                    distance=qmodels.Distance.COSINE,
                ),
            )
            logger.info(
                "Created Qdrant collection '%s' (dim=%d, distance=COSINE)",
                self._collection,
                self._vector_dim,
            )
        else:
            logger.debug("Qdrant collection '%s' already exists.", self._collection)

        self._collection_ready = True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def upsert_chunks(self, chunks: list[Chunk]) -> None:
        """Batch-upsert chunks with their embeddings and metadata payload.

        Args:
            chunks: Chunks to upsert. Every chunk must have a non-null
                ``embedding`` field; chunks without embeddings are skipped
                with a warning.

        Raises:
            RuntimeError: If the Qdrant upsert operation fails.
        """
        await self._ensure_collection()

        points: list[qmodels.PointStruct] = []
        skipped = 0
        for chunk in chunks:
            if chunk.embedding is None:
                logger.warning(
                    "Chunk %s has no embedding — skipping upsert.", chunk.chunk_id
                )
                skipped += 1
                continue

            points.append(
                qmodels.PointStruct(
                    id=chunk.chunk_id,
                    vector=chunk.embedding,
                    payload={
                        "doc_id": chunk.doc_id,
                        "filename": chunk.filename or "",
                        "format": chunk.format or "",
                        "ingested_at": chunk.ingested_at.isoformat()
                        if chunk.ingested_at
                        else datetime.utcnow().isoformat(),
                        "chunk_index": chunk.chunk_index,
                        "char_start": chunk.char_start,
                        "char_end": chunk.char_end,
                        "strategy": chunk.strategy,
                        "text": chunk.text,
                    },
                )
            )

        if not points:
            logger.warning("upsert_chunks called but no embeddable chunks found.")
            return

        await self._client.upsert(
            collection_name=self._collection,
            points=points,
            wait=True,
        )
        logger.info(
            "Upserted %d point(s) to collection '%s' (%d skipped, no embedding).",
            len(points),
            self._collection,
            skipped,
        )

    async def search(
        self,
        query_vector: list[float],
        top_k: int = 10,
    ) -> list[ScoredChunk]:
        """ANN search returning the top-k most similar chunks.

        Args:
            query_vector: Dense query embedding.
            top_k: Maximum number of results to return.

        Returns:
            List of :class:`~ingestion.models.ScoredChunk` objects sorted by
            descending cosine similarity score.
        """
        await self._ensure_collection()

        results = await self._client.search(
            collection_name=self._collection,
            query_vector=query_vector,
            limit=top_k,
            with_payload=True,
        )

        scored_chunks: list[ScoredChunk] = []
        for rank, hit in enumerate(results, start=1):
            payload = hit.payload or {}
            chunk = Chunk(
                chunk_id=str(hit.id),
                doc_id=payload.get("doc_id", ""),
                filename=payload.get("filename") or None,
                format=payload.get("format") or None,
                ingested_at=datetime.fromisoformat(payload["ingested_at"])
                if payload.get("ingested_at")
                else None,
                chunk_index=payload.get("chunk_index", 0),
                text=payload.get("text", ""),
                char_start=payload.get("char_start", 0),
                char_end=payload.get("char_end", 0),
                strategy=payload.get("strategy", "fixed"),
                embedding=None,  # not returned from the store
            )
            scored_chunks.append(
                ScoredChunk(
                    chunk=chunk,
                    score=hit.score,
                    rank=rank,
                    dense_score=hit.score,
                )
            )

        return scored_chunks

    async def list_documents(self) -> list[DocumentMeta]:
        """Aggregate document-level metadata from the payload store.

        Scrolls through all points in the collection and groups them by
        ``doc_id``, counting chunks per document and collecting ``filename``,
        ``format``, and ``ingested_at`` from the first chunk seen for each doc.

        Returns:
            A list of :class:`~ingestion.models.DocumentMeta` objects, one per
            unique document.
        """
        await self._ensure_collection()

        # Accumulate metadata keyed by doc_id
        docs: dict[str, dict] = {}
        offset = None

        while True:
            batch, next_offset = await self._client.scroll(
                collection_name=self._collection,
                scroll_filter=None,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )

            for point in batch:
                payload = point.payload or {}
                doc_id = payload.get("doc_id", "")
                if not doc_id:
                    continue

                if doc_id not in docs:
                    docs[doc_id] = {
                        "doc_id": doc_id,
                        "filename": payload.get("filename", ""),
                        "format": payload.get("format", ""),
                        "ingested_at": payload.get(
                            "ingested_at", datetime.utcnow().isoformat()
                        ),
                        "chunk_count": 0,
                    }
                docs[doc_id]["chunk_count"] += 1

            if next_offset is None:
                break
            offset = next_offset

        return [
            DocumentMeta(
                doc_id=d["doc_id"],
                filename=d["filename"],
                format=d["format"],
                chunk_count=d["chunk_count"],
                ingested_at=datetime.fromisoformat(d["ingested_at"])
                if isinstance(d["ingested_at"], str)
                else d["ingested_at"],
            )
            for d in docs.values()
        ]
