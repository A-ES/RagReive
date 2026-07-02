"""
Ingestion pipeline orchestrator for the Hybrid RAG pipeline.

Wires together all ingestion and indexing stages:

    DocumentParser → get_chunker → deduplicate_chunks → Embedder.embed_batch
        → VectorStoreClient.upsert_chunks + BM25Index.build / BM25Index.save

Usage::

    from pathlib import Path
    from config import PipelineConfig
    from ingestion.pipeline import run_ingestion

    config = PipelineConfig()
    result = await run_ingestion(file_paths=[Path("doc.md")], config=config)
    print(result)

The function is async because both ``Embedder.embed_batch`` and
``VectorStoreClient.upsert_chunks`` are async.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from config import PipelineConfig
from ingestion.chunkers import get_chunker
from ingestion.deduplication import deduplicate_chunks
from ingestion.models import Chunk, FailedFile, IngestionResult
from ingestion.parsers import DocumentParser, UnsupportedFormatError
from retrieval.bm25_index import BM25Index
from retrieval.embedder import Embedder
from retrieval.vector_store import VectorStoreClient

logger = logging.getLogger(__name__)


async def run_ingestion(
    file_paths: list[Path],
    config: PipelineConfig,
) -> IngestionResult:
    """
    Run the full ingestion pipeline for a list of files.

    Steps
    -----
    1. Parse each file into a ``ParsedDocument``; log and skip on failure.
    2. Chunk each document using the strategy from ``config.default_chunking_strategy``.
    3. Deduplicate chunks by cosine similarity (requires embeddings — see note).
    4. Embed all chunks in a single batched call to ``Embedder.embed_batch``.
    5. Deduplicate post-embedding so similarity comparisons are meaningful.
    6. Upsert embedded chunks into the Qdrant vector store.
    7. Build the BM25 index over all kept chunks and persist it to disk.

    Note on deduplication ordering: embeddings are needed for cosine-similarity
    deduplication, so deduplication runs *after* embedding.  The Embedder call
    therefore operates on all chunks (before dedup); dedup then reduces the set
    before upserting and indexing.

    Parameters
    ----------
    file_paths:
        Ordered list of file paths to ingest.
    config:
        ``PipelineConfig`` instance controlling chunking strategy, embedding
        provider, Qdrant connection, BM25 persistence path, etc.

    Returns
    -------
    IngestionResult
        Totals for documents processed, chunks indexed, any failed files, and
        total wall-clock time in seconds.
    """
    wall_start = time.perf_counter()

    failed_files: list[FailedFile] = []
    all_chunks: list[Chunk] = []

    # ------------------------------------------------------------------
    # Stage 1 + 2: Parse and chunk each file
    # ------------------------------------------------------------------
    parser = DocumentParser()

    # Resolve chunking strategy — semantic needs an embedder instance passed
    # to get_chunker, so we create the embedder once up front.
    embedder = Embedder(
        openai_api_key=config.openai_api_key,
        openai_model=config.openai_embedding_model,
        fallback_model=config.fallback_embedding_model,
        embedding_workers=config.embedding_workers,
        batch_size=config.embedding_batch_size,
    )

    strategy = config.default_chunking_strategy
    chunker = get_chunker(strategy, config, embedder=embedder)

    logger.info(
        "Ingestion started: %d file(s), strategy='%s'",
        len(file_paths),
        strategy,
    )

    for file_path in file_paths:
        try:
            doc = parser.parse(file_path)
        except UnsupportedFormatError as exc:
            logger.warning("Skipping unsupported file '%s': %s", file_path.name, exc)
            failed_files.append(
                FailedFile(
                    filename=file_path.name,
                    error=str(exc),
                    error_type=type(exc).__name__,
                )
            )
            continue
        except FileNotFoundError as exc:
            logger.warning("File not found '%s': %s", file_path, exc)
            failed_files.append(
                FailedFile(
                    filename=file_path.name,
                    error=str(exc),
                    error_type=type(exc).__name__,
                )
            )
            continue
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Failed to parse '%s' (%s): %s",
                file_path.name,
                type(exc).__name__,
                exc,
                exc_info=True,
            )
            failed_files.append(
                FailedFile(
                    filename=file_path.name,
                    error=str(exc),
                    error_type=type(exc).__name__,
                )
            )
            continue

        try:
            chunks = chunker.chunk(doc)
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Failed to chunk '%s' (%s): %s",
                file_path.name,
                type(exc).__name__,
                exc,
                exc_info=True,
            )
            failed_files.append(
                FailedFile(
                    filename=file_path.name,
                    error=str(exc),
                    error_type=type(exc).__name__,
                )
            )
            continue

        logger.debug("'%s' → %d chunk(s)", file_path.name, len(chunks))
        all_chunks.extend(chunks)

    total_documents = len(file_paths) - len(failed_files)

    if not all_chunks:
        wall_seconds = time.perf_counter() - wall_start
        logger.warning(
            "No chunks produced from %d file(s). Ingestion complete in %.3fs.",
            total_documents,
            wall_seconds,
        )
        return IngestionResult(
            total_documents=total_documents,
            total_chunks=0,
            failed_files=failed_files,
            wall_clock_seconds=wall_seconds,
        )

    # ------------------------------------------------------------------
    # Stage 3: Embed all chunks
    # ------------------------------------------------------------------
    logger.info("Embedding %d chunk(s)…", len(all_chunks))
    embed_start = time.perf_counter()

    texts = [chunk.text for chunk in all_chunks]
    embeddings = await embedder.embed_batch(texts)

    embed_elapsed = time.perf_counter() - embed_start
    avg_embed_latency_ms = (embed_elapsed / len(all_chunks)) * 1000

    logger.info(
        "Embedding complete: %d chunk(s) in %.3fs (avg %.2f ms/chunk)",
        len(all_chunks),
        embed_elapsed,
        avg_embed_latency_ms,
    )

    # Attach embeddings to chunks (in-place via model_copy to stay immutable-friendly)
    for chunk, embedding in zip(all_chunks, embeddings):
        chunk.embedding = embedding

    # ------------------------------------------------------------------
    # Stage 4: Deduplicate (now that embeddings are populated)
    # ------------------------------------------------------------------
    pre_dedup_count = len(all_chunks)
    all_chunks = deduplicate_chunks(all_chunks, threshold=config.dedup_similarity_threshold)
    post_dedup_count = len(all_chunks)

    if pre_dedup_count != post_dedup_count:
        logger.info(
            "Deduplication removed %d near-duplicate chunk(s) (threshold=%.2f). "
            "%d chunk(s) remaining.",
            pre_dedup_count - post_dedup_count,
            config.dedup_similarity_threshold,
            post_dedup_count,
        )

    # ------------------------------------------------------------------
    # Stage 5: Upsert into Qdrant vector store
    # ------------------------------------------------------------------
    vector_store = VectorStoreClient(
        host=config.qdrant_host,
        port=config.qdrant_port,
        collection_name=config.qdrant_collection,
        vector_dim=embedder.embedding_dim,
    )

    logger.info("Upserting %d chunk(s) to Qdrant…", len(all_chunks))
    await vector_store.upsert_chunks(all_chunks)

    # ------------------------------------------------------------------
    # Stage 6: Build and persist the BM25 index
    # ------------------------------------------------------------------
    bm25 = BM25Index()
    bm25.build(all_chunks)
    bm25.save(config.bm25_index_path)

    logger.info(
        "BM25 index built (%d chunk(s)) and saved to '%s'.",
        len(all_chunks),
        config.bm25_index_path,
    )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    wall_seconds = time.perf_counter() - wall_start

    logger.info(
        "Ingestion complete: %d document(s), %d chunk(s) indexed, "
        "%d failed file(s), total wall-clock %.3fs, avg embed latency %.2f ms/chunk.",
        total_documents,
        len(all_chunks),
        len(failed_files),
        wall_seconds,
        avg_embed_latency_ms,
    )

    return IngestionResult(
        total_documents=total_documents,
        total_chunks=len(all_chunks),
        failed_files=failed_files,
        wall_clock_seconds=wall_seconds,
    )
