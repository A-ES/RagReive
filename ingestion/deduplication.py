"""
Near-duplicate chunk removal via pairwise cosine similarity.

Chunks must have embeddings set before deduplication is called.
Any chunk whose cosine similarity to a previously-seen chunk exceeds
`threshold` is discarded. Chunks without embeddings are always kept.
"""

from __future__ import annotations

import logging

import numpy as np

from ingestion.models import Chunk

logger = logging.getLogger(__name__)


def _cosine_similarity_matrix(matrix: np.ndarray) -> np.ndarray:
    """
    Compute the pairwise cosine similarity matrix for a 2-D embedding matrix.

    Parameters
    ----------
    matrix:
        Shape (N, D).  Every row is expected to be non-zero.

    Returns
    -------
    np.ndarray
        Shape (N, N) with values in [-1, 1].
    """
    # L2-normalise each row so dot product == cosine similarity
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    # Avoid division by zero for zero-vectors (treat them as orthogonal to everything)
    norms = np.where(norms == 0, 1.0, norms)
    normalised = matrix / norms
    return normalised @ normalised.T


def deduplicate_chunks(
    chunks: list[Chunk],
    threshold: float = 0.95,
) -> list[Chunk]:
    """
    Remove near-duplicate chunks based on pairwise cosine similarity.

    A chunk is considered a near-duplicate of an *earlier* chunk when
    their cosine similarity strictly exceeds `threshold`.  The first
    occurrence is always kept; subsequent near-duplicates are dropped.

    Chunks without embeddings (``embedding is None``) are passed through
    unchanged and never compared.

    Parameters
    ----------
    chunks:
        Ordered list of ``Chunk`` objects.  Embeddings must be populated
        for similarity-based filtering to apply.
    threshold:
        Similarity threshold above which a chunk is considered a near-
        duplicate and discarded.  Defaults to 0.95 (from
        ``PipelineConfig.dedup_similarity_threshold``).

    Returns
    -------
    list[Chunk]
        Deduplicated list preserving the relative order of kept chunks.
    """
    if not chunks:
        return []

    # Separate chunks that have embeddings from those that don't.
    # Chunks without embeddings are always retained.
    embedded_indices: list[int] = []
    no_embedding_indices: list[int] = []

    for i, chunk in enumerate(chunks):
        if chunk.embedding is not None:
            embedded_indices.append(i)
        else:
            no_embedding_indices.append(i)

    if not embedded_indices:
        logger.debug(
            "deduplicate_chunks: no chunks have embeddings — returning all %d chunks unchanged.",
            len(chunks),
        )
        return list(chunks)

    # Build the embedding matrix for chunks that have embeddings.
    embedding_matrix = np.array(
        [chunks[i].embedding for i in embedded_indices], dtype=np.float32
    )  # shape: (M, D)

    sim_matrix = _cosine_similarity_matrix(embedding_matrix)  # shape: (M, M)

    # Greedy forward pass: keep a chunk unless it is too similar to any
    # *earlier* kept chunk.
    kept_local: list[bool] = [False] * len(embedded_indices)
    kept_local[0] = True  # first embedded chunk always kept

    for j in range(1, len(embedded_indices)):
        # Check similarity against all earlier *kept* embedded chunks.
        duplicate = False
        for k in range(j):
            if kept_local[k] and sim_matrix[j, k] > threshold:
                duplicate = True
                break
        kept_local[j] = not duplicate

    # Count how many were removed for logging.
    n_removed = sum(1 for kept in kept_local if not kept)
    if n_removed:
        logger.info(
            "deduplicate_chunks: removed %d near-duplicate chunk(s) "
            "(threshold=%.3f, total_input=%d).",
            n_removed,
            threshold,
            len(chunks),
        )

    # Rebuild the final ordered list using the original chunk ordering.
    kept_original_indices: set[int] = set(no_embedding_indices)
    for local_idx, original_idx in enumerate(embedded_indices):
        if kept_local[local_idx]:
            kept_original_indices.add(original_idx)

    return [chunk for i, chunk in enumerate(chunks) if i in kept_original_indices]


# Public alias matching the function signature specified in the design.
deduplicate = deduplicate_chunks
