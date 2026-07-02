"""Reciprocal Rank Fusion for dense and sparse retrieval results."""

from __future__ import annotations

from ingestion.models import ScoredChunk


def _rank_map(results: list[ScoredChunk]) -> dict[str, int]:
    return {
        result.chunk.chunk_id: result.rank if result.rank is not None else idx
        for idx, result in enumerate(results, start=1)
    }


def reciprocal_rank_fusion(
    dense_results: list[ScoredChunk],
    sparse_results: list[ScoredChunk],
    alpha: float = 0.7,
    k: int = 60,
) -> list[ScoredChunk]:
    """Fuse two ranked lists using weighted RRF and return at most 20 candidates."""
    alpha = max(0.0, min(1.0, alpha))
    dense_ranks = _rank_map(dense_results)
    sparse_ranks = _rank_map(sparse_results)
    by_id = {result.chunk.chunk_id: result for result in dense_results + sparse_results}

    fused: list[ScoredChunk] = []
    for chunk_id, source in by_id.items():
        dense_rank = dense_ranks.get(chunk_id, len(dense_results) + 1)
        sparse_rank = sparse_ranks.get(chunk_id, len(sparse_results) + 1)
        dense_component = alpha * (1 / (k + dense_rank))
        sparse_component = (1 - alpha) * (1 / (k + sparse_rank))
        dense_source = next((r for r in dense_results if r.chunk.chunk_id == chunk_id), None)
        sparse_source = next((r for r in sparse_results if r.chunk.chunk_id == chunk_id), None)
        fused.append(
            ScoredChunk(
                chunk=source.chunk,
                score=dense_component + sparse_component,
                dense_score=dense_source.dense_score if dense_source else None,
                sparse_score=sparse_source.sparse_score if sparse_source else None,
            )
        )

    fused.sort(key=lambda result: result.score, reverse=True)
    for rank, result in enumerate(fused[:20], start=1):
        result.rank = rank
    return fused[:20]
