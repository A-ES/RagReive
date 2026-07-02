from uuid import uuid4

from ingestion.models import Chunk, ScoredChunk
from retrieval.rrf_fusion import reciprocal_rank_fusion


def _result(label: str, rank: int, score: float = 1.0) -> ScoredChunk:
    return ScoredChunk(
        chunk=Chunk(
            chunk_id=label,
            doc_id=str(uuid4()),
            chunk_index=rank,
            text=label,
            char_start=0,
            char_end=len(label),
            strategy="fixed",
        ),
        score=score,
        rank=rank,
    )


def test_higher_alpha_shifts_toward_dense_result():
    dense = [_result("dense-favored", 1), _result("shared", 2)]
    sparse = [_result("sparse-favored", 1), _result("shared", 2)]
    low_alpha_top = reciprocal_rank_fusion(dense, sparse, alpha=0.0)[0].chunk.chunk_id
    high_alpha_top = reciprocal_rank_fusion(dense, sparse, alpha=1.0)[0].chunk.chunk_id
    assert low_alpha_top == "sparse-favored"
    assert high_alpha_top == "dense-favored"


def test_chunk_higher_in_both_lists_ranks_at_least_as_high():
    dense = [_result("better", 1), _result("worse", 2)]
    sparse = [_result("better", 1), _result("worse", 2)]
    fused = reciprocal_rank_fusion(dense, sparse)
    ranks = {result.chunk.chunk_id: result.rank for result in fused}
    assert ranks["better"] < ranks["worse"]


def test_result_count_is_capped_at_twenty():
    dense = [_result(f"d{i}", i) for i in range(1, 31)]
    sparse = [_result(f"s{i}", i) for i in range(1, 31)]
    assert len(reciprocal_rank_fusion(dense, sparse)) <= 20
