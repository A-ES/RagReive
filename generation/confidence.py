"""Composite confidence score calculation."""

from __future__ import annotations

from ingestion.models import Citation


def compute_confidence_score(
    top5_relevance_scores: list[float],
    citations: list[Citation],
    completeness_score: float,
) -> float:
    retrieval = sum(top5_relevance_scores) / len(top5_relevance_scores) if top5_relevance_scores else 0.0
    retrieval = max(0.0, min(1.0, retrieval))
    completeness = max(0.0, min(1.0, completeness_score))
    if citations:
        supported = sum(1 for c in citations if c.verification_status == "supported")
        citation_component = supported / len(citations)
        score = 0.4 * retrieval + 0.4 * citation_component + 0.2 * completeness
    else:
        score = 0.4 * retrieval + 0.2 * completeness
    return max(0.0, min(1.0, score))
