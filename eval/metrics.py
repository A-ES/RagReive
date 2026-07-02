"""Evaluation metrics for generated answers and retrieval traces."""

from __future__ import annotations

from ingestion.models import Citation, ScoredChunk


def correctness(answer: str, expected: str) -> float:
    try:
        from rouge_score import rouge_scorer

        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        return float(scorer.score(expected, answer)["rougeL"].fmeasure)
    except Exception:
        expected_tokens = set(expected.lower().split())
        answer_tokens = set(answer.lower().split())
        return len(expected_tokens & answer_tokens) / max(1, len(expected_tokens))


def faithfulness(citations: list[Citation]) -> float:
    return citation_accuracy(citations)


def context_relevance(scored_chunks: list[ScoredChunk]) -> float:
    top = scored_chunks[:5]
    return sum(c.reranker_score if c.reranker_score is not None else c.score for c in top) / len(top) if top else 0.0


def citation_accuracy(citations: list[Citation]) -> float:
    if not citations:
        return 0.0
    return sum(1 for c in citations if c.verification_status == "supported") / len(citations)
