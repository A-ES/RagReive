"""Candidate reranking with cross-encoder or lightweight LLM-judge mode."""

from __future__ import annotations

import logging
import os
import time

from ingestion.models import ScoredChunk

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", mode: str = "cross_encoder") -> None:
        self.model_name = model_name
        self.mode = mode
        self._model = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            logger.info("Loading reranker model '%s'.", self.model_name)
            self._model = CrossEncoder(self.model_name)
        return self._model

    async def rerank(self, query: str, candidates: list[ScoredChunk], top_k: int = 5) -> list[ScoredChunk]:
        start = time.perf_counter()
        if not candidates:
            return []
        if self.mode == "llm_judge":
            scored = await self._rerank_llm_judge(query, candidates)
        else:
            scored = self._rerank_cross_encoder(query, candidates)
        scored.sort(key=lambda result: result.reranker_score if result.reranker_score is not None else result.score, reverse=True)
        for rank, result in enumerate(scored[:top_k], start=1):
            result.rank = rank
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info("Reranked %d candidate(s) in %.2f ms.", len(candidates), elapsed_ms)
        return scored[:top_k]

    def _rerank_cross_encoder(self, query: str, candidates: list[ScoredChunk]) -> list[ScoredChunk]:
        try:
            model = self._get_model()
            scores = model.predict([(query, c.chunk.text) for c in candidates])
            for candidate, score in zip(candidates, scores):
                candidate.reranker_score = float(score)
                candidate.score = float(score)
                logger.debug("Reranker score %.4f for chunk %s.", candidate.reranker_score, candidate.chunk.chunk_id)
            return candidates
        except Exception as exc:  # noqa: BLE001
            logger.warning("Cross-encoder reranking failed (%s); using retrieval scores.", exc)
            return candidates

    async def _rerank_llm_judge(self, query: str, candidates: list[ScoredChunk]) -> list[ScoredChunk]:
        if not os.environ.get("OPENAI_API_KEY"):
            return candidates
        try:
            from openai import AsyncOpenAI

            client = AsyncOpenAI()
            for candidate in candidates:
                prompt = (
                    "Rate relevance from 1 to 10. Return only a number.\n"
                    f"Query: {query}\nPassage: {candidate.chunk.text[:2000]}"
                )
                response = await client.chat.completions.create(
                    model=os.environ.get("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                )
                text = response.choices[0].message.content or "0"
                candidate.reranker_score = max(0.0, min(10.0, float(text.strip()))) / 10
                candidate.score = candidate.reranker_score
            return candidates
        except Exception as exc:  # noqa: BLE001
            logger.warning("LLM-judge reranking failed (%s); using retrieval scores.", exc)
            return candidates
