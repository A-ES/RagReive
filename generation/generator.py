"""Grounded answer generation with bracketed citations."""

from __future__ import annotations

import os
import re

from ingestion.models import Citation, GenerationResult, ScoredChunk

_CITATION_RE = re.compile(r"\[(\d+)\]")


class Generator:
    def __init__(self, model: str = "gpt-4o-mini") -> None:
        self.model = model

    async def generate(
        self,
        query: str,
        chunks: list[ScoredChunk],
        min_relevance_threshold: float = 0.3,
    ) -> GenerationResult:
        highest = max((chunk.score for chunk in chunks), default=0.0)
        if not chunks or highest < min_relevance_threshold:
            return GenerationResult(
                answer="I don't know based on the available context.",
                citations=[],
                is_grounded=False,
                reason=f"highest_relevance_score={highest:.3f}",
            )

        prompt = self._build_prompt(query, chunks)
        if os.environ.get("OPENAI_API_KEY"):
            answer, prompt_tokens, completion_tokens = await self._call_openai(prompt)
        else:
            answer = self._offline_answer(query, chunks)
            prompt_tokens = len(prompt.split())
            completion_tokens = len(answer.split())

        citations = self._extract_citations(answer, chunks)
        return GenerationResult(
            answer=answer,
            citations=citations,
            is_grounded=True,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    def _build_prompt(self, query: str, chunks: list[ScoredChunk]) -> str:
        context = "\n\n".join(
            f"[{idx}] Source: {chunk.chunk.filename or chunk.chunk.doc_id}\n{chunk.chunk.text}"
            for idx, chunk in enumerate(chunks, start=1)
        )
        return (
            "Answer using only the context. Cite each factual claim with [n]. "
            "If the answer is not present, say you don't know.\n\n"
            f"Question: {query}\n\nContext:\n{context}\n\nAnswer:"
        )

    async def _call_openai(self, prompt: str) -> tuple[str, int, int]:
        from openai import AsyncOpenAI

        client = AsyncOpenAI()
        response = await client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )
        usage = response.usage
        return (
            response.choices[0].message.content or "",
            usage.prompt_tokens if usage else 0,
            usage.completion_tokens if usage else 0,
        )

    def _offline_answer(self, query: str, chunks: list[ScoredChunk]) -> str:
        excerpts = []
        for idx, chunk in enumerate(chunks[:3], start=1):
            text = chunk.chunk.text.strip().replace("\n", " ")
            excerpts.append(f"{text[:260]} [{idx}]")
        return " ".join(excerpts) if excerpts else "I don't know based on the available context."

    def _extract_citations(self, answer: str, chunks: list[ScoredChunk]) -> list[Citation]:
        seen: set[int] = set()
        citations: list[Citation] = []
        for match in _CITATION_RE.finditer(answer):
            idx = int(match.group(1))
            if idx in seen or idx < 1 or idx > len(chunks):
                continue
            seen.add(idx)
            chunk = chunks[idx - 1].chunk
            citations.append(
                Citation(
                    index=idx,
                    chunk_id=chunk.chunk_id,
                    chunk_text=chunk.text,
                    source=chunk.filename or chunk.doc_id,
                    claim=_sentence_around(answer, match.start()),
                )
            )
        return citations


def _sentence_around(text: str, pos: int) -> str:
    start = max(text.rfind(".", 0, pos), text.rfind("\n", 0, pos)) + 1
    end_candidates = [i for i in (text.find(".", pos), text.find("\n", pos)) if i != -1]
    end = min(end_candidates) + 1 if end_candidates else len(text)
    return text[start:end].strip()
