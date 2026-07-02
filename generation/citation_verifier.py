"""Citation verification using NLI with a lexical fallback."""

from __future__ import annotations

import re
from collections import Counter
from typing import Literal

from ingestion.models import Citation

VerificationStatus = Literal["supported", "partial", "unsupported"]


class CitationVerifier:
    def __init__(self, model_name: str = "cross-encoder/nli-deberta-v3-small") -> None:
        self.model_name = model_name
        self._model = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)
        return self._model

    async def verify(self, claim: str, chunk_text: str) -> VerificationStatus:
        try:
            model = self._get_model()
            scores = model.predict([(chunk_text, claim)])
            labels = ["contradiction", "neutral", "entailment"]
            label = labels[int(max(range(len(scores)), key=lambda i: scores[i]))]
            return {"entailment": "supported", "neutral": "partial", "contradiction": "unsupported"}[label]  # type: ignore[return-value]
        except Exception:
            return self._lexical_verify(claim, chunk_text)

    async def verify_citations(self, citations: list[Citation]) -> list[Citation]:
        for citation in citations:
            if citation.claim:
                citation.verification_status = await self.verify(citation.claim, citation.chunk_text)
        return citations

    def _lexical_verify(self, claim: str, chunk_text: str) -> VerificationStatus:
        claim_tokens = _tokens(claim)
        chunk_tokens = _tokens(chunk_text)
        if not claim_tokens:
            return "partial"
        overlap = sum((claim_tokens & chunk_tokens).values()) / sum(claim_tokens.values())
        if _has_simple_contradiction(claim, chunk_text):
            return "unsupported"
        if overlap >= 0.55:
            return "supported"
        if overlap >= 0.2:
            return "partial"
        return "unsupported"


def extract_claims(answer: str) -> list[tuple[str, int]]:
    claims: list[tuple[str, int]] = []
    for sentence in re.split(r"(?<=[.!?])\s+", answer):
        for marker in re.findall(r"\[(\d+)\]", sentence):
            claims.append((sentence.strip(), int(marker)))
    return claims


def _tokens(text: str) -> Counter[str]:
    stop = {"the", "a", "an", "and", "or", "to", "of", "in", "is", "are", "with", "for", "on", "by", "it", "this", "that"}
    return Counter(t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in stop)


def _has_simple_contradiction(claim: str, chunk_text: str) -> bool:
    claim_lower = claim.lower()
    chunk_lower = chunk_text.lower()
    pairs = [("enabled", "disabled"), ("true", "false"), ("required", "optional"), ("supports", "does not support")]
    return any(a in claim_lower and b in chunk_lower for a, b in pairs) or any(b in claim_lower and a in chunk_lower for a, b in pairs)
