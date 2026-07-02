import pytest

from generation.citation_verifier import CitationVerifier
from generation.confidence import compute_confidence_score
from generation.generator import Generator
from ingestion.models import Chunk, Citation, ScoredChunk


@pytest.mark.asyncio
async def test_citation_verifier_supported_with_lexical_fallback():
    verifier = CitationVerifier()
    verifier._get_model = lambda: (_ for _ in ()).throw(RuntimeError("skip model"))
    status = await verifier.verify("Qdrant stores embeddings.", "Qdrant stores embeddings and chunk payload metadata.")
    assert status == "supported"


@pytest.mark.asyncio
async def test_citation_verifier_unsupported_with_lexical_fallback():
    verifier = CitationVerifier()
    verifier._get_model = lambda: (_ for _ in ()).throw(RuntimeError("skip model"))
    status = await verifier.verify("The API key is required.", "The API key is optional because local fallback is available.")
    assert status == "unsupported"


@pytest.mark.asyncio
async def test_citation_verifier_partial_with_lexical_fallback():
    verifier = CitationVerifier()
    verifier._get_model = lambda: (_ for _ in ()).throw(RuntimeError("skip model"))
    status = await verifier.verify("The dashboard shows reports.", "The dashboard displays answers and citations.")
    assert status == "partial"


@pytest.mark.asyncio
async def test_generator_returns_i_dont_know_below_threshold():
    chunk = Chunk(doc_id="doc", chunk_index=0, text="Irrelevant", char_start=0, char_end=10, strategy="fixed")
    result = await Generator().generate("question", [ScoredChunk(chunk=chunk, score=0.1)], min_relevance_threshold=0.3)
    assert result.is_grounded is False
    assert "I don't know" in result.answer


def test_confidence_score_is_bounded():
    citation = Citation(index=1, chunk_id="c", chunk_text="text", source="doc", verification_status="supported")
    assert 0.0 <= compute_confidence_score([0.9, 0.8], [citation], 1.0) <= 1.0


def test_confidence_increases_with_supported_fraction():
    unsupported = Citation(index=1, chunk_id="c1", chunk_text="text", source="doc", verification_status="unsupported")
    supported = Citation(index=2, chunk_id="c2", chunk_text="text", source="doc", verification_status="supported")
    low = compute_confidence_score([0.5], [unsupported], 0.5)
    high = compute_confidence_score([0.5], [supported], 0.5)
    assert high > low
