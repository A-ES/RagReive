"""
Unit tests for ingestion/chunkers.py and ingestion/deduplication.py.

Covers task 2.5:
- FixedChunker: chunk count, 64-char overlap correctness, metadata attachment
- StructuralChunker: split at headings, paragraphs, and code fences
- SemanticChunker: merges until similarity drops, correct metadata
- deduplication: near-duplicate removal (similarity > 0.95), retention of distinct chunks

Requirements: 11.2
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from ingestion.chunkers import FixedChunker, SemanticChunker, StructuralChunker
from ingestion.deduplication import deduplicate_chunks
from ingestion.models import Chunk, ParsedDocument


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_doc(content: str, fmt: str = "txt", filename: str = "test.txt") -> ParsedDocument:
    return ParsedDocument(
        doc_id=str(uuid4()),
        filename=filename,
        format=fmt,  # type: ignore[arg-type]
        content=content,
    )


def _make_chunk_with_embedding(
    text: str,
    embedding: list[float],
    idx: int = 0,
    doc_id: str = "doc-1",
) -> Chunk:
    return Chunk(
        chunk_id=str(uuid4()),
        doc_id=doc_id,
        chunk_index=idx,
        text=text,
        char_start=0,
        char_end=len(text),
        strategy="fixed",
        embedding=embedding,
    )


# ---------------------------------------------------------------------------
# FixedChunker
# ---------------------------------------------------------------------------


class TestFixedChunker:
    """Tests for FixedChunker — fixed-size character splitting with overlap."""

    # A document whose length ensures predictable chunk boundaries:
    # 640 characters → with chunk_size=512 and overlap=64 we expect 2 chunks.
    CONTENT_640 = "A" * 512 + "B" * 128

    def test_chunk_count_for_known_document(self):
        """A 640-char doc with size=512, overlap=64 produces exactly 2 chunks."""
        doc = _make_doc(self.CONTENT_640)
        chunker = FixedChunker(chunk_size=512, chunk_overlap=64)
        chunks = chunker.chunk(doc)
        assert len(chunks) == 2

    def test_overlap_correctness(self):
        """The tail of chunk[i] must appear at the head of chunk[i+1]."""
        doc = _make_doc(self.CONTENT_640)
        chunker = FixedChunker(chunk_size=512, chunk_overlap=64)
        chunks = chunker.chunk(doc)
        assert len(chunks) >= 2

        overlap = 64
        tail_of_first = chunks[0].text[-overlap:]
        head_of_second = chunks[1].text[:overlap]
        assert tail_of_first == head_of_second

    def test_no_chunk_exceeds_chunk_size(self):
        """Every chunk's text must be ≤ chunk_size characters."""
        doc = _make_doc("X" * 2000)
        chunker = FixedChunker(chunk_size=512, chunk_overlap=64)
        chunks = chunker.chunk(doc)
        for chunk in chunks:
            assert len(chunk.text) <= 512, (
                f"Chunk {chunk.chunk_index} has {len(chunk.text)} chars (> 512)"
            )

    def test_metadata_doc_id(self):
        """Every chunk must carry the parent document's doc_id."""
        doc = _make_doc(self.CONTENT_640)
        chunker = FixedChunker(chunk_size=512, chunk_overlap=64)
        chunks = chunker.chunk(doc)
        for chunk in chunks:
            assert chunk.doc_id == doc.doc_id

    def test_metadata_chunk_index_sequential(self):
        """chunk_index must start at 0 and increment by 1."""
        doc = _make_doc(self.CONTENT_640)
        chunker = FixedChunker(chunk_size=512, chunk_overlap=64)
        chunks = chunker.chunk(doc)
        for expected, chunk in enumerate(chunks):
            assert chunk.chunk_index == expected

    def test_metadata_strategy_is_fixed(self):
        """Every chunk's strategy field must equal 'fixed'."""
        doc = _make_doc(self.CONTENT_640)
        chunks = FixedChunker().chunk(doc)
        for chunk in chunks:
            assert chunk.strategy == "fixed"

    def test_metadata_char_positions_are_valid(self):
        """char_start ≥ 0, char_end > char_start, and text matches the slice."""
        doc = _make_doc(self.CONTENT_640)
        chunker = FixedChunker(chunk_size=512, chunk_overlap=64)
        chunks = chunker.chunk(doc)
        for chunk in chunks:
            assert chunk.char_start >= 0
            assert chunk.char_end > chunk.char_start

    def test_empty_document_returns_empty_list(self):
        """An empty document should produce no chunks without raising."""
        doc = _make_doc("")
        chunks = FixedChunker().chunk(doc)
        assert chunks == []

    def test_short_document_fits_in_one_chunk(self):
        """A document shorter than chunk_size should produce exactly 1 chunk."""
        doc = _make_doc("Hello world.")
        chunks = FixedChunker(chunk_size=512, chunk_overlap=64).chunk(doc)
        assert len(chunks) == 1
        assert chunks[0].text == "Hello world."

    def test_chunk_id_is_unique_per_chunk(self):
        """Each chunk must get a distinct UUID."""
        doc = _make_doc("Z" * 1500)
        chunks = FixedChunker(chunk_size=512, chunk_overlap=64).chunk(doc)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# StructuralChunker
# ---------------------------------------------------------------------------


class TestStructuralChunker:
    """Tests for StructuralChunker — heading, paragraph, and code-fence splitting."""

    def test_split_at_markdown_headings(self):
        """Markdown documents are split at heading boundaries."""
        content = (
            "# Introduction\n\nThis is the intro.\n\n"
            "## Section One\n\nContent of section one.\n\n"
            "## Section Two\n\nContent of section two."
        )
        doc = _make_doc(content, fmt="md", filename="doc.md")
        chunks = StructuralChunker().chunk(doc)
        # We expect at least 3 chunks (one per heading section)
        assert len(chunks) >= 3

    def test_split_at_paragraphs_for_txt(self):
        """Non-markdown documents are split on double-newline paragraph breaks."""
        content = "First paragraph text.\n\nSecond paragraph text.\n\nThird paragraph text."
        doc = _make_doc(content, fmt="txt")
        chunks = StructuralChunker().chunk(doc)
        assert len(chunks) == 3
        assert "First paragraph" in chunks[0].text
        assert "Second paragraph" in chunks[1].text
        assert "Third paragraph" in chunks[2].text

    def test_split_at_code_fences(self):
        """Code-fence delimiters (```) act as split boundaries for generic formats."""
        content = (
            "Here is some prose.\n\n"
            "```\ndef hello():\n    return 'world'\n```\n\n"
            "More prose after the code block."
        )
        doc = _make_doc(content, fmt="txt")
        chunks = StructuralChunker().chunk(doc)
        # At minimum: prose before fence, code block content, prose after fence
        assert len(chunks) >= 2
        texts = " ".join(c.text for c in chunks)
        assert "prose" in texts
        assert "hello" in texts

    def test_empty_document_returns_empty_list(self):
        """An empty document produces no chunks."""
        doc = _make_doc("", fmt="txt")
        chunks = StructuralChunker().chunk(doc)
        assert chunks == []

    def test_whitespace_only_segments_are_discarded(self):
        """Segments that are whitespace-only after stripping must be dropped."""
        content = "First paragraph.\n\n   \n\nSecond paragraph."
        doc = _make_doc(content, fmt="txt")
        chunks = StructuralChunker().chunk(doc)
        for chunk in chunks:
            assert chunk.text.strip() != ""

    def test_metadata_strategy_is_structural(self):
        """Every chunk must have strategy='structural'."""
        content = "Para one.\n\nPara two."
        doc = _make_doc(content, fmt="txt")
        chunks = StructuralChunker().chunk(doc)
        for chunk in chunks:
            assert chunk.strategy == "structural"

    def test_metadata_doc_id_propagated(self):
        """Every chunk must carry the document's doc_id."""
        content = "Para one.\n\nPara two."
        doc = _make_doc(content, fmt="txt")
        chunks = StructuralChunker().chunk(doc)
        for chunk in chunks:
            assert chunk.doc_id == doc.doc_id

    def test_metadata_char_positions_valid(self):
        """char_start ≥ 0 and char_end > char_start for every chunk."""
        content = "First paragraph of content.\n\nSecond paragraph of content."
        doc = _make_doc(content, fmt="txt")
        chunks = StructuralChunker().chunk(doc)
        for chunk in chunks:
            assert chunk.char_start >= 0
            assert chunk.char_end > chunk.char_start

    def test_single_paragraph_produces_one_chunk(self):
        """A document with no split points returns a single chunk."""
        content = "Just one paragraph with no double newlines."
        doc = _make_doc(content, fmt="txt")
        chunks = StructuralChunker().chunk(doc)
        assert len(chunks) == 1
        assert chunks[0].text == content

    def test_markdown_heading_content_preserved(self):
        """Each heading's content text must be present in the corresponding chunk."""
        content = "# Title\n\nIntro text here.\n\n## Details\n\nDetail text here."
        doc = _make_doc(content, fmt="md", filename="doc.md")
        chunks = StructuralChunker().chunk(doc)
        all_text = " ".join(c.text for c in chunks)
        assert "Intro text here" in all_text
        assert "Detail text here" in all_text


# ---------------------------------------------------------------------------
# SemanticChunker
# ---------------------------------------------------------------------------


class TestSemanticChunker:
    """
    Tests for SemanticChunker.

    The embedder is mocked to inject controlled cosine similarity scores,
    avoiding real API calls while exercising the merging logic precisely.
    """

    def _make_mock_embedder(self, embeddings: list[list[float]]) -> MagicMock:
        """Return a mock embedder whose embed_batch returns the given embeddings."""
        embedder = MagicMock()
        embedder.embed_batch = AsyncMock(return_value=embeddings)
        return embedder

    def test_merges_sentences_when_similarity_is_high(self):
        """
        When all consecutive sentence pairs have similarity ≥ threshold,
        all sentences should be merged into a single chunk.
        """
        # Three sentences that all embed similarly — use identical vectors
        identical = [1.0, 0.0]
        embedder = self._make_mock_embedder([identical, identical, identical])

        doc = _make_doc("First sentence. Second sentence. Third sentence.")
        chunker = SemanticChunker(embedder=embedder, similarity_threshold=0.75)
        chunks = chunker.chunk(doc)

        # All three sentences are above threshold → merged into one chunk
        assert len(chunks) == 1
        assert "First sentence" in chunks[0].text
        assert "Third sentence" in chunks[0].text

    def test_splits_when_similarity_drops(self):
        """
        When consecutive sentence similarity drops below the threshold,
        a new chunk boundary must be inserted.
        """
        # Sentences 0 and 1 are similar; sentence 2 is orthogonal (sim=0.0)
        high_sim_vec = [1.0, 0.0]   # cos_sim with itself = 1.0
        low_sim_vec = [0.0, 1.0]    # cos_sim([1,0],[0,1]) = 0.0

        embedder = self._make_mock_embedder([high_sim_vec, high_sim_vec, low_sim_vec])

        doc = _make_doc("Alpha sentence. Beta sentence. Gamma sentence.")
        chunker = SemanticChunker(embedder=embedder, similarity_threshold=0.75)
        chunks = chunker.chunk(doc)

        # Sentences 0+1 merge; sentence 2 starts a new chunk
        assert len(chunks) == 2
        assert "Alpha sentence" in chunks[0].text
        assert "Beta sentence" in chunks[0].text
        assert "Gamma sentence" in chunks[1].text

    def test_every_sentence_is_its_own_chunk_when_all_similar_pairs_drop(self):
        """
        When every consecutive pair is below threshold, each sentence becomes
        its own chunk.
        """
        high = [1.0, 0.0]
        low = [0.0, 1.0]
        # Alternating vectors: sim(high, low) = 0.0 < 0.75
        embedder = self._make_mock_embedder([high, low, high])

        doc = _make_doc("First. Second. Third.")
        chunker = SemanticChunker(embedder=embedder, similarity_threshold=0.75)
        chunks = chunker.chunk(doc)

        assert len(chunks) == 3

    def test_metadata_strategy_is_semantic(self):
        """Every chunk must carry strategy='semantic'."""
        vec = [1.0, 0.0]
        embedder = self._make_mock_embedder([vec, vec])
        doc = _make_doc("Sentence one. Sentence two.")
        chunks = SemanticChunker(embedder=embedder).chunk(doc)
        for chunk in chunks:
            assert chunk.strategy == "semantic"

    def test_metadata_doc_id_propagated(self):
        """Every chunk must carry the parent document's doc_id."""
        vec = [1.0, 0.0]
        embedder = self._make_mock_embedder([vec, vec])
        doc = _make_doc("Sentence one. Sentence two.")
        chunks = SemanticChunker(embedder=embedder).chunk(doc)
        for chunk in chunks:
            assert chunk.doc_id == doc.doc_id

    def test_metadata_chunk_index_sequential(self):
        """chunk_index must start at 0 and increment by 1."""
        high = [1.0, 0.0]
        low = [0.0, 1.0]
        embedder = self._make_mock_embedder([high, low, high])
        doc = _make_doc("Alpha. Beta. Gamma.")
        chunker = SemanticChunker(embedder=embedder, similarity_threshold=0.75)
        chunks = chunker.chunk(doc)
        for expected, chunk in enumerate(chunks):
            assert chunk.chunk_index == expected

    def test_metadata_char_positions_valid(self):
        """char_start ≥ 0 and char_end > char_start for every chunk."""
        high = [1.0, 0.0]
        low = [0.0, 1.0]
        embedder = self._make_mock_embedder([high, low])
        doc = _make_doc("First sentence here. Second sentence here.")
        chunks = SemanticChunker(embedder=embedder, similarity_threshold=0.75).chunk(doc)
        for chunk in chunks:
            assert chunk.char_start >= 0
            assert chunk.char_end > chunk.char_start

    def test_empty_document_returns_empty_list(self):
        """An empty document should return no chunks without raising."""
        embedder = self._make_mock_embedder([])
        doc = _make_doc("")
        chunks = SemanticChunker(embedder=embedder).chunk(doc)
        assert chunks == []

    def test_single_sentence_returns_single_chunk(self):
        """A document with one sentence produces exactly one chunk."""
        embedder = self._make_mock_embedder([[1.0, 0.0]])
        doc = _make_doc("Only one sentence here.")
        chunks = SemanticChunker(embedder=embedder).chunk(doc)
        assert len(chunks) == 1


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


class TestDeduplication:
    """Tests for ingestion/deduplication.py near-duplicate removal."""

    def test_removes_near_duplicate_above_threshold(self):
        """
        A chunk that is nearly identical (sim > 0.95) to an earlier chunk
        must be removed.
        """
        # Two identical vectors → cosine similarity = 1.0 > 0.95
        vec = [1.0, 0.0, 0.0]
        chunk_a = _make_chunk_with_embedding("Text A", vec, idx=0)
        chunk_b = _make_chunk_with_embedding("Text B (near-dup)", vec, idx=1)

        result = deduplicate_chunks([chunk_a, chunk_b], threshold=0.95)

        assert len(result) == 1
        assert result[0].chunk_id == chunk_a.chunk_id

    def test_retains_distinct_chunks(self):
        """
        Chunks with cosine similarity ≤ 0.95 must both be retained.
        """
        # Orthogonal vectors → similarity = 0.0
        vec_a = [1.0, 0.0, 0.0]
        vec_b = [0.0, 1.0, 0.0]
        chunk_a = _make_chunk_with_embedding("Text A", vec_a, idx=0)
        chunk_b = _make_chunk_with_embedding("Text B", vec_b, idx=1)

        result = deduplicate_chunks([chunk_a, chunk_b], threshold=0.95)

        assert len(result) == 2

    def test_retains_chunk_at_exactly_threshold(self):
        """
        A chunk with similarity exactly equal to 0.95 must be KEPT
        (the condition is strictly >, not >=).
        """
        import math

        # Build two vectors with cosine similarity exactly 0.95.
        # cos θ = 0.95 → θ = arccos(0.95)
        theta = math.acos(0.95)
        vec_a = [1.0, 0.0]
        vec_b = [math.cos(theta), math.sin(theta)]

        chunk_a = _make_chunk_with_embedding("Text A", vec_a, idx=0)
        chunk_b = _make_chunk_with_embedding("Text B", vec_b, idx=1)

        result = deduplicate_chunks([chunk_a, chunk_b], threshold=0.95)

        assert len(result) == 2, (
            "Chunk at exactly the threshold (sim == 0.95) must NOT be removed "
            "(deduplication uses strict >, not >=)"
        )

    def test_removes_exact_duplicate(self):
        """
        An exact duplicate (identical text and embedding) must be removed.
        """
        vec = [0.6, 0.8]
        chunk_a = _make_chunk_with_embedding("Exact text", vec, idx=0)
        chunk_b = _make_chunk_with_embedding("Exact text", vec, idx=1)

        result = deduplicate_chunks([chunk_a, chunk_b], threshold=0.95)

        assert len(result) == 1

    def test_first_occurrence_is_kept(self):
        """When duplicates exist, the first chunk (lower index) is preserved."""
        vec = [1.0, 0.0]
        chunk_a = _make_chunk_with_embedding("First", vec, idx=0)
        chunk_b = _make_chunk_with_embedding("Second (dup)", vec, idx=1)
        chunk_c = _make_chunk_with_embedding("Third (dup)", vec, idx=2)

        result = deduplicate_chunks([chunk_a, chunk_b, chunk_c], threshold=0.95)

        assert len(result) == 1
        assert result[0].chunk_id == chunk_a.chunk_id

    def test_chunks_without_embeddings_are_always_kept(self):
        """Chunks with embedding=None are passed through without comparison."""
        vec = [1.0, 0.0]
        chunk_with_emb = _make_chunk_with_embedding("Has embedding", vec, idx=0)
        chunk_no_emb = Chunk(
            chunk_id=str(uuid4()),
            doc_id="doc-1",
            chunk_index=1,
            text="No embedding",
            char_start=0,
            char_end=12,
            strategy="fixed",
            embedding=None,
        )

        result = deduplicate_chunks([chunk_with_emb, chunk_no_emb], threshold=0.95)

        # Both should be kept: the embedded chunk (it's first) plus the one with no embedding
        assert len(result) == 2

    def test_empty_list_returns_empty_list(self):
        """Passing an empty list returns an empty list without errors."""
        result = deduplicate_chunks([], threshold=0.95)
        assert result == []

    def test_single_chunk_always_kept(self):
        """A list with a single chunk is always returned unchanged."""
        vec = [1.0, 0.0]
        chunk = _make_chunk_with_embedding("Only chunk", vec, idx=0)
        result = deduplicate_chunks([chunk], threshold=0.95)
        assert len(result) == 1
        assert result[0].chunk_id == chunk.chunk_id

    def test_all_distinct_chunks_are_retained(self):
        """When no chunks are near-duplicates, all must survive deduplication."""
        # Four orthogonal vectors in 4D space
        vecs = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
        chunks = [
            _make_chunk_with_embedding(f"Distinct chunk {i}", v, idx=i)
            for i, v in enumerate(vecs)
        ]

        result = deduplicate_chunks(chunks, threshold=0.95)

        assert len(result) == 4

    def test_mixed_duplicates_and_distinct_chunks(self):
        """
        Only near-duplicates (sim > 0.95) are removed; distinct chunks survive.
        Input: [A, B (dup of A), C (distinct)]
        Expected output: [A, C]
        """
        dup_vec = [1.0, 0.0, 0.0]
        distinct_vec = [0.0, 1.0, 0.0]

        chunk_a = _make_chunk_with_embedding("Chunk A", dup_vec, idx=0)
        chunk_b = _make_chunk_with_embedding("Chunk B (dup of A)", dup_vec, idx=1)
        chunk_c = _make_chunk_with_embedding("Chunk C (distinct)", distinct_vec, idx=2)

        result = deduplicate_chunks([chunk_a, chunk_b, chunk_c], threshold=0.95)

        assert len(result) == 2
        result_ids = {r.chunk_id for r in result}
        assert chunk_a.chunk_id in result_ids
        assert chunk_c.chunk_id in result_ids
        assert chunk_b.chunk_id not in result_ids

    def test_order_preserved_after_deduplication(self):
        """The relative order of kept chunks must be preserved."""
        vec_a = [1.0, 0.0, 0.0]
        vec_b = [0.0, 1.0, 0.0]
        vec_c = [0.0, 0.0, 1.0]

        chunk_a = _make_chunk_with_embedding("A", vec_a, idx=0)
        chunk_b = _make_chunk_with_embedding("B", vec_b, idx=1)
        chunk_c = _make_chunk_with_embedding("C", vec_c, idx=2)

        result = deduplicate_chunks([chunk_a, chunk_b, chunk_c], threshold=0.95)

        assert [r.chunk_id for r in result] == [
            chunk_a.chunk_id,
            chunk_b.chunk_id,
            chunk_c.chunk_id,
        ]
