"""
Chunking strategies for the Hybrid RAG ingestion pipeline.

Three strategies are provided:

- ``FixedChunker``     — wraps LangChain ``CharacterTextSplitter``.
                         Default: 512 chars, 64-char overlap.
- ``StructuralChunker``— uses LangChain ``MarkdownHeaderTextSplitter`` for
                         Markdown documents; paragraph/code-fence splitting
                         for all other formats.
- ``SemanticChunker``  — encodes sentences individually, then greedily merges
                         adjacent sentences until cosine similarity between
                         consecutive sentence embeddings drops below a
                         configurable threshold (default 0.75).

All chunkers attach full ``Chunk`` metadata: ``doc_id``, ``chunk_index``,
``char_start``, ``char_end``, and ``strategy``.

Factory function ``get_chunker(strategy, config)`` returns the appropriate
``BaseChunker`` instance configured from ``PipelineConfig``.
"""

from __future__ import annotations

import asyncio
import logging
import re
from abc import ABC, abstractmethod
from typing import Any
from uuid import uuid4

from ingestion.models import Chunk, ParsedDocument

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors (pure Python, no numpy dep)."""
    try:
        import numpy as np  # type: ignore[import-untyped]

        va = np.array(a, dtype=np.float32)
        vb = np.array(b, dtype=np.float32)
        norm_a = np.linalg.norm(va)
        norm_b = np.linalg.norm(vb)
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return float(np.dot(va, vb) / (norm_a * norm_b))
    except ImportError:
        # Pure-Python fallback (slow, only used if numpy is missing)
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot / (norm_a * norm_b)


def _find_char_start(content: str, text: str, search_from: int = 0) -> int:
    """
    Return the character index of *text* within *content*, starting the search
    at *search_from*.  Returns -1 if not found (caller must handle).
    """
    return content.find(text, search_from)


def _make_chunk(
    *,
    doc: ParsedDocument,
    text: str,
    index: int,
    char_start: int,
    char_end: int,
    strategy: str,
) -> Chunk:
    """Construct a ``Chunk`` with all required metadata fields."""
    return Chunk(
        chunk_id=str(uuid4()),
        doc_id=doc.doc_id,
        filename=doc.filename,
        format=doc.format,
        ingested_at=doc.ingested_at,
        chunk_index=index,
        text=text,
        char_start=char_start,
        char_end=char_end,
        strategy=strategy,  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class BaseChunker(ABC):
    """Abstract base class for all chunking strategies."""

    @abstractmethod
    def chunk(self, doc: ParsedDocument) -> list[Chunk]:
        """Split *doc* into a list of ``Chunk`` objects with full metadata."""
        ...


# ---------------------------------------------------------------------------
# FixedChunker
# ---------------------------------------------------------------------------


class FixedChunker(BaseChunker):
    """
    Splits documents into fixed-size character chunks using LangChain's
    ``CharacterTextSplitter``.

    Parameters
    ----------
    chunk_size:
        Maximum number of characters per chunk (default 512).
    chunk_overlap:
        Number of characters to overlap between consecutive chunks (default 64).
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, doc: ParsedDocument) -> list[Chunk]:
        from langchain_text_splitters import CharacterTextSplitter  # type: ignore[import-untyped]

        splitter = CharacterTextSplitter(
            separator="",           # split on any character boundary
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )

        texts = splitter.split_text(doc.content)

        if not texts:
            return []

        chunks: list[Chunk] = []
        search_from = 0

        for idx, text in enumerate(texts):
            char_start = _find_char_start(doc.content, text, search_from)
            if char_start == -1:
                # Fallback: best-effort position for overlapping edge cases
                char_start = max(0, search_from - self.chunk_overlap)

            char_end = char_start + len(text)

            chunks.append(
                _make_chunk(
                    doc=doc,
                    text=text,
                    index=idx,
                    char_start=char_start,
                    char_end=char_end,
                    strategy="fixed",
                )
            )

            # Advance search cursor; allow look-back for overlap window
            search_from = max(search_from, char_end - self.chunk_overlap)

        return chunks


# ---------------------------------------------------------------------------
# StructuralChunker
# ---------------------------------------------------------------------------


class StructuralChunker(BaseChunker):
    """
    Splits documents at structural boundaries.

    - For Markdown (``format == "md"``): uses LangChain's
      ``MarkdownHeaderTextSplitter`` to split at heading boundaries.
    - For all other formats: splits on paragraph boundaries (double newlines
      ``\\n\\n``) and code fences (triple back-ticks).

    Empty chunks (after stripping whitespace) are discarded.
    """

    # Headers to split on for Markdown (from h1 to h4)
    _MD_HEADERS = [
        ("#", "h1"),
        ("##", "h2"),
        ("###", "h3"),
        ("####", "h4"),
    ]

    # Regex that matches paragraph breaks *or* code-fence delimiters
    # A "split point" is either:
    #   - two or more consecutive newlines  →  paragraph boundary
    #   - a line that is exactly ```         →  code fence boundary
    _SPLIT_RE = re.compile(r"(?m)(?:\n\n+|^```)")

    def chunk(self, doc: ParsedDocument) -> list[Chunk]:
        if doc.format == "md":
            return self._chunk_markdown(doc)
        return self._chunk_generic(doc)

    # ------------------------------------------------------------------
    # Markdown path
    # ------------------------------------------------------------------

    def _chunk_markdown(self, doc: ParsedDocument) -> list[Chunk]:
        from langchain_text_splitters import MarkdownHeaderTextSplitter  # type: ignore[import-untyped]

        splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=self._MD_HEADERS,
            strip_headers=False,
        )

        lc_docs = splitter.split_text(doc.content)

        if not lc_docs:
            return []

        chunks: list[Chunk] = []
        search_from = 0
        content = doc.content

        for idx, lc_doc in enumerate(lc_docs):
            # LangChain may append "  \n" (two trailing spaces) after heading
            # lines, which don't exist in the original.  Normalise by stripping
            # trailing whitespace from every line, then stripping outer space.
            raw_page = lc_doc.page_content
            text = "\n".join(line.rstrip() for line in raw_page.splitlines()).strip()
            if not text:
                continue

            # Locate the chunk's position inside the original content.
            # Strategy: find the first significant line (usually the heading or
            # the first sentence) starting from search_from.
            char_start = self._locate_chunk_in_content(content, text, search_from)
            if char_start == -1:
                logger.warning(
                    "StructuralChunker (md): could not locate chunk text in document '%s' (idx %d); skipping.",
                    doc.filename,
                    idx,
                )
                continue

            char_end = char_start + len(text)
            chunks.append(
                _make_chunk(
                    doc=doc,
                    text=text,
                    index=len(chunks),
                    char_start=char_start,
                    char_end=char_end,
                    strategy="structural",
                )
            )
            search_from = char_end

        return chunks

    @staticmethod
    def _locate_chunk_in_content(content: str, chunk_text: str, search_from: int) -> int:
        """
        Locate *chunk_text* inside *content* starting at *search_from*.

        Because LangChain's splitter may lightly reformat whitespace (e.g.,
        trailing spaces after headings), we first try an exact search, then fall
        back to finding the first non-empty line of the chunk in the original.
        """
        # 1. Exact search from search_from
        pos = content.find(chunk_text, search_from)
        if pos != -1:
            return pos

        # 2. Exact search from beginning
        pos = content.find(chunk_text, 0)
        if pos != -1:
            return pos

        # 3. Anchor on the first non-empty line (handles whitespace differences)
        first_line = next(
            (line.strip() for line in chunk_text.splitlines() if line.strip()),
            None,
        )
        if first_line:
            pos = content.find(first_line, search_from)
            if pos != -1:
                return pos
            pos = content.find(first_line, 0)
            if pos != -1:
                return pos

        return -1

    # ------------------------------------------------------------------
    # Generic (non-markdown) path
    # ------------------------------------------------------------------

    def _chunk_generic(self, doc: ParsedDocument) -> list[Chunk]:
        """Split on paragraph boundaries (\\n\\n) and code fences (```)."""
        content = doc.content
        segments = self._split_on_boundaries(content)

        chunks: list[Chunk] = []
        search_from = 0

        for raw_segment in segments:
            text = raw_segment.strip()
            if not text:
                continue

            char_start = _find_char_start(content, text, search_from)
            if char_start == -1:
                char_start = _find_char_start(content, text, 0)
            if char_start == -1:
                logger.warning(
                    "StructuralChunker (generic): could not locate segment in document '%s'; skipping.",
                    doc.filename,
                )
                continue

            char_end = char_start + len(text)
            chunks.append(
                _make_chunk(
                    doc=doc,
                    text=text,
                    index=len(chunks),
                    char_start=char_start,
                    char_end=char_end,
                    strategy="structural",
                )
            )
            search_from = char_end

        return chunks

    def _split_on_boundaries(self, content: str) -> list[str]:
        """
        Split *content* at paragraph boundaries and code fences.

        The code-fence splitter preserves fence context: lines before the
        opening ``` become one segment, the fenced block (including the
        closing ```) becomes the next segment.
        """
        # We build segments manually to handle code fences correctly.
        segments: list[str] = []
        # Split on double newlines first to get paragraph-like pieces
        para_pieces = re.split(r"\n\n+", content)

        for piece in para_pieces:
            # Further split each paragraph piece on code-fence boundaries
            fence_pieces = re.split(r"(?m)^```", piece)
            for fp in fence_pieces:
                if fp.strip():
                    segments.append(fp)

        return segments


# ---------------------------------------------------------------------------
# SemanticChunker
# ---------------------------------------------------------------------------


class SemanticChunker(BaseChunker):
    """
    Splits documents at semantic topic boundaries.

    Algorithm:
    1. Split the document content into sentences using a simple regex.
    2. Embed all sentences in a single batch via the provided ``embedder``.
    3. Greedily merge adjacent sentences into a growing chunk as long as the
       cosine similarity between consecutive sentence embeddings is ≥
       ``similarity_threshold``.
    4. When similarity drops below the threshold, flush the current chunk and
       start a new one.

    Parameters
    ----------
    embedder:
        Any object with an ``embed_batch(texts: list[str]) -> list[list[float]]``
        async method (typically an instance of ``retrieval.embedder.Embedder``).
        Accepted as ``Any`` to avoid circular imports.
    similarity_threshold:
        Cosine similarity threshold below which a sentence boundary becomes a
        chunk boundary (default 0.75).
    """

    # Sentence splitter: split after `.`, `!`, or `?` followed by whitespace
    # or end-of-string.  Keeps the delimiter with the preceding sentence.
    _SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")

    def __init__(
        self,
        embedder: Any,
        similarity_threshold: float = 0.75,
    ) -> None:
        self.embedder = embedder
        self.similarity_threshold = similarity_threshold

    def chunk(self, doc: ParsedDocument) -> list[Chunk]:
        sentences = self._split_sentences(doc.content)

        if not sentences:
            return []

        # Embed all sentences synchronously (chunking is a sync operation)
        embeddings = self._embed_sentences(sentences)

        if not embeddings or len(embeddings) != len(sentences):
            logger.warning(
                "SemanticChunker: embedder returned %d embeddings for %d sentences in '%s'; "
                "falling back to one-sentence-per-chunk.",
                len(embeddings) if embeddings else 0,
                len(sentences),
                doc.filename,
            )
            embeddings = None  # type: ignore[assignment]

        return self._build_chunks(doc, sentences, embeddings)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _split_sentences(self, content: str) -> list[str]:
        """Split *content* into individual sentences."""
        raw = self._SENTENCE_RE.split(content.strip())
        return [s.strip() for s in raw if s.strip()]

    def _embed_sentences(self, sentences: list[str]) -> list[list[float]] | None:
        """Embed *sentences* synchronously by running the async method.

        We need to bridge from sync code into an async coroutine.  Three cases:

        1. No running event loop (typical script / pytest without asyncio mode):
           ``asyncio.run()`` creates a fresh loop, runs the coroutine, and tears
           the loop down.

        2. A running event loop exists (e.g. inside ``pytest-asyncio``,
           ``IPython``, ``Jupyter``):
           ``asyncio.run()`` would raise ``RuntimeError: This event loop is
           already running``.  We fall back to a worker thread where we can
           safely call ``asyncio.run()``.
        """
        import concurrent.futures

        coro = self.embedder.embed_batch(sentences)

        try:
            # Check whether we are already inside a running loop.
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None

        try:
            if running_loop is not None:
                # We are inside a running event loop — offload to a thread.
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    future = pool.submit(asyncio.run, coro)
                    return future.result()
            else:
                return asyncio.run(coro)
        except Exception as exc:
            logger.error(
                "SemanticChunker: embedding failed: %s", exc, exc_info=True
            )
            return None

    def _build_chunks(
        self,
        doc: ParsedDocument,
        sentences: list[str],
        embeddings: list[list[float]] | None,
    ) -> list[Chunk]:
        """
        Greedily merge sentences into chunks.

        A new chunk is started when:
        - embeddings are available AND the cosine similarity between
          sentence[i-1] and sentence[i] drops below ``similarity_threshold``, OR
        - embeddings are unavailable (each sentence becomes its own chunk).
        """
        chunks: list[Chunk] = []
        current_sentences: list[str] = []
        search_from = 0

        def _flush(sents: list[str]) -> None:
            nonlocal search_from
            if not sents:
                return
            text = " ".join(sents)
            char_start = _find_char_start(doc.content, sents[0], search_from)
            if char_start == -1:
                char_start = _find_char_start(doc.content, sents[0], 0)
            if char_start == -1:
                logger.warning(
                    "SemanticChunker: could not locate sentence in document '%s'; using approximate offset.",
                    doc.filename,
                )
                char_start = search_from

            char_end = char_start + len(text)
            chunks.append(
                _make_chunk(
                    doc=doc,
                    text=text,
                    index=len(chunks),
                    char_start=char_start,
                    char_end=char_end,
                    strategy="semantic",
                )
            )
            search_from = char_end

        for i, sentence in enumerate(sentences):
            if i == 0:
                current_sentences.append(sentence)
                continue

            # Determine whether to split before this sentence
            should_split = False
            if embeddings is None:
                # No embeddings — treat every sentence as its own chunk
                should_split = True
            else:
                sim = _cosine_similarity(embeddings[i - 1], embeddings[i])
                if sim < self.similarity_threshold:
                    should_split = True

            if should_split:
                _flush(current_sentences)
                current_sentences = [sentence]
            else:
                current_sentences.append(sentence)

        # Flush the last group
        _flush(current_sentences)

        return chunks


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------


def get_chunker(strategy: str, config: Any, embedder: Any = None) -> BaseChunker:
    """
    Return a configured ``BaseChunker`` for the given *strategy*.

    Parameters
    ----------
    strategy:
        One of ``"fixed"``, ``"structural"``, or ``"semantic"``.
    config:
        A ``PipelineConfig`` instance providing chunking parameters.
    embedder:
        Required when *strategy* is ``"semantic"``; an object with an async
        ``embed_batch`` method.  Ignored for other strategies.

    Raises
    ------
    ValueError:
        If *strategy* is not one of the recognised values.
    ValueError:
        If *strategy* is ``"semantic"`` and *embedder* is ``None``.
    """
    if strategy == "fixed":
        return FixedChunker(
            chunk_size=config.fixed_chunk_size,
            chunk_overlap=config.fixed_chunk_overlap,
        )

    if strategy == "structural":
        return StructuralChunker()

    if strategy == "semantic":
        if embedder is None:
            raise ValueError(
                "SemanticChunker requires an 'embedder' argument. "
                "Pass an Embedder instance to get_chunker()."
            )
        return SemanticChunker(
            embedder=embedder,
            similarity_threshold=config.semantic_similarity_threshold,
        )

    raise ValueError(
        f"Unknown chunking strategy '{strategy}'. "
        "Valid values: 'fixed', 'structural', 'semantic'."
    )
