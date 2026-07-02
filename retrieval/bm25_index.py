"""
BM25 sparse retrieval index backed by rank_bm25.

Tokenization uses simple whitespace + punctuation splitting. The index
stores a parallel list of Chunk objects so that BM25 scores can be mapped
back to the originating chunk at query time.

Persistence is handled via pickle so both the BM25Okapi object and the
chunk list are serialised together to a single file.
"""

from __future__ import annotations

import logging
import pickle
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

from ingestion.models import Chunk, ScoredChunk

logger = logging.getLogger(__name__)

# Matches one or more whitespace characters OR one or more punctuation chars.
# Using a negative character class keeps the pattern simple and avoids
# importing a full NLP library just for tokenisation.
_TOKEN_SPLIT_RE = re.compile(r"[\s\W]+")


def _tokenize(text: str) -> list[str]:
    """Split text on whitespace and punctuation; discard empty tokens."""
    tokens = _TOKEN_SPLIT_RE.split(text.lower())
    return [t for t in tokens if t]


class BM25Index:
    """
    Wrapper around BM25Okapi with disk persistence.

    Usage::

        index = BM25Index()
        index.build(chunks)
        results = index.search("my query", top_k=5)
        index.save(Path("./data/bm25_index.pkl"))

        # Later, in a fresh process:
        index2 = BM25Index()
        index2.load(Path("./data/bm25_index.pkl"))
        results2 = index2.search("my query", top_k=5)
    """

    def __init__(self) -> None:
        self._bm25: BM25Okapi | None = None
        self._chunks: list[Chunk] = []
        self._built = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self, chunks: list[Chunk]) -> None:
        """Tokenise *chunks* and build the BM25Okapi index in memory.

        The method is idempotent — calling it again replaces the
        existing index with a fresh one built from the supplied chunks.

        Args:
            chunks: The corpus to index.  Every chunk must have a
                non-empty ``text`` field.
        """
        if not chunks:
            logger.warning("BM25Index.build called with an empty chunk list; index will be empty.")
            self._bm25 = BM25Okapi([[]])
            self._chunks = []
            self._built = True
            return

        tokenised_corpus = [_tokenize(chunk.text) for chunk in chunks]
        self._bm25 = BM25Okapi(tokenised_corpus)
        self._chunks = list(chunks)
        self._built = True
        logger.info("BM25 index built with %d chunks.", len(self._chunks))

    def search(self, query: str, top_k: int = 10) -> list[ScoredChunk]:
        """Return the *top_k* chunks most relevant to *query*.

        Scores are raw BM25 scores (higher is more relevant).  Results
        are sorted in descending score order and wrapped in
        :class:`~ingestion.models.ScoredChunk` objects with ``rank``
        set to 1-based position and ``sparse_score`` mirroring ``score``.

        Args:
            query: The search query string.
            top_k: Maximum number of results to return.

        Returns:
            A list of at most *top_k* :class:`~ingestion.models.ScoredChunk`
            objects sorted by descending BM25 score.

        Raises:
            RuntimeError: If the index has not been built or loaded yet.
        """
        if self._bm25 is None or not self._built:
            raise RuntimeError(
                "BM25Index has not been built or loaded. "
                "Call build() or load() before searching."
            )
        if not self._chunks:
            return []

        query_tokens = _tokenize(query)
        if not query_tokens:
            logger.warning("BM25Index.search received an empty query after tokenisation.")
            return []

        scores: list[float] = self._bm25.get_scores(query_tokens).tolist()

        # Pair each score with its position, sort descending, take top_k.
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]

        results: list[ScoredChunk] = []
        for rank, (idx, score) in enumerate(ranked, start=1):
            results.append(
                ScoredChunk(
                    chunk=self._chunks[idx],
                    score=score,
                    rank=rank,
                    sparse_score=score,
                )
            )

        return results

    def save(self, path: Path) -> None:
        """Serialise the index and chunk list to *path* using pickle.

        The parent directory is created automatically if it does not
        exist.

        Args:
            path: Destination file path (e.g. ``Path("./data/bm25_index.pkl")``).

        Raises:
            RuntimeError: If the index has not been built yet.
        """
        if self._bm25 is None:
            raise RuntimeError(
                "Cannot save an uninitialised BM25Index. Call build() first."
            )

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "bm25": self._bm25,
            "chunks": self._chunks,
        }
        with path.open("wb") as fh:
            pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)

        logger.info("BM25 index saved to %s (%d chunks).", path, len(self._chunks))

    def load(self, path: Path) -> None:
        """Restore the index and chunk list from a pickle file.

        Args:
            path: Path to the pickle file written by :meth:`save`.

        Raises:
            FileNotFoundError: If *path* does not exist.
            ValueError: If the pickle payload is missing expected keys.
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"BM25 index file not found: {path}")

        with path.open("rb") as fh:
            payload: dict = pickle.load(fh)  # noqa: S301 — trusted internal artifact

        if "bm25" not in payload or "chunks" not in payload:
            raise ValueError(
                f"Invalid BM25 index file at {path}: "
                "expected keys 'bm25' and 'chunks'."
            )

        self._bm25 = payload["bm25"]
        self._chunks = payload["chunks"]
        self._built = True
        logger.info("BM25 index loaded from %s (%d chunks).", path, len(self._chunks))

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    @property
    def is_built(self) -> bool:
        """True when the index has been built or loaded."""
        return self._bm25 is not None and self._built

    def __len__(self) -> int:
        return len(self._chunks)
