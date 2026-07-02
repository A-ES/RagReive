"""
Pipeline configuration using Pydantic Settings.

All settings can be overridden via environment variables or a `.env` file.
Field names map directly to environment variable names (uppercased).

Example `.env`:
    OPENAI_API_KEY=sk-...
    QDRANT_HOST=localhost
    QDRANT_PORT=6333
"""

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class PipelineConfig(BaseSettings):
    # Embedder
    openai_api_key: str | None = None
    openai_embedding_model: str = "text-embedding-3-small"
    fallback_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_batch_size: int = 64
    embedding_workers: int = 8

    # Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "hybrid_rag_chunks"

    # BM25
    bm25_index_path: Path = Path("./data/bm25_index.pkl")

    # Chunking
    default_chunking_strategy: str = "structural"
    fixed_chunk_size: int = 512
    fixed_chunk_overlap: int = 64
    semantic_similarity_threshold: float = 0.75

    # Retrieval
    default_rrf_weight: float = 0.7
    retrieval_top_k: int = 10
    dedup_similarity_threshold: float = 0.95

    # Generation
    openai_chat_model: str = "gpt-4o-mini"
    min_relevance_threshold: float = 0.3
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_mode: Literal["cross_encoder", "llm_judge"] = "cross_encoder"

    # Confidence weights
    confidence_retrieval_weight: float = 0.4
    confidence_citation_weight: float = 0.4
    confidence_completeness_weight: float = 0.2

    model_config = SettingsConfigDict(env_file=".env")
