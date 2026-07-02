"""
Core domain models for the Hybrid RAG ingestion pipeline.

These models are shared across the ingestion, retrieval, generation, and api packages.
"""

from datetime import datetime
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field


class ParsedDocument(BaseModel):
    doc_id: str = Field(default_factory=lambda: str(uuid4()))
    filename: str
    format: Literal["md", "txt", "html", "pdf"]
    content: str
    source_url: str | None = None
    ingested_at: datetime = Field(default_factory=datetime.utcnow)


class Chunk(BaseModel):
    chunk_id: str = Field(default_factory=lambda: str(uuid4()))
    doc_id: str
    filename: str | None = None
    format: str | None = None
    ingested_at: datetime | None = None
    chunk_index: int
    text: str
    char_start: int
    char_end: int
    strategy: Literal["fixed", "structural", "semantic"]
    embedding: list[float] | None = None


class ScoredChunk(BaseModel):
    chunk: Chunk
    score: float
    rank: int | None = None
    dense_score: float | None = None
    sparse_score: float | None = None
    reranker_score: float | None = None


class Citation(BaseModel):
    index: int
    chunk_id: str
    chunk_text: str
    source: str
    verification_status: Literal["supported", "partial", "unsupported"] | None = None
    claim: str | None = None


class GenerationResult(BaseModel):
    answer: str
    citations: list[Citation]
    is_grounded: bool
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reason: str | None = None


class FailedFile(BaseModel):
    filename: str
    error: str
    error_type: str


class IngestionResult(BaseModel):
    total_documents: int
    total_chunks: int
    failed_files: list[FailedFile]
    wall_clock_seconds: float


class DocumentMeta(BaseModel):
    doc_id: str
    filename: str
    format: str
    chunk_count: int
    ingested_at: datetime
