"""Pydantic models for the FastAPI layer."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    query: str = Field(..., min_length=1)
    dense_only: bool = False
    rrf_weight: float = Field(0.7, ge=0.0, le=1.0)
    chunking_strategy: Literal["fixed", "structural", "semantic"] = "structural"


class CitationResponse(BaseModel):
    index: int
    chunk_text: str
    source: str
    verification_status: str | None = None


class TraceChunk(BaseModel):
    rank: int
    text: str
    source: str
    relevance_score: float
    reranker_score: float | None = None
    strategy: str


class LatencyBreakdown(BaseModel):
    embed_query: float = 0.0
    dense_retrieval: float = 0.0
    sparse_retrieval: float = 0.0
    rrf_fusion: float = 0.0
    reranking: float = 0.0
    generation: float = 0.0
    citation_verification: float = 0.0
    total: float = 0.0


class AskResponse(BaseModel):
    answer: str
    citations: list[CitationResponse]
    confidence_score: float
    latency_ms: LatencyBreakdown
    retrieved_chunks: list[TraceChunk] = []


class DocumentMeta(BaseModel):
    doc_id: str
    filename: str
    format: str
    chunk_count: int
    ingested_at: datetime


class IngestResponse(BaseModel):
    total_documents: int
    total_chunks: int
    failed_files: list[dict]
    wall_clock_seconds: float
