"""API routes for the Hybrid RAG pipeline."""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

from fastapi import APIRouter, File, UploadFile

from api.models import (
    AskRequest,
    AskResponse,
    CitationResponse,
    IngestResponse,
    LatencyBreakdown,
    TraceChunk,
)
from config import PipelineConfig
from generation.citation_verifier import CitationVerifier
from generation.confidence import compute_confidence_score
from generation.generator import Generator
from ingestion.pipeline import run_ingestion
from retrieval.bm25_index import BM25Index
from retrieval.dense_retriever import DenseRetriever
from retrieval.embedder import Embedder
from retrieval.rrf_fusion import reciprocal_rank_fusion
from retrieval.reranker import CrossEncoderReranker
from retrieval.sparse_retriever import SparseRetriever
from retrieval.vector_store import VectorStoreClient

router = APIRouter(prefix="/v1")


def _config() -> PipelineConfig:
    return PipelineConfig()


@router.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest) -> AskResponse:
    config = _config()
    total_start = time.perf_counter()
    latency = LatencyBreakdown()

    embedder = Embedder(
        openai_api_key=config.openai_api_key,
        openai_model=config.openai_embedding_model,
        fallback_model=config.fallback_embedding_model,
        embedding_workers=config.embedding_workers,
        batch_size=config.embedding_batch_size,
    )
    vector_store = VectorStoreClient(config.qdrant_host, config.qdrant_port, config.qdrant_collection, embedder.embedding_dim)
    bm25 = BM25Index()
    if config.bm25_index_path.exists():
        bm25.load(config.bm25_index_path)

    start = time.perf_counter()
    query_vector = await embedder.embed_query(request.query)
    latency.embed_query = _ms(start)

    start = time.perf_counter()
    dense_results = await DenseRetriever(vector_store).search(query_vector, top_k=config.retrieval_top_k)
    latency.dense_retrieval = _ms(start)

    sparse_results = []
    if not request.dense_only and bm25.is_built:
        start = time.perf_counter()
        sparse_results = await SparseRetriever(bm25).search(request.query, top_k=config.retrieval_top_k)
        latency.sparse_retrieval = _ms(start)

    start = time.perf_counter()
    candidates = dense_results if request.dense_only else reciprocal_rank_fusion(dense_results, sparse_results, alpha=request.rrf_weight)
    latency.rrf_fusion = _ms(start)

    start = time.perf_counter()
    reranked = await CrossEncoderReranker(config.reranker_model, config.reranker_mode).rerank(request.query, candidates, top_k=5)
    latency.reranking = _ms(start)

    start = time.perf_counter()
    generation = await Generator(config.openai_chat_model).generate(request.query, reranked, config.min_relevance_threshold)
    latency.generation = _ms(start)

    start = time.perf_counter()
    citations = await CitationVerifier().verify_citations(generation.citations)
    latency.citation_verification = _ms(start)

    confidence = compute_confidence_score([r.score for r in reranked[:5]], citations, completeness_score=1.0 if generation.is_grounded else 0.0)
    latency.total = _ms(total_start)

    return AskResponse(
        answer=generation.answer,
        citations=[CitationResponse(index=c.index, chunk_text=c.chunk_text, source=c.source, verification_status=c.verification_status) for c in citations],
        confidence_score=confidence,
        latency_ms=latency,
        retrieved_chunks=[
            TraceChunk(
                rank=i,
                text=r.chunk.text,
                source=r.chunk.filename or r.chunk.doc_id,
                relevance_score=r.score,
                reranker_score=r.reranker_score,
                strategy=r.chunk.strategy,
            )
            for i, r in enumerate(reranked, start=1)
        ],
    )


@router.get("/documents")
async def documents():
    config = _config()
    embedder = Embedder(openai_api_key=config.openai_api_key)
    vector_store = VectorStoreClient(config.qdrant_host, config.qdrant_port, config.qdrant_collection, embedder.embedding_dim)
    return await vector_store.list_documents()


@router.post("/ingest", response_model=IngestResponse)
async def ingest(file: UploadFile = File(...)) -> IngestResponse:
    config = _config()
    suffix = Path(file.filename or "upload.txt").suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as fh:
        path = Path(fh.name)
        fh.write(await file.read())
    result = await run_ingestion([path], config)
    return IngestResponse(
        total_documents=result.total_documents,
        total_chunks=result.total_chunks,
        failed_files=[f.model_dump() for f in result.failed_files],
        wall_clock_seconds=result.wall_clock_seconds,
    )


def _ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000
