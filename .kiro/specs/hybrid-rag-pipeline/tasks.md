# Implementation Plan: Hybrid RAG Pipeline

## Overview

Implement a production-grade Hybrid RAG pipeline organized as clean Python packages (`ingestion`, `retrieval`, `generation`, `eval`, `api`) with a Streamlit dashboard and full Docker Compose orchestration. Tasks build incrementally: project scaffold → ingestion → retrieval → generation → evaluation → API → dashboard → containerization.

## Tasks

- [x] 1. Scaffold project structure and shared configuration
  - [x] 1.1 Create Python package directories with `__init__.py` files
    - Create directories: `ingestion/`, `retrieval/`, `generation/`, `eval/`, `api/`, `dashboard/`, `data/corpus/`, `data/eval_cache/`
    - Add `__init__.py` to each package directory
    - _Requirements: 11.1_
  - [x] 1.2 Create `config.py` with `PipelineConfig` Pydantic Settings model
    - Implement `PipelineConfig` using `pydantic-settings` with all fields from design: embedder, Qdrant, BM25, chunking, retrieval, generation, and confidence weight settings
    - Support `.env` file loading via `SettingsConfigDict(env_file=".env")`
    - _Requirements: 10.4, 3.2, 2.5_
  - [x] 1.3 Create `pyproject.toml` with pinned dependencies and `ruff` linter config
    - Pin all production dependencies: `fastapi`, `uvicorn`, `qdrant-client`, `rank-bm25`, `sentence-transformers`, `openai`, `pdfplumber`, `pypdf`, `beautifulsoup4`, `markdownify`, `langchain`, `pydantic`, `pydantic-settings`, `streamlit`, `numpy`, `pytest`, `ruff`
    - Add `[tool.ruff]` config section for PEP 8 enforcement
    - _Requirements: 11.3, 11.4_

- [x] 2. Implement document ingestion and normalization
  - [x] 2.1 Create `ingestion/models.py` with `ParsedDocument`, `Chunk`, `FailedFile`, and `IngestionResult` Pydantic models
    - Implement all models exactly as specified in the design's Data Models section
    - _Requirements: 1.1, 1.4, 2.4_
  - [x] 2.2 Implement `ingestion/parsers.py` with format-specific document parsers
    - Implement `MarkdownParser` using `markdownify` or regex to strip frontmatter and convert to plaintext
    - Implement `TextParser` with UTF-8 reading and encoding detection fallback
    - Implement `HtmlParser` using `BeautifulSoup` preserving heading hierarchy
    - Implement `PdfParser` using `pdfplumber` with `pypdf` fallback
    - Implement `DocumentParser` dispatcher that routes by file extension and raises a structured error for unsupported extensions
    - Each parser populates `ParsedDocument` with `doc_id`, `filename`, `format`, `content`, `source_url`, and `ingested_at`
    - _Requirements: 1.1, 1.2_
  - [x] 2.3 Implement `ingestion/chunkers.py` with all three chunking strategies
    - Implement `BaseChunker` ABC with `chunk(doc: ParsedDocument) -> list[Chunk]`
    - Implement `FixedChunker` wrapping `LangChain CharacterTextSplitter` (default: 512 chars, 64 overlap)
    - Implement `StructuralChunker` using `LangChain MarkdownHeaderTextSplitter` for markdown; paragraph/code-fence splitting for other formats
    - Implement `SemanticChunker` that encodes sentences, then greedily merges until cosine similarity drops below threshold (default 0.75)
    - Each chunker attaches full `Chunk` metadata: `doc_id`, `chunk_index`, `char_start`, `char_end`, `strategy`
    - Expose a `get_chunker(strategy: str, config: PipelineConfig) -> BaseChunker` factory function
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_
  - [x] 2.4 Implement `ingestion/deduplication.py` for near-duplicate chunk removal
    - Compute pairwise cosine similarity (batched via numpy) over chunk embeddings
    - Discard any chunk whose similarity to an earlier chunk exceeds `dedup_similarity_threshold` (default 0.95)
    - _Requirements: 2.6_
  - [x] 2.5 Write unit tests for chunking strategies
    - Test `FixedChunker`: verify chunk count, overlap correctness (64-char overlap), and metadata attachment for a known document
    - Test `StructuralChunker`: verify split at headings, paragraphs, and code fences
    - Test `SemanticChunker`: verify chunks are merged until similarity drops, metadata is correct
    - Test `deduplication.py`: verify near-duplicate removal (similarity > 0.95) and retention of distinct chunks
    - _Requirements: 11.2_

- [x] 3. Implement embedding and indexing
  - [x] 3.1 Implement `retrieval/embedder.py` with OpenAI and sentence-transformers support
    - Implement `Embedder` class with `embed_batch` and `embed_query` async methods
    - On init, check for `OPENAI_API_KEY`; if absent, log warning and fall back to `sentence-transformers/all-MiniLM-L6-v2`
    - Use `asyncio.gather` with a `asyncio.Semaphore` (default 8 workers) for parallel batch embedding during ingestion
    - Log average embedding latency per chunk
    - _Requirements: 3.1, 3.2, 3.5, 3.6, 10.4_
  - [x] 3.2 Implement `retrieval/vector_store.py` as a Qdrant client wrapper
    - Implement `VectorStoreClient` with `upsert_chunks`, `search`, and `list_documents` async methods
    - Store chunk UUID as point ID; payload dict contains `{doc_id, filename, format, chunk_index, char_start, char_end, strategy, text}`
    - Create collection on first use with correct vector dimension (1536 for OpenAI, 384 for sentence-transformers)
    - _Requirements: 3.3_
  - [x] 3.3 Implement `retrieval/bm25_index.py` as a `rank_bm25` wrapper with disk persistence
    - Implement `BM25Index` with `build`, `search`, `save`, and `load` methods
    - Tokenize with whitespace + punctuation splitting; store parallel chunk list for score mapping
    - Persist via pickle to `bm25_index_path` from config; load on startup if file exists
    - _Requirements: 3.4_
  - [x] 3.4 Wire ingestion pipeline: parse → chunk → deduplicate → embed → index
    - Create `ingestion/pipeline.py` orchestrating: `DocumentParser` → `get_chunker` → `deduplication` → `Embedder.embed_batch` → `VectorStoreClient.upsert_chunks` + `BM25Index.build/save`
    - Log total wall-clock time and per-chunk embedding latency at completion
    - Return `IngestionResult` with totals and any `FailedFile` entries
    - _Requirements: 1.4, 3.5, 3.6_

- [x] 4. Implement hybrid retrieval and reranking
  - [x] 4.1 Implement `retrieval/dense_retriever.py` and `retrieval/sparse_retriever.py`
    - `DenseRetriever.search(query_vector, top_k=10)` queries `VectorStoreClient` and returns `list[ScoredChunk]`
    - `SparseRetriever.search(query, top_k=10)` queries `BM25Index` and returns `list[ScoredChunk]`
    - Both log per-query latency
    - _Requirements: 4.1, 4.2, 4.6_
  - [x] 4.2 Implement `retrieval/rrf_fusion.py` with configurable RRF weighting
    - Implement `reciprocal_rank_fusion(dense_results, sparse_results, alpha=0.7, k=60)` as a pure function
    - RRF score = `alpha * (1 / (k + dense_rank)) + (1 - alpha) * (1 / (k + sparse_rank))`
    - Chunks missing from one list get rank `len(list) + 1` in that list
    - Return up to 20 candidates sorted by descending RRF score
    - _Requirements: 4.3, 4.4_
  - [x] 4.3 Write property tests for RRF fusion
    - Property: higher `alpha` should increase weight of dense results (top result shifts toward dense-favored chunks as alpha → 1.0)
    - Property: rank ordering is consistent — a chunk ranked higher in both lists must rank ≥ as high in fused output
    - Property: result count is always ≤ 20 regardless of input sizes
    - _Requirements: 4.3, 11.2_
  - [x] 4.4 Implement `retrieval/reranker.py` with cross-encoder and LLM-judge modes
    - Implement `CrossEncoderReranker` loading `cross-encoder/ms-marco-MiniLM-L-6-v2` via `sentence-transformers CrossEncoder`
    - `rerank(query, candidates, top_k=5)` scores all candidates and returns top-5 by cross-encoder score
    - Implement LLM-judge mode: structured prompt rating relevance 1–10, selected via `RERANKER_MODE=llm_judge`
    - Log reranking latency and per-chunk cross-encoder score
    - _Requirements: 5.1, 5.2, 5.3, 5.4_

- [x] 5. Checkpoint — Ensure ingestion and retrieval tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 6. Implement generation and citation verification
  - [x] 6.1 Implement `generation/generator.py` with grounded citation generation
    - Implement `Generator.generate(query, chunks, min_relevance_threshold=0.3)` as an async method
    - Build the generation prompt with `[n]`-indexed context chunks using the template from the design
    - If all top-5 chunk scores are below `min_relevance_threshold`, return a structured "I don't know" `GenerationResult` with `is_grounded=False`
    - Parse `[n]` references from the answer to build `Citation` objects
    - Return `GenerationResult` with `answer`, `citations`, `is_grounded`, `prompt_tokens`, `completion_tokens`
    - _Requirements: 6.1, 6.2, 6.5_
  - [x] 6.2 Implement `generation/citation_verifier.py` using NLI model
    - Implement `CitationVerifier.verify(claim, chunk_text)` as an async method
    - Load `cross-encoder/nli-deberta-v3-small` for NLI inference
    - Map NLI labels: ENTAILMENT → `supported`, NEUTRAL → `partial`, CONTRADICTION → `unsupported`
    - Extract claim text by regex-parsing the sentence surrounding each `[n]` citation marker
    - _Requirements: 6.3, 6.4_
  - [x] 6.3 Implement `generation/confidence.py` composite confidence score calculation
    - Implement `compute_confidence_score(top5_relevance_scores, citations, completeness_score)` as a pure function
    - Formula: `0.4 * mean(top5_relevance_scores) + 0.4 * (supported_count / total_citations) + 0.2 * completeness_score`
    - Handle edge case: zero citations (return retrieval-only component)
    - _Requirements: 6.6_
  - [x] 6.4 Write unit tests for citation verifier
    - Test `supported` classification: chunk that clearly entails the claim
    - Test `unsupported` classification: chunk that contradicts the claim
    - Test `partial` classification: chunk that is tangentially related
    - Test "I don't know" guard: verify `is_grounded=False` returned when all scores are below threshold
    - _Requirements: 6.3, 6.4, 6.5, 11.2_
  - [x] 6.5 Write property tests for citation coverage
    - Property: `confidence_score` is always in [0.0, 1.0]
    - Property: confidence score increases monotonically as the fraction of `supported` citations increases (other inputs held constant)
    - _Requirements: 6.6_

- [x] 7. Implement evaluation framework
  - [x] 7.1 Create `eval/dataset.py` with 50+ Q&A pairs covering all categories
    - Define `QAPair` model and load dataset from a JSON file at `data/eval_dataset.json`
    - Create `data/eval_dataset.json` with ≥50 Q&A pairs covering: factual lookups, multi-hop reasoning, no-answer questions, and ambiguous queries
    - _Requirements: 7.1_
  - [x] 7.2 Implement `eval/metrics.py` for correctness, faithfulness, context relevance, and citation accuracy
    - `correctness(answer, expected)`: ROUGE-L score with optional LLM-as-judge override
    - `faithfulness(citations)`: fraction of answer citations with `supported` status
    - `context_relevance(scored_chunks)`: mean reranker score of top-5 chunks
    - `citation_accuracy(citations)`: fraction of citations with `supported` status
    - _Requirements: 7.2_
  - [x] 7.3 Implement `eval/harness.py` to run all configs and produce an `EvalReport`
    - Implement `EvalHarness.run(configs, dataset, use_cache=True)`
    - Evaluate all 6 configurations: 3 chunking strategies × 2 retrieval modes (hybrid, dense-only)
    - Cache LLM responses to `data/eval_cache/responses_{hash}.json`; load from cache when `use_cache=True` for offline reproducibility
    - Return `EvalReport` with `results`, `best_config`, and `hybrid_vs_dense_delta`
    - Print summary to stdout on completion
    - _Requirements: 7.2, 7.3, 7.5, 7.6_
  - [x] 7.4 Implement `eval/report_generator.py` to write `/eval/report.md` with tables and charts
    - Generate per-configuration metric tables in Markdown
    - Generate hybrid-vs-dense-only comparison section
    - Generate chunking-strategy comparison section
    - Export chart images (e.g., bar charts via `matplotlib`) and embed in the report
    - Write output to `eval/report.md`
    - _Requirements: 7.4_

- [x] 8. Implement the FastAPI API layer
  - [x] 8.1 Create `api/models.py` with all Pydantic request/response models
    - Implement `AskRequest`, `AskResponse`, `LatencyBreakdown`, `DocumentMeta`, `IngestResponse` as defined in the design's Data Models section
    - _Requirements: 8.1, 8.2, 8.3, 8.4_
  - [x] 8.2 Implement `api/middleware.py` with request ID injection and global error handling
    - Inject a UUID `request_id` into each request's state via middleware
    - Register a global exception handler that catches unhandled errors, logs the full stack trace with `request_id`, and returns HTTP 500 with sanitized message and `request_id`
    - Return HTTP 422 for Pydantic validation errors with structured error body
    - _Requirements: 8.6, 8.7_
  - [x] 8.3 Implement `api/routes.py` with all three endpoints
    - `POST /v1/ask`: accept `AskRequest`, run the full pipeline (embed query → dense+sparse retrieve → RRF fusion or dense-only bypass → rerank → generate → verify → score), return `AskResponse` with per-stage `latency_ms`
    - `GET /v1/documents`: call `VectorStoreClient.list_documents()` and return `list[DocumentMeta]`
    - `POST /v1/ingest`: accept multipart file upload, save to temp file, run ingestion pipeline, return `IngestResponse`
    - All route handlers are `async def`
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 4.4, 4.5_
  - [x] 8.4 Implement `api/main.py` FastAPI app factory with OpenAPI docs
    - Create FastAPI app with title and description for portfolio presentation
    - Register middleware and routes
    - Serve interactive OpenAPI docs at `/docs`
    - _Requirements: 8.5_

- [x] 9. Checkpoint — Ensure API starts and all unit tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 10. Implement the Streamlit dashboard
  - [x] 10.1 Create `dashboard/app.py` with query input and response display
    - Query input field that submits to `POST /v1/ask` via `httpx`
    - Display generated answer with inline citation links that expand to show chunk text, source document, and verification status
    - Show low-confidence warning (visual indicator) when `confidence_score < 0.4`
    - _Requirements: 9.1, 9.2, 9.6_
  - [x] 10.2 Add retrieval trace panel and hybrid/dense-only toggle
    - Display retrieved and reranked chunks in ranked order with relevance score, reranker score, source document, and chunking strategy
    - Toggle switch for hybrid vs. dense-only mode that re-submits the current query and displays both results for comparison
    - _Requirements: 9.3, 9.4_
  - [x] 10.3 Add evaluation report tab rendering `/eval/report.md`
    - Render the chunking strategy comparison report from `eval/report.md`
    - Display metric tables and embed chart images inline
    - _Requirements: 9.5_

- [x] 11. Add sample corpus and seed script
  - [x] 11.1 Create sample corpus files in `data/corpus/`
    - Create 8–12 files covering: API documentation (`.md`), deployment runbook (`.md`), onboarding FAQ (`.txt`), changelog (`.md`), and at least one `.html` file
    - Files should contain realistic technical content that supports varied Q&A types (factual, multi-hop, no-answer)
    - _Requirements: 1.3_
  - [x] 11.2 Implement `seed.py` one-command corpus seeding script
    - Load `PipelineConfig` from environment
    - Ingest all files from `data/corpus/` using the default chunking strategy
    - Build and persist both the Qdrant collection and the BM25 index
    - Log ingestion summary and exit cleanly before Uvicorn starts
    - _Requirements: 10.3_

- [x] 12. Containerization and Docker Compose setup
  - [x] 12.1 Write `Dockerfile.api` for the FastAPI service
    - Multi-stage build: install dependencies, copy packages, run `seed.py` then `uvicorn api.main:app --host 0.0.0.0 --port 8000`
    - Include healthcheck for the `/docs` endpoint
    - _Requirements: 10.1, 10.2_
  - [x] 12.2 Write `Dockerfile.dash` for the Streamlit dashboard
    - Install Streamlit and dashboard dependencies
    - Expose port 3000 (set via `STREAMLIT_SERVER_PORT=3000`)
    - _Requirements: 10.1, 10.2_
  - [x] 12.3 Write `docker-compose.yml` defining all three services
    - Service `qdrant`: `qdrant/qdrant` image, port 6333, persistent volume mount at `./qdrant_storage`
    - Service `api`: builds `./Dockerfile.api`, port 8000, `depends_on: qdrant` with health condition, passes env vars including `OPENAI_API_KEY`
    - Service `dashboard`: builds `./Dockerfile.dash`, port 3000, `depends_on: api`
    - System must be up within 5 minutes of `docker-compose up --build`
    - _Requirements: 10.1, 10.2_
  - [x] 12.4 Add graceful startup fallback for missing `OPENAI_API_KEY`
    - In `config.py` / embedder init, detect missing key at startup, log a clear error identifying the variable, and switch to sentence-transformers without raising an exception
    - _Requirements: 10.4_

- [x] 13. Write `README.md` with results, architecture, and setup instructions
  - [x] 13.1 Create `README.md` with all required sections
    - Results table: faithfulness, citation accuracy, hybrid vs. dense-only delta (populated after eval run)
    - ASCII or Mermaid architecture diagram matching the high-level data flow from the design
    - "Why hybrid search" section with before/after evaluation numbers
    - "Key engineering decisions" section: chunking strategy tradeoffs, RRF weighting rationale, citation verification design
    - One-command setup: `docker-compose up --build`
    - _Requirements: 10.5_

- [x] 14. Final checkpoint — Full test suite and end-to-end validation
  - Ensure `pytest` reports 0 failures with `pytest --tb=short`
  - Verify `ruff check .` passes with 0 errors
  - Ask the user if any questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for a faster MVP
- Each task references specific requirements for traceability
- The design does not include a formal Correctness Properties section, so no PBT tasks are included for ingestion or retrieval beyond the RRF fusion property tests (task 4.3) and confidence score properties (task 6.5), which validate well-defined mathematical invariants
- The `SemanticChunker` in task 2.3 depends on `Embedder` being available — instantiate a shared `Embedder` and pass it into the chunker to avoid a circular dependency
- Docker services start in order: `qdrant` → `api` (runs `seed.py`) → `dashboard`
- The eval harness (task 7.3) should be run once after full ingestion to populate `eval/report.md` before the dashboard renders it

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1", "1.2", "1.3"] },
    { "id": 1, "tasks": ["2.1"] },
    { "id": 2, "tasks": ["2.2", "2.3"] },
    { "id": 3, "tasks": ["2.4", "3.1", "3.2", "3.3"] },
    { "id": 4, "tasks": ["2.5", "3.4"] },
    { "id": 5, "tasks": ["4.1", "4.2"] },
    { "id": 6, "tasks": ["4.3", "4.4"] },
    { "id": 7, "tasks": ["6.1", "8.1"] },
    { "id": 8, "tasks": ["6.2", "6.3", "8.2", "8.3"] },
    { "id": 9, "tasks": ["6.4", "6.5", "8.4", "7.1"] },
    { "id": 10, "tasks": ["7.2", "11.1"] },
    { "id": 11, "tasks": ["7.3", "11.2"] },
    { "id": 12, "tasks": ["7.4", "10.1"] },
    { "id": 13, "tasks": ["10.2", "12.1", "12.2"] },
    { "id": 14, "tasks": ["10.3", "12.3"] },
    { "id": 15, "tasks": ["12.4", "13.1"] }
  ]
}
```
