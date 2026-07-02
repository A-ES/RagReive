# Requirements Document

## Introduction

A production-grade Hybrid Retrieval-Augmented Generation (RAG) pipeline designed as a portfolio showcase for AI/ML engineering skills. The system combines dense vector search (OpenAI embeddings via Qdrant) with sparse keyword search (BM25) using Reciprocal Rank Fusion, followed by cross-encoder reranking, grounded generation with bracketed citation verification, and a comprehensive evaluation framework. The deliverable is a fully containerized application (Docker Compose) with a FastAPI backend, a web dashboard, and reproducible evaluation results demonstrating measurable quality improvements over dense-only retrieval.

## Glossary

- **Pipeline**: The end-to-end Hybrid RAG system, from document ingestion through answer generation.
- **Corpus**: The collection of ingested documents (markdown, text, HTML, PDF).
- **Chunk**: A sub-document fragment produced by a chunking strategy, stored with metadata.
- **Chunking_Strategy**: One of three swappable algorithms that split documents into chunks: `fixed`, `structural`, or `semantic`.
- **Embedder**: The component that converts text into dense vector representations using OpenAI `text-embedding-3-small` or a local sentence-transformers fallback.
- **Vector_Store**: Qdrant, the containerized vector database storing dense embeddings and metadata.
- **BM25_Index**: The in-memory sparse retrieval index built from chunk text using `rank_bm25`.
- **Dense_Retriever**: The component that performs approximate nearest-neighbor search against the Vector_Store.
- **Sparse_Retriever**: The component that performs BM25 keyword search against the BM25_Index.
- **RRF_Fusion**: Reciprocal Rank Fusion — the algorithm that merges dense and sparse ranked lists into a single hybrid ranking, with configurable weighting.
- **Reranker**: The cross-encoder model (`cross-encoder/ms-marco-MiniLM-L-6-v2`) that rescores candidate chunks against the query.
- **Generator**: The LLM-backed component that produces grounded answers with bracketed `[n]` citations.
- **Citation_Verifier**: The NLI/LLM-as-judge component that validates each citation claim against its source chunk.
- **Confidence_Score**: The composite score combining retrieval relevance, citation coverage, and answer completeness.
- **Eval_Harness**: The offline evaluation framework that runs 50+ Q&A pairs and reports faithfulness, context relevance, and citation accuracy metrics.
- **API**: The FastAPI application exposing `/v1/ask`, `/v1/documents`, and `/v1/ingest` endpoints.
- **Dashboard**: The web UI showing query input, inline citations, ranked chunks with scores, and evaluation reports.
- **Deduplication**: The process of removing near-duplicate chunks whose cosine similarity exceeds 0.95.
- **NLI**: Natural Language Inference — used to verify whether a chunk entails or supports a cited claim.
- **RRF_Weight**: The scalar `α` controlling the blend of dense vs. sparse scores in RRF fusion (default `α = 0.7` for dense, `1 - α = 0.3` for sparse).

---

## Requirements

### Requirement 1: Document Ingestion and Normalization

**User Story:** As an AI/ML engineer, I want to ingest documents in multiple formats (markdown, plain text, HTML, PDF), so that the corpus can be built from realistic heterogeneous sources.

#### Acceptance Criteria

1. WHEN a file with extension `.md`, `.txt`, `.html`, or `.pdf` is submitted to the ingestion pipeline, THE Pipeline SHALL parse the file and produce normalized plaintext with per-document metadata (filename, format, ingestion timestamp, source URL if present).
2. IF a submitted file has an unsupported extension, THEN THE Pipeline SHALL return a structured error response identifying the file and the unsupported format, without halting ingestion of other files.
3. THE Pipeline SHALL ship with a sample corpus of 8–12 files covering API documentation, runbooks, FAQs, and changelogs.
4. WHEN the ingestion pipeline completes, THE Pipeline SHALL report the total number of documents loaded, the number of chunks produced, and any files that failed to parse.

---

### Requirement 2: Chunking Strategies

**User Story:** As an AI/ML engineer, I want three swappable chunking strategies, so that I can compare their effect on retrieval and generation quality.

#### Acceptance Criteria

1. THE Chunking_Strategy named `fixed` SHALL split documents into chunks of a configurable character count with a configurable overlap, defaulting to 512 characters with 64-character overlap.
2. THE Chunking_Strategy named `structural` SHALL split documents at structural boundaries (headings, paragraphs, list items, code fences) detected from the normalized plaintext.
3. THE Chunking_Strategy named `semantic` SHALL split documents at topic boundaries using sentence-level embedding similarity, grouping sentences until a cosine-similarity drop below a configurable threshold (default 0.75) is detected.
4. WHEN any Chunking_Strategy produces a chunk, THE Chunking_Strategy SHALL attach metadata to each chunk: parent document ID, chunk index, character offsets (start, end), and strategy name.
5. THE Pipeline SHALL expose a configuration parameter allowing the active Chunking_Strategy to be selected at ingestion time without code changes.
6. WHEN two or more chunks from the same or different documents have a pairwise cosine similarity above 0.95, THE Pipeline SHALL retain only one chunk and discard the near-duplicate.

---

### Requirement 3: Embedding and Indexing

**User Story:** As an AI/ML engineer, I want every chunk embedded and stored in a vector database alongside a BM25 index, so that both dense and sparse retrieval are available at query time.

#### Acceptance Criteria

1. WHEN chunks are produced by the active Chunking_Strategy, THE Embedder SHALL generate a dense embedding for each chunk using OpenAI `text-embedding-3-small`.
2. WHERE the OpenAI API is unavailable or not configured, THE Embedder SHALL fall back to a local `sentence-transformers` model to generate embeddings, ensuring offline demo capability.
3. THE Vector_Store SHALL store each chunk's embedding, plaintext content, and metadata in a named Qdrant collection.
4. THE BM25_Index SHALL be built from the plaintext of all chunks at ingestion completion and persisted to disk so that it survives application restarts.
5. THE Pipeline SHALL embed and index chunks in parallel, utilizing concurrent workers to reduce total ingestion time.
6. WHEN ingestion completes, THE Pipeline SHALL log the total ingestion wall-clock time and the average embedding latency per chunk.

---

### Requirement 4: Hybrid Retrieval

**User Story:** As an AI/ML engineer, I want hybrid dense + sparse retrieval with configurable RRF fusion, so that the system handles both semantic and keyword-sensitive queries.

#### Acceptance Criteria

1. WHEN a query is received, THE Dense_Retriever SHALL return the top-10 chunks by cosine similarity from the Vector_Store.
2. WHEN a query is received, THE Sparse_Retriever SHALL return the top-10 chunks by BM25 score from the BM25_Index.
3. WHEN dense and sparse result lists are available, THE RRF_Fusion component SHALL merge them using Reciprocal Rank Fusion with a configurable RRF_Weight (default `α = 0.7` dense, `0.3` sparse) to produce a unified ranked list of up to 20 candidates.
4. THE API SHALL expose the RRF_Weight as a runtime parameter on the `/v1/ask` endpoint, allowing callers to override the default blend without restarting the service.
5. WHEN a `dense_only` flag is set to `true` on a request, THE Pipeline SHALL bypass RRF_Fusion and return only the Dense_Retriever results, enabling A/B comparison.
6. THE Pipeline SHALL log the per-stage latency (dense retrieval, sparse retrieval, RRF fusion) for every query.

---

### Requirement 5: Reranking

**User Story:** As an AI/ML engineer, I want a cross-encoder reranker applied to the fusion candidates, so that the top-5 chunks passed to generation are the most relevant.

#### Acceptance Criteria

1. WHEN the RRF_Fusion produces a candidate list, THE Reranker SHALL score every candidate using `cross-encoder/ms-marco-MiniLM-L-6-v2` against the original query.
2. THE Reranker SHALL return the top-5 highest-scoring chunks to the Generator.
3. WHERE an `llm_judge` mode is enabled via configuration, THE Reranker SHALL score candidates using an LLM-as-judge prompt instead of the cross-encoder model.
4. THE Pipeline SHALL log the reranking latency and the cross-encoder score of each returned chunk for every query.

---

### Requirement 6: Grounded Answer Generation with Citations

**User Story:** As a user, I want grounded answers with verifiable bracketed citations, so that every claim in the response can be traced back to a source chunk.

#### Acceptance Criteria

1. WHEN the top-5 reranked chunks are available, THE Generator SHALL produce an answer using a grounded generation prompt that instructs the LLM to cite each factual claim with a bracketed reference `[n]` corresponding to a chunk index.
2. THE Generator SHALL include the plaintext of each cited chunk as inline context in the generation prompt, and SHALL NOT fabricate information beyond what the context provides.
3. WHEN the Generator produces an answer, THE Citation_Verifier SHALL evaluate each `[n]` citation by checking whether the cited chunk entails the associated claim, using an NLI model or LLM-as-judge.
4. THE Citation_Verifier SHALL assign each citation a verification status of `supported`, `partial`, or `unsupported`.
5. WHEN no retrieved chunk has a relevance score above a configurable threshold (default 0.3), THE Generator SHALL return a structured "I don't know" response instead of generating an answer, including the reason and the highest relevance score observed.
6. THE Pipeline SHALL compute a Confidence_Score for each response as a weighted composite of: mean retrieval relevance of top-5 chunks (weight 0.4), citation coverage rate (weight 0.4), and answer completeness score from the LLM judge (weight 0.2).

---

### Requirement 7: Evaluation Framework

**User Story:** As an AI/ML engineer, I want an offline-reproducible evaluation harness, so that I can report real performance numbers comparing chunking strategies and hybrid vs. dense-only retrieval.

#### Acceptance Criteria

1. THE Eval_Harness SHALL include a dataset of at least 50 question-answer pairs covering: direct fact lookups, multi-hop reasoning, no-answer questions, and ambiguous queries.
2. WHEN the Eval_Harness is executed, THE Eval_Harness SHALL compute the following metrics for each evaluated configuration: correctness, faithfulness, context relevance, and citation accuracy.
3. THE Eval_Harness SHALL evaluate all three Chunking_Strategies (`fixed`, `structural`, `semantic`) and both retrieval modes (hybrid, dense-only) in a single run, producing a comparison table.
4. THE Eval_Harness SHALL produce a report at `/eval/report.md` containing: per-configuration metric tables, a hybrid-vs-dense-only comparison, a chunking-strategy comparison, and exported chart images.
5. THE Eval_Harness SHALL run fully offline without requiring live LLM API calls when a cached responses file is present, ensuring reproducibility.
6. WHEN evaluation completes, THE Eval_Harness SHALL print a summary to stdout including the best-performing configuration and the margin by which hybrid retrieval outperforms dense-only on technical-term queries.

---

### Requirement 8: API Layer

**User Story:** As a developer, I want a fully async FastAPI service with OpenAPI documentation, so that the pipeline is accessible programmatically and integrates with the dashboard.

#### Acceptance Criteria

1. THE API SHALL expose a `POST /v1/ask` endpoint accepting a JSON body with fields: `query` (string, required), `dense_only` (boolean, default `false`), `rrf_weight` (float, default `0.7`), and `chunking_strategy` (string, default `structural`).
2. WHEN `POST /v1/ask` returns successfully, THE API SHALL respond with a JSON body containing: `answer` (string), `citations` (array of objects with `index`, `chunk_text`, `source`, `verification_status`), `confidence_score` (float), and `latency_ms` (object with per-stage breakdowns).
3. THE API SHALL expose a `GET /v1/documents` endpoint returning the list of ingested documents with metadata including document ID, filename, format, chunk count, and ingestion timestamp.
4. THE API SHALL expose a `POST /v1/ingest` endpoint accepting a multipart file upload, triggering ingestion of the uploaded file using the currently configured Chunking_Strategy.
5. THE API SHALL serve interactive OpenAPI documentation at `/docs`.
6. WHEN any endpoint receives a malformed or invalid request, THE API SHALL return an HTTP 422 response with a structured error body describing the validation failure.
7. WHEN an unhandled internal error occurs during request processing, THE API SHALL return an HTTP 500 response with a request ID and a non-sensitive error message, and SHALL log the full stack trace internally.

---

### Requirement 9: Dashboard

**User Story:** As a hiring manager or engineer reviewing the portfolio, I want a web dashboard that demonstrates all pipeline capabilities visually, so that I can evaluate the system's quality and engineering depth without reading code.

#### Acceptance Criteria

1. THE Dashboard SHALL provide a query input field from which users can submit questions to `POST /v1/ask`.
2. WHEN a response is returned, THE Dashboard SHALL display the generated answer with inline clickable citation links that expand to show the source chunk text, source document, and citation verification status.
3. THE Dashboard SHALL display the retrieved and reranked chunks in ranked order, showing each chunk's relevance score, reranker score, source document, and chunking strategy.
4. THE Dashboard SHALL provide a toggle to switch between hybrid and dense-only retrieval mode, re-submitting the current query and displaying both results side-by-side or sequentially for comparison.
5. THE Dashboard SHALL render the chunking strategy comparison report from `/eval/report.md`, including metric tables and chart images.
6. WHEN the Confidence_Score for a response is below 0.4, THE Dashboard SHALL visually indicate a low-confidence warning alongside the answer.

---

### Requirement 10: Containerization and One-Command Setup

**User Story:** As a developer or recruiter, I want to run the entire system with a single command from a clean clone, so that setup friction is eliminated and the system is immediately evaluable.

#### Acceptance Criteria

1. THE Pipeline SHALL include a `docker-compose.yml` that defines services for: the Qdrant vector store, the FastAPI API, and the Dashboard.
2. WHEN `docker-compose up --build` is run from the repository root on a machine with Docker and Docker Compose installed, THE Pipeline SHALL start all services, run `seed.py` to auto-ingest the sample corpus, and expose the API at `http://localhost:8000` and the Dashboard at `http://localhost:3000` within 5 minutes on a standard developer laptop.
3. THE Pipeline SHALL include a `seed.py` script that ingests all sample corpus files using the default Chunking_Strategy and builds both the Vector_Store collection and the BM25_Index before the API begins serving traffic.
4. IF any required environment variable (e.g., `OPENAI_API_KEY`) is missing at startup, THEN THE Pipeline SHALL log a clear error message identifying the missing variable and SHALL fall back to the offline sentence-transformers Embedder rather than failing to start.
5. THE Pipeline SHALL include a `README.md` at the repository root containing: a results table (faithfulness, citation accuracy, hybrid vs. dense-only delta), an architecture diagram in ASCII or Mermaid format, a "Why hybrid search" section with before/after evaluation numbers, a "Key engineering decisions" section covering chunking strategy tradeoffs, RRF weighting rationale, and citation verification design, and a one-command setup instruction.

---

### Requirement 11: Code Quality and Testing

**User Story:** As an AI/ML engineer, I want a clean Python package structure with tests for critical components, so that the codebase demonstrates production engineering standards.

#### Acceptance Criteria

1. THE Pipeline SHALL organize source code into Python packages: `ingestion`, `retrieval`, `generation`, `eval`, and `api`, each with an `__init__.py` and no circular imports.
2. THE Pipeline SHALL include unit tests for: each Chunking_Strategy (verifying chunk count, metadata attachment, and overlap correctness), RRF_Fusion (verifying rank ordering and weight sensitivity), and Citation_Verifier (verifying `supported`/`partial`/`unsupported` classification on known examples).
3. THE Pipeline SHALL include a `pyproject.toml` or `requirements.txt` with pinned dependency versions for all production dependencies.
4. THE Pipeline SHALL follow PEP 8 style conventions, verified by a linter configuration (e.g., `ruff` or `flake8`) included in the repository.
5. WHEN the test suite is run with `pytest`, THE Pipeline SHALL report 0 test failures on a clean environment with all dependencies installed.
