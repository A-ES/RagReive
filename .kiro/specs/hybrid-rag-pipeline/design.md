# Design Document: Hybrid RAG Pipeline

## Overview

The Hybrid RAG Pipeline is a production-grade Retrieval-Augmented Generation system that demonstrates measurable quality improvements over dense-only retrieval by combining dense vector search with sparse BM25 retrieval, merged through Reciprocal Rank Fusion (RRF), then reranked by a cross-encoder, and finally grounded answer generation with citation verification.

The system is designed as a portfolio showcase that evaluates all major design decisions with real numbers. Every component — chunking strategy, retrieval mode, RRF weighting — is swappable and observable, and an offline eval harness reports concrete metrics.

### Design Goals

- **Measurable correctness**: Every design decision (chunking strategy, RRF weight, reranker) is evaluated against 50+ Q&A pairs, and results are reported in the README.
- **Observability**: Per-stage latency is logged for every query; confidence scores are surfaced to the user.
- **Reproducibility**: A single `docker-compose up --build` brings up the entire stack, seeds the corpus, and runs evaluation offline.
- **Offline capability**: If no OpenAI API key is present, the system degrades gracefully to local sentence-transformers embeddings.
- **Portfolio clarity**: Architecture is visible in the dashboard's retrieval trace; code is organized into clean, importable packages.

### High-Level Data Flow

```
User query
    │
    ▼
┌─────────────┐
│  FastAPI     │  POST /v1/ask
│  (async)     │
└──────┬──────┘
       │
       ├──────────────────────────────────────────────┐
       ▼                                              ▼
┌─────────────┐                              ┌─────────────────┐
│Dense Retriever│                            │Sparse Retriever  │
│ Qdrant ANN   │                            │ BM25 Index       │
│ top-10       │                            │ top-10           │
└──────┬───────┘                            └────────┬────────┘
       │                                             │
       └──────────────────┬──────────────────────────┘
                          ▼
                 ┌─────────────────┐
                 │   RRF Fusion    │
                 │   α=0.7/0.3     │
                 │   up to 20 cand.│
                 └────────┬────────┘
                          ▼
                 ┌─────────────────┐
                 │ Cross-Encoder   │
                 │  Reranker       │
                 │  top-5          │
                 └────────┬────────┘
                          ▼
                 ┌─────────────────┐
                 │  Generator      │
                 │  (OpenAI LLM)   │
                 │  [n] citations  │
                 └────────┬────────┘
                          ▼
                 ┌─────────────────┐
                 │Citation Verifier│
                 │ NLI / LLM judge │
                 └────────┬────────┘
                          ▼
                 ┌─────────────────┐
                 │Confidence Score │
                 │ 0.4+0.4+0.2    │
                 └────────┬────────┘
                          ▼
                    API Response
```

---

## Architecture

### Package Structure

```
hybrid-rag-pipeline/
├── ingestion/
│   ├── __init__.py
│   ├── parsers.py          # Format-specific parsers (MD, TXT, HTML, PDF)
│   ├── chunkers.py         # FixedChunker, StructuralChunker, SemanticChunker
│   └── deduplication.py    # Cosine-similarity-based near-duplicate removal
├── retrieval/
│   ├── __init__.py
│   ├── embedder.py         # OpenAI + sentence-transformers fallback
│   ├── vector_store.py     # Qdrant client wrapper
│   ├── bm25_index.py       # rank_bm25 wrapper with persistence
│   ├── dense_retriever.py  # ANN search against Qdrant
│   ├── sparse_retriever.py # BM25 search
│   ├── rrf_fusion.py       # Reciprocal Rank Fusion
│   └── reranker.py         # Cross-encoder reranker + LLM-judge mode
├── generation/
│   ├── __init__.py
│   ├── generator.py        # Grounded generation with [n] citations
│   ├── citation_verifier.py# NLI/LLM-as-judge citation verification
│   └── confidence.py       # Composite confidence score calculation
├── eval/
│   ├── __init__.py
│   ├── harness.py          # Eval runner: all configs × all metrics
│   ├── dataset.py          # 50+ Q&A pair dataset loader
│   ├── metrics.py          # Correctness, faithfulness, context relevance, citation accuracy
│   └── report.md           # Auto-generated eval report (output artifact)
├── api/
│   ├── __init__.py
│   ├── main.py             # FastAPI app factory
│   ├── routes.py           # /v1/ask, /v1/documents, /v1/ingest
│   ├── models.py           # Pydantic request/response models
│   └── middleware.py       # Error handling, request ID injection
├── dashboard/              # Streamlit or Next.js frontend
├── seed.py                 # One-command corpus seeding script
├── docker-compose.yml
├── pyproject.toml
└── README.md
```

### Service Architecture

The system runs as three Docker Compose services:

| Service      | Image / Build        | Port  | Role                                   |
|--------------|----------------------|-------|----------------------------------------|
| `qdrant`     | `qdrant/qdrant`      | 6333  | Vector store (persistent volume mount) |
| `api`        | `./Dockerfile.api`   | 8000  | FastAPI async backend                  |
| `dashboard`  | `./Dockerfile.dash`  | 3000  | Web dashboard (Streamlit or Next.js)   |

The API container depends on `qdrant` being healthy. `seed.py` runs as an init step within the API container's `CMD` before the Uvicorn server starts.

### Technology Decisions

**FastAPI over Flask**: Native async support (critical for concurrent embedding calls), automatic OpenAPI generation, Pydantic v2 validation with precise error messages.

**Qdrant over Chroma/Weaviate**: Qdrant's native payload filtering and named collections make it trivial to store chunking-strategy metadata alongside embeddings. Its Docker image is lightweight and production-ready.

**rank_bm25 over Elasticsearch**: For a portfolio-scale corpus (hundreds to thousands of chunks), an in-memory BM25 index is sufficient, avoids a fourth Docker service, and keeps setup one-command.

**sentence-transformers as fallback**: `all-MiniLM-L6-v2` is compact (~90 MB), ships offline, and yields embeddings in the same 384-dimension space regardless of OpenAI availability.

**cross-encoder/ms-marco-MiniLM-L-6-v2**: Purpose-trained for passage reranking on MS MARCO, small enough for CPU inference (<200 ms per batch of 20 candidates on a developer laptop).

**Streamlit for dashboard**: Minimal boilerplate, renders dataframes and charts natively, exposes a Python-native interface to the FastAPI backend. Next.js is the alternative if richer interactivity is desired.

---

## Components and Interfaces

### 1. Ingestion Pipeline

```python
class DocumentParser:
    def parse(self, file_path: Path) -> ParsedDocument:
        """Returns normalized plaintext + metadata for a given file."""

class ParsedDocument(BaseModel):
    doc_id: str              # UUID
    filename: str
    format: Literal["md", "txt", "html", "pdf"]
    content: str             # Normalized plaintext
    source_url: str | None
    ingested_at: datetime

class IngestionResult(BaseModel):
    total_documents: int
    total_chunks: int
    failed_files: list[FailedFile]
    wall_clock_seconds: float
```

Parser implementations:
- `MarkdownParser`: strips frontmatter, converts to plaintext via `markdownify` or regex
- `TextParser`: reads UTF-8 with encoding detection fallback
- `HtmlParser`: uses `BeautifulSoup` to extract body text, preserving heading hierarchy
- `PdfParser`: uses `pdfplumber` for text extraction; falls back to `pypdf`

### 2. Chunking Strategies

All chunkers implement a common interface:

```python
class BaseChunker(ABC):
    @abstractmethod
    def chunk(self, doc: ParsedDocument) -> list[Chunk]:
        ...

class Chunk(BaseModel):
    chunk_id: str            # UUID
    doc_id: str
    chunk_index: int
    text: str
    char_start: int
    char_end: int
    strategy: str            # "fixed" | "structural" | "semantic"
    embedding: list[float] | None = None
```

**FixedChunker**: Wraps `LangChain CharacterTextSplitter` with configurable `chunk_size` (default 512) and `chunk_overlap` (default 64). Straightforward but ignores semantic boundaries.

**StructuralChunker**: Wraps `LangChain MarkdownHeaderTextSplitter` for markdown; for other formats, splits on paragraph boundaries (double newlines) and code fences. Respects document structure.

**SemanticChunker**: Encodes sentences individually, then greedily merges adjacent sentences until cosine similarity between consecutive sentence embeddings drops below a configurable threshold (default 0.75). Uses the same `Embedder` instance to avoid a separate embedding call.

**Deduplication**: After all chunks are produced, `deduplication.py` computes pairwise cosine similarity (batched via numpy) and removes any chunk whose similarity to an earlier chunk exceeds 0.95.

### 3. Embedder

```python
class Embedder:
    def __init__(self, provider: Literal["openai", "sentence_transformers"] = "openai"):
        ...

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Returns embeddings for a batch of texts. Async to support concurrent workers."""

    async def embed_query(self, query: str) -> list[float]:
        """Single-query embedding for retrieval."""
```

Provider selection: On startup, the `Embedder` checks for `OPENAI_API_KEY`. If absent, it logs a warning and falls back to `sentence-transformers/all-MiniLM-L6-v2` loaded locally.

Concurrency: Ingestion uses `asyncio.gather` with a semaphore (configurable worker count, default 8) to parallelize embedding API calls.

### 4. Vector Store (Qdrant)

```python
class VectorStoreClient:
    def __init__(self, host: str, port: int, collection_name: str):
        ...

    async def upsert_chunks(self, chunks: list[Chunk]) -> None:
        """Batch-upsert chunks with embeddings and metadata payload."""

    async def search(self, query_vector: list[float], top_k: int = 10) -> list[ScoredChunk]:
        """ANN search returning top-k chunks with cosine similarity scores."""

    async def list_documents(self) -> list[DocumentMeta]:
        """Aggregate metadata from the payload store."""
```

Collection schema: Each Qdrant point stores the chunk UUID as ID, the dense embedding as the vector, and a payload dict containing `{doc_id, filename, format, chunk_index, char_start, char_end, strategy, text}`.

### 5. BM25 Index

```python
class BM25Index:
    def build(self, chunks: list[Chunk]) -> None:
        """Tokenizes chunk texts and builds the BM25Okapi index."""

    def search(self, query: str, top_k: int = 10) -> list[ScoredChunk]:
        """Returns top-k chunks by BM25 score."""

    def save(self, path: Path) -> None:
        """Persists index and chunk mapping to disk using pickle."""

    def load(self, path: Path) -> None:
        """Restores index from disk."""
```

The BM25 index stores both the `BM25Okapi` object and a parallel list of chunks (indexed by position) so scores can be mapped back to `Chunk` objects at query time. Tokenization uses simple whitespace + punctuation splitting; stop-word removal is optional via configuration.

### 6. RRF Fusion

```python
def reciprocal_rank_fusion(
    dense_results: list[ScoredChunk],
    sparse_results: list[ScoredChunk],
    alpha: float = 0.7,
    k: int = 60,
) -> list[ScoredChunk]:
    """
    Merges two ranked lists using RRF.
    RRF score = alpha * (1 / (k + dense_rank)) + (1 - alpha) * (1 / (k + sparse_rank))
    Returns up to 20 candidates sorted by descending RRF score.
    """
```

The `k=60` constant is the standard RRF smoothing parameter. Chunks appearing in only one list receive a rank of `len(list) + 1` in the missing list. The `alpha` parameter is passed directly from the API request.

### 7. Reranker

```python
class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: list[ScoredChunk], top_k: int = 5) -> list[ScoredChunk]:
        """Scores all candidates against the query; returns top_k by cross-encoder score."""
```

In `llm_judge` mode, the reranker uses a structured LLM prompt that asks the model to rate relevance 1–10, then sorts by that rating. This mode is configurable via `RERANKER_MODE=llm_judge`.

### 8. Generator

```python
class Generator:
    async def generate(
        self,
        query: str,
        chunks: list[ScoredChunk],
        min_relevance_threshold: float = 0.3,
    ) -> GenerationResult:
        ...

class GenerationResult(BaseModel):
    answer: str
    citations: list[Citation]
    is_grounded: bool        # False when "I don't know" fallback fires
    prompt_tokens: int
    completion_tokens: int

class Citation(BaseModel):
    index: int               # [n] reference
    chunk_id: str
    chunk_text: str
    source: str              # filename
    verification_status: Literal["supported", "partial", "unsupported"] | None
```

The generation prompt template:
```
You are a precise technical assistant. Answer the following question using ONLY the provided context chunks.
For each factual claim, cite the source using bracketed notation [1], [2], etc.
If the context does not contain sufficient information to answer, respond with "I don't have enough information to answer this question."

Context:
[1] {chunk_1_text}
[2] {chunk_2_text}
...

Question: {query}
Answer:
```

The "I don't know" guard: if all top-5 chunk scores are below `min_relevance_threshold` (default 0.3), generation is skipped and a structured fallback response is returned immediately.

### 9. Citation Verifier

```python
class CitationVerifier:
    async def verify(
        self,
        claim: str,
        chunk_text: str,
    ) -> Literal["supported", "partial", "unsupported"]:
        """
        Uses an NLI model (cross-encoder/nli-deberta-v3-small) or LLM-as-judge
        to classify whether chunk_text entails, partially supports, or contradicts the claim.
        """
```

NLI label mapping: ENTAILMENT → `supported`, NEUTRAL → `partial`, CONTRADICTION → `unsupported`. Each `[n]` citation in the answer is parsed by regex, the surrounding sentence is extracted as the claim, and the referenced chunk is the hypothesis source.

### 10. Confidence Score

```python
def compute_confidence_score(
    top5_relevance_scores: list[float],  # from reranker
    citations: list[Citation],
    completeness_score: float,           # from LLM judge (0–1)
) -> float:
    retrieval_relevance = mean(top5_relevance_scores)    # weight 0.4
    citation_coverage = supported_count / total_citations # weight 0.4
    return 0.4 * retrieval_relevance + 0.4 * citation_coverage + 0.2 * completeness_score
```

### 11. API Layer

```python
# POST /v1/ask
class AskRequest(BaseModel):
    query: str
    dense_only: bool = False
    rrf_weight: float = Field(default=0.7, ge=0.0, le=1.0)
    chunking_strategy: Literal["fixed", "structural", "semantic"] = "structural"

class AskResponse(BaseModel):
    answer: str
    citations: list[Citation]
    confidence_score: float
    latency_ms: LatencyBreakdown
    is_grounded: bool

class LatencyBreakdown(BaseModel):
    dense_retrieval_ms: float
    sparse_retrieval_ms: float
    rrf_fusion_ms: float
    reranking_ms: float
    generation_ms: float
    citation_verification_ms: float
    total_ms: float
```

All endpoints are implemented as `async def` route handlers. A global exception handler catches unhandled errors, logs the stack trace with a `request_id` (UUID injected by middleware), and returns HTTP 500 with a sanitized message.

### 12. Eval Harness

```python
class EvalHarness:
    def run(
        self,
        configs: list[EvalConfig],
        dataset: list[QAPair],
        use_cache: bool = True,
    ) -> EvalReport:
        ...

class EvalConfig(BaseModel):
    chunking_strategy: str
    retrieval_mode: Literal["hybrid", "dense_only"]
    rrf_weight: float = 0.7

class QAPair(BaseModel):
    question: str
    expected_answer: str
    category: Literal["factual", "multi_hop", "no_answer", "ambiguous"]

class EvalReport(BaseModel):
    configs: list[EvalConfig]
    results: dict[str, ConfigMetrics]  # key = config name
    best_config: str
    hybrid_vs_dense_delta: float
```

Metrics implementation:
- **Correctness**: LLM-as-judge or ROUGE-L against expected answer
- **Faithfulness**: Fraction of answer claims that are `supported` by citations
- **Context relevance**: Mean reranker score of top-5 chunks
- **Citation accuracy**: Fraction of citations with `supported` status

---

## Data Models

### Core Domain Models

```python
# ingestion/models.py

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

# generation/models.py

class Citation(BaseModel):
    index: int
    chunk_id: str
    chunk_text: str
    source: str
    verification_status: Literal["supported", "partial", "unsupported"] | None = None

class GenerationResult(BaseModel):
    answer: str
    citations: list[Citation]
    confidence_score: float
    is_grounded: bool
    prompt_tokens: int
    completion_tokens: int

# api/models.py

class AskRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    dense_only: bool = False
    rrf_weight: float = Field(default=0.7, ge=0.0, le=1.0)
    chunking_strategy: Literal["fixed", "structural", "semantic"] = "structural"

class AskResponse(BaseModel):
    answer: str
    citations: list[Citation]
    confidence_score: float
    latency_ms: LatencyBreakdown
    is_grounded: bool

class DocumentMeta(BaseModel):
    doc_id: str
    filename: str
    format: str
    chunk_count: int
    ingested_at: datetime

class IngestResponse(BaseModel):
    doc_id: str
    filename: str
    chunk_count: int
    ingestion_time_seconds: float

class LatencyBreakdown(BaseModel):
    dense_retrieval_ms: float
    sparse_retrieval_ms: float
    rrf_fusion_ms: float
    reranking_ms: float
    generation_ms: float
    citation_verification_ms: float
    total_ms: float

# eval/models.py

class QAPair(BaseModel):
    question_id: str
    question: str
    expected_answer: str
    category: Literal["factual", "multi_hop", "no_answer", "ambiguous"]
    source_doc: str | None = None

class ConfigMetrics(BaseModel):
    correctness: float       # 0–1
    faithfulness: float      # 0–1
    context_relevance: float # 0–1
    citation_accuracy: float # 0–1
    mean_latency_ms: float
    sample_count: int
```

### Configuration Model

```python
# config.py (Pydantic Settings)

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
```

### Persistence Layout

```
data/
├── bm25_index.pkl        # Serialized BM25Okapi + chunk list
├── corpus/               # Sample documents (8–12 files)
│   ├── api_reference.md
│   ├── runbook_deploy.md
│   ├── faq_onboarding.txt
│   └── ...
└── eval_cache/           # Cached LLM responses for offline eval
    └── responses_{hash}.json

qdrant_storage/           # Docker volume mount for Qdrant persistence
```

---

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system — essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

The following properties are derived from the acceptance criteria. Each is universally quantified and intended to be implemented as a property-based test (minimum 100 iterations, using `hypothesis` for Python).

---

### Property 1: Document Parsing Produces Complete Metadata

*For any* file of a supported format (`.md`, `.txt`, `.html`, `.pdf`) with any valid content, the ingestion parser SHALL produce a `ParsedDocument` with non-null, non-empty values for `doc_id`, `filename`, `format`, `content`, and `ingested_at`.

**Validates: Requirements 1.1**

---

### Property 2: Unsupported Files Produce Structured Errors Without Halting Valid Files

*For any* batch of files containing a mix of valid-format and invalid-format files, the ingestion pipeline SHALL produce: a structured error for every invalid-format file, and a valid `ParsedDocument` for every valid-format file. The total `failed_files` count in `IngestionResult` SHALL equal the number of invalid-format files submitted, and `total_documents` SHALL equal the number of valid files.

**Validates: Requirements 1.2, 1.4**

---

### Property 3: Fixed Chunker Respects Size and Overlap Invariants

*For any* document with any character-length content and any configurable `chunk_size` ≥ 1 and `chunk_overlap` ≥ 0, the `FixedChunker` SHALL produce chunks where: every chunk's `len(text)` ≤ `chunk_size`, and for every pair of consecutive chunks, the last `chunk_overlap` characters of chunk[i] appear in chunk[i+1].

**Validates: Requirements 2.1**

---

### Property 4: Semantic Chunker Splits at Similarity Drops

*For any* sequence of sentences where the embedder returns controlled similarity scores, the `SemanticChunker` SHALL insert a chunk boundary when and only when the cosine similarity between consecutive sentence embeddings drops below the configured threshold. Above-threshold pairs SHALL NOT be split; below-threshold pairs SHALL be split.

**Validates: Requirements 2.3**

---

### Property 5: All Chunking Strategies Attach Complete Metadata to Every Chunk

*For any* document and any of the three chunking strategies (`fixed`, `structural`, `semantic`), every produced chunk SHALL have non-null values for: `doc_id`, `chunk_index`, `char_start`, `char_end`, and `strategy`. Furthermore, `char_start` SHALL be ≥ 0, `char_end` SHALL be > `char_start`, and `strategy` SHALL match the strategy that produced it.

**Validates: Requirements 2.4**

---

### Property 6: Deduplication Removes All Near-Duplicates

*For any* set of chunks (including exact duplicates and near-duplicates with cosine similarity > 0.95), after deduplication, no two remaining chunks SHALL have pairwise cosine similarity exceeding 0.95. Furthermore, all chunks that were not near-duplicates of any other chunk SHALL be retained.

**Validates: Requirements 2.6**

---

### Property 7: Embedder Produces Non-Null Vectors for All Chunks

*For any* non-empty list of chunks, `embed_batch()` SHALL return a list of the same length where every element is a non-null list of floats with the correct embedding dimension (1536 for OpenAI `text-embedding-3-small`, 384 for `all-MiniLM-L6-v2`).

**Validates: Requirements 3.1**

---

### Property 8: Vector Store Round-Trip Preserves Chunk Data

*For any* chunk with a non-null embedding, after upserting to the Qdrant vector store and fetching by `chunk_id`, the retrieved payload SHALL contain values equal to the original `text`, `doc_id`, `chunk_index`, `char_start`, `char_end`, and `strategy` fields.

**Validates: Requirements 3.3**

---

### Property 9: BM25 Index Persistence Round-Trip

*For any* set of chunks used to build a BM25 index, saving the index to disk and loading it into a fresh instance SHALL produce search results identical (same chunk IDs in same rank order) to the results from the original in-memory index for any query string.

**Validates: Requirements 3.4**

---

### Property 10: Retrieval Results Are Sorted and Bounded

*For any* query vector (dense) or query string (sparse) issued to the Dense_Retriever or Sparse_Retriever, the returned results SHALL be: sorted in descending order by score, and contain at most `top_k` items.

**Validates: Requirements 4.1, 4.2**

---

### Property 11: RRF Fusion Correctly Merges Ranked Lists

*For any* two non-empty ranked lists (dense and sparse results) and any `alpha` in [0.0, 1.0]:
- The output SHALL be sorted in descending order by RRF score.
- A chunk appearing in both lists SHALL have a higher or equal RRF score than a chunk appearing in only one list, given equivalent ranks.
- When `alpha = 1.0`, the output order SHALL match the dense list order.
- When `alpha = 0.0`, the output order SHALL match the sparse list order.

**Validates: Requirements 4.3**

---

### Property 12: Reranker Selects Top-5 by Score

*For any* list of candidate chunks scored by the reranker, the output SHALL contain exactly `min(5, len(candidates))` items, all selected chunks SHALL have reranker scores ≥ every non-selected chunk's score, and the output SHALL be sorted in descending order by reranker score.

**Validates: Requirements 5.1, 5.2**

---

### Property 13: Generator Prompt Contains All Chunk Texts and Citation Instructions

*For any* set of top-5 reranked chunks, the prompt sent by the Generator to the LLM SHALL contain: the full plaintext of every chunk as inline context with its `[n]` index label, and explicit instructions to cite claims using bracketed `[n]` notation.

**Validates: Requirements 6.1, 6.2**

---

### Property 14: Citation Verifier Produces Valid Status for All Citations

*For any* generated answer containing `[n]` citations, the `CitationVerifier` SHALL be called once for each distinct citation index, and every call SHALL return exactly one of `{"supported", "partial", "unsupported"}`. No citation in the final response SHALL have a `null` verification_status.

**Validates: Requirements 6.3, 6.4**

---

### Property 15: Low-Relevance Guard Prevents Generation

*For any* set of retrieved chunks where all relevance scores are strictly below the configured `min_relevance_threshold`, the Generator SHALL return a `GenerationResult` with `is_grounded = False` and SHALL NOT invoke the LLM. Conversely, if at least one chunk score is ≥ `min_relevance_threshold`, the LLM SHALL be invoked.

**Validates: Requirements 6.5**

---

### Property 16: Confidence Score Matches the Weighted Formula

*For any* triple of `(mean_retrieval_relevance, citation_coverage_rate, completeness_score)` all in [0.0, 1.0], `compute_confidence_score()` SHALL return exactly `0.4 * mean_retrieval_relevance + 0.4 * citation_coverage_rate + 0.2 * completeness_score`, and the result SHALL be in [0.0, 1.0].

**Validates: Requirements 6.6**

---

### Property 17: API Response Contains All Required Fields for Any Valid Query

*For any* well-formed `POST /v1/ask` request with a valid non-empty `query`, the API SHALL return HTTP 200 with a response body containing non-null `answer` (string), `citations` (array), `confidence_score` (float in [0,1]), `latency_ms` (object with all stage breakdowns), and `is_grounded` (boolean).

**Validates: Requirements 8.2**

---

### Property 18: API Returns 422 for All Malformed Requests

*For any* request body that violates the schema of any API endpoint (missing required fields, wrong types, `rrf_weight` outside [0.0, 1.0], empty `query`, etc.), the API SHALL return HTTP 422 with a structured JSON error body describing the specific validation failure.

**Validates: Requirements 8.6**

---

## Error Handling

### Error Categories and Responses

| Category | Trigger | Response |
|---|---|---|
| Unsupported file format | Extension not in `.md .txt .html .pdf` | `IngestionError` with filename and format; ingestion continues |
| Parse failure | Corrupted or unreadable file | `IngestionError` with filename and exception type; ingestion continues |
| Embedding API failure | OpenAI unavailable or key missing | Fallback to sentence-transformers; log warning |
| Qdrant unavailable | Connection refused at startup | Service fails to start; Docker Compose restarts |
| BM25 index missing | `bm25_index.pkl` not found | Rebuild from Qdrant payloads on startup; log warning |
| No relevant chunks | All scores below `min_relevance_threshold` | Structured "I don't know" response; `is_grounded = False` |
| LLM generation failure | OpenAI API error during generation | HTTP 500 with request_id; full stack trace in logs only |
| Malformed request | Pydantic validation failure | HTTP 422 with field-level error details |
| Unhandled exception | Any unexpected error in a route | HTTP 500 with request_id and sanitized message; full trace in logs |

### Error Response Schema

```python
class ErrorResponse(BaseModel):
    request_id: str
    error_code: str         # e.g., "VALIDATION_ERROR", "INTERNAL_ERROR"
    message: str            # Human-readable, non-sensitive
    details: dict | None    # Field-level details for 422s
```

### Graceful Degradation Chain

```
OpenAI API key present?
  YES → Use text-embedding-3-small
  NO  → Warn + use sentence-transformers/all-MiniLM-L6-v2

OpenAI chat model available?
  YES → Use gpt-4o-mini for generation
  NO  → Return structured error (generation is not optional)

BM25 index on disk?
  YES → Load it
  NO  → Rebuild from Qdrant payloads (slower startup, logged)
```

---

## Testing Strategy

### Dual Testing Approach

The pipeline uses both unit/property-based tests and integration tests. Property-based tests validate universal invariants across a large input space; unit/example tests cover specific scenarios and edge cases. Together they provide comprehensive coverage without over-investing in redundant example tests.

### Property-Based Testing

**Library**: [`hypothesis`](https://hypothesis.readthedocs.io/) — the standard Python PBT library with composite strategies, database for shrinking, and Pytest integration.

**Configuration**: Each property test runs a minimum of 100 examples (set via `@settings(max_examples=100)`). Slow tests (reranker, citation verifier) mock the ML model and run 100 examples in < 5 seconds.

**Tag format** for each test: `# Feature: hybrid-rag-pipeline, Property {N}: {property_text}`

```python
# Example property test structure
from hypothesis import given, settings, strategies as st

# Feature: hybrid-rag-pipeline, Property 16: Confidence score matches weighted formula
@given(
    retrieval=st.floats(min_value=0.0, max_value=1.0),
    coverage=st.floats(min_value=0.0, max_value=1.0),
    completeness=st.floats(min_value=0.0, max_value=1.0),
)
@settings(max_examples=100)
def test_confidence_score_formula(retrieval, coverage, completeness):
    result = compute_confidence_score([retrieval], coverage, completeness)
    expected = 0.4 * retrieval + 0.4 * coverage + 0.2 * completeness
    assert abs(result - expected) < 1e-9
    assert 0.0 <= result <= 1.0
```

### Test Organization

```
tests/
├── unit/
│   ├── test_chunkers.py          # Properties 3, 4, 5 — chunking invariants
│   ├── test_deduplication.py     # Property 6 — near-duplicate removal
│   ├── test_rrf_fusion.py        # Property 11 — RRF correctness
│   ├── test_confidence.py        # Property 16 — confidence formula
│   ├── test_citation_verifier.py # Property 14 — verifier output validity
│   └── test_generator.py         # Properties 13, 15 — prompt + guard
├── integration/
│   ├── test_ingestion.py         # Properties 1, 2 — parsing + error handling
│   ├── test_embedder.py          # Property 7 — embedding output
│   ├── test_vector_store.py      # Property 8 — Qdrant round-trip (requires Qdrant)
│   ├── test_bm25.py              # Property 9 — BM25 persistence
│   ├── test_retrieval.py         # Property 10 — sorted/bounded results
│   ├── test_reranker.py          # Property 12 — top-5 selection
│   └── test_api.py               # Properties 17, 18 — API response contract
└── smoke/
    ├── test_corpus.py            # Req 1.3 — corpus file count and categories
    ├── test_config.py            # Req 2.5 — strategy selection by config
    ├── test_openapi.py           # Req 8.5 — /docs is accessible
    └── test_eval_harness.py      # Req 7.1, 7.3, 7.4, 7.6 — harness outputs
```

### What Unit Tests Focus On

- **Chunkers**: Specific documents with known structural boundaries; edge cases like empty documents, single-sentence documents, documents with no headings.
- **RRF**: Specific rank lists where the correct merged rank is known by hand calculation.
- **Citation Verifier**: Known entailment examples from the MS MARCO or NLI literature.
- **API Error Handling**: Specific malformed payloads (null query, rrf_weight=2.0, unknown chunking strategy).

### Integration Test Strategy

Integration tests run against real services via Docker Compose `--profile test`. The Qdrant service is started; BM25 index is built from a small 5-chunk test corpus. These tests are tagged `@pytest.mark.integration` and excluded from the default `pytest` run; CI runs them separately.

### Eval Harness Tests

The eval harness is tested with a 5-question subset and fully mocked LLM responses to verify: metric computation is correct, the report structure is valid, and the offline cache mechanism works. The full 50+ Q&A dataset is validated with a smoke test (correct count and category distribution).

### Coverage Targets

| Package | Target |
|---|---|
| `ingestion` | ≥ 90% line coverage |
| `retrieval` | ≥ 85% line coverage |
| `generation` | ≥ 85% line coverage |
| `api` | ≥ 80% line coverage |
| `eval` | ≥ 75% line coverage |
