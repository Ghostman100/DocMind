# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DocMind is a RAG (Retrieval-Augmented Generation) system that ingests documents, chunks them, creates embeddings, and enables semantic search. All documents are stored in a single Qdrant collection named "documents". Each document is assigned a unique UUID (document_id), and all chunks from that document share this ID.

**Data flow:**
```
Document → Chunking → Embeddings → Qdrant (single "documents" collection)
                                         ↓
                              All chunks tagged with document_id
                                         ↓
                          Query (filters by document_id) → Results
```

## Running the Application

```bash
# Install dependencies
pip install -r requirements.txt

# Setup environment (required before first run)
cp .env.example .env
# Edit .env to set QDRANT_URL

# Run the server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Or via Python
python -m app.main
```

API documentation available at `http://localhost:8000/docs`

## Configuration System

The application uses environment-based configuration via `.env` file:

- **QDRANT_URL**: Required. Qdrant vector database URL
- **QDRANT_API_KEY**: Optional. API key for Qdrant authentication
- **COLLECTION_NAME**: Collection name in Qdrant (default: `"documents2"`)
- **CHUNKING_STRATEGY**: `"paragraph"` or `"recursive"` - switches chunking algorithms
- **EMBEDDING_MODEL**: Legacy single model setting (kept for backward compatibility)
- **EMBEDDING_MODELS**: List of models to load at startup (all kept in RAM). Default: `["sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "intfloat/multilingual-e5-base", "deepvk/USER-bge-m3"]`
- **DEFAULT_EMBEDDING_MODEL**: Model used by `/ingest` and `/query` endpoints. Default: `"sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"`
- **DEFAULT_VECTOR_NAME**: Named vector in Qdrant for default model. Default: `"fast-paraphrase-multilingual-minilm-l12-v2"`
- **CHUNK_SIZE** / **CHUNK_OVERLAP**: Used only when `CHUNKING_STRATEGY=recursive`
- **API_HOST** / **API_PORT**: Server binding (default: `0.0.0.0:8000`)

Configuration is loaded via `app/config.py` using pydantic-settings and validated at startup.

## Architecture & Key Patterns

### Service Initialization Pattern

All services (embeddings, Qdrant, chunker) are initialized **once** during FastAPI lifespan startup and stored as global variables. This ensures:
- Embedding models are loaded only once (expensive operation)
- Qdrant connection is reused
- Chunker is selected based on config at startup

See `app/main.py` lifespan context manager (lines 38-78).

### Multi-Model Embedding System (EmbeddingRegistry Pattern)

The system supports loading and managing multiple embedding models simultaneously via `EmbeddingRegistry`:
- **EmbeddingRegistry** (`app/embeddings.py`): Central registry that manages multiple `EmbeddingService` instances
- All models listed in `EMBEDDING_MODELS` config are loaded at startup and kept in RAM
- Each model is wrapped in an `EmbeddingService` singleton
- Default model is used by `/ingest` and `/query` endpoints automatically
- `/embed` endpoint allows using any registered model explicitly

The registry pattern enables:
- Model reuse across requests (no reloading)
- Support for comparing different embedding models
- Easy addition of new models without code changes

See `app/embeddings.py` for `EmbeddingRegistry` and `get_embedding_registry()` singleton factory.

### Chunking Strategy System

Two chunking strategies with a common interface (`BaseChunker`):
- **ParagraphChunker** (`app/chunking/paragraph.py`): Splits on `\n\n`
- **RecursiveChunker** (`app/chunking/recursive.py`): LangChain-style recursive splitting with overlap

The chunker is selected at startup based on `settings.chunking_strategy`. To add a new strategy:
1. Extend `BaseChunker` in `app/chunking/`
2. Update `app/chunking/__init__.py`
3. Add logic in `lifespan()` startup to instantiate it
4. Update `config.py` Literal type hint

### Singleton Service Pattern

`embeddings.py` and `qdrant_client.py` use singleton pattern via `get_*_service()` functions to ensure single instances across the application.

### Schema Location

Pydantic models are in `app/schemas.py` (not `models.py`). This includes:
- `IngestRequest`, `IngestResponse` - Document ingestion
- `QueryRequest`, `QueryResponse` - Semantic search
- `SearchResult` - Individual search result with text, score, metadata
- `HealthResponse` - System health status
- `EmbedRequest`, `EmbedResponse` - Direct embedding generation

All schemas include JSON examples in `model_config` for OpenAPI documentation.

## LangChain Integration

The system now includes **parallel LangChain-powered functionality** alongside the existing custom implementation. This provides advanced features like document summarization and point extraction.

### Architecture: Dual-Track Approach

- **Original endpoints** (`/ingest`, `/query`, `/embed`, `/test`): Continue to work using custom implementation
- **LangChain endpoints** (`/langchain/*`): New alternative implementation with LLM-powered features
- **Shared collection**: Both systems use the same Qdrant collection ("documents2")
- **Compatible metadata**: Document IDs and metadata work across both systems

### LangChain Components

**app/langchain_integration/** module includes:
- `vector_store.py`: QdrantVectorStore wrapper that integrates with existing Qdrant connection
- `summarizer.py`: Document summarization and point extraction using LangChain chains
- `schemas.py`: Pydantic schemas for LangChain endpoints

### Configuration

LangChain features require additional configuration in `.env`:
- **LANGCHAIN_ENABLED**: Set to `true` to enable LangChain endpoints that require LLM
- **LLM_PROVIDER**: "openai" or "anthropic"
- **LLM_API_KEY**: API key for the LLM provider (required for summarization)
- **LLM_MODEL**: Model name (default: "gpt-4o-mini")
- **LLM_TEMPERATURE**: Temperature for LLM responses (default: 0.0 for deterministic)

Note: `/langchain/ingest` and `/langchain/query` work without LLM API key. Only summarization and extraction (with summarize=True) require LLM configuration.

## API Endpoints

### GET /
Root endpoint, returns API info and available endpoints.

### GET /health
Returns system status including Qdrant connectivity and loaded models.

```json
{
  "status": "healthy",
  "qdrant_connected": true,
  "embedding_models": ["model1", "model2", "model3"],
  "default_embedding_model": "model1",
  "chunking_strategy": "paragraph"
}
```

### POST /ingest
Receives document name and text, generates UUID, chunks it, creates embeddings using default model, stores in Qdrant. Returns the generated document_id.

```python
{"document_name": "Tender #123", "text": "document content..."}
# Returns: {"success": true, "message": "...", "chunks_count": 15, "document_id": "550e8400-..."}
```

### POST /query
Performs semantic search within a specific document (by document_id) using default embedding model.

```python
{"document_id": "550e8400-...", "query": "search query", "top_k": 5}
# Returns: {"success": true, "results": [...], "query": "...", "document_id": "..."}
```

### POST /embed
Creates embedding vector for arbitrary text using any registered model. Useful for testing models or external integrations.

```python
{"model": "intfloat/multilingual-e5-base", "text": "some text to embed"}
# Returns: {"embedding": [0.1, 0.2, ...], "model": "...", "dimension": 768}
```

### POST /test
Test endpoint that extracts text from `test.docx` file (must exist in project root) and ingests it. Useful for development/testing.

Note: Requires `app/scripts/doc_to_text.py` module (600KB+ file with document parsing logic).

---

## LangChain API Endpoints

### POST /langchain/ingest
Alternative document ingestion using LangChain RecursiveCharacterTextSplitter. Compatible with existing `/ingest` - documents can be queried by either system.

```python
{"document_name": "Tender #123", "text": "document content..."}
# Returns: {"success": true, "message": "...", "chunks_count": 15, "document_id": "550e8400-..."}
```

No LLM API key required.

### POST /langchain/query
Semantic search using LangChain retriever interface. Works with documents ingested via either `/ingest` or `/langchain/ingest`.

```python
{"document_id": "550e8400-...", "query": "search query", "top_k": 5}
# Returns: {"success": true, "results": [...], "query": "...", "document_id": "..."}
```

No LLM API key required.

### POST /langchain/summarize
**New feature**: Create a shortened version of a document using LLM-powered summarization. Retrieves all chunks of a document and applies MapReduce/Stuff/Refine summarization strategy.

```python
{
  "document_id": "550e8400-...",
  "strategy": "map_reduce",  # "stuff", "map_reduce", or "refine"
  "max_chunks": 100
}
# Returns: {"success": true, "summary": "...", "chunks_processed": 15, ...}
```

**Requires**: `LANGCHAIN_ENABLED=true` and `LLM_API_KEY` in `.env`

**Strategies**:
- `stuff`: For short documents, all chunks in one prompt (fastest, limited by context window)
- `map_reduce`: Summarize each chunk, then combine summaries (best for long documents)
- `refine`: Iterative refinement of summary (most thorough, slowest)

### POST /langchain/extract_points
**New feature**: Extract specific points from a document by topics. Performs semantic search for each topic within the document and optionally summarizes findings.

```python
{
  "document_id": "550e8400-...",
  "topics": [
    "Требования к участникам",
    "Сроки подачи заявок",
    "Критерии оценки"
  ],
  "chunks_per_topic": 3,
  "summarize": true
}
# Returns: {
#   "success": true,
#   "extracted_points": [
#     {"topic": "...", "relevant_chunks": [...], "summary": "..."},
#     ...
#   ],
#   "total_chunks_retrieved": 9
# }
```

**Requires**: `LLM_API_KEY` only if `summarize=true`. Without summarization, returns raw relevant chunks.

## Qdrant Integration Details

Single collection named "documents2" by default (configurable via `COLLECTION_NAME` in .env). The collection is created automatically on application startup with support for multiple named vectors.

Each chunk is stored as a point with:
- UUID as point ID (unique per chunk)
- Named embedding vectors (one per registered model, dimension determined by each model)
- Payload:
  - `text`: chunk content
  - `chunk_index`: position in document
  - `document_id`: UUID identifying the source document
  - `document_name`: human-readable document name
  - `upload_timestamp`: ISO timestamp of when document was ingested
  - Additional metadata from chunking strategy (e.g., `type`, `char_count`)

**Multi-vector support**: The system uses Qdrant's named vectors feature to store embeddings from multiple models in the same collection. Each model has its own vector name (derived from `DEFAULT_VECTOR_NAME` config).

Search is filtered by `document_id` using Qdrant's Filter with FieldCondition. Distance metric: COSINE similarity. Query embeddings use the same named vector as was used during ingestion.

## Embedding Model Behavior

First run downloads models from HuggingFace (can be large, requires internet). Models are cached by sentence-transformers library in `~/.cache/huggingface` and `~/.cache/sentence-transformers`. All models listed in `EMBEDDING_MODELS` are loaded synchronously at startup - expect significant delay on first run.

**Cache locations** (configurable via env vars):
- `TRANSFORMERS_CACHE`: HuggingFace transformers cache
- `SENTENCE_TRANSFORMERS_HOME`: Sentence-transformers cache

Embeddings are normalized (`normalize_embeddings=True`) for cosine similarity search.

**Model size considerations**: Loading multiple large models keeps them all in RAM. Consider memory requirements:
- `paraphrase-multilingual-MiniLM-L12-v2`: ~420MB
- `multilingual-e5-base`: ~1.1GB
- `USER-bge-m3`: ~2.2GB

## Docker Deployment

A multi-stage Dockerfile is provided with CPU-only PyTorch optimization:

```bash
# Build image
docker build -t docmind:latest .

# Run container
docker run -p 8000:8000 --env-file .env docmind:latest
```

**Dockerfile features**:
- Multi-stage build reduces image size
- CPU-only PyTorch installation (~2GB savings vs CUDA version)
- Non-root user (`appuser`) for security
- Model cache directories pre-created
- Health check endpoint configured (`/health`)
- Environment variables for cache locations

**Environment variables in container**:
- `TRANSFORMERS_CACHE=/home/appuser/.cache/huggingface`
- `SENTENCE_TRANSFORMERS_HOME=/home/appuser/.cache/sentence-transformers`

Models will be downloaded on first container run unless you mount a pre-populated cache directory.

## Development Notes

- Pydantic models are in `app/schemas.py` (not `models.py`). Use `from .schemas import ...`
- LangChain schemas are in `app/langchain_integration/schemas.py`
- All services (EmbeddingRegistry, Qdrant, chunker) are initialized globally in `main.py` lifespan during startup
- `EmbeddingRegistry` pattern allows multiple models to coexist; access via `embedding_registry.get_model(name)`
- LangChain vector store service uses singleton pattern via `get_langchain_vector_store()`
- Chunking happens synchronously; for very large documents, consider async chunking
- Single Qdrant collection with named vectors for multiple models - no per-document collections
- Document IDs are UUIDs generated server-side on ingest (not client-provided)
- Search always requires document_id - no cross-document search currently
- Each chunk payload includes document_id, document_name, upload_timestamp, and chunking metadata
- **LangChain compatibility**: Documents ingested via `/ingest` can be queried/summarized via LangChain endpoints and vice versa
- **LLM costs**: Summarization endpoints use external LLM APIs (OpenAI/Anthropic) - monitor usage and costs
- **Russian language**: All LangChain summarization prompts are in Russian for optimal results
- CORS is wide open (`allow_origins=["*"]`) - restrict in production
- `app/scripts/doc_to_text.py` is a large (600KB+) file with document parsing utilities; avoid reading entire file unless necessary
- PyTorch CPU-only installation recommended: use `--index-url https://download.pytorch.org/whl/cpu` when installing torch

## LangChain Dependencies

Additional dependencies for LangChain functionality:
```
langchain==0.3.7
langchain-core==0.3.15
langchain-qdrant==0.2.0
langchain-community==0.3.5
langchain-text-splitters==0.3.2
langchain-openai==0.2.5
```

These are included in `requirements.txt` and will be installed automatically.
