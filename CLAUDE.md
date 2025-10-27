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
- **CHUNKING_STRATEGY**: `"paragraph"` or `"recursive"` - switches chunking algorithms
- **EMBEDDING_MODEL**: sentence-transformers model name (default: `intfloat/multilingual-e5-large`)
- **CHUNK_SIZE** / **CHUNK_OVERLAP**: Used only when `CHUNKING_STRATEGY=recursive`

Configuration is loaded via `app/config.py` using pydantic-settings and validated at startup.

## Architecture & Key Patterns

### Service Initialization Pattern

All services (embeddings, Qdrant, chunker) are initialized **once** during FastAPI lifespan startup and stored as global variables. This ensures:
- Embedding model is loaded only once (expensive operation)
- Qdrant connection is reused
- Chunker is selected based on config at startup

See `app/main.py` lifespan context manager (lines 34-64).

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
- `IngestRequest`, `IngestResponse`
- `QueryRequest`, `QueryResponse`
- `SearchResult`, `HealthResponse`

## API Endpoints

### POST /ingest
Receives document name and text, generates UUID, chunks it, creates embeddings, stores in Qdrant. Returns the generated document_id.

```python
{"document_name": "Tender #123", "text": "document content..."}
# Returns: {"success": true, "chunks_count": 15, "document_id": "550e8400-..."}
```

### POST /query
Performs semantic search within a specific document (by document_id).

```python
{"document_id": "550e8400-...", "query": "search query", "top_k": 5}
```

### GET /health
Returns system status including Qdrant connectivity.

## Qdrant Integration Details

Single collection named "documents" (configurable via `COLLECTION_NAME` in .env). The collection is created automatically on application startup.

Each chunk is stored as a point with:
- UUID as point ID (unique per chunk)
- Embedding vector (dimension determined by embedding model)
- Payload:
  - `text`: chunk content
  - `chunk_index`: position in document
  - `document_id`: UUID identifying the source document
  - `document_name`: human-readable document name
  - `upload_timestamp`: ISO timestamp of when document was ingested
  - Additional metadata from chunking strategy

Search is filtered by `document_id` using Qdrant's Filter with FieldCondition. Distance metric: COSINE similarity.

## Embedding Model Behavior

First run downloads the model from HuggingFace (can be large, requires internet). Models are cached by sentence-transformers library. The model is loaded synchronously at startup - expect delay on first run.

Embeddings are normalized for cosine similarity search.

## Development Notes

- Models imports: Use `from .schemas import ...` (not `.models`)
- All services are initialized globally in `main.py` after settings load
- Chunking happens synchronously; for very large documents, consider async chunking
- Single Qdrant collection is created at startup - no per-document collections
- Document IDs are UUIDs generated server-side on ingest
- Search always requires document_id - no cross-document search
- Each chunk payload includes document_id, document_name, and upload_timestamp
- CORS is wide open (`allow_origins=["*"]`) - restrict in production
