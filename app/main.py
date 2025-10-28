from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
import uuid
from datetime import datetime, UTC

from .config import settings
from .schemas import (
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    HealthResponse,
    SearchResult,
)
from .qdrant_client import get_qdrant_service
from .embeddings import get_embedding_service
from .chunking import ParagraphChunker, RecursiveChunker
from .scripts.doc_to_text import extract_text_from_file

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Глобальные сервисы
qdrant_service = None
embedding_service = None
chunker = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """События жизненного цикла приложения"""
    global qdrant_service, embedding_service, chunker

    # Запуск
    logger.info("Starting DocMind application...")
    logger.info(f"Chunking strategy: {settings.chunking_strategy}")

    # Инициализация сервисов
    embedding_service = get_embedding_service()
    qdrant_service = get_qdrant_service()

    # Инициализация чанкера на основе конфигурации
    if settings.chunking_strategy == "paragraph":
        chunker = ParagraphChunker()
        logger.info("Using ParagraphChunker")
    elif settings.chunking_strategy == "recursive":
        chunker = RecursiveChunker(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap
        )
        logger.info(f"Using RecursiveChunker (size={settings.chunk_size}, overlap={settings.chunk_overlap})")
    else:
        raise ValueError(f"Unknown chunking strategy: {settings.chunking_strategy}")

    logger.info("Application started successfully")

    yield

    # Завершение
    logger.info("Shutting down DocMind application...")


# Создание FastAPI приложения
app = FastAPI(
    title="DocMind",
    description="RAG система для загрузки документов и семантического поиска",
    version="1.0.0",
    lifespan=lifespan
)

# Добавление CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Health"])
async def root():
    """Корневой endpoint"""
    return {
        "message": "DocMind API is running",
        "docs_url": "/docs",
        "health_check": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Endpoint проверки здоровья"""
    try:
        # Проверить подключение к Qdrant
        qdrant_connected = True
        try:
            qdrant_service.client.get_collections()
        except Exception as e:
            logger.error(f"Qdrant connection failed: {e}")
            qdrant_connected = False

        return HealthResponse(
            status="healthy" if qdrant_connected else "degraded",
            qdrant_connected=qdrant_connected,
            embedding_model=settings.embedding_model,
            chunking_strategy=settings.chunking_strategy
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e)
        )


@app.post("/ingest", response_model=IngestResponse, tags=["Documents"])
async def ingest_document(request: IngestRequest):
    """
    Загрузить документ в систему

    Этот endpoint:
    1. Получает текст документа
    2. Разбивает его на чанки на основе настроенной стратегии
    3. Создает embeddings для каждого чанка
    4. Сохраняет чанки в Qdrant
    """
    try:
        # Генерировать UUID для документа
        document_id = str(uuid.uuid4())
        upload_timestamp = datetime.now(UTC)

        logger.info(f"Ingesting document: {request.document_name}")
        logger.info(f"Document ID: {document_id}")
        logger.info(f"Document length: {len(request.text)} characters")

        # Разбить текст на чанки
        chunks = chunker.chunk(request.text)
        logger.info(f"Created {len(chunks)} chunks")

        if not chunks:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No chunks were created from the document. The text might be too short."
            )

        # Извлечь текст и метаданные из чанков
        chunk_texts = [chunk.text for chunk in chunks]
        chunk_metadata = [chunk.metadata for chunk in chunks]

        # Загрузить в Qdrant
        chunks_count = qdrant_service.ingest_chunks(
            document_id=document_id,
            document_name=request.document_name,
            upload_timestamp=upload_timestamp,
            chunks=chunk_texts,
            metadata=chunk_metadata
        )

        logger.info(f"Successfully ingested {chunks_count} chunks for document {document_id}")

        return IngestResponse(
            success=True,
            message=f"Successfully ingested {chunks_count} chunks",
            chunks_count=chunks_count,
            document_id=document_id
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error ingesting document: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to ingest document: {str(e)}"
        )


@app.post("/query", response_model=QueryResponse, tags=["Search"])
async def query_documents(request: QueryRequest):
    """
    Запросить документы используя семантический поиск

    Этот endpoint:
    1. Получает текст запроса
    2. Создает embedding для запроса
    3. Ищет похожие чанки в Qdrant
    4. Возвращает наиболее релевантные чанки
    """
    try:
        logger.info(f"Querying document: {request.document_id}")
        logger.info(f"Query: {request.query}")

        # Поиск в Qdrant с фильтрацией по document_id
        results = qdrant_service.search(
            document_id=request.document_id,
            query=request.query,
            top_k=request.top_k
        )

        logger.info(f"Found {len(results)} results")

        # Форматировать результаты
        search_results = [
            SearchResult(
                text=result["text"],
                score=result["score"],
                metadata=result["metadata"]
            )
            for result in results
        ]

        return QueryResponse(
            success=True,
            results=search_results,
            query=request.query,
            document_id=request.document_id
        )

    except Exception as e:
        logger.error(f"Error querying documents: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to query documents: {str(e)}"
        )


@app.post("/test", response_model=IngestResponse, tags=["Documents"])
async def test_ingest_document():
    """
    Тестовый endpoint для загрузки test.docx

    Извлекает текст из test.docx и загружает его в систему
    """
    try:
        # Извлечь текст из test.docx
        text = extract_text_from_file('test.docx')

        if not text or not text.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to extract text from test.docx or file is empty"
            )

        # Создать запрос для ingest
        ingest_request = IngestRequest(
            document_name="test.docx",
            text=text
        )

        # Вызвать основной endpoint для загрузки
        return await ingest_document(ingest_request)

    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="test.docx file not found"
        )

    except Exception as e:
        logger.error(f"Error in test_ingest_document: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process test.docx: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True
    )
