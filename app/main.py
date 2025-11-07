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
    EmbedRequest,
    EmbedResponse,
)
from .qdrant_client import QdrantService
from .embeddings import get_embedding_registry
from .chunking import ParagraphChunker, RecursiveChunker
from .scripts.doc_to_text import extract_text_from_file

# LangChain интеграция
from .langchain_integration.schemas import (
    LangChainIngestRequest,
    LangChainIngestResponse,
    LangChainQueryRequest,
    LangChainQueryResponse,
    LangChainSearchResult,
    SummarizeRequest,
    SummarizeResponse,
    ExtractPointsRequest,
    ExtractPointsResponse,
    ExtractedPoint,
)
from .langchain_integration.vector_store import get_langchain_vector_store
from .langchain_integration.summarizer import summarize_document, extract_points_from_document

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Глобальные сервисы
qdrant_service = None
embedding_registry = None
chunker = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """События жизненного цикла приложения"""
    global qdrant_service, embedding_registry, chunker

    # Запуск
    logger.info("Starting DocMind application...")
    logger.info(f"Chunking strategy: {settings.chunking_strategy}")

    # Инициализация embedding registry и загрузка моделей
    logger.info(f"Loading {len(settings.embedding_models)} embedding models...")
    embedding_registry = get_embedding_registry()

    for model_name in settings.embedding_models:
        logger.info(f"Registering model: {model_name}")
        embedding_registry.register_model(model_name)

    logger.info(f"All models loaded. Default model: {settings.default_embedding_model}")

    # Инициализация Qdrant service с registry
    qdrant_service = QdrantService(embedding_registry)

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
            embedding_models=embedding_registry.list_models(),
            default_embedding_model=settings.default_embedding_model,
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

        # Загрузить в Qdrant используя модель по умолчанию
        chunks_count = qdrant_service.ingest_chunks(
            document_id=document_id,
            document_name=request.document_name,
            upload_timestamp=upload_timestamp,
            chunks=chunk_texts,
            model_name=settings.default_embedding_model,
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

        # Поиск в Qdrant с фильтрацией по document_id, используя модель по умолчанию
        results = qdrant_service.search(
            document_id=request.document_id,
            query=request.query,
            model_name=settings.default_embedding_model,
            top_k=request.top_k
        )

        logger.info(f"Found {len(results)} results")

        # Форматировать результаты
        search_results = [
            SearchResult(
                document=result["document"],
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


@app.post("/embed", response_model=EmbedResponse, tags=["Embeddings"])
async def get_embedding(request: EmbedRequest):
    """
    Получить embedding для текста

    Этот endpoint:
    1. Получает текст и название модели
    2. Создает embedding используя указанную модель
    3. Возвращает вектор embedding с метаданными
    """
    try:
        logger.info(f"Creating embedding with model: {request.model}")
        logger.info(f"Text length: {len(request.text)} characters")

        # Проверить что модель зарегистрирована
        if not embedding_registry.has_model(request.model):
            available_models = embedding_registry.list_models()
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Model '{request.model}' not available. Available models: {available_models}"
            )

        # Получить embedding service для модели
        embedding_service = embedding_registry.get_model(request.model)

        # Создать embedding
        embedding_vector = embedding_service.encode_single(request.text)

        logger.info(f"Successfully created embedding with dimension {len(embedding_vector)}")

        return EmbedResponse(
            embedding=embedding_vector.tolist(),
            model=request.model,
            dimension=len(embedding_vector)
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating embedding: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create embedding: {str(e)}"
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


@app.post("/test_langchain", response_model=LangChainIngestResponse, tags=["LangChain"])
async def test_ingest_langchain():
    """
    Тестовый endpoint для загрузки test.docx через LangChain

    Извлекает текст из test.docx и загружает его в систему через LangChain
    """
    try:
        # Извлечь текст из test.docx
        text = extract_text_from_file('test.docx')

        if not text or not text.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to extract text from test.docx or file is empty"
            )

        # Создать запрос для LangChain ingest
        ingest_request = LangChainIngestRequest(
            document_name="test.docx",
            text=text
        )

        # Вызвать LangChain endpoint для загрузки
        return await langchain_ingest_document(ingest_request)

    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="test.docx file not found"
        )

    except Exception as e:
        logger.error(f"Error in test_ingest_langchain: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process test.docx via LangChain: {str(e)}"
        )


# ============================================================================
# LangChain Endpoints
# ============================================================================

@app.post("/langchain/ingest", response_model=LangChainIngestResponse, tags=["LangChain"])
async def langchain_ingest_document(request: LangChainIngestRequest):
    """
    Загрузить документ в систему через LangChain

    Этот endpoint использует LangChain RecursiveCharacterTextSplitter
    для разбиения текста и QdrantVectorStore для сохранения.
    """
    try:
        # Генерировать UUID для документа
        document_id = str(uuid.uuid4())

        logger.info(f"[LangChain] Ingesting document: {request.document_name}")
        logger.info(f"[LangChain] Document ID: {document_id}")
        logger.info(f"[LangChain] Document length: {len(request.text)} characters")

        # Использовать LangChain text splitter
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]
        )

        texts = text_splitter.split_text(request.text)
        logger.info(f"[LangChain] Created {len(texts)} chunks")

        if not texts:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No chunks were created from the document. The text might be too short."
            )

        # Загрузить в Qdrant через LangChain
        langchain_service = get_langchain_vector_store()
        chunks_count = langchain_service.ingest_texts(
            texts=texts,
            document_id=document_id,
            document_name=request.document_name
        )

        logger.info(f"[LangChain] Successfully ingested {chunks_count} chunks for document {document_id}")

        return LangChainIngestResponse(
            success=True,
            message=f"Successfully ingested {chunks_count} chunks via LangChain",
            chunks_count=chunks_count,
            document_id=document_id
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[LangChain] Error ingesting document: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to ingest document via LangChain: {str(e)}"
        )


@app.post("/langchain/query", response_model=LangChainQueryResponse, tags=["LangChain"])
async def langchain_query_documents(request: LangChainQueryRequest):
    """
    Запросить документы используя LangChain retriever

    Этот endpoint использует LangChain QdrantVectorStore.as_retriever()
    для поиска с фильтрацией по document_id.
    """
    try:
        logger.info(f"[LangChain] Querying document: {request.document_id}")
        logger.info(f"[LangChain] Query: {request.query}")

        # Поиск через LangChain
        langchain_service = get_langchain_vector_store()
        results = langchain_service.search(
            document_id=request.document_id,
            query=request.query,
            top_k=request.top_k
        )

        logger.info(f"[LangChain] Found {len(results)} results")

        # Форматировать результаты
        search_results = [
            LangChainSearchResult(
                page_content=result["page_content"],
                score=result.get("score", 0.0),
                metadata=result["metadata"]
            )
            for result in results
        ]

        return LangChainQueryResponse(
            success=True,
            results=search_results,
            query=request.query,
            document_id=request.document_id
        )

    except Exception as e:
        logger.error(f"[LangChain] Error querying documents: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to query documents via LangChain: {str(e)}"
        )


@app.post("/langchain/summarize", response_model=SummarizeResponse, tags=["LangChain"])
async def langchain_summarize_document(request: SummarizeRequest):
    """
    Создать сокращенную версию документа через суммаризацию

    Этот endpoint извлекает все чанки документа и использует
    LangChain summarization chains для создания резюме.

    Требуется настройка LLM_API_KEY в .env файле.
    """
    try:
        # Проверить что LangChain включен и API ключ настроен
        if not settings.langchain_enabled:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="LangChain functionality is disabled. Set LANGCHAIN_ENABLED=true in .env"
            )

        if not settings.llm_api_key:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="LLM API key is not configured. Set LLM_API_KEY in .env"
            )

        logger.info(f"[LangChain] Summarizing document: {request.document_id}")
        logger.info(f"[LangChain] Strategy: {request.strategy}")

        # Выполнить суммаризацию
        result = summarize_document(
            document_id=request.document_id,
            strategy=request.strategy,
            max_chunks=request.max_chunks
        )

        logger.info(f"[LangChain] Summarization completed")

        return SummarizeResponse(
            success=True,
            summary=result["summary"],
            document_id=result["document_id"],
            chunks_processed=result["chunks_processed"],
            strategy=result["strategy"]
        )

    except HTTPException:
        raise
    except ValueError as e:
        logger.error(f"[LangChain] Validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"[LangChain] Error summarizing document: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to summarize document: {str(e)}"
        )


@app.post("/langchain/extract_points", response_model=ExtractPointsResponse, tags=["LangChain"])
async def langchain_extract_points(request: ExtractPointsRequest):
    """
    Извлечь ключевые пункты из документа по заданным темам

    Этот endpoint выполняет семантический поиск по каждой теме
    в рамках документа и опционально суммаризует найденные фрагменты.

    Требуется настройка LLM_API_KEY в .env файле (только если summarize=True).
    """
    try:
        # Проверить настройки только если требуется суммаризация
        if request.summarize:
            if not settings.langchain_enabled:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="LangChain functionality is disabled. Set LANGCHAIN_ENABLED=true in .env"
                )

            if not settings.llm_api_key:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="LLM API key is not configured. Set LLM_API_KEY in .env"
                )

        logger.info(f"[LangChain] Extracting points from document: {request.document_id}")
        logger.info(f"[LangChain] Topics: {len(request.topics)}")
        logger.info(f"[LangChain] Summarize: {request.summarize}")

        # Выполнить извлечение
        result = extract_points_from_document(
            document_id=request.document_id,
            topics=request.topics,
            chunks_per_topic=request.chunks_per_topic,
            summarize=request.summarize
        )

        logger.info(f"[LangChain] Extraction completed")

        # Преобразовать в Pydantic модели
        extracted_points = [
            ExtractedPoint(
                topic=point["topic"],
                relevant_chunks=point["relevant_chunks"],
                summary=point.get("summary")
            )
            for point in result["extracted_points"]
        ]

        return ExtractPointsResponse(
            success=True,
            document_id=result["document_id"],
            extracted_points=extracted_points,
            total_chunks_retrieved=result["total_chunks_retrieved"]
        )

    except HTTPException:
        raise
    except ValueError as e:
        logger.error(f"[LangChain] Validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"[LangChain] Error extracting points: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to extract points: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True
    )
