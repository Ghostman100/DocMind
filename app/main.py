from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
import uuid
from datetime import datetime, UTC
import langchain
import time

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
    SummarizeWithAnalysisResponse,
    ExtractPointsRequest,
    ExtractPointsResponse,
    ExtractedPoint,
)
from .langchain_integration.vector_store import get_langchain_vector_store
from .langchain_integration.summarizer import summarize_document, extract_points_from_document, summarize_text, apply_prompt_to_text

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

    # Включить глобальный debug режим LangChain если настроен
    if settings.langchain_debug:
        langchain.debug = True
        logger.info("LangChain DEBUG mode enabled - detailed output will be shown")
    else:
        langchain.debug = False

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
            chunking_strategy=settings.chunking_strategy,
            langchain_debug=settings.langchain_debug
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


@app.post("/test_summarize", response_model=SummarizeWithAnalysisResponse, tags=["LangChain"])
async def test_summarize_document(target_tokens: int = 70000):
    """
    Тестовый endpoint для суммаризации test.docx с последующим анализом

    Извлекает текст из test.docx, создает резюме и применяет PROMPT для анализа
    """
    start_total = time.time()

    try:
        # Извлечь текст из test.docx
        start_extraction = time.time()
        text = extract_text_from_file('test.docx')
        extraction_time = time.time() - start_extraction

        if not text or not text.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to extract text from test.docx or file is empty"
            )

        logger.info(f"Extracted {len(text)} characters from test.docx in {extraction_time:.2f}s")

        # Создать запрос для summarize
        summarize_request = SummarizeRequest(
            text=text,
            target_tokens=target_tokens
        )

        # Вызвать суммаризацию
        start_summarization = time.time()
        summary_result = await langchain_summarize_text(summarize_request)
        summarization_time = time.time() - start_summarization

        logger.info(f"Summary completed in {summarization_time:.2f}s, now applying PROMPT for analysis")

        # Применить PROMPT к суммаризованному тексту
        start_analysis = time.time()
        analysis_result = apply_prompt_to_text(
            text=summary_result.summary,
            prompt=PROMPT
        )
        analysis_time = time.time() - start_analysis

        total_time = time.time() - start_total

        logger.info(f"Analysis completed in {analysis_time:.2f}s")
        logger.info(f"Total time: {total_time:.2f}s (extraction: {extraction_time:.2f}s, summarization: {summarization_time:.2f}s, analysis: {analysis_time:.2f}s)")

        # Вернуть результат с суммаризацией и анализом
        return SummarizeWithAnalysisResponse(
            success=True,
            summary=summary_result.summary,
            analysis=analysis_result["result"],
            summary_tokens=summary_result.output_tokens,
            analysis_input_tokens=analysis_result["input_tokens"],
            analysis_output_tokens=analysis_result["output_tokens"],
            parts_processed=summary_result.parts_processed,
            strategy_used=summary_result.strategy_used,
            extraction_time_seconds=round(extraction_time, 2),
            summarization_time_seconds=round(summarization_time, 2),
            analysis_time_seconds=round(analysis_time, 2),
            total_time_seconds=round(total_time, 2)
        )

    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="test.docx file not found"
        )

    except Exception as e:
        logger.error(f"Error in test_summarize_document: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to summarize test.docx: {str(e)}"
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
async def langchain_summarize_text(request: SummarizeRequest):
    """
    Создать резюме текста с автоматическим выбором стратегии

    Если текст ≤ 90 000 токенов - использует "stuff" за один запрос.
    Если текст > 90 000 токенов - делит на 2-3 части и делает несколько "stuff" сводок.

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

        logger.info(f"[LangChain] Summarizing text, target tokens: {request.target_tokens}")

        # Выполнить суммаризацию
        result = summarize_text(
            text=request.text,
            target_tokens=request.target_tokens
        )

        logger.info(f"[LangChain] Summarization completed: {result['input_tokens']} -> {result['output_tokens']} tokens")

        return SummarizeResponse(
            success=True,
            summary=result["summary"],
            input_tokens=result["input_tokens"],
            output_tokens=result["output_tokens"],
            parts_processed=result["parts_processed"],
            strategy_used=result["strategy_used"]
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
        logger.error(f"[LangChain] Error summarizing text: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to summarize text: {str(e)}"
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


PROMPT = """
Мы оказываем услуги предоставления разработчиков на проект (на разработку). Ты должен проверить тендерную документацию и выдать мне результаты проверки с цитатами из документов (если нарушается пункт из чек-листа далее) на сл. моменты в Таблицу ”Чек-лист на критичные моменты”:
Пункт 1. Мы не оказываем услуги предпроектное обследование, сопровождение, поддержка, лицензии, предоставление права, продажу программ, соответственно эти задачи и виды услуг не должны быть указаны в тендерной документации как те, которые нужно выполнять в рамках тендера
Пункт 2.Мы не имеем статус и соответственно эти статусы не должны быть обязательными к участию в тендере по документации (не должно быть явно прописано таких статусов):
Партнер «Центр компетенции 1С: КОРП
Кандидат в «Центры компетенции 1С: КОРП»
Наличие официального статуса на сайте Фирмы 1С https://1c.ru/ в разделе Рейтинг партнеров «1С: Центры компетенции по ERP-решениям»:
Официальный статус 1С: Франчайзинг.
Статус Центр реальной автоматизации
Центр компетенции по ERP-решениям для управления предприятием
­Партнер по бухгалтерскому консалтингу
­Партнер по управленческому консалтингу
­Центр сопровождения программ и информационных продуктов фирмы 1C
­Центр сертифицированного обучения фирмы 1C
статус официального партнера 1С «Центр ERP-решений
Кандидатский статус 1С - «Кандидат в 1С: Консалтинг»
«Кандидат в 1С: Центр ERP»
«Партнер по внедрению и комплексному обслуживанию Решений на платформе 1С: Предприятие 8».
Пункт 3. Мы не оказываем поддержку SLA 1 и 2 линии, 24/7, соответственно ее не должно быть указано в документации, а также не должно быть того, что явн к этому ведет( наличие кол-центра у исполнителя, выделенная телефонная линия поддержки и т.д)
Пункт 4. В тендере не должно требоваться обязательно Очное присутствие на территории Заказчика для оказания услуг, выезды на территорию и др. формулировки явно подразумевающие выезд на территории Заказчика для оказания услуг, не должно быть выездов по-необходимости и т.д. Вся работа по оказанию услуг должна проводиться удаленно. Требования к наличию офиса в России не является для нас ограничением.
Пункт 5. В тендере не должно быть запрещено привлечение соисполнителей  субподрядчиков (и др. термины этой же сути), а также не должно быть требования по работе только штатными специалистами, не должно быть требований подтверждения факта трудоустройства, не должно быть требования последующего трудоустройства. 
Выведи в начале “Общую информацию” следующими данными таблицей “Общее” (в 2 колонки - в первой название пункта, во второй - содержание):
Пункт 1. Суть тендера (на что)
Пункт 2. Определи тип тендера (жирным) и по каким ключевым словам ты так решил (приведи цитатами в этой же ячейке)
Пункт 3. Разработку каких систем, интеграций с чем и т.д. нужно выполнить
Пункт 4. Какие специалисты нужны с какими компетенциями в каком объеме людей или человеко-часов
Пункт 5. Что конкретно нужно делать (какие задачи решать)
Пункт 6. Какие сертификаты у специалистов обязательны к наличию
Пункт 7. Сроки оказания услуг
Пункт 8. Стоимость за час или сумма тендера
Пункт 9. Сроки оплаты
Далее выведи в начале итоговую таблицу с пунктами чек-листа (таблицу ”Чек-лист на критичные моменты”). 
Пункт чек-листа
Состояние
Цитата


Если в тендере не указано не подходящего требования для нас, значит он подходит - рисуй в Состояние галочку зеленым цветом ✅ . Если есть неподходящее требование - рисуй в Состояние крестик красным цветом ❌. Если пункт по наличию статусов, и там они не обязательны, а желательны, то рисуй в Состояние восклицательный знак желтым цветом ⚠️. Если ты не уверен - нарисуй в Состояние знак вопроса синий круг и знак вопроса🔵❓. Также добавь колонку “цитата”- если есть требование, по которому мы не походим критерий участия, то в эту колонку пропиши цитату из документа, на основании которой ты решил.
После итоговой таблицы выведи более подробные рассуждения по заполненным выше данным в раздел  “Рассуждения”.
Тип тендера определять по следующим вариантам
Разработка 1С

Автоматизация на 1С, Адаптация 1С, Аутсорсинг 1С, Аутстаффинг 1С, Доработка 1С, Интеграция 1С, КОНФИГУРАЦИЯ 1С, Модификация 1С, Настройка 1С, Перенос 1С, Перевод 1С, Программирование 1С, Разработка на 1С, Сопровождение 1С, Создание конфигураций 1С, Внедрение 1С, Программист 1С, 1С:Предприятие 8, 1С:ERP, 1С:ЗУП, Документооборот 1С, ДО 1С
2.     Разработка на MS Dynamics 365 / AX (Axapta) 

MS Dynamics AX, MS Dynamics 365, Axapta, разработка Axapta, внедрение Dynamics, сопровождение Dynamics, программист Dynamics, интеграция Dynamics
3.     Разработка на Python 

программист Python, разработка ПО на Python, веб-разработка Python, автоматизация на Python
4.     Разработка на Java 

программист Java, разработка ПО на Java, веб-разработка Java, enterprise разработка на Java
5.     Аутстаффинг программистов 

аренда программистов, удаленные программисты, ит-аутстаффинг, аутстаффинг разработчиков, аутстаффинг 1С, аутстаффинг Java, аутстаффинг Python, квалификационный отбор, техподдержка
6.     Аутсорсинг разработки 

аутсорсинг программирования, аутсорсинг ит, передача разработки на аутсорсинг, разработка по на аутсорсинг
Интеграция систем 

интеграция по,  интеграция 1С с другими системами, разработка интеграционных решений, шина данных, datareon

Обучение для разработчиков 

курсы для программистов 1С, обучение разработке, повышение квалификации программистов, корпоративное обучение программированию, Обучение 1С
Zool.ai (Видеоаналитика)
видеоаналитика, AI видеоаналитика, нейросетевая видеоаналитика, контроль охраны труда, промышленная безопасность, контроль качества продукции, подсчет посетителей, анализ потока клиентов, контроль кассовой зоны, машинное зрение, контроль за, внутреннего контроля

Facereg.ru (WFM & Биометрия)
WFM система, учет рабочего времени, автоматизация учета рабочего времени, распознавание лиц, Face ID, биометрический контроль доступа, интеграция WFM и 1С, планирование персонала, составление графиков смен, контроль дисциплины
Prostoskud.ru (Интеграция)
интеграция СКУД и 1С, синхронизация 1С и СКУД, автоматическое формирование табеля, СКУД Perco, СКУД Болид, СКУД Parsec, 1С: ЗУП, 1С: ERP, 1С: КА, сокращение трудозатрат кадровой службы
document_id='fe5aae47-2b4a-4257-b322-12db71fad450'
"""