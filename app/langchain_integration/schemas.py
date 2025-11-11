from pydantic import BaseModel, Field
from typing import List, Dict, Any, Literal


class LangChainIngestRequest(BaseModel):
    """Модель запроса для загрузки документа через LangChain"""

    document_name: str = Field(
        ...,
        description="Название документа",
        min_length=1
    )
    text: str = Field(
        ...,
        description="Текст документа для загрузки",
        min_length=1
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "document_name": "Тендер №123 - Закупка оборудования",
                    "text": "Очень длинный текст тендерной документации"
                }
            ]
        }
    }


class LangChainIngestResponse(BaseModel):
    """Модель ответа для загрузки документа через LangChain"""

    success: bool
    message: str
    chunks_count: int = Field(
        ...,
        description="Количество созданных и загруженных чанков"
    )
    document_id: str = Field(
        ...,
        description="Уникальный идентификатор документа (UUID)"
    )


class LangChainQueryRequest(BaseModel):
    """Модель запроса для поиска через LangChain"""

    document_id: str = Field(
        ...,
        description="Идентификатор документа для поиска (UUID)",
        min_length=1
    )
    query: str = Field(
        ...,
        description="Текст поискового запроса",
        min_length=1
    )
    top_k: int = Field(
        default=5,
        description="Количество результатов для возврата",
        ge=1,
        le=50
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "document_id": "550e8400-e29b-41d4-a716-446655440000",
                    "query": "Каковы требования к участникам тендера?",
                    "top_k": 5
                }
            ]
        }
    }


class LangChainSearchResult(BaseModel):
    """Один результат поиска из LangChain"""

    page_content: str = Field(..., description="Текстовое содержимое чанка")
    score: float = Field(..., description="Оценка схожести (0-1)")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Метаданные документа"
    )


class LangChainQueryResponse(BaseModel):
    """Модель ответа для запроса через LangChain"""

    success: bool
    results: List[LangChainSearchResult] = Field(
        default_factory=list,
        description="Список результатов поиска"
    )
    query: str = Field(..., description="Исходный текст запроса")
    document_id: str = Field(..., description="Идентификатор документа")


class SummarizeRequest(BaseModel):
    """Модель запроса для суммаризации текста"""

    text: str = Field(
        ...,
        description="Текст для суммаризации",
        min_length=1
    )
    target_tokens: int = Field(
        default=70000,
        description="Целевое количество токенов в резюме",
        ge=1000,
        le=100000
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "text": "Очень длинный текст для суммаризации...",
                    "target_tokens": 70000
                }
            ]
        }
    }


class SummarizeResponse(BaseModel):
    """Модель ответа суммаризации"""

    success: bool
    summary: str = Field(..., description="Суммаризованный текст")
    input_tokens: int = Field(..., description="Количество токенов в исходном тексте")
    output_tokens: int = Field(..., description="Количество токенов в резюме")
    parts_processed: int = Field(..., description="Количество частей, на которые был разделен текст")
    strategy_used: str = Field(..., description="Использованная стратегия (stuff или multi-part stuff)")


class ExtractPointsRequest(BaseModel):
    """Модель запроса для извлечения ключевых пунктов"""

    document_id: str = Field(
        ...,
        description="Идентификатор документа (UUID)",
        min_length=1
    )
    topics: List[str] = Field(
        ...,
        description="Список тем/вопросов для извлечения",
        min_length=1
    )
    chunks_per_topic: int = Field(
        default=3,
        description="Количество чанков на каждую тему",
        ge=1,
        le=20
    )
    summarize: bool = Field(
        default=False,
        description="Суммаризовать ли найденные чанки"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "document_id": "550e8400-e29b-41d4-a716-446655440000",
                    "topics": [
                        "Требования к участникам",
                        "Сроки подачи заявок",
                        "Критерии оценки"
                    ],
                    "chunks_per_topic": 3,
                    "summarize": True
                }
            ]
        }
    }


class ExtractedPoint(BaseModel):
    """Извлеченный пункт по теме"""

    topic: str = Field(..., description="Тема/вопрос")
    relevant_chunks: List[str] = Field(..., description="Релевантные чанки")
    summary: str | None = Field(None, description="Суммаризация (если запрошена)")


class ExtractPointsResponse(BaseModel):
    """Модель ответа извлечения пунктов"""

    success: bool
    document_id: str = Field(..., description="Идентификатор документа")
    extracted_points: List[ExtractedPoint] = Field(
        ...,
        description="Извлеченные пункты по темам"
    )
    total_chunks_retrieved: int = Field(
        ...,
        description="Общее количество извлеченных чанков"
    )


class SummarizeWithAnalysisResponse(BaseModel):
    """Модель ответа суммаризации с анализом"""

    success: bool
    summary: str = Field(..., description="Суммаризованный текст")
    analysis: str = Field(..., description="Результат анализа с помощью PROMPT")
    summary_tokens: int = Field(..., description="Количество токенов в резюме")
    analysis_input_tokens: int = Field(..., description="Количество токенов на входе анализа")
    analysis_output_tokens: int = Field(..., description="Количество токенов в результате анализа")
    parts_processed: int = Field(..., description="Количество частей, на которые был разделен текст")
    strategy_used: str = Field(..., description="Использованная стратегия суммаризации")
    # Время выполнения
    extraction_time_seconds: float = Field(..., description="Время извлечения текста из документа (секунды)")
    summarization_time_seconds: float = Field(..., description="Время суммаризации (секунды)")
    analysis_time_seconds: float = Field(..., description="Время анализа с PROMPT (секунды)")
    total_time_seconds: float = Field(..., description="Общее время выполнения (секунды)")
