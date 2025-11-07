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
    """Модель запроса для суммаризации документа"""

    document_id: str = Field(
        ...,
        description="Идентификатор документа для суммаризации (UUID)",
        min_length=1
    )
    strategy: Literal["stuff", "map_reduce", "refine"] = Field(
        default="map_reduce",
        description="Стратегия суммаризации: stuff (для коротких), map_reduce (для длинных), refine (итеративная)"
    )
    max_chunks: int = Field(
        default=100,
        description="Максимальное количество чанков для обработки",
        ge=1,
        le=1000
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "document_id": "550e8400-e29b-41d4-a716-446655440000",
                    "strategy": "map_reduce",
                    "max_chunks": 100
                }
            ]
        }
    }


class SummarizeResponse(BaseModel):
    """Модель ответа суммаризации"""

    success: bool
    summary: str = Field(..., description="Суммаризованный текст документа")
    document_id: str = Field(..., description="Идентификатор документа")
    chunks_processed: int = Field(..., description="Количество обработанных чанков")
    strategy: str = Field(..., description="Использованная стратегия")


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
