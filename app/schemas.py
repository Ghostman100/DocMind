from pydantic import BaseModel, Field
from typing import List, Dict, Any


class IngestRequest(BaseModel):
    """Модель запроса для загрузки документа"""

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


class IngestResponse(BaseModel):
    """Модель ответа для загрузки документа"""

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


class QueryRequest(BaseModel):
    """Модель запроса для поиска по документам"""

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


class SearchResult(BaseModel):
    """Один результат поиска"""

    text: str = Field(..., description="Текстовое содержимое чанка")
    score: float = Field(..., description="Оценка схожести (0-1)")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Дополнительные метаданные"
    )


class QueryResponse(BaseModel):
    """Модель ответа для запроса"""

    success: bool
    results: List[SearchResult] = Field(
        default_factory=list,
        description="Список результатов поиска"
    )
    query: str = Field(..., description="Исходный текст запроса")
    document_id: str = Field(..., description="Идентификатор документа, по которому производился поиск")


class HealthResponse(BaseModel):
    """Ответ проверки здоровья"""

    status: str
    qdrant_connected: bool
    embedding_models: List[str] = Field(
        ...,
        description="Список загруженных моделей embeddings"
    )
    default_embedding_model: str = Field(
        ...,
        description="Модель по умолчанию"
    )
    chunking_strategy: str


class EmbedRequest(BaseModel):
    """Модель запроса для получения embeddings"""

    model: str = Field(
        ...,
        description="Название модели для создания embeddings",
        min_length=1
    )
    text: str = Field(
        ...,
        description="Текст для кодирования",
        min_length=1
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "model": "deepvk/USER-bge-m3",
                    "text": "Какой-то текст для получения эмбеддинга"
                }
            ]
        }
    }


class EmbedResponse(BaseModel):
    """Модель ответа с embeddings"""

    embedding: List[float] = Field(
        ...,
        description="Вектор embeddings"
    )
    model: str = Field(
        ...,
        description="Название использованной модели"
    )
    dimension: int = Field(
        ...,
        description="Размерность вектора embeddings"
    )
