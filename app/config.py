from pydantic_settings import BaseSettings
from typing import Literal, List


class Settings(BaseSettings):
    """Настройки приложения"""

    # Настройки Qdrant
    qdrant_url: str
    qdrant_api_key: str | None = None
    collection_name: str = "documents2"  # Единая коллекция для всех документов

    # Настройки моделей для embeddings
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"  # Для обратной совместимости

    # Модели которые нужно загрузить при старте (всегда в RAM)
    embedding_models: List[str] = [
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "intfloat/multilingual-e5-base",
        "deepvk/USER-bge-m3"
    ]

    # Модель по умолчанию для /ingest и /query эндпоинтов
    default_embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    default_vector_name: str = "fast-paraphrase-multilingual-minilm-l12-v2"

    # Стратегия чанкинга: "paragraph" или "recursive"
    chunking_strategy: Literal["paragraph", "recursive"] = "paragraph"

    # Настройки рекурсивного чанкинга (используются когда chunking_strategy="recursive")
    chunk_size: int = 500
    chunk_overlap: int = 50

    # Настройки API
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
