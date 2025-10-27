from pydantic_settings import BaseSettings
from typing import Literal


class Settings(BaseSettings):
    """Настройки приложения"""

    # Настройки Qdrant
    qdrant_url: str
    qdrant_api_key: str | None = None
    collection_name: str = "documents"  # Единая коллекция для всех документов

    # Настройки модели для embeddings
    embedding_model: str = "intfloat/multilingual-e5-large"

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
