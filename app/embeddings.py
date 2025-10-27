from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np
from .config import settings


class EmbeddingService:
    """Сервис для создания embeddings используя sentence-transformers"""

    def __init__(self):
        """Инициализация модели для embeddings"""
        print(f"Loading embedding model: {settings.embedding_model}")
        self.model = SentenceTransformer(settings.embedding_model)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"Model loaded. Embedding dimension: {self.embedding_dim}")

    def encode_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Создать embeddings для списка текстов

        Args:
            texts: Список текстовых строк для кодирования
            batch_size: Размер батча для кодирования

        Returns:
            numpy массив embeddings с формой (len(texts), embedding_dim)
        """
        if not texts:
            return np.array([])

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 10,
            normalize_embeddings=True  # Нормализация для косинусного сходства
        )

        return embeddings

    def encode_single(self, text: str) -> np.ndarray:
        """
        Создать embedding для одного текста

        Args:
            text: Текстовая строка для кодирования

        Returns:
            numpy массив embedding с формой (embedding_dim,)
        """
        return self.encode_texts([text])[0]


# Глобальный экземпляр (паттерн singleton)
_embedding_service = None


def get_embedding_service() -> EmbeddingService:
    """Получить или создать глобальный экземпляр сервиса embeddings"""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service
