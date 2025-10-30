from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional
import numpy as np


class EmbeddingService:
    """Сервис для создания embeddings используя sentence-transformers"""

    def __init__(self, model_name: str):
        """Инициализация модели для embeddings

        Args:
            model_name: Название модели из HuggingFace
        """
        print(f"Loading embedding model: {model_name}")
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
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


class EmbeddingRegistry:
    """Реестр для управления несколькими моделями embeddings"""

    def __init__(self):
        """Инициализация пустого реестра"""
        self._models: Dict[str, EmbeddingService] = {}

    def register_model(self, model_name: str) -> EmbeddingService:
        """
        Зарегистрировать и загрузить новую модель

        Args:
            model_name: Название модели из HuggingFace

        Returns:
            Экземпляр EmbeddingService для этой модели

        Raises:
            ValueError: Если модель уже зарегистрирована
        """
        if model_name in self._models:
            print(f"Model {model_name} already loaded, skipping...")
            return self._models[model_name]

        service = EmbeddingService(model_name)
        self._models[model_name] = service
        return service

    def get_model(self, model_name: str) -> Optional[EmbeddingService]:
        """
        Получить сервис embeddings по названию модели

        Args:
            model_name: Название модели

        Returns:
            EmbeddingService или None если модель не найдена
        """
        return self._models.get(model_name)

    def list_models(self) -> List[str]:
        """
        Получить список всех зарегистрированных моделей

        Returns:
            Список названий моделей
        """
        return list(self._models.keys())

    def has_model(self, model_name: str) -> bool:
        """
        Проверить, зарегистрирована ли модель

        Args:
            model_name: Название модели

        Returns:
            True если модель зарегистрирована
        """
        return model_name in self._models


# Глобальный реестр (паттерн registry)
_embedding_registry: Optional[EmbeddingRegistry] = None


def get_embedding_registry() -> EmbeddingRegistry:
    """Получить или создать глобальный реестр embeddings"""
    global _embedding_registry
    if _embedding_registry is None:
        _embedding_registry = EmbeddingRegistry()
    return _embedding_registry
