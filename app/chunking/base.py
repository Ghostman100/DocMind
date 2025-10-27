from abc import ABC, abstractmethod
from typing import List


class Chunk:
    """Представляет текстовый чанк с метаданными"""

    def __init__(self, text: str, index: int, metadata: dict = None):
        self.text = text
        self.index = index
        self.metadata = metadata or {}

    def __repr__(self):
        return f"Chunk(index={self.index}, text_length={len(self.text)})"


class BaseChunker(ABC):
    """Абстрактный базовый класс для стратегий разбиения текста на чанки"""

    @abstractmethod
    def chunk(self, text: str) -> List[Chunk]:
        """
        Разбить текст на чанки

        Args:
            text: Входной текст для разбиения

        Returns:
            Список объектов Chunk
        """
        pass

    def _clean_text(self, text: str) -> str:
        """Очистить и нормализовать текст"""
        # Удалить избыточные пробелы
        text = " ".join(text.split())
        return text.strip()
