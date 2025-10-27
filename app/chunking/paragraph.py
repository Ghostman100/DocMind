import re
from typing import List
from .base import BaseChunker, Chunk


class ParagraphChunker(BaseChunker):
    """
    Разбивает текст по абзацам (разделенным двойными переносами строк)
    """

    def __init__(self, min_chunk_length: int = 10):
        """
        Args:
            min_chunk_length: Минимальная длина чанка в символах
        """
        self.min_chunk_length = min_chunk_length

    def chunk(self, text: str) -> List[Chunk]:
        """
        Разбить текст на чанки по абзацам

        Args:
            text: Входной текст для разбиения

        Returns:
            Список объектов Chunk
        """
        # Разбить по двойным переносам строк (абзацы)
        paragraphs = re.split(r'\n\s*\n', text)

        chunks = []
        for i, paragraph in enumerate(paragraphs):
            # Очистить абзац
            cleaned = self._clean_text(paragraph)

            # Пропустить очень короткие абзацы
            if len(cleaned) < self.min_chunk_length:
                continue

            chunk = Chunk(
                text=cleaned,
                index=i,
                metadata={
                    "type": "paragraph",
                    "char_count": len(cleaned)
                }
            )
            chunks.append(chunk)

        return chunks
