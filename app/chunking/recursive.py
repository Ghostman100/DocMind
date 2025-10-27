from typing import List
from .base import BaseChunker, Chunk


class RecursiveChunker(BaseChunker):
    """
    Рекурсивно разбивает текст с наложением, аналогично RecursiveCharacterTextSplitter из LangChain.
    Пытается разбить по различным разделителям в порядке предпочтения.
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Args:
            chunk_size: Максимальный размер каждого чанка в символах
            chunk_overlap: Количество символов наложения между чанками
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Разделители в порядке предпочтения (от крупных к мелким единицам)
        self.separators = [
            "\n\n",  # Двойной перенос строки (абзацы)
            "\n",    # Одиночный перенос строки
            ". ",    # Конец предложения
            "! ",    # Восклицание
            "? ",    # Вопрос
            "; ",    # Точка с запятой
            ", ",    # Запятая
            " ",     # Пробел
            ""       # Посимвольно (последнее средство)
        ]

    def chunk(self, text: str) -> List[Chunk]:
        """
        Рекурсивно разбить текст на чанки с наложением

        Args:
            text: Входной текст для разбиения

        Returns:
            Список объектов Chunk
        """
        return self._recursive_split(text)

    def _recursive_split(self, text: str, separator_index: int = 0) -> List[Chunk]:
        """
        Рекурсивно разбить текст используя разделители

        Args:
            text: Текст для разбиения
            separator_index: Индекс текущего разделителя для попытки

        Returns:
            Список объектов Chunk
        """
        if separator_index >= len(self.separators):
            # Последнее средство: разбить по символам
            return self._split_by_character(text)

        separator = self.separators[separator_index]

        # Если разделитель пустой, мы на уровне символов
        if separator == "":
            return self._split_by_character(text)

        # Разбить по текущему разделителю
        splits = text.split(separator) if separator else [text]

        # Восстановить разбиения с разделителем
        final_chunks = []
        current_chunk = ""

        for i, split in enumerate(splits):
            # Добавить разделитель обратно (кроме последнего разбиения)
            if i < len(splits) - 1 and separator:
                split = split + separator

            # Проверить, не превысит ли добавление этого разбиения размер чанка
            if len(current_chunk) + len(split) <= self.chunk_size:
                current_chunk += split
            else:
                # Если текущий чанк не пустой, сохранить его
                if current_chunk:
                    final_chunks.append(current_chunk.strip())

                # Если это разбиение все еще слишком большое, попробовать следующий разделитель
                if len(split) > self.chunk_size:
                    sub_chunks = self._recursive_split(split, separator_index + 1)
                    final_chunks.extend([chunk.text for chunk in sub_chunks])
                    current_chunk = ""
                else:
                    current_chunk = split

        # Добавить оставшийся чанк
        if current_chunk:
            final_chunks.append(current_chunk.strip())

        # Применить наложение
        return self._apply_overlap(final_chunks)

    def _split_by_character(self, text: str) -> List[Chunk]:
        """Разбить текст по символам как последнее средство"""
        chunks = []
        for i in range(0, len(text), self.chunk_size):
            chunk_text = text[i:i + self.chunk_size]
            chunks.append(Chunk(
                text=chunk_text,
                index=len(chunks),
                metadata={"type": "character_split", "char_count": len(chunk_text)}
            ))
        return chunks

    def _apply_overlap(self, text_chunks: List[str]) -> List[Chunk]:
        """
        Применить наложение между чанками

        Args:
            text_chunks: Список текстовых строк

        Returns:
            Список объектов Chunk с наложением
        """
        if not text_chunks:
            return []

        chunks = []
        for i, text in enumerate(text_chunks):
            # Добавить наложение из предыдущего чанка
            if i > 0 and self.chunk_overlap > 0:
                prev_text = text_chunks[i - 1]
                overlap = prev_text[-self.chunk_overlap:] if len(prev_text) >= self.chunk_overlap else prev_text
                text = overlap + " " + text

            chunk = Chunk(
                text=self._clean_text(text),
                index=i,
                metadata={
                    "type": "recursive",
                    "char_count": len(text),
                    "has_overlap": i > 0 and self.chunk_overlap > 0
                }
            )
            chunks.append(chunk)

        return chunks
