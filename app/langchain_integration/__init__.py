"""
LangChain интеграция для DocMind

Этот модуль предоставляет альтернативные методы работы с Qdrant через LangChain,
включая суммаризацию и извлечение ключевых пунктов из документов.
"""

from .vector_store import get_langchain_vector_store
from .summarizer import summarize_document, extract_points_from_document

__all__ = [
    "get_langchain_vector_store",
    "summarize_document",
    "extract_points_from_document",
]
