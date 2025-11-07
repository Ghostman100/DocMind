"""
LangChain QdrantVectorStore обертка для интеграции с существующей системой
"""

from langchain_qdrant import QdrantVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from typing import List, Dict, Any, Optional
import uuid
from datetime import datetime, UTC

from ..config import settings
from ..embeddings import get_embedding_registry


class LangChainVectorStoreService:
    """Сервис для работы с Qdrant через LangChain"""

    def __init__(self):
        """Инициализация сервиса"""
        self.client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key
        )
        self.collection_name = settings.collection_name
        self.embedding_registry = get_embedding_registry()

        # Создать LangChain embeddings wrapper для модели по умолчанию
        self.embeddings = HuggingFaceEmbeddings(
            model_name=settings.default_embedding_model,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

    def get_vector_store(self) -> QdrantVectorStore:
        """
        Получить QdrantVectorStore для работы с существующей коллекцией

        Returns:
            QdrantVectorStore instance
        """
        vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.embeddings,
            vector_name=settings.default_vector_name,  # Используем named vector
        )

        return vector_store

    def ingest_texts(
        self,
        texts: List[str],
        document_id: str,
        document_name: str,
        metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> int:
        """
        Загрузить тексты в Qdrant через LangChain

        Args:
            texts: Список текстов для загрузки
            document_id: UUID документа
            document_name: Название документа
            metadatas: Дополнительные метаданные для каждого текста

        Returns:
            Количество загруженных чанков
        """
        upload_timestamp = datetime.now(UTC).isoformat()

        # Подготовить метаданные
        if metadatas is None:
            metadatas = [{}] * len(texts)

        # Добавить обязательные поля в метаданные
        for i, metadata in enumerate(metadatas):
            metadata.update({
                "document_id": document_id,
                "document_name": document_name,
                "chunk_index": i,
                "upload_timestamp": upload_timestamp,
                "embedding_model": settings.default_embedding_model
            })

        # Получить vector store
        vector_store = self.get_vector_store()

        # Загрузить тексты
        ids = [str(uuid.uuid4()) for _ in texts]
        vector_store.add_texts(
            texts=texts,
            metadatas=metadatas,
            ids=ids
        )

        return len(texts)

    def search(
        self,
        document_id: str,
        query: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Поиск с фильтрацией по document_id

        Args:
            document_id: UUID документа
            query: Поисковый запрос
            top_k: Количество результатов

        Returns:
            Список результатов с метаданными
        """
        vector_store = self.get_vector_store()

        # LangChain использует retriever для фильтрации
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": top_k,
                "filter": Filter(
                    must=[
                        FieldCondition(
                            key="document_id",
                            match=MatchValue(value=document_id)
                        )
                    ]
                )
            }
        )

        # Выполнить поиск
        documents = retriever.invoke(query)

        # Преобразовать в формат результатов
        results = []
        for doc in documents:
            results.append({
                "page_content": doc.page_content,
                "metadata": doc.metadata,
                "score": doc.metadata.get("score", 0.0)  # Score может быть в metadata
            })

        return results

    def get_all_documents_by_id(
        self,
        document_id: str,
        max_chunks: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Получить все чанки документа по document_id

        Args:
            document_id: UUID документа
            max_chunks: Максимальное количество чанков

        Returns:
            Список всех чанков документа
        """
        # Используем прямой API Qdrant для получения всех точек с фильтром
        scroll_result = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(value=document_id)
                    )
                ]
            ),
            limit=max_chunks,
            with_payload=True,
            with_vectors=False
        )

        points, _ = scroll_result

        # Преобразовать в формат документов
        results = []
        for point in points:
            payload = point.payload or {}
            results.append({
                "page_content": payload.get("document", ""),
                "metadata": {
                    "chunk_index": payload.get("chunk_index", 0),
                    "document_id": payload.get("document_id"),
                    "document_name": payload.get("document_name"),
                    "upload_timestamp": payload.get("upload_timestamp")
                }
            })

        # Сортировать по chunk_index
        results.sort(key=lambda x: x["metadata"]["chunk_index"])

        return results


# Синглтон для LangChain vector store service
_langchain_vector_store_service: Optional[LangChainVectorStoreService] = None


def get_langchain_vector_store() -> LangChainVectorStoreService:
    """Получить или создать LangChain vector store service"""
    global _langchain_vector_store_service
    if _langchain_vector_store_service is None:
        _langchain_vector_store_service = LangChainVectorStoreService()
    return _langchain_vector_store_service
