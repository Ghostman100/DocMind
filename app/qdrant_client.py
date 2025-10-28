from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from typing import List, Dict, Any
import uuid
from .config import settings
from .embeddings import get_embedding_service


class QdrantService:
    """Сервис для взаимодействия с векторной базой данных Qdrant"""

    def __init__(self):
        """Инициализация клиента Qdrant"""
        self.client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key
        )
        self.embedding_service = get_embedding_service()
        print(f"Connected to Qdrant at {settings.qdrant_url}")

        # Убедиться, что единая коллекция существует
        self.ensure_collection()

    def ensure_collection(self) -> None:
        """
        Убедиться, что единая коллекция существует, создать если не существует
        """
        collections = self.client.get_collections().collections
        collection_names = [col.name for col in collections]

        if settings.collection_name not in collection_names:
            print(f"Creating collection: {settings.collection_name}")
            self.client.create_collection(
                collection_name=settings.collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_service.embedding_dim,
                    distance=Distance.COSINE
                )
            )
            print(f"Collection {settings.collection_name} created")
        else:
            print(f"Collection {settings.collection_name} already exists")

    def ingest_chunks(
        self,
        document_id: str,
        document_name: str,
        upload_timestamp: str,
        chunks: List[str],
        metadata: List[Dict[str, Any]] = None
    ) -> int:
        """
        Загрузить текстовые чанки в Qdrant

        Args:
            document_id: Уникальный идентификатор документа (UUID)
            document_name: Название документа
            upload_timestamp: Временная метка загрузки (ISO format)
            chunks: Список текстовых чанков
            metadata: Опциональный список словарей метаданных для каждого чанка

        Returns:
            Количество загруженных чанков
        """
        if not chunks:
            return 0

        # Создать embeddings
        print(f"Creating embeddings for {len(chunks)} chunks...")
        embeddings = self.embedding_service.encode_texts(chunks)

        # Подготовить точки для Qdrant
        points = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            point_id = str(uuid.uuid4())
            payload = {
                "text": chunk,
                "chunk_index": i,
                "document_id": document_id,
                "document_name": document_name,
                "upload_timestamp": upload_timestamp
            }

            # Добавить метаданные чанка если предоставлены
            if metadata and i < len(metadata):
                payload.update(metadata[i])

            point = PointStruct(
                id=point_id,
                vector={settings.embedding_model: embedding.tolist()},
                payload=payload
            )
            points.append(point)

        # Загрузить в Qdrant
        print(f"Uploading {len(points)} points to collection {settings.collection_name}...")
        self.client.upsert(
            collection_name=settings.collection_name,
            points=points
        )

        print(f"Successfully ingested {len(points)} chunks for document {document_id}")
        return len(points)

    def search(
        self,
        document_id: str,
        query: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Поиск похожих чанков в документе

        Args:
            document_id: Идентификатор документа для поиска
            query: Текст запроса
            top_k: Количество результатов для возврата

        Returns:
            Список результатов поиска с текстом и метаданными
        """
        # Создать embedding запроса
        query_embedding = self.embedding_service.encode_single(query)

        # Поиск в Qdrant с фильтрацией по document_id
        results = self.client.query_points(
            collection_name=settings.collection_name,
            using=settings.embedding_model,
            query=query_embedding.tolist(),
            limit=top_k,
            query_filter=Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(value=document_id)
                    )
                ]
            )
        ).points

        # Форматировать результаты
        formatted_results = []
        for result in results:
            # print(r)
            formatted_results.append({
                "text": result.payload.get("text", ""),
                "score": result.score,
                "metadata": {
                    k: v for k, v in result.payload.items()
                    if k != "text"
                }
            })

        return formatted_results


# Глобальный экземпляр (паттерн singleton)
_qdrant_service = None


def get_qdrant_service() -> QdrantService:
    """Получить или создать глобальный экземпляр сервиса Qdrant"""
    global _qdrant_service
    if _qdrant_service is None:
        _qdrant_service = QdrantService()
    return _qdrant_service
