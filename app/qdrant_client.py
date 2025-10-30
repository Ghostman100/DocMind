from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from typing import List, Dict, Any
import uuid
from .config import settings
from .embeddings import EmbeddingRegistry


class QdrantService:
    """Сервис для взаимодействия с векторной базой данных Qdrant"""

    def __init__(self, embedding_registry: EmbeddingRegistry):
        """Инициализация клиента Qdrant

        Args:
            embedding_registry: Реестр моделей embeddings
        """
        self.client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key
        )
        self.embedding_registry = embedding_registry
        print(f"Connected to Qdrant at {settings.qdrant_url}")

        # Убедиться, что единая коллекция существует
        self.ensure_collection()

    def ensure_collection(self) -> None:
        """
        Убедиться, что единая коллекция существует, создать если не существует
        Поддерживает несколько named vectors (по одному на модель)
        """
        collections = self.client.get_collections().collections
        collection_names = [col.name for col in collections]

        if settings.collection_name not in collection_names:
            print(f"Creating collection: {settings.collection_name}")

            # Создать named vectors config для всех зарегистрированных моделей
            vectors_config = {}
            for model_name in self.embedding_registry.list_models():
                embedding_service = self.embedding_registry.get_model(model_name)
                vectors_config[model_name] = VectorParams(
                    size=embedding_service.embedding_dim,
                    distance=Distance.COSINE
                )

            self.client.create_collection(
                collection_name=settings.collection_name,
                vectors_config=vectors_config
            )
            print(f"Collection {settings.collection_name} created with {len(vectors_config)} vector configs")
        else:
            print(f"Collection {settings.collection_name} already exists")

    def ingest_chunks(
        self,
        document_id: str,
        document_name: str,
        upload_timestamp: str,
        chunks: List[str],
        model_name: str,
        metadata: List[Dict[str, Any]] = None
    ) -> int:
        """
        Загрузить текстовые чанки в Qdrant

        Args:
            document_id: Уникальный идентификатор документа (UUID)
            document_name: Название документа
            upload_timestamp: Временная метка загрузки (ISO format)
            chunks: Список текстовых чанков
            model_name: Название модели для создания embeddings
            metadata: Опциональный список словарей метаданных для каждого чанка

        Returns:
            Количество загруженных чанков

        Raises:
            ValueError: Если модель не найдена в registry
        """
        if not chunks:
            return 0

        # Получить embedding service для модели
        embedding_service = self.embedding_registry.get_model(model_name)
        if embedding_service is None:
            raise ValueError(f"Model {model_name} not found in registry")

        # Создать embeddings
        print(f"Creating embeddings for {len(chunks)} chunks using model {model_name}...")
        embeddings = embedding_service.encode_texts(chunks)

        # Подготовить точки для Qdrant
        points = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            point_id = str(uuid.uuid4())
            payload = {
                "text": chunk,
                "chunk_index": i,
                "document_id": document_id,
                "document_name": document_name,
                "upload_timestamp": upload_timestamp,
                "embedding_model": model_name  # Сохранить информацию о модели
            }

            # Добавить метаданные чанка если предоставлены
            if metadata and i < len(metadata):
                payload.update(metadata[i])

            point = PointStruct(
                id=point_id,
                vector={model_name: embedding.tolist()},
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
        model_name: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Поиск похожих чанков в документе

        Args:
            document_id: Идентификатор документа для поиска
            query: Текст запроса
            model_name: Название модели для создания query embedding
            top_k: Количество результатов для возврата

        Returns:
            Список результатов поиска с текстом и метаданными

        Raises:
            ValueError: Если модель не найдена в registry
        """
        # Получить embedding service для модели
        embedding_service = self.embedding_registry.get_model(model_name)
        if embedding_service is None:
            raise ValueError(f"Model {model_name} not found in registry")

        # Создать embedding запроса
        query_embedding = embedding_service.encode_single(query)

        # Поиск в Qdrant с фильтрацией по document_id
        results = self.client.query_points(
            collection_name=settings.collection_name,
            using=model_name,
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
