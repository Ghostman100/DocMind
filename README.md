# DocMind

DocMind - это RAG (Retrieval-Augmented Generation) система для загрузки документов, разбиения их на чанки, создания embeddings и semantic search.

## Возможности

- Загрузка и обработка текстовых документов
- Два метода разбиения на чанки:
  - **ParagraphChunker**: разбиение по абзацам (двойной перенос строки)
  - **RecursiveChunker**: умное рекурсивное разбиение с overlap
- Создание embeddings с использованием sentence-transformers
- Хранение векторов в Qdrant
- Semantic search для поиска релевантных чанков
- REST API на FastAPI

## Архитектура

```
Документ → Chunking → Embeddings → Qdrant (единая коллекция "documents")
              ↓
    document_id присваивается всем чанкам
              ↓
    Query (по document_id) → Semantic Search → Результаты
```

## Установка

### 1. Клонирование и установка зависимостей

```bash
cd DocMind
pip install -r requirements.txt
```

### 2. Настройка окружения

Создайте файл `.env` на основе `.env.example`:

```bash
cp .env.example .env
```

Отредактируйте `.env`:

```env
# URL вашего Qdrant
QDRANT_URL=http://your-qdrant-url:6333
QDRANT_API_KEY=your-api-key-if-needed

# Модель для embeddings
EMBEDDING_MODEL=intfloat/multilingual-e5-large

# Стратегия чанкинга: paragraph или recursive
CHUNKING_STRATEGY=paragraph

# Настройки для recursive chunking (если используется)
CHUNK_SIZE=500
CHUNK_OVERLAP=50
```

### 3. Запуск

```bash
# Запуск через uvicorn
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Или через Python
python -m app.main
```

API будет доступен на `http://localhost:8000`

## API Endpoints

### Health Check

```bash
GET /health
```

Проверка состояния сервиса.

**Ответ:**
```json
{
  "status": "healthy",
  "qdrant_connected": true,
  "embedding_model": "intfloat/multilingual-e5-large",
  "chunking_strategy": "paragraph"
}
```

### Загрузка документа (Ingest)

```bash
POST /ingest
```

Загружает документ, разбивает на чанки и сохраняет в Qdrant. Автоматически генерирует уникальный `document_id`.

**Запрос:**
```json
{
  "document_name": "Тендер №123 - Закупка оборудования",
  "text": "Очень длинный текст тендерной документации..."
}
```

**Ответ:**
```json
{
  "success": true,
  "message": "Successfully ingested 15 chunks",
  "chunks_count": 15,
  "document_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

**Пример с curl:**
```bash
curl -X POST "http://localhost:8000/ingest" \
  -H "Content-Type: application/json" \
  -d '{
    "document_name": "Тендер №123",
    "text": "Текст вашего документа здесь..."
  }'
```

**Пример с Python:**
```python
import requests

response = requests.post(
    "http://localhost:8000/ingest",
    json={
        "document_name": "Тендер №123",
        "text": "Текст вашего документа здесь..."
    }
)
result = response.json()
document_id = result["document_id"]  # Сохраните для последующих запросов
print(f"Document ID: {document_id}")
```

### Поиск по документам (Query)

```bash
POST /query
```

Выполняет semantic search по конкретному документу (по `document_id`).

**Запрос:**
```json
{
  "document_id": "550e8400-e29b-41d4-a716-446655440000",
  "query": "Каковы требования к участникам тендера?",
  "top_k": 5
}
```

**Ответ:**
```json
{
  "success": true,
  "results": [
    {
      "text": "Участники тендера должны соответствовать следующим требованиям...",
      "score": 0.89,
      "metadata": {
        "type": "paragraph",
        "char_count": 245,
        "document_id": "550e8400-e29b-41d4-a716-446655440000",
        "document_name": "Тендер №123",
        "upload_timestamp": "2024-10-27T12:00:00.000000"
      }
    }
  ],
  "query": "Каковы требования к участникам тендера?",
  "document_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

**Пример с curl:**
```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "document_id": "550e8400-e29b-41d4-a716-446655440000",
    "query": "Каковы требования к участникам тендера?",
    "top_k": 5
  }'
```

**Пример с Python:**
```python
import requests

response = requests.post(
    "http://localhost:8000/query",
    json={
        "document_id": "550e8400-e29b-41d4-a716-446655440000",
        "query": "Каковы требования к участникам тендера?",
        "top_k": 5
    }
)

results = response.json()
for result in results["results"]:
    print(f"Score: {result['score']:.2f}")
    print(f"Text: {result['text'][:200]}...")
    print(f"Document: {result['metadata']['document_name']}")
    print("---")
```

## Конфигурация

### Выбор модели для embeddings

В `.env` можно указать различные модели:

```env
# Multilingual модели (хорошо работают с русским):
EMBEDDING_MODEL=intfloat/multilingual-e5-large
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

# Русские модели:
EMBEDDING_MODEL=cointegrated/rubert-tiny2
```

### Переключение стратегии чанкинга

**Paragraph (по абзацам):**
```env
CHUNKING_STRATEGY=paragraph
```

**Recursive (с overlap):**
```env
CHUNKING_STRATEGY=recursive
CHUNK_SIZE=500
CHUNK_OVERLAP=50
```

## Интеграция с MCP

Для интеграции с Model Context Protocol (MCP), можно использовать endpoints через HTTP:

```python
# В вашем MCP сервере:
async def get_relevant_chunks(area: str, query: str) -> list:
    """Получить релевантные чанки из DocMind"""
    response = await http_client.post(
        "http://localhost:8000/query",
        json={"area": area, "query": query, "top_k": 5}
    )
    return response.json()["results"]
```

## Структура проекта

```
DocMind/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI приложение
│   ├── config.py            # Конфигурация
│   ├── models.py            # Pydantic модели
│   ├── embeddings.py        # Сервис для создания embeddings
│   ├── qdrant_client.py     # Клиент для Qdrant
│   └── chunking/
│       ├── __init__.py
│       ├── base.py          # Базовый класс
│       ├── paragraph.py     # Разбиение по абзацам
│       └── recursive.py     # Рекурсивное разбиение
├── .env                     # Переменные окружения
├── .env.example             # Шаблон .env
├── requirements.txt
└── README.md
```

## Примеры использования

### Полный рабочий пример

```python
import requests

BASE_URL = "http://localhost:8000"

# 1. Загрузка документа
document_text = """
Тендерная документация.

Требования к участникам:
1. Наличие лицензии
2. Опыт работы не менее 5 лет
3. Финансовая устойчивость

Сроки подачи заявок:
Заявки принимаются до 31 декабря 2024 года.
"""

ingest_response = requests.post(
    f"{BASE_URL}/ingest",
    json={
        "document_name": "Тендер №456 - Строительство",
        "text": document_text
    }
)

ingest_data = ingest_response.json()
document_id = ingest_data["document_id"]
print(f"Загружено чанков: {ingest_data['chunks_count']}")
print(f"Document ID: {document_id}")

# 2. Поиск по документу
query_response = requests.post(
    f"{BASE_URL}/query",
    json={
        "document_id": document_id,
        "query": "Какие требования к участникам?",
        "top_k": 3
    }
)

for result in query_response.json()["results"]:
    print(f"\nScore: {result['score']:.3f}")
    print(f"Text: {result['text']}")
    print(f"Document: {result['metadata']['document_name']}")
```

## Troubleshooting

### Ошибка подключения к Qdrant

Проверьте URL и доступность Qdrant:
```bash
curl http://your-qdrant-url:6333/collections
```

### Модель embeddings не загружается

При первом запуске модель будет скачана автоматически. Убедитесь, что есть доступ к интернету и достаточно места на диске.

### Слишком маленькие или большие чанки

Настройте параметры в `.env`:
- Для `paragraph`: нет дополнительных параметров
- Для `recursive`: измените `CHUNK_SIZE` и `CHUNK_OVERLAP`

## Лицензия

MIT
