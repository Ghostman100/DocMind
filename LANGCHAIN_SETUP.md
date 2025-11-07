# LangChain Integration - Quick Start Guide

## Обзор

DocMind теперь поддерживает два способа работы с документами:

1. **Оригинальная система** (`/ingest`, `/query`) - Быстрая и эффективная система без внешних зависимостей
2. **LangChain система** (`/langchain/*`) - Расширенный функционал включая суммаризацию и извлечение ключевых пунктов

Оба подхода совместимы и используют одну коллекцию Qdrant.

## Установка

### 1. Установите зависимости

```bash
pip install -r requirements.txt
```

Это установит все необходимые LangChain пакеты:
- `langchain==0.3.7`
- `langchain-qdrant==0.2.0`
- `langchain-community==0.3.5`
- `langchain-text-splitters==0.3.2`
- `langchain-openai==0.2.5`

### 2. Настройте .env файл

Скопируйте `.env.example`:
```bash
cp .env.example .env
```

**Для базового функционала** (ingest, query) настройте только Qdrant:
```env
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=your-api-key-if-needed
```

**Для суммаризации** добавьте настройки LLM:
```env
LANGCHAIN_ENABLED=true
LLM_PROVIDER=openai
LLM_API_KEY=sk-your-openai-api-key
LLM_MODEL=gpt-4o-mini
LLM_TEMPERATURE=0.0
```

### 3. Запустите сервер

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

API документация доступна на `http://localhost:8000/docs`

## Использование

### Сценарий 1: Базовая загрузка и поиск (без LLM API ключа)

```python
import requests

BASE_URL = "http://localhost:8000"

# 1. Загрузить документ через LangChain
response = requests.post(
    f"{BASE_URL}/langchain/ingest",
    json={
        "document_name": "Тендер №123",
        "text": "Длинный текст тендерной документации..."
    }
)
document_id = response.json()["document_id"]
print(f"Document ID: {document_id}")

# 2. Поиск по документу
response = requests.post(
    f"{BASE_URL}/langchain/query",
    json={
        "document_id": document_id,
        "query": "Каковы требования к участникам?",
        "top_k": 5
    }
)
results = response.json()["results"]
for result in results:
    print(f"Score: {result['score']:.2f}")
    print(f"Text: {result['page_content'][:200]}...")
```

### Сценарий 2: Суммаризация документа (требуется LLM API ключ)

```python
import requests

BASE_URL = "http://localhost:8000"

# Суммаризовать уже загруженный документ
response = requests.post(
    f"{BASE_URL}/langchain/summarize",
    json={
        "document_id": "550e8400-e29b-41d4-a716-446655440000",
        "strategy": "map_reduce",  # или "stuff" / "refine"
        "max_chunks": 100
    }
)

result = response.json()
print(f"Резюме ({result['chunks_processed']} чанков):")
print(result['summary'])
```

**Стратегии суммаризации:**
- `stuff` - Все чанки в один промпт (быстро, для коротких документов)
- `map_reduce` - Суммаризация каждого чанка, затем объединение (лучше для длинных)
- `refine` - Итеративное уточнение (самое тщательное, медленнее)

### Сценарий 3: Извлечение ключевых пунктов

```python
import requests

BASE_URL = "http://localhost:8000"

# Извлечь конкретные пункты по темам
response = requests.post(
    f"{BASE_URL}/langchain/extract_points",
    json={
        "document_id": "550e8400-e29b-41d4-a716-446655440000",
        "topics": [
            "Требования к участникам тендера",
            "Сроки подачи заявок",
            "Критерии оценки предложений"
        ],
        "chunks_per_topic": 3,
        "summarize": True  # Опционально: суммаризовать найденные чанки
    }
)

result = response.json()
for point in result['extracted_points']:
    print(f"\n📌 Тема: {point['topic']}")
    print(f"Найдено чанков: {len(point['relevant_chunks'])}")
    if point['summary']:
        print(f"Резюме: {point['summary']}")
```

**Без суммаризации** (не требуется LLM API ключ):
```python
response = requests.post(
    f"{BASE_URL}/langchain/extract_points",
    json={
        "document_id": document_id,
        "topics": ["Требования к участникам"],
        "chunks_per_topic": 5,
        "summarize": False  # Только извлечение, без суммаризации
    }
)

# Получите raw relevant chunks
for point in result['extracted_points']:
    for chunk in point['relevant_chunks']:
        print(chunk)
```

## Совместимость между системами

Документы загруженные через `/ingest` можно суммаризовать через LangChain:

```python
# 1. Загрузить через оригинальную систему
response = requests.post(
    f"{BASE_URL}/ingest",
    json={"document_name": "Doc", "text": "..."}
)
document_id = response.json()["document_id"]

# 2. Суммаризовать через LangChain
response = requests.post(
    f"{BASE_URL}/langchain/summarize",
    json={"document_id": document_id, "strategy": "map_reduce"}
)
print(response.json()["summary"])
```

И наоборот - документы из `/langchain/ingest` можно искать через `/query`.

## Стоимость и лимиты

### LLM API Costs

LangChain endpoints (`/langchain/summarize`, `/langchain/extract_points` с `summarize=true`) используют внешние LLM API:

**OpenAI gpt-4o-mini** (рекомендуется):
- Input: $0.15 / 1M tokens
- Output: $0.60 / 1M tokens
- Примерная стоимость суммаризации 50-чанкового документа: $0.01-0.05

**OpenAI gpt-4o**:
- Дороже, но более качественные резюме
- Input: $2.50 / 1M tokens

### Рекомендации

1. **Для тестирования**: Используйте `gpt-4o-mini` с небольшими документами
2. **Для production**:
   - Добавьте rate limiting
   - Мониторьте использование через OpenAI dashboard
   - Кэшируйте суммаризации для повторных запросов
3. **Для экономии**:
   - Используйте `max_chunks` параметр для ограничения обработки
   - Используйте `extract_points` вместо полной суммаризации где возможно

## Troubleshooting

### Ошибка: "LangChain functionality is disabled"

Установите в `.env`:
```env
LANGCHAIN_ENABLED=true
```

### Ошибка: "LLM API key is not configured"

Добавьте в `.env`:
```env
LLM_API_KEY=sk-your-openai-api-key
```

### Ошибка: "No chunks found for document_id"

Проверьте что документ существует:
```bash
curl http://localhost:8000/langchain/query \
  -H "Content-Type: application/json" \
  -d '{"document_id": "your-id", "query": "test", "top_k": 1}'
```

### Медленная суммаризация

- Используйте `strategy="stuff"` для коротких документов (быстрее)
- Уменьшите `max_chunks` параметр
- Проверьте latency до OpenAI API

## Примеры использования

### Полный workflow: загрузка → поиск → суммаризация → извлечение

```python
import requests

BASE_URL = "http://localhost:8000"

# 1. Загрузить документ
doc_response = requests.post(
    f"{BASE_URL}/langchain/ingest",
    json={
        "document_name": "Тендерная документация 2024",
        "text": open("tender_document.txt").read()
    }
)
doc_id = doc_response.json()["document_id"]
print(f"✅ Документ загружен: {doc_id}")

# 2. Быстрый поиск по ключевым словам
query_response = requests.post(
    f"{BASE_URL}/langchain/query",
    json={"document_id": doc_id, "query": "сроки", "top_k": 3}
)
print(f"✅ Найдено {len(query_response.json()['results'])} релевантных чанков")

# 3. Полная суммаризация документа
summary_response = requests.post(
    f"{BASE_URL}/langchain/summarize",
    json={"document_id": doc_id, "strategy": "map_reduce"}
)
print(f"✅ Резюме документа:")
print(summary_response.json()['summary'])

# 4. Извлечь конкретные пункты
extract_response = requests.post(
    f"{BASE_URL}/langchain/extract_points",
    json={
        "document_id": doc_id,
        "topics": ["Требования", "Сроки", "Критерии оценки"],
        "summarize": True
    }
)
print(f"✅ Извлечено {len(extract_response.json()['extracted_points'])} пунктов")
```

## Дополнительные ресурсы

- **API Documentation**: `http://localhost:8000/docs`
- **LangChain Docs**: https://python.langchain.com/docs/
- **OpenAI Pricing**: https://openai.com/api/pricing/
- **CLAUDE.md**: Полная техническая документация архитектуры
