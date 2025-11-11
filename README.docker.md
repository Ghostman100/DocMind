# DocMind - Инструкция по запуску через Docker

## Быстрый старт
Проверка работы Qdrant:
```bash
curl http://localhost:6333/health
```

### 1. Сборка образа приложения

```bash
docker build -t docmind-app .
```

### 2. Запуск приложения

**С использованием существующего .env файла:**

```bash
docker run -d \
  --name docmind \
  -p 127.0.0.1:8001:8000 \
  --env-file .env \
  -v $(pwd)/app:/app/app \
  -v model_cache:/home/appuser/.cache \
  --network qdrant_net \
  -e QDRANT_URL=http://qdrant:6333 \
  docmind-app
```


### 3. Запуск приложения local

**С использованием существующего .env файла:**

```bash
docker run -d \
  --name docmind \
  -p 127.0.0.1:8000:8000 \
  --env-file .env \
  -v $(pwd)/app:/app/app \
  -v model_cache:/home/appuser/.cache \
  --network host \
  docmind-app
```