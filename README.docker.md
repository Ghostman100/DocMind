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
  -v model_cache:/home/appuser/.cache \
  docmind-app
```