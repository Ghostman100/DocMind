# Multi-stage build для уменьшения размера образа
FROM python:3.12-slim as builder

# Установка системных зависимостей
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Создание директории для приложения
WORKDIR /app

# Копирование файлов зависимостей
COPY requirements.txt .

# Установка PyTorch CPU-only (значительно меньше размер, чем с CUDA)
RUN pip install --no-cache-dir --user \
    torch==2.5.1 \
    torchvision==0.20.1 \
    torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cpu

# Установка остальных зависимостей (пропускаем уже установленные torch пакеты)
RUN pip install --no-cache-dir --user -r requirements.txt

# Финальный образ
FROM python:3.12-slim

# Установка необходимых системных библиотек для sentence-transformers
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Создание непривилегированного пользователя
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Копирование установленных зависимостей из builder stage
COPY --from=builder /root/.local /home/appuser/.local

# Копирование кода приложения
COPY --chown=appuser:appuser . .

# Добавление путей Python в PATH
ENV PATH=/home/appuser/.local/bin:$PATH

# Переключение на непривилегированного пользователя
USER appuser

# Создание директории для кэша моделей
RUN mkdir -p /home/appuser/.cache

# Переменные окружения
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TRANSFORMERS_CACHE=/home/appuser/.cache/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/home/appuser/.cache/sentence-transformers

# Порт приложения
EXPOSE 8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health').read()" || exit 1

# Запуск приложения
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
