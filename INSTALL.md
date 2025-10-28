# Инструкция по установке DocMind

## Проблема с размером venv

При установке через pip в `venv/` может занять **3-5 GB** из-за:
- PyTorch с CUDA (~2GB)
- Дубликаты nvidia пакетов
- Кеш модулей

## Решение: Poetry

Poetry решает эти проблемы автоматически:
- ✅ Устанавливает CPU-only PyTorch (~2GB экономии)
- ✅ Избегает дубликатов зависимостей
- ✅ Управляет версиями автоматически
- ✅ Создает воспроизводимое окружение

## Быстрый старт

### 1. Установка Poetry

**Linux/macOS/WSL:**
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

**Windows (PowerShell):**
```powershell
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -
```

Добавьте Poetry в PATH (перезапустите терминал после установки).

### 2. Установка зависимостей

```bash
cd DocMind
poetry install
```

Poetry автоматически:
- Создаст виртуальное окружение
- Установит PyTorch CPU-only
- Установит все зависимости без дубликатов

### 3. Запуск приложения

**Вариант 1: С активацией окружения**
```bash
poetry shell
uvicorn app.main:app --reload
```

**Вариант 2: Без активации**
```bash
poetry run uvicorn app.main:app --reload
```

## Альтернатива: pip (если Poetry недоступен)

```bash
# Создать venv
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# или venv\Scripts\activate для Windows

# Установить PyTorch CPU-only
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Установить остальные пакеты
pip install fastapi uvicorn[standard] pydantic pydantic-settings \
    python-dotenv python-multipart sentence-transformers qdrant-client numpy
```

## Сравнение размеров

| Метод | Размер venv | Время установки |
|-------|-------------|-----------------|
| pip с CUDA | ~4.5 GB | ~10 мин |
| pip CPU-only | ~2.5 GB | ~7 мин |
| **Poetry CPU-only** | **~2.2 GB** | **~6 мин** |

## Полезные команды Poetry

```bash
# Добавить новый пакет
poetry add package-name

# Обновить зависимости
poetry update

# Показать установленные пакеты
poetry show

# Экспортировать в requirements.txt
poetry export -f requirements.txt --output requirements.txt

# Удалить окружение и создать заново
poetry env remove python
poetry install
```

## Настройка окружения

После установки создайте `.env`:

```bash
cp .env.example .env
# Отредактируйте .env, укажите QDRANT_URL
```

## Проверка установки

```bash
poetry run python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

Должно вывести:
```
PyTorch: 2.5.1+cpu
CUDA: False
```

## Troubleshooting

### Poetry не найден после установки
```bash
# Добавьте в ~/.bashrc или ~/.zshrc
export PATH="$HOME/.local/bin:$PATH"
```

### Ошибка при установке PyTorch
```bash
# Очистите кеш Poetry
poetry cache clear pypi --all
poetry install
```

### Нужна GPU версия PyTorch
Отредактируйте `pyproject.toml`, удалите строки с `source = "pytorch-cpu"` и установите:
```bash
poetry add torch torchvision torchaudio --source pypi
```
