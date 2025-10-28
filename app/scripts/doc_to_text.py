from pathlib import Path

import requests


def extract_text_from_file(file_path: str):
    """
    Извлекает текст из файла

    Args:
        file_path: Путь к файлу

    Returns:
        dict: Ответ сервера с текстом или ошибкой
    """
    file_path = Path(file_path)

    if not file_path.exists():
        return {"error": f"Файл не найден: {file_path}"}

    try:
        with open(file_path, 'rb') as file:
            files = {'file': (file_path.name, file, 'application/octet-stream')}
            response = requests.post("http://127.0.0.1:8877/extract", files=files)

        return response.json()['text']

    except Exception as e:
        return {"error": f"Ошибка отправки запроса: {str(e)}"}
