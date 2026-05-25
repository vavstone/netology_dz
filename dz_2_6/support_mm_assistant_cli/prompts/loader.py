import json
from pathlib import Path
from jinja2 import Template

from models import ImageInfo

# Определяем корень пакета
PACKAGE_DIR = Path(__file__).parent

def _read_prompt_file(filename: str) -> str:
    """Читает текстовый файл из той же директории."""
    return (PACKAGE_DIR / filename).read_text(encoding="utf-8").strip()

SYSTEM_PROMPT_TEMPLATE = _read_prompt_file("system_prompt.txt")
CLASSIFIER_SYSTEM_PROMPT = _read_prompt_file("classifier_system_prompt.txt")
CLASSIFIER_FEW_SHOTS = json.loads(_read_prompt_file("classifier_few_shots.json"))
ANSWERS_FEW_SHOTS = json.loads(_read_prompt_file("answers_few_shots.json"))

def _render_template(template_text: str, context: dict) -> str:
    """Рендерит Jinja-шаблон."""
    return Template(template_text).render(**context)

def build_system_prompt() -> str:
    """Собирает системный промпт для чат-ассистента."""
    return _render_template(SYSTEM_PROMPT_TEMPLATE, {"answers_few_shots": ANSWERS_FEW_SHOTS})

def build_classifier_system_prompt() -> str:
    """Собирает системный промпт для классификатора."""
    return _render_template(CLASSIFIER_SYSTEM_PROMPT, {"classifier_few_shots": CLASSIFIER_FEW_SHOTS})

def _build_messages(system_prompt: str, user_message: str, user_image:ImageInfo = None, history: list[dict[str, str]] | None = None) -> list[dict[str, str]]:
    """Общая функция для построения списка сообщений."""
    messages = [{'role': 'system', 'content': system_prompt}]
    if history:
        messages.extend(history)
    #вопрос с изображением
    if user_image:
        image_data_url = f'data:{user_image.get_mime_type()};base64,{user_image.base64content}'
        messages.append({
            'role': 'user',
            'content': [
                {'type': 'text', 'text': user_message},
                {'type': 'image_url', 'image_url':
                    {'url': image_data_url, 'detail': 'high'}
                 }
            ]})
    else:
        messages.append({'role': 'system', 'content': user_message})
    return messages

def build_answer_messages(system_prompt: str, history: list[dict[str, str]], user_message: str, userImage: ImageInfo = None) -> list[dict[str, str]]:
    """Сообщения для генерации ответа с учётом истории."""
    return _build_messages(system_prompt, user_message, userImage, history)

def build_classifier_messages(system_prompt: str, user_message: str) -> list[dict[str, str]]:
    """Сообщения для классификации (без истории)."""
    return _build_messages(system_prompt, user_message, None, None)