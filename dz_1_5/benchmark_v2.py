import json
import time
import os
from dotenv import load_dotenv
from transformers import pipeline
import openai
from eval_tasks import EVAL_TASKS
from tee_print import tee_print

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def ner_pipeline_to_string(entities):
    """Преобразует выход NER пайплайна в единую строку"""
    if not entities:
        return ""
    grouped = {}
    for ent in entities:
        grp = ent['entity_group']
        word = ent['word'].strip()
        grouped.setdefault(grp, []).append(word)
    parts = []
    for grp, words in grouped.items():
        parts.append(f"{grp}: {', '.join(words)}")
    return "; ".join(parts)


def json_entities_to_string(data: dict) -> str:
    """
    Универсальный конвертер JSON-ответа облачной модели.
    Принимает словарь с любыми ключами (типами сущностей) и значениями-списками/строками.
    Возвращает строку вида 'TYPE: val1, val2; ANOTHER_TYPE: val3; ...'
    """
    parts = []
    for tag, value in data.items():
        if isinstance(value, str):
            value = value.strip()
        elif isinstance(value, list):
            value = ', '.join([v.strip() for v in value if v])
        else:
            value = ''
        parts.append(f"{tag}: {value}")
    return '; '.join(parts)


def benchmark_model(model_name: str, eval_tasks):
    ner = pipeline(
        "ner",
        model=model_name,
        aggregation_strategy="simple",
        token=HF_TOKEN,
    )
    correct = 0
    total_time = 0.0
    measures = []

    for task in eval_tasks:
        text = task["prompt"]
        start = time.perf_counter()
        entities = ner(text)
        elapsed = time.perf_counter() - start
        total_time += elapsed

        extracted_str = ner_pipeline_to_string(entities)
        is_correct = task["check"](extracted_str)
        if is_correct:
            correct += 1

        measures.append({
            "name": task["name"],
            "correct": is_correct,
            "time": elapsed,
            "extracted": extracted_str,
            "original_text": text
        })

    return {
        'total_time': total_time,
        'success': correct,
        'measures': measures
    }


def benchmark_cloud_model(
    model_name: str,
    eval_tasks,
    api_key: str = None,
    max_tokens: int = 150,
    temperature: float = 0.0,
    entity_types: list[str] | None = None,
    use_json_response: bool = True,
    base_url: str = "https://api.openai.com/v1"
):
    """
    Тестирует модель NER через OpenRouter API, используя библиотеку openai.

    Параметры:
    - entity_types: список типов сущностей, например ['PER', 'ORG', 'LOC'].
      Если None — модель сама решает, какие типы извлекать.
    - use_json_response: запрашивать ответ в формате JSON (рекомендуется).
    - base_url: базовый URL для API (по умолчанию OpenRouter).
    """
    if api_key is None:
        api_key = OPENAI_API_KEY
    if not api_key:
        raise ValueError("Open API key is missing. Set OPENAI_API_KEY in .env or pass api_key parameter.")

    # Инициализация клиента OpenAI с кастомным base_url
    client = openai.OpenAI(api_key=api_key, base_url=base_url)

    # Формируем системный промпт динамически
    if entity_types:
        types_str = ", ".join(entity_types)
        json_example = {t: [f"пример_{t}"] for t in entity_types}
        system_prompt = (
            f"Ты — система извлечения именованных сущностей (NER). "
            f"Извлекай строго только сущности следующих типов: {types_str}.\n"
            f"Ответ верни в виде JSON-объекта, где ключи — это типы сущностей ({types_str}), "
            f"а значения — списки найденных строк (или пустые списки, если сущностей нет).\n"
            f"Пример формата ответа: {json.dumps(json_example, ensure_ascii=False)}\n"
            f"Никаких комментариев, только JSON."
        )
    else:
        system_prompt = (
            "Ты — система извлечения именованных сущностей (NER). "
            "Извлеки все сущности любых типов, которые найдёшь в тексте. "
            "Ответ верни в виде JSON-объекта, где ключи — это названия типов сущностей, "
            "а значения — списки строк. Пример: {\"PER\": [\"Иван\"], \"DATE\": [\"2023 год\"]}. "
            "Никаких комментариев, только JSON."
        )

    correct = 0
    total_time = 0.0
    measures = []

    for task in eval_tasks:
        text = task["prompt"]
        user_prompt = f"Извлеки сущности из текста:\n\n{text}"

        # Параметры запроса
        request_params = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if use_json_response:
            request_params["response_format"] = {"type": "json_object"}

        start = time.perf_counter()
        try:
            response = client.chat.completions.create(**request_params)
            content = response.choices[0].message.content.strip()
        except Exception as e:
            elapsed = time.perf_counter() - start
            total_time += elapsed
            measures.append({
                "name": task["name"],
                "correct": False,
                "time": elapsed,
                "extracted": f"ERROR: {str(e)[:100]}",
                "original_text": text
            })
            continue

        elapsed = time.perf_counter() - start
        total_time += elapsed

        # Парсим JSON-ответ
        if use_json_response:
            try:
                parsed = json.loads(content)
                extracted_str = json_entities_to_string(parsed)
            except json.JSONDecodeError:
                # Если модель не вернула чистый JSON, используем как есть
                extracted_str = content.replace("\n", " ").strip()
        else:
            extracted_str = content

        is_correct = task["check"](extracted_str)
        if is_correct:
            correct += 1

        measures.append({
            "name": task["name"],
            "correct": is_correct,
            "time": elapsed,
            "extracted": extracted_str,
            "original_text": text
        })

    return {
        'total_time': total_time,
        'success': correct,
        'measures': measures
    }


# ----------------------------------------
# Основной блок тестирования
# ----------------------------------------
local_models = [
    "Babelscape/wikineural-multilingual-ner",
    "oliverguhr/fullstop-punctuation-multilang-large",
    "Gherman/bert-base-NER-Russian",
]

models_results = {}

# Локальные модели
for m in local_models:
    result = benchmark_model(m, EVAL_TASKS)
    models_results[m] = result
    tee_print(f"Model: {m:50}  | time: {result['total_time']:5.2f}s | success: {result['success']}")

# Облачная модель
if OPENAI_API_KEY:
    cloud_model = "gpt-4o-mini"
    #tee_print(f"\nModel: {cloud_model:50}  | (cloud, JSON mode, entities: PER/ORG/LOC)  | testing...")
    try:
        cloud_result = benchmark_cloud_model(
            cloud_model, EVAL_TASKS,
            entity_types=None,#["PER", "ORG", "LOC"],
            use_json_response=True
        )
        models_results[cloud_model] = cloud_result
        tee_print(f"Model: {cloud_model:43} (cloud) | time: {cloud_result['total_time']:5.2f}s | success: {cloud_result['success']}")
    except Exception as e:
        tee_print(f"Cloud model test failed: {e}")

# Вывод детализации
tee_print()
tee_print("="*100)
tee_print("Детализация:")
tee_print("="*100)
tee_print()

for m, res in models_results.items():
    tee_print(f"Модель {m}:")
    tee_print()
    for i, measure in enumerate(res['measures']):
        tee_print(f" {(i+1):3} | task: {measure['name']:45} | correct: {measure['correct']:3} | orig: {measure['original_text']:80} | res: {measure['extracted']}\n")