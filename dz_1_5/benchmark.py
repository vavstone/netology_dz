from transformers import pipeline
import time
import os
from dotenv import load_dotenv
from eval_tasks import EVAL_TASKS
from tee_print import tee_print

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

def ner_pipeline_to_string(entities):
    """
    Преобразует выход NER пайплайна (список словарей) в строку,
    с которой работают check-функции.
    """
    if not entities:
        return ""
    # Группируем сущности: entity_group -> список слов
    grouped = {}
    for ent in entities:
        grp = ent['entity_group']
        word = ent['word'].strip()
        grouped.setdefault(grp, []).append(word)
    # Формируем строку вида "PER: Иван Петров, ORG: Яндекс, LOC: Москва"
    parts = []
    for grp, words in grouped.items():
        parts.append(f"{grp}: {', '.join(words)}")
    return "; ".join(parts)

def benchmark_model(model_name: str,  eval_tasks):
    max_new_tokens = 100
    ner = pipeline(
        "ner",
        model=model_name,
        aggregation_strategy="simple",
        #max_new_tokens=max_new_tokens,
        token=HF_TOKEN,
    )
    # Прогрев
    #ner('test', max_new_tokens=10)

    correct = 0
    total_time = 0.0
    measures = []

    for task in eval_tasks:
        text = task["prompt"]
        start = time.perf_counter()
        entities = ner(text)
        elapsed = time.perf_counter() - start
        total_time += elapsed

        # Преобразуем выход пайплайна в строку
        extracted_str = ner_pipeline_to_string(entities)

        # Проверяем через check-функцию
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
    result = {
        'total_time': total_time,
        'success': correct,
        'measures': measures
    }
    return result

models = [
    "Babelscape/wikineural-multilingual-ner",
    "oliverguhr/fullstop-punctuation-multilang-large",
    "Gherman/bert-base-NER-Russian",
]

models_results = dict()

for m in models:
    #tee_print(f"Тест модели {m}:")
    result = benchmark_model(m, EVAL_TASKS)
    models_results[m] = result
    tee_print(f"Model: {m:50}  | time: {result['total_time']:5.2f}s | success: {result['success']}")

tee_print()
tee_print("="*100)
tee_print("Детализация:")
tee_print("="*100)
tee_print()
for m in models:
    tee_print(f"Модель {m}:")
    tee_print()
    measures = models_results[m]['measures']
    for i, measure in enumerate(measures):
        tee_print(f" {(i+1):3} | task: {measure['name']:45} | correct: {measure['correct']:3} | orig: {measure['original_text']:80} | res: {measure['extracted']}\n")