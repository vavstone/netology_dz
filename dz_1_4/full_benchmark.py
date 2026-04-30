import time
import os
import sys
from openai import OpenAI
from eval_tasks import EVAL_TASKS
from openai.types.chat import ChatCompletionUserMessageParam
from openai.types.chat import ChatCompletionStreamOptionsParam
from dotenv import load_dotenv
from pathlib import Path


# Определяем папку приложения (там, где находится этот скрипт)
APP_DIR = Path(__file__).parent
LOG_FILE = APP_DIR / "output.log"   # имя файла для сохранения вывода


def tee_print(*args, sep=' ', end='\n', file=sys.stdout, flush=False):
    """
    Аналог print, который:
      - выводит в консоль (или в указанный file)
      - дописывает такую же строку в текстовый файл (LOG_FILE)
    """
    # 1. Вывод в обычный print (консоль или другой поток)
    print(*args, sep=sep, end=end, file=file, flush=flush)

    # 2. Формируем ту же строку для записи в файл
    line = sep.join(str(arg) for arg in args) + end

    # 3. Пишем в файл (в режиме добавления, UTF-8)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(line)
        if flush:
            f.flush()


def benchmark_model(model_desc, evals) -> dict:
    """Полный бенчмарк с прогоном для каждого теста в evals"""
    load_dotenv()
    results = []
    details = []
    passed = 0
    counter = 0
    client: OpenAI
    if model_desc['type'] == 'ollama':
        client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
    elif model_desc['type'] == 'openrouter':
        openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=openrouter_api_key)
    elif model_desc['type'] == 'openai':
        openai_api_key = os.getenv("OPENAI_API_KEY")
        client = OpenAI(base_url="https://api.openai.com/v1", api_key=openai_api_key)

    for task in evals:
        start = time.perf_counter()
        stream_options: ChatCompletionStreamOptionsParam = {"include_usage": True}
        stream = client.chat.completions.create(
            model = model_desc['name'],
            messages=[
                ChatCompletionUserMessageParam(role="user", content=task['prompt'])
            ],
            stream=True,
            stream_options=stream_options,
            temperature=0
        )

        answer=''
        ttft = None
        tokens = 0
        usage_input_tokens = 0
        usage_output_tokens = 0
        usage_all_tokens = 0

        for chunk in stream:
            if ttft is None:
                ttft = time.perf_counter() - start
            tokens += 1
            # Извлекаем текстовый контент из фрагмента
            if chunk.choices and chunk.choices[0].delta.content:
                content_piece = chunk.choices[0].delta.content
                answer+=content_piece
            # Когда дойдём до последнего фрагмента, в нём будет статистика использования
            elif chunk.usage:
                usage_input_tokens = chunk.usage.prompt_tokens
                usage_output_tokens = chunk.usage.completion_tokens
                usage_all_tokens = chunk.usage.total_tokens

        success = task['check'](answer)
        counter+=1
        if success:
            passed+=1
        details.append(f' {counter}. {'✓' if success else '✗'} Task: {task['name']}\nPrompt: {task['prompt']}\nModel result: {answer}')

        total = time.perf_counter() - start
        gen_time = total - ttft if ttft else total
        throughput = tokens / gen_time if gen_time > 0 else 0

        results.append({
            "ttft": ttft,
            "total": total,
            "tokens": tokens,
            "throughput": throughput,
            "usage_input_tokens": usage_input_tokens,
            "usage_output_tokens": usage_output_tokens,
            "usage_all_tokens": usage_all_tokens,
        })

    # Усредняем результаты
    avg = {
        "model_type": model_desc['type'],
        "model_name": model_desc['name'],
        "avg_ttft": round(sum(r["ttft"] for r in results) / len(evals), 3),
        "avg_total": round(sum(r["total"] for r in results) / len(evals), 3),
        "avg_tokens": round(sum(r["tokens"] for r in results) / len(evals)),
        "avg_throughput": round(sum(r["throughput"] for r in results) / len(evals), 1),
        "avg_usage_input_tokens": round(sum(r["usage_input_tokens"] for r in results) / len(evals)),
        "avg_usage_output_tokens": round(sum(r["usage_output_tokens"] for r in results) / len(evals)),
        "avg_usage_all_tokens": round(sum(r["usage_all_tokens"] for r in results) / len(evals)),
        "passed":passed,
        "details":details
    }
    return avg


# Сравниваем три модели
models = [
    {'type': 'ollama', 'name': 'gemma3:1b'},
    {'type':'ollama','name':'qwen3:1.7b'},
    {'type':'ollama','name':'gemma3:4b'},
    #{'type':'ollama','name':'qwen3:8b'},
    {'type':'openai','name':'gpt-4o-mini'},
    #{'type':'openrouter','name':'openrouter/free'},
]

tasks_to_process = EVAL_TASKS[:]

tee_print(f"Бенчмарк моделей по eval набору ({len(tasks_to_process)} задач):\n")
results = []
for model_desc in models:
    result = benchmark_model(model_desc, tasks_to_process)
    tee_print(
        f"{result['model_type']:12s} | {result['model_name']:16s} |  TTFT: {result['avg_ttft']:.3f}s | "
        f"Avg total: {result['avg_total']:5.2f}s | Avg tokens: {result['avg_tokens']:5} | {result['avg_throughput']:5.1f} tok/s | "
        f"Avg usage_input_tokens: {result['avg_usage_input_tokens']:5} | "
        f"Avg usage_output_tokens: {result['avg_usage_output_tokens']:5} | "
        f"Avg usage_all_tokens: {result['avg_usage_all_tokens']:5} | "
        f"Eval passed: {result['passed']} ")
    results.append(result)

tee_print("\nДетали прохождения тестов:\n")
for result in results:
    tee_print()
    tee_print(f'Провайдер: {result['model_type']}, модель: {result['model_name']}\n')
    for detail in result['details']:
        tee_print(detail)

