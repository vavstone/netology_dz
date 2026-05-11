import time
import os
import sys
from typing import Any
from pathlib import Path

# Определяем папку приложения (там, где находится этот скрипт)
APP_DIR = Path(__file__).parent
LOG_FILE = APP_DIR / "output.log"  # имя файла для сохранения вывода

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

def get_client(provider:str = 'openai') -> tuple[Any,str]:

    try:
        from openai import OpenAI
    except ImportError as ex:
        raise SystemExit('Установите зависимость openai: pip install openai') from ex
    configs = {
        "openai": {
            "base_url": "https://api.openai.com/v1",
            "api_key": os.getenv("OPENAI_API_KEY"),
            "model": "gpt-4o-mini",
        },
        "openrouter": {
            "base_url": "https://openrouter.ai/api/v1",
            "api_key": os.getenv("OPENROUTER_API_KEY"),
            "model": "nvidia/nemotron-3-super-120b-a12b:free",
        },
        "ollama": {
            "base_url": "http://localhost:11434/v1",
            "api_key": "ollama",
            "model": "gemma3:1b",
        },
    }
    if provider not in configs:
        available = ", ".join(configs)
        raise SystemExit(f"Неизвестный провайдер {provider}. Доступно: {available}")
    cfg = configs[provider]
    if not cfg["api_key"]:
        raise SystemExit(f"Для провайдера {provider} не настроен API ключ")
    client = OpenAI(base_url=cfg["base_url"], api_key=cfg["api_key"])
    return client, cfg["model"]

def build_messages() -> list[dict[str, str]]:
    return [
        {"role": "system", "content": "Ты — опытный Python-разработчик."},
        {"role": "user", "content": "Объясни в 3 предложениях, что такое списковые включения."},
    ]

def ask_model(provider: str, temperature:float, is_stream:bool)->None:
    start = time.perf_counter()
    client, model = get_client(provider)
    response = client.chat.completions.create(
        model=model,
        messages=build_messages(),
        stream=is_stream,
        temperature=temperature,
    )
    ttft = None
    tee_print(f"\nПровайдер: {provider}")
    tee_print(f"Модель: {model}")
    tee_print(f"Температура: {temperature}")
    tee_print(f"Stream: {is_stream}\n")

    if is_stream:
        full_response = ""
        for chunk in response:
            if ttft is None:
                ttft = time.perf_counter() - start  # Первый chunk = TTFT
            delta = chunk.choices[0].delta.content or ""
            if delta:
                tee_print(delta, end="", flush=True)
                full_response += delta
    else:
        tee_print(response.choices[0].message.content)
    tee_print()
    total_time = time.perf_counter() - start
    if is_stream:
        tee_print(f'TTFT: {ttft:.2f} сек.')
    tee_print(f'Total time: {total_time:.2f} сек.')
    tee_print('_'*30)


def main() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError as ex:
        raise SystemExit('Установите зависимость python-dotenv: pip install python-dotenv') from ex
    load_dotenv()
    provider = os.getenv("LLM_PROVIDER", "ollama")
    ask_model(provider, 0, False)
    ask_model(provider, 0.7, False)
    ask_model(provider, 1.5, False)
    ask_model(provider, 0, True)
    ask_model(provider, 0.7, True)
    ask_model(provider, 1.5, True)

if __name__ == "__main__":
    main()