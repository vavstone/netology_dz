import time

import ollama


def measure_ttft(model: str, prompt: str) -> dict:
    """Измеряем время до первого токена через стриминг"""
    start = time.perf_counter()  # perf_counter точнее, чем time.time()
    client = ollama.Client(host='http://localhost:11434')
    stream = client.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        stream=True,  # Обязательно! Без stream нельзя измерить TTFT
    )

    ttft = None
    full_response = ""
    token_count = 0

    for chunk in stream:
        if ttft is None:
            ttft = time.perf_counter() - start  # Первый chunk = TTFT
        content = chunk["message"]["content"]
        full_response += content
        token_count += 1  # Примерный подсчёт по чанкам

    total_time = time.perf_counter() - start

    return {
        "model": model,
        "ttft": round(ttft, 3),
        "total_time": round(total_time, 3),
        "tokens": token_count,
        "throughput": round(token_count / (total_time - ttft), 1) if total_time > ttft else 0,
    }


# Запуск
result = measure_ttft("qwen3:8b", "Объясни что такое REST API в трёх предложениях")
print(
    f"TTFT: {result['ttft']}s | Total: {result['total_time']}s | "
    f"Throughput: {result['throughput']} tok/s"
)
# Пример вывода: TTFT: 0.234s | Total: 2.871s | Throughput: 68.3 tok/s
