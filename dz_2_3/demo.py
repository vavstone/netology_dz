#!/usr/bin/env python3
"""
Демонстрация работы RobustLLMClient:
- обычный вызов,
- симуляция ошибок (с помощью подмены URL для тестирования retry),
- fallback на OpenRouter,
- трекинг стоимости и статистики.
"""

import sys
import time
from robust_llm_client import RobustLLMClient


def demo_normal_usage():
    """Демонстрация обычного успешного вызова."""
    print("\n=== Демонстрация 1: обычный вызов ===")
    client = RobustLLMClient(daily_budget_limit=1.0)  # лимит $1 на демо
    messages = [
        {"role": "system", "content": "Ты — полезный ассистент. Отвечай кратко."},
        {"role": "user", "content": "Что такое exponential backoff?"}
    ]
    answer = client.chat(messages)
    print(f"Ответ: {answer}")
    print(f"Статистика: {client.get_session_stats()}")


def demo_fallback():
    """Демонстрация fallback: принудительно отключаем OpenAI (подменяем base_url)."""
    print("\n=== Демонстрация 2: Fallback на OpenRouter ===")
    client = RobustLLMClient()
    # Временно подменяем клиента OpenAI на нерабочий URL
    original_client = client.providers[0]["client"]
    client.providers[0]["client"].base_url = "https://invalid.api.openai.com/v1"
    try:
        messages = [{"role": "user", "content": "Привет! Кто ты?"}]
        answer = client.chat(messages)
        print(f"Ответ (после fallback): {answer}")
    finally:
        # Восстанавливаем (для дальнейших тестов)
        client.providers[0]["client"].base_url = original_client.base_url


def simulate_retry_sequence():
    """Симулирует последовательность retry с логами (без реального вызова, только для показа)."""
    print("\n=== Демонстрация 3: Лог ретраев (симуляция) ===")
    print("Ниже приведён пример лога, который будет сгенерирован при возникновении ошибки 429:\n")
    # Выводим пример ожидаемых логов
    log_example = """
2025-05-14 10:00:01,123 - __main__ - WARNING - Попытка 1/5 для OpenAI: RateLimitError (код: 429) — Rate limit exceeded
2025-05-14 10:00:01,124 - __main__ - INFO - Повтор через 1.23 секунд...
2025-05-14 10:00:02,357 - __main__ - WARNING - Попытка 2/5 для OpenAI: RateLimitError (код: 429) — Rate limit exceeded
2025-05-14 10:00:02,358 - __main__ - INFO - Повтор через 2.45 секунд...
2025-05-14 10:00:04,808 - __main__ - WARNING - Попытка 3/5 для OpenAI: APIError (код: 500) — Internal server error
2025-05-14 10:00:04,809 - __main__ - INFO - Повтор через 4.67 секунд...
2025-05-14 10:00:09,479 - __main__ - INFO - Успешный ответ от OpenAI | Токены: 15+32 | Стоимость: $0.000015 | За сегодня: $0.000015
"""
    print(log_example)
    print("В реальности, если API вернёт ошибку, вы увидите подобные записи в логах.")


if __name__ == "__main__":
    print("Запуск демонстрации RobustLLMClient")
    print("Убедитесь, что переменные окружения OPENAI_API_KEY и OPENROUTER_API_KEY установлены в .env\n")

    demo_normal_usage()
    time.sleep(1)
    demo_fallback()
    simulate_retry_sequence()

    print("\nДемонстрация завершена.")