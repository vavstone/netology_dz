import os
import time
import random
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Tuple

import requests
from openai import OpenAI, RateLimitError, APIError, APIConnectionError, AuthenticationError
from dotenv import load_dotenv

# Настройка логирования
# Создаём handlers: и в консоль, и в файл
console_handler = logging.StreamHandler()
file_handler = logging.FileHandler('llm_client.log.txt', encoding='utf-8')

# Формат сообщений
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Настраиваем корневой логгер
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
root_logger.addHandler(console_handler)
root_logger.addHandler(file_handler)

logger = logging.getLogger(__name__)

# Примерные цены за 1M токенов (обновляются под используемые модели)
PRICES_PER_1M_TOKENS = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.6},
    "deepseek/deepseek-chat": {"input": 0.14, "output": 0.28},  # OpenRouter
}


class CircuitBreaker:
    """Circuit breaker для отдельного провайдера.
    После N последовательных ошибок провайдер блокируется на указанное время.
    """

    def __init__(self, failure_threshold: int = 3, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.blocked_until: Optional[datetime] = None

    def record_failure(self) -> None:
        """Регистрирует ошибку. При превышении порога блокирует провайдера."""
        self.failure_count += 1
        if self.failure_count >= self.failure_threshold:
            self.blocked_until = datetime.now() + timedelta(seconds=self.recovery_timeout)
            logger.warning(f"Circuit breaker открыт. Провайдер заблокирован до {self.blocked_until}")

    def record_success(self) -> None:
        """Сбрасывает счётчик ошибок при успешном вызове."""
        self.failure_count = 0
        self.blocked_until = None

    def is_blocked(self) -> bool:
        """Проверяет, заблокирован ли провайдер в данный момент."""
        if self.blocked_until and datetime.now() < self.blocked_until:
            return True
        if self.blocked_until:  # время блокировки истекло
            self.blocked_until = None
            self.failure_count = 0
        return False


class RateLimiter:
    """Простой токен-бакетный rate limiter для клиентских запросов.
    Ограничивает частоту запросов к API (например, 10 запросов в секунду).
    """

    def __init__(self, requests_per_second: float = 10.0):
        self.rate = requests_per_second
        self.interval = 1.0 / requests_per_second
        self.last_request_time: Optional[float] = None

    def wait_if_needed(self) -> None:
        """Приостанавливает выполнение, если следующий запрос должен быть отложен."""
        if self.last_request_time is None:
            self.last_request_time = time.time()
            return
        elapsed = time.time() - self.last_request_time
        if elapsed < self.interval:
            sleep_time = self.interval - elapsed
            logger.debug(f"Rate limiter: ожидание {sleep_time:.3f} секунд")
            time.sleep(sleep_time)
        self.last_request_time = time.time()


class RobustLLMClient:
    """Надёжный клиент для LLM API с retry, fallback, circuit breaker и трекингом стоимости.

    Особенности:
    - Автоматический retry при ошибках 429 (Rate Limit) и 5xx (серверные ошибки)
      с экспоненциальной задержкой и jitter (максимум 5 попыток).
    - Fallback-цепочка: OpenAI → OpenRouter → fallback-сообщение.
    - Circuit breaker для каждого провайдера: 3 ошибки подряд блокируют провайдера на 60 секунд.
    - Rate limiter на стороне клиента (10 запросов/сек по умолчанию).
    - Логирование каждой попытки (timestamp, код ошибки, номер попытки).
    - Подсчёт стоимости запроса на основе токенов и текущей модели.
    """

    def __init__(self, daily_budget_limit: Optional[float] = None):
        """Инициализирует клиент, загружает API-ключи из .env, настраивает провайдеров."""
        load_dotenv()
        self._validate_dependencies()

        # Провайдеры с их конфигурацией
        self.providers = [
            {
                "name": "OpenAI",
                "client": OpenAI(
                    api_key=os.getenv("OPENAI_API_KEY"),
                    base_url="https://api.openai.com/v1"
                ),
                "model": "gpt-4o-mini",
                "circuit_breaker": CircuitBreaker(),
            },
            {
                "name": "OpenRouter",
                "client": OpenAI(
                    api_key=os.getenv("OPENROUTER_API_KEY"),
                    base_url="https://openrouter.ai/api/v1"
                ),
                "model": "deepseek/deepseek-chat",
                "circuit_breaker": CircuitBreaker(),
            }
        ]

        # Фильтруем провайдеров без валидных ключей
        self.providers = [p for p in self.providers if p["client"].api_key]
        if not self.providers:
            raise SystemExit("Нет доступных провайдеров. Проверьте API-ключи в .env")

        self.rate_limiter = RateLimiter(requests_per_second=10.0)
        self.daily_budget_limit = daily_budget_limit
        self.daily_cost = 0.0
        self.calls_today = 0

        logger.info(f"Инициализирован RobustLLMClient с провайдерами: {[p['name'] for p in self.providers]}")

    @staticmethod
    def _validate_dependencies() -> None:
        """Проверяет наличие необходимых библиотек."""
        try:
            import openai  # noqa
            import dotenv  # noqa
        except ImportError as e:
            raise SystemExit("Установите зависимости: pip install openai python-dotenv") from e

    def _call_with_retry(self, provider: Dict[str, Any], messages: List[Dict[str, str]]) -> Tuple[str, Dict[str, int]]:
        """Вызов конкретного провайдера с exponential backoff + jitter.

        Args:
            provider: Конфигурация провайдера (client, model, circuit_breaker)
            messages: Список сообщений для API

        Returns:
            Tuple[str, dict]: (текст ответа, usage словарь с токенами)

        Raises:
            Exception: Последняя ошибка после всех попыток
        """
        max_retries = 5
        last_exception = None

        for attempt in range(1, max_retries + 1):
            # Проверяем circuit breaker перед вызовом
            if provider["circuit_breaker"].is_blocked():
                raise Exception(f"Провайдер {provider['name']} временно заблокирован circuit breaker'ом")

            try:
                # Rate limiter
                self.rate_limiter.wait_if_needed()

                # Вызов API (таймаут 30 секунд)
                response = provider["client"].chat.completions.create(
                    model=provider["model"],
                    messages=messages,
                    timeout=30,
                )

                # Успех — сбрасываем circuit breaker
                provider["circuit_breaker"].record_success()
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }
                return response.choices[0].message.content, usage

            except (RateLimitError, APIError, APIConnectionError, requests.exceptions.Timeout) as e:
                last_exception = e
                status_code = getattr(e, 'status_code', None) if isinstance(e, APIError) else (
                    429 if isinstance(e, RateLimitError) else 500
                )
                error_type = type(e).__name__

                # Логируем попытку
                logger.warning(
                    f"Попытка {attempt}/{max_retries} для {provider['name']}: "
                    f"{error_type} (код: {status_code}) — {str(e)[:100]}"
                )

                # Регистрируем ошибку в circuit breaker
                provider["circuit_breaker"].record_failure()

                if attempt == max_retries:
                    break

                # Экспоненциальная задержка + jitter
                base_delay = 2 ** (attempt - 1)  # 1, 2, 4, 8 секунд (для 5 попыток последняя 16)
                jitter = random.uniform(0, base_delay * 0.5)
                delay = base_delay + jitter
                logger.info(f"Повтор через {delay:.2f} секунд...")
                time.sleep(delay)

            except AuthenticationError as e:
                # Ошибка аутентификации — не повторяем, сразу пробрасываем
                logger.error(f"Ошибка аутентификации {provider['name']}: {e}")
                raise
            except Exception as e:
                # Неожиданная ошибка — тоже не повторяем
                logger.error(f"Необработанная ошибка {provider['name']}: {e}")
                raise

        raise Exception(
            f"Не удалось выполнить запрос к {provider['name']} после {max_retries} попыток: {last_exception}")

    def _calculate_cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        """Рассчитывает стоимость запроса на основе модели и токенов."""
        price = PRICES_PER_1M_TOKENS.get(model, {"input": 2.00, "output": 8.00})
        cost_input = prompt_tokens / 1_000_000 * price["input"]
        cost_output = completion_tokens / 1_000_000 * price["output"]
        return cost_input + cost_output

    def chat(self, messages: List[Dict[str, str]]) -> str:
        """Отправляет сообщения провайдерам в порядке приоритета с fallback.

        Args:
            messages: Список сообщений в формате OpenAI (role, content)

        Returns:
            str: Текст ответа или сообщение о недоступности сервиса
        """
        last_error = None

        for provider in self.providers:
            try:
                logger.info(f"Пробуем провайдера: {provider['name']}")
                response_text, usage = self._call_with_retry(provider, messages)

                # Логируем usage и стоимость
                cost = self._calculate_cost(provider["model"], usage["prompt_tokens"], usage["completion_tokens"])
                self.daily_cost += cost
                self.calls_today += 1

                # Проверка дневного бюджета
                if self.daily_budget_limit and self.daily_cost > self.daily_budget_limit:
                    logger.warning(f"Превышен дневной бюджет: ${self.daily_cost:.4f} / ${self.daily_budget_limit:.2f}")

                logger.info(
                    f"Успешный ответ от {provider['name']} | Токены: {usage['prompt_tokens']}+{usage['completion_tokens']} | "
                    f"Стоимость: ${cost:.6f} | За сегодня: ${self.daily_cost:.6f}"
                )
                return response_text

            except Exception as e:
                last_error = e
                logger.error(f"Провайдер {provider['name']} не сработал: {e}")
                continue

        # Если все провайдеры недоступны
        logger.critical("Все провайдеры недоступны")
        return "Сервис временно недоступен. Пожалуйста, попробуйте позже."

    def get_session_stats(self) -> Dict[str, Any]:
        """Возвращает статистику текущей сессии (стоимость, количество вызовов)."""
        return {
            "calls_today": self.calls_today,
            "total_cost_usd": round(self.daily_cost, 6),
            "daily_budget_limit": self.daily_budget_limit,
        }