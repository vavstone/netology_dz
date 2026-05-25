import time
from uuid import uuid4
from loguru import logger
from typing import Generator

from config import Settings
from models import AssistantResponse, SessionStats, LLMStat, ChatResult, Category, ProviderType
from infrastructure.cache import RedisCache
from infrastructure.llm import FALLBACK_ANSWER, RobustLLMClient
from prompts.loader import build_answer_messages, build_classifier_messages, build_classifier_system_prompt, build_system_prompt
from core.classification import should_escalate


class SupportAssistantApp:
    """Основное приложение чат-ассистента поддержки."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.system_prompt = build_system_prompt()
        self.classification_prompt = build_classifier_system_prompt()
        self.history: list[dict[str, str]] = []
        self.failed_attempts = 0
        self.stats = SessionStats()
        self.cache = RedisCache(settings.redis_host, settings.redis_port, settings.redis_ttl)
        self.client = RobustLLMClient(settings)


    def handle_command(self, command: str) -> str | None:
        """Обрабатывает команды пользователя (/clear, /stats, /quit)."""
        if command == "/clear":
            self.history.clear()
            self.failed_attempts = 0
            return "История очищена"
        if command == "/quit":
            return None
        if command == "/stats":
            cache_info = self.cache.stats()
            return (
                f"Запросов: {self.stats.total_queries} | "
                f"LLM вызовов (cloud): {self.stats.cloud_llm_calls} | "
                f"Токенов (cloud): {self.stats.cloud_total_tokens} | "
                f"Кеш-токенов (cloud): {self.stats.cloud_cache_tokens} | "
                f"LLM вызовов (local): {self.stats.local_llm_calls} | "
                f"Токенов (local): {self.stats.local_total_tokens} | "
                f"Эскалаций: {self.stats.escalations} | "
                f"Категории: FAQ={self.stats.categories_faq}, TECHNICAL={self.stats.categories_technical}, "
                f"COMPLAINT={self.stats.categories_complaint}, ESCALATION={self.stats.categories_escalation} | "
                f"Redis: {cache_info['keys']} ключей, hit rate: {cache_info['hit_rate']} "
                f"({cache_info['hits']}/{cache_info['hits'] + cache_info['misses']})"
            )
        return "Доступные команды: /clear, /quit, /stats"

    def _yield_single_response(self, text: str, category: Category, from_cache: bool,
                               latency: float, provider: str | None, model: str | None,
                               used_fallback: bool) -> Generator[str, None, AssistantResponse]:
        """Вспомогательный генератор для ответов, которые не требуют потокового вывода."""
        yield text
        return AssistantResponse(text, category, from_cache, latency, provider, model, used_fallback)

    def _update_stats_from_chat_result(self, chat_result: ChatResult) -> None:
        """Обновляет статистику на основе результата LLM."""
        if RobustLLMClient.get_provider_type(chat_result.provider) == ProviderType.OPENAIREADY:
            self.stats.cloud_llm_calls += 1
            self.stats.cloud_total_tokens += chat_result.stat.total_tokens
            self.stats.cloud_cache_tokens += chat_result.stat.cached_tokens
        elif RobustLLMClient.get_provider_type(chat_result.provider) == ProviderType.OLLAMA:
            self.stats.local_llm_calls += 1
            self.stats.local_total_tokens += chat_result.stat.total_tokens

    def respond(self, user_message: str) -> Generator[str, None, AssistantResponse]:
        """
        Основной метод обработки сообщения.
        Возвращает генератор, выдающий чанки ответа, а после завершения – AssistantResponse.
        """
        started_at = time.perf_counter()
        self.stats.total_queries += 1

        # ---------- 1. Классификация ----------
        try:
            category_result = self.client.classify(
                build_classifier_messages(self.classification_prompt, user_message)
            )
        except Exception as e:
            logger.error(f"Ошибка классификации: {e}, используется эвристика")
            from core.classification import heuristic_classify
            category_result = CategoryResult(
                category=heuristic_classify(user_message),
                stat=LLMStat(),
                provider=None,
                model=None,
                used_fallback=True
            )

        # Обновляем статистику по категориям
        cat = category_result.category
        if cat == Category.FAQ:
            self.stats.categories_faq += 1
        elif cat == Category.TECHNICAL:
            self.stats.categories_technical += 1
        elif cat == Category.COMPLAINT:
            self.stats.categories_complaint += 1
        elif cat == Category.ESCALATION:
            self.stats.categories_escalation += 1

        # ---------- 2. Эскалация ----------
        if should_escalate(user_message, category_result.category, self.failed_attempts):
            self.stats.escalations += 1
            self.failed_attempts = 0
            ticket_id = uuid4().hex[:8].upper()
            text = f"Передаю вопрос специалисту. Номер обращения: {ticket_id}."
            latency = time.perf_counter() - started_at
            self._log(user_message, str(category_result.category), text, 0, latency, False, "router", "escalation")
            return self._yield_single_response(
                text, category_result.category, False, latency,
                "router", "escalation", False
            )

        # ---------- 3. Кеш ----------
        cached = self.cache.get(user_message)
        if cached is not None:
            self.stats.cache_hits += 1
            self._remember_turn(user_message, cached)
            latency = time.perf_counter() - started_at
            self._log(user_message, str(category_result.category), cached, 0, latency, True, "cache", "cache")
            return self._yield_single_response(
                cached, category_result.category, True, latency,
                "cache", "cache", False
            )

        # ---------- 4. Основной потоковый ответ ----------
        self.stats.cache_misses += 1
        messages = build_answer_messages(self.system_prompt, self.history, user_message)

        try:
            answer_stream = self.client.answer(messages)
        except Exception as e:
            logger.error(f"Не удалось инициализировать поток ответа: {e}")
            fallback_text = FALLBACK_ANSWER
            latency = time.perf_counter() - started_at
            self._log(user_message, str(category_result.category), fallback_text, 0, latency, False, None, "error")
            return self._yield_single_response(
                fallback_text, category_result.category, False, latency,
                None, None, True
            )

        def _stream_gen() -> Generator[str, None, AssistantResponse]:
            full_text_parts = []
            chat_result: ChatResult | None = None

            try:
                while True:
                    try:
                        chunk = next(answer_stream)
                        if chunk:
                            full_text_parts.append(chunk)
                            yield chunk
                    except StopIteration as e:
                        chat_result = e.value
                        break
            except Exception as e:
                logger.error(f"Ошибка во время стрима: {e}")
                chat_result = ChatResult(
                    text=FALLBACK_ANSWER,
                    stat=LLMStat(),
                    provider=None,
                    model=None,
                    used_fallback=True
                )

            full_text = "".join(full_text_parts).strip()
            if not chat_result:
                chat_result = ChatResult(
                    text=full_text or FALLBACK_ANSWER,
                    stat=LLMStat(),
                    provider=None,
                    model=None,
                    used_fallback=True
                )

            # Кешируем итоговый ответ
            self.cache.set(user_message, chat_result.text)

            # Обновляем счётчик неудачных попыток
            if chat_result.text.strip() == FALLBACK_ANSWER:
                self.failed_attempts += 1
            else:
                self.failed_attempts = 0

            latency = time.perf_counter() - started_at
            self._remember_turn(user_message, chat_result.text)
            self._update_stats_from_chat_result(chat_result)

            self._log(
                user_message, str(category_result.category), chat_result.text,
                chat_result.stat.total_tokens, latency, False,
                chat_result.provider, chat_result.model
            )

            return AssistantResponse(
                chat_result.text, category_result.category, False, latency,
                chat_result.provider, chat_result.model, chat_result.used_fallback
            )

        return _stream_gen()

    def _remember_turn(self, user_message: str, answer: str) -> None:
        """Сохраняет диалог в историю."""
        self.history.append({'role': 'user', 'content': user_message})
        self.history.append({'role': 'assistant', 'content': answer})
        if len(self.history) > self.settings.history_limit:
            self.history = self.history[-self.settings.history_limit:]

    @staticmethod
    def _log(user_message: str, category: str, answer: str,
             tokens: int, latency_seconds: float, from_cache: bool,
             provider: str | None, model: str | None) -> None:
        """Записывает событие в лог."""
        logger.info(
            f"{category} | {provider}/{model} | {tokens} tok | {latency_seconds:.3f}s | "
            f"cache={from_cache} | Q: {user_message} | A: {answer}"
        )