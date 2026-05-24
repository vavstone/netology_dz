import time
from uuid import uuid4
from loguru import logger
from typing import Generator

from config import Settings
from models import AssistantResponse, SessionStats, LLMStat, ChatResult, ProviderType
from infrastructure.cache import RedisCache
from infrastructure.llm import FALLBACK_ANSWER, RobustLLMClient
from prompts.loader import build_answer_messages, build_classifier_messages, build_classifier_system_prompt, build_system_prompt
from core.classification import should_escalate

class SupportAssistantApp:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.system_prompt = build_system_prompt()
        self.classification_prompt = build_classifier_system_prompt()
        self.history: list[dict[str,str]] = []
        self.failed_attempts = 0
        self.stats = SessionStats()
        self.cache = RedisCache(settings.redis_host, settings.redis_port, settings.redis_ttl)
        self.client = RobustLLMClient(settings)
        logger.remove()
        logger.add(settings.log_path, format="{time} {message}", rotation="10 MB")

    def handle_command(self, command: str) -> str|None:
        if command=="/clear":
            self.history.clear()
            self.failed_attempts = 0
            return "История очищена"
        if command=="/quit":
            return None
        if command=="/stats":
            cache_info = self.cache.stats()
            return (
                f"Запросов: {self.stats.total_queries} | "
                f"LLM вызовов (cloud): {self.stats.cloud_llm_calls} | "
                f"Токенов (cloud): {self.stats.cloud_total_tokens} | "
                f"Кеш-токенов (cloud): {self.stats.cloud_cache_tokens} | "
                f"LLM вызовов (local): {self.stats.local_llm_calls} | "
                f"Токенов (local): {self.stats.local_total_tokens} | "
                f"Эскалаций: {self.stats.escalations} | "
                f"Redis: {cache_info['keys']} ключей, "
                f"hit rate: {cache_info['hit_rate']} "
                f"({cache_info['hits']}/{cache_info['hits'] + cache_info['misses']})"
            )
        return "Доступные команды: /clear, /quit, /stats"

    def respond(self, user_message: str) -> Generator[str, None, AssistantResponse]:
        started_at = time.perf_counter()
        self.stats.total_queries += 1
        latency:float
        # Классификация всегда остаётся моментальной
        # TODO учесть в статистике category_result
        category_result = self.client.classify(build_classifier_messages(self.classification_prompt, user_message))

        # --- Эскалация ---
        if should_escalate(
                user_message=user_message,
                category=category_result.category,
                failed_attempts=self.failed_attempts,
        ):
            self.stats.escalations += 1
            self.failed_attempts = 0
            text = f"Передаю вопрос специалисту. Номер обращения: {uuid4().hex[:8].upper()}."
            latency = time.perf_counter() - started_at
            self._log(user_message, category_result.category, text, 0, latency, False, "router", "escalation")

            def _escalation_gen() -> Generator[str, None, AssistantResponse]:
                yield text
                return AssistantResponse(text, category_result.category, False, latency, "router", "escalation", False)

            return _escalation_gen()

        # --- Кеш ---
        cached = self.cache.get(user_message)
        if cached is not None:
            cached_val:str = cached
            self.stats.cache_hits += 1
            self._remember_turn(user_message, cached)
            latency = time.perf_counter() - started_at
            self._log(user_message, category_result.category, cached, 0, latency, True, "cache", "cache")

            def _cache_gen() -> Generator[str, None, AssistantResponse]:
                yield cached_val
                return AssistantResponse(cached_val, category_result.category, True, latency, "cache", "cache", False)

            return _cache_gen()

        # --- Основной потоковый ответ ---
        self.stats.cache_misses += 1
        messages = build_answer_messages(self.system_prompt, self.history, user_message)
        answer_stream = self.client.answer(messages)  # возвращает Generator[str, None, ChatResult]

        def _stream_gen() -> Generator[str, None, AssistantResponse]:
            full_text_parts = []
            chat_result:ChatResult
            latency2:float
            # Пробрасываем все чанки наружу и ловим завершение внутреннего генератора
            try:
                while True:
                    try:
                        chunk = next(answer_stream)
                        if chunk:
                            full_text_parts.append(chunk)
                            yield chunk
                    except StopIteration as exc:
                        chat_result = exc.value  # ChatResult
                        break
            except Exception:
                raise  # ошибка стрима пробрасывается дальше

            # Постобработка после получения полного ответа
            full_text = "".join(full_text_parts).strip()
            if not chat_result:
                # Запасной вариант (теоретически не нужен)
                chat_result = ChatResult(
                    text=full_text or FALLBACK_ANSWER,
                    stat=LLMStat(),
                    provider=None,
                    model=None,
                    used_fallback=True
                )

            # Кешируем итоговый ответ
            self.cache.set(user_message, chat_result.text)

            # Счётчик неудачных попыток
            if chat_result.text.strip() == FALLBACK_ANSWER:
                self.failed_attempts += 1
            else:
                self.failed_attempts = 0

            latency2 = time.perf_counter() - started_at
            self._remember_turn(user_message, chat_result.text)
            if RobustLLMClient.get_provider_type(chat_result.provider) == ProviderType.OPENAIREADY:
                self.stats.cloud_llm_calls += 1
                self.stats.cloud_total_tokens += chat_result.stat.total_tokens
                self.stats.cloud_cache_tokens += chat_result.stat.cached_tokens
            elif RobustLLMClient.get_provider_type(chat_result.provider) == ProviderType.OLLAMA:
                self.stats.local_llm_calls += 1
                self.stats.local_total_tokens += chat_result.stat.total_tokens
            self._log(
                user_message, category_result.category, chat_result.text,
                chat_result.stat.total_tokens, latency2, False,
                chat_result.provider, chat_result.model
            )

            return AssistantResponse(
                chat_result.text, category_result.category, False, latency2,
                chat_result.provider, chat_result.model, chat_result.used_fallback
            )

        return _stream_gen()

    def _remember_turn(self, user_message:str, answer:str)->None:
        self.history.append({'role': 'user','content' : user_message})
        self.history.append({'role': 'assistant', 'content': answer})
        if len(self.history)>self.settings.history_limit:
            self.history = self.history[-self.settings.history_limit:]

    @staticmethod
    def _log(user_message:str,
             category:str,
             answer:str,
             tokens:int,
             latency_seconds:float,
             from_cache:bool,
             provider:str|None,
             model:str|None):
        logger.info(f"{category} | {provider}/{model} | {tokens} tok | {latency_seconds:.3f}s | cache={from_cache} | Q: {user_message} | A: {answer}")
