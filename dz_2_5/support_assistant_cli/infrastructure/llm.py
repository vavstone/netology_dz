from collections.abc import Iterator
from loguru import logger
from typing import Generator
from openai import OpenAI, RateLimitError, APIStatusError
from tenacity import retry, stop_after_attempt, retry_if_exception_type, wait_exponential
from ollama import Options
import ollama

from config import Settings
from models import Category, LLMStat, LLMAnswerWithStat, ChatResult, CategoryResult, WorkType, ProviderType
from core.classification import heuristic_classify

FALLBACK_ANSWER = "Передаю вопрос специалисту."

def _build_client(api_key:str|None, base_url:str|None)->OpenAI:
    return OpenAI(api_key=api_key, base_url=base_url)

class RobustLLMClient:
    def __init__(self,settings:Settings):
        self.settings = settings
        self.main = _build_client(api_key=self.settings.main_provider_api_key, base_url=self.settings.main_provider_base_url)
        self.classifier = _build_client(api_key=self.settings.classifier_provider_api_key, base_url=self.settings.classifier_provider_base_url)
        self.fallback = _build_client(api_key=self.settings.fallback_provider_api_key, base_url=self.settings.fallback_provider_base_url)

    @staticmethod
    def get_provider_type(provider:str|None)->ProviderType|None:
        if provider=='openai' or provider=='openrouter':
            return ProviderType.OPENAIREADY
        if provider=='ollama':
            return ProviderType.OLLAMA
        return None

    def _provider_chain(self, work_type: WorkType) -> Iterator[tuple[OpenAI,str, str,bool]]:
        if work_type == WorkType.CHAT and self.main is not None:
            yield self.main, self.settings.main_provider, self.settings.main_provider_model, False
        if work_type == WorkType.CLASSIFICATION and self.classifier is not None:
            yield self.classifier, self.settings.classifier_provider, self.settings.classifier_provider_model, False
        if self.fallback is not None:
            yield self.fallback, self.settings.fallback_provider, self.settings.fallback_provider_model, True

    def _call(self,
              provider_type: ProviderType,
              client: OpenAI,
              model: str,
              messages: list[dict[str, str]],
              temperature: float = 0.2,
              max_tokens: int = 350) -> LLMAnswerWithStat:
        """
        Синхронный вызов LLM – возвращает готовый ответ со статистикой.
        """

        @retry(
            wait=wait_exponential(multiplier=1, min=1, max=60),
            stop=stop_after_attempt(self.settings.retry_attempts),
            retry=retry_if_exception_type((RateLimitError, APIStatusError))
        )
        def _do() -> LLMAnswerWithStat:
            answer: str = ''
            prompt_tokens: int = 0
            completion_tokens: int = 0
            total_tokens: int = 0
            cached_tokens: int = 0

            if provider_type == ProviderType.OPENAIREADY:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=self.settings.request_timeout_seconds
                )
                answer = (response.choices[0].message.content or "").strip()
                usage_data = response.usage
                prompt_tokens = getattr(usage_data, 'prompt_tokens', 0)
                completion_tokens = getattr(usage_data, 'completion_tokens', 0)
                total_tokens = getattr(usage_data, 'total_tokens', 0)
                if hasattr(usage_data, 'prompt_tokens_details') and usage_data.prompt_tokens_details:
                    cached_tokens = getattr(usage_data.prompt_tokens_details, 'cached_tokens', 0)

            elif provider_type == ProviderType.OLLAMA:
                response = ollama.chat(
                    model=model,
                    messages=messages,
                    options=Options(temperature=temperature, num_predict=max_tokens)
                )
                answer = (response['message']['content'] or "").strip()
                prompt_tokens = response['prompt_eval_count']
                completion_tokens = response['eval_count']
                total_tokens = prompt_tokens + completion_tokens

            stat = LLMStat(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                cached_tokens=cached_tokens
            )
            return LLMAnswerWithStat(answer or FALLBACK_ANSWER, stat)

        return _do()

    def _call_stream(self,
                     provider_type: ProviderType,
                     client: OpenAI,
                     model: str,
                     messages: list[dict[str, str]],
                     temperature: float = 0.2,
                     max_tokens: int = 350) -> Generator[str, None, LLMAnswerWithStat]:
        """
        Потоковый вызов LLM – возвращает генератор токенов.
        После завершения генерации в StopIteration.value лежит итоговый LLMAnswerWithStat со статистикой.
        """

        @retry(
            wait=wait_exponential(multiplier=1, min=1, max=60),
            stop=stop_after_attempt(self.settings.retry_attempts),
            retry=retry_if_exception_type((RateLimitError, APIStatusError))
        )
        def _create_stream():
            if provider_type == ProviderType.OPENAIREADY:
                return client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=self.settings.request_timeout_seconds,
                    stream=True,
                    stream_options={"include_usage": True}
                )
            elif provider_type == ProviderType.OLLAMA:
                return ollama.chat(
                    model=model,
                    messages=messages,
                    options=Options(temperature=temperature, num_predict=max_tokens),
                    stream=True
                )

        stream_iter = _create_stream()

        def generator() -> Generator[str, None, LLMAnswerWithStat]:
            answer_parts = []
            prompt_tokens = 0
            completion_tokens = 0
            total_tokens = 0
            cached_tokens = 0

            if provider_type == ProviderType.OPENAIREADY:
                for chunk in stream_iter:
                    delta = chunk.choices[0].delta if chunk.choices else None
                    content = delta.content if delta and delta.content else ""
                    if content:
                        answer_parts.append(content)
                        yield content
                    if hasattr(chunk, 'usage') and chunk.usage:
                        usage_data = chunk.usage
                        prompt_tokens = getattr(usage_data, 'prompt_tokens', 0)
                        completion_tokens = getattr(usage_data, 'completion_tokens', 0)
                        total_tokens = getattr(usage_data, 'total_tokens', 0)
                        if hasattr(usage_data, 'prompt_tokens_details') and usage_data.prompt_tokens_details:
                            cached_tokens = getattr(usage_data.prompt_tokens_details, 'cached_tokens', 0)

            elif provider_type == ProviderType.OLLAMA:
                for chunk in stream_iter:
                    content = chunk.get('message', {}).get('content', '')
                    if content:
                        answer_parts.append(content)
                        yield content
                    if chunk.get('done', False):
                        prompt_tokens = chunk.get('prompt_eval_count', 0)
                        completion_tokens = chunk.get('eval_count', 0)
                        total_tokens = prompt_tokens + completion_tokens

            full_answer = ''.join(answer_parts).strip()
            stat = LLMStat(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                cached_tokens=cached_tokens
            )
            return LLMAnswerWithStat(full_answer or FALLBACK_ANSWER, stat)

        return generator()


    def classify(self, messages: list[dict[str,str]]) ->CategoryResult:
        for client, provider, model, used_fallback in self._provider_chain(WorkType.CLASSIFICATION):
            try:
                if used_fallback:
                    logger.info(f"Для классификации переключаюсь на fallback ({provider}, {model})")
                raw = self._call(provider_type=self.get_provider_type(provider),
                                 client=client,
                                 model=model,
                                 messages=messages,
                                 temperature=0,
                                 max_tokens=10)
                return CategoryResult(
                    category=Category(raw.text.strip().lower()),
                    stat=raw.stat,
                    provider=provider,
                    model=model,
                    used_fallback=used_fallback)
            except Exception as e:
                logger.warning(f"Ошибка классификации провайдером {provider}, {model}: {e}")

        category = heuristic_classify(messages[-1]['content'])

        return CategoryResult(category=category,
                              stat=LLMStat(),
                              provider=None,
                              model=None,
                              used_fallback=True)

    def answer(self, messages: list[dict[str, str]]) -> Generator[str, None, ChatResult]:
        """
        Стриминговый метод получения ответа от LLM.
        Перебирает цепочку провайдеров, при успешном подключении начинает потоковую выдачу токенов.
        После исчерпания генератора в StopIteration.value возвращается итоговый ChatResult.
        """
        for client, provider, model, used_fallback in self._provider_chain(WorkType.CHAT):
            try:
                if used_fallback:
                    logger.info(f"Для ответа переключаюсь на fallback ({provider}, {model})")

                # Пытаемся получить стрим-генератор от конкретного провайдера
                stream_gen = self._call_stream(
                    provider_type=self.get_provider_type(provider),
                    client=client,
                    model=model,
                    messages=messages
                )

                # Пробуем извлечь первый чанк, чтобы проверить работоспособность соединения
                first_chunk = next(stream_gen)
                if first_chunk:
                    yield first_chunk

                # Последующие чанки передаём наружу, пока внутренний генератор не завершится
                while True:
                    try:
                        chunk = next(stream_gen)
                        if chunk:
                            yield chunk
                    except StopIteration as e:
                        llm_answer: LLMAnswerWithStat = e.value
                        result = ChatResult(
                            text=llm_answer.text,
                            stat=llm_answer.stat,
                            provider=provider,
                            model=model,
                            used_fallback=used_fallback
                        )
                        # Завершаем внешний генератор, передавая итоговый ChatResult
                        return result

            except Exception as e:
                logger.warning(f"Провайдер {provider}, {model} недоступен: {e}")
                # Переходим к следующему провайдеру в цепочке
                continue

        # Все провайдеры отказали — возвращаем fallback-ответ без единого чанка
        fallback_result = ChatResult(
            text=FALLBACK_ANSWER,
            stat=LLMStat(),
            provider=None,
            model=None,
            used_fallback=True
        )
        return fallback_result