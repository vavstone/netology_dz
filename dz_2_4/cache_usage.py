from LLMCache import LLMCache
from loguru import logger
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
import time
import os

# Настраиваем хранение лога файл
log_file = Path(__file__).parent / "my_log.txt"
logger.add(log_file, encoding="utf-8")

def build_client() -> OpenAI:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Не найден OPENAI_API_KEY в переменных окружения или .env")
    return OpenAI(api_key=api_key)

def query_model(model:str, client:OpenAI, cache:LLMCache, req:str):
    logger.info(f'Вопрос пользователя: {req}')
    sys_prompt = 'Ты - специалист по Python. Отвечай на вопросы кратко, одним предложением (7-10 слов)'
    messages = messages=[
            {'role': 'system', 'content': sys_prompt},
            {'role': 'user', 'content': req}
        ]
    answer = ''
    answer = cache.get(model, messages)
    if (answer):
        logger.info(f'Ответ из кеша: {answer}')
    else:
        res = client.chat.completions.create(
            model=model,
            temperature=0,
            messages=messages,
            max_tokens=30
        )
        answer = res.choices[0].message.content
        cache.set(model, messages, 0, answer)
        logger.info(f'Ответ от модели: {answer}')


def main() -> None:
    """Демонстрация работы LLMCache."""
    #для демонстрации установим TTL = 5 секунд
    cache = LLMCache(ttl_seconds=5)
    model = "gpt-4.1-nano"
    messages = []
    client = build_client()
    logger.info('Демонстрация работы кеша LLM:\n')
    logger.info('Запрос 1:')
    query_model(model, client, cache, 'Что такое декоратор?')
    time.sleep(3)
    logger.info('Запрос 2:')
    query_model(model, client, cache, 'Что такое декоратор?')
    time.sleep(6)
    logger.info('Запрос 3:')
    query_model(model, client, cache, 'Что такое декоратор?')
    time.sleep(6)
    logger.info('Запрос 4:')
    query_model(model, client, cache, 'Что такое функция?')
    time.sleep(1)
    logger.info('Запрос 5:')
    query_model(model, client, cache, 'Что такое функция?')

    logger.info("GET (hits):  {}", cache.hits)
    logger.info("GET (misses):  {}", cache.misses)
    logger.info("Hit rate:   {:.2f}%", cache.stats)


if __name__ == "__main__":
    main()