from collections.abc import Iterator
from loguru import logger
from openai import OpenAI, RateLimitError, APIStatusError
from tenacity import retry, stop_after_attempt, retry_if_exception_type

from config import Settings
from models import Category, LLMResult