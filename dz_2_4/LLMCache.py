import hashlib
import json
import time
from loguru import logger


class LLMCache:
    """In-memory кеш с TTL."""

    def __init__(self, ttl_seconds: int = 3600) -> None:
        self._cache: dict[str, tuple[str, float]] = {}
        self.ttl = ttl_seconds
        self.hits = 0
        self.misses = 0

    def _make_key(
        self, model: str, messages: list[dict], temperature: float = 0
    ) -> str:
        """Ключ = хеш(модель + параметры + промпт)."""
        data = json.dumps(
            {"model": model, "messages": messages, "temperature": temperature},
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(data.encode()).hexdigest()

    def get(self, model: str, messages: list[dict], temperature: float = 0) -> str | None:
        key = self._make_key(model, messages, temperature)
        if key in self._cache:
            value, created_at = self._cache[key]
            if time.time() - created_at < self.ttl:
                self.hits += 1
                return value
            del self._cache[key]  # TTL истёк
        self.misses += 1
        return None

    def set(self, model: str, messages: list[dict], temperature: float, response: str) -> None:
        key = self._make_key(model, messages, temperature)
        self._cache[key] = (response, time.time())

    @property
    def stats(self) -> float:
        total = self.hits + self.misses
        return self.hits / total * 100 if total > 0 else 0.0