import hashlib
import redis
from loguru import logger

def _normalize_url(question: str) -> str:
    """Нормализует вопрос: нижний регистр, удаляет лишние пробелы."""
    return " ".join(question.strip().lower().split())


class RedisCache:
    """Кеш ответов на базе Redis с автоматическим отключением при недоступности."""

    def __init__(self, host: str = 'localhost', port: int = 6379, ttl: int = 3600):
        self.ttl = ttl
        self.available = True
        try:
            self.client = redis.Redis(host=host, port=port, decode_responses=True)
            self.client.ping()
        except Exception as e:
            logger.warning(f"Redis недоступен ({e}), кеш отключён")
            self.available = False
            self.client = None

    @staticmethod
    def _make_key(question: str) -> str:
        normalized = _normalize_url(question)
        return f"support:{hashlib.sha256(normalized.encode()).hexdigest()}"

    def get(self, question: str) -> str | None:
        """Возвращает ответ из кеша или None."""
        if not self.available:
            return None
        try:
            return self.client.get(self._make_key(question))
        except Exception as e:
            logger.warning(f"Ошибка чтения из Redis: {e}")
            return None

    def set(self, question: str, answer: str) -> None:
        """Сохраняет ответ в кеш."""
        if not self.available:
            return
        try:
            key = self._make_key(question)
            self.client.setex(key, self.ttl, answer)
        except Exception as e:
            logger.warning(f"Ошибка записи в Redis: {e}")

    def clear(self) -> int:
        """Очищает все ключи приложения в Redis."""
        if not self.available:
            return 0
        try:
            deleted = 0
            for key in self.client.scan_iter('support:*'):
                self.client.delete(key)
                deleted += 1
            return deleted
        except Exception as e:
            logger.warning(f"Ошибка очистки Redis: {e}")
            return 0

    def stats(self) -> dict[str, str | int]:
        """Возвращает статистику кеша (hits/misses/keys)."""
        if not self.available:
            return {"hits": 0, "misses": 0, "hit_rate": "", "keys": 0, "error": "Redis unavailable"}
        try:
            info = self.client.info("stats")
            hits = info.get("keyspace_hits", 0)
            misses = info.get("keyspace_misses", 0)
            total = hits + misses
            return {
                "hits": hits,
                "misses": misses,
                "hit_rate": f"{hits / total * 100:.1f}%" if total else "",
                "keys": self.client.dbsize()
            }
        except Exception as e:
            logger.warning(f"Ошибка получения статистики Redis: {e}")
            return {"hits": 0, "misses": 0, "hit_rate": "", "keys": 0, "error": str(e)}