import hashlib
import redis

def _normalize_url(question: str) -> str:
    return " ".join(question.strip().lower().split())

class RedisCache:

    def __init__(self, host:str='localhost', port:int=6379, ttl:int=3600):
        self.client = redis.Redis(host=host, port=port, decode_responses=True)
        self.ttl = ttl

    @staticmethod
    def _make_key(question:str)->str:
        normalized_question = _normalize_url(question)
        return f"support:{hashlib.sha256(normalized_question.encode()).hexdigest()}"

    def get(self,question:str)->str|None:
        return self.client.get(self._make_key(question))

    def set(self,question:str,answer:str):
        key = self._make_key(question)
        self.client.setex(key,self.ttl,answer)

    def clear(self) -> int:
        deleted = 0
        for key in self.client.scan_iter('support:*'):
            self.client.delete(key)
            deleted += 1
        return deleted

    def reset_stats(self) -> None:
        self.client.config_resetstat()

    def stats(self) -> dict[str, str|int]:
        info = self.client.info("stats")
        hits = info.get("keyspace_hits",0)
        misses = info.get("keyspace_misses", 0)
        total = hits + misses
        return {
            "hits": hits,
            "misses": misses,
            "hit_rate": f"{hits/total*100:.1f}%" if total else "",
            "keys": self.client.dbsize()
        }

