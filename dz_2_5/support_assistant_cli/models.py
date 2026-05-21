from dataclasses import dataclass
from enum import StrEnum

class Category(StrEnum):
    FAQ = "faq"
    TECHNICAL = "technical"
    COMPLAINT = "complaint"
    ESCALATION = "escalation"


@dataclass(slots=True)
class SessionStats:
    total_queries: int = 0
    escalations: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    cloud_total_tokens: int = 0
    cloud_llm_calls: int = 0
    local_total_tokens: int = 0
    local_llm_calls: int = 0

@dataclass(slots=True)
class LLMResult:
    text: str
    tokens: int
    provider: str
    model: str
    used_fallback: bool
    is_local: bool

@dataclass(slots=True)
class AssistantResponse:
    text: str
    category: Category
    from_cache: bool
    latency_seconds: float
    provider: str
    model: str
    used_fallback: bool