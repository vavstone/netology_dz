from dataclasses import dataclass
from enum import StrEnum

class Category(StrEnum):
    FAQ = "faq"
    TECHNICAL = "technical"
    COMPLAINT = "complaint"
    ESCALATION = "escalation"

class WorkType(StrEnum):
    CHAT = "chat"
    CLASSIFICATION = "classification"

class ProviderType(StrEnum):
    OPENAIREADY = "openaiready"
    OLLAMA = "ollama"


@dataclass(slots=True)
class SessionStats:
    total_queries: int = 0
    escalations: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    cloud_total_tokens: int = 0
    cloud_cache_tokens: int = 0
    cloud_llm_calls: int = 0
    local_total_tokens: int = 0
    local_llm_calls: int = 0

@dataclass(slots=True)
class LLMStat:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0

@dataclass(slots=True)
class LLMAnswerWithStat:
    text: str
    stat: LLMStat

@dataclass(slots=True)
class ChatResult:
    text: str
    stat: LLMStat
    provider: str|None
    model: str|None
    used_fallback: bool

@dataclass(slots=True)
class CategoryResult:
    category: Category
    stat: LLMStat
    provider: str|None
    model: str|None
    used_fallback: bool

@dataclass(slots=True)
class AssistantResponse:
    text: str
    category: Category
    from_cache: bool
    latency_seconds: float
    provider: str|None
    model: str|None
    used_fallback: bool