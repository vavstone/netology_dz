import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

@dataclass(slots=True)
class Settings:
    """Настройки приложения из переменных окружения."""
    service_name: str
    # основная модель чата
    main_provider: str
    main_provider_api_key: str | None
    main_provider_base_url: str | None
    main_provider_model: str
    # модель для классификации
    classifier_provider: str
    classifier_provider_api_key: str | None
    classifier_provider_base_url: str | None
    classifier_provider_model: str
    # Fallback-модель
    fallback_provider: str
    fallback_provider_api_key: str | None
    fallback_provider_base_url: str | None
    fallback_provider_model: str
    # Общие настройки
    request_timeout_seconds: float
    retry_attempts: int
    history_limit: int
    log_path: Path
    redis_host: str
    redis_port: int
    redis_ttl: int

    @classmethod
    def from_env(cls) -> "Settings":
        base_dir = Path(__file__).resolve().parent.parent

        # Функция для безопасного получения ключа: сначала своя переменная, затем общая OPENROUTER_API_KEY
        def get_api_key(var_name: str) -> str | None:
            value = os.getenv(var_name)
            if value and value != "...":
                return value
            # Если не задана, пробуем взять из OPENROUTER_API_KEY (общий ключ)
            return os.getenv("OPENROUTER_API_KEY") or None

        return cls(
            service_name=os.getenv("SUPPORT_SERVICE_NAME", "CORP_DOCUMENTATION"),
            main_provider=os.getenv("MAIN_PROVIDER", "openrouter"),
            main_provider_api_key=get_api_key("MAIN_PROVIDER_API_KEY"),
            main_provider_base_url=os.getenv("MAIN_PROVIDER_BASE_URL", "https://openrouter.ai/api/v1"),
            main_provider_model=os.getenv("MAIN_PROVIDER_MODEL", "openrouter/free"),
            classifier_provider=os.getenv("CLASSIFIER_PROVIDER", "openrouter"),
            classifier_provider_api_key=get_api_key("CLASSIFIER_PROVIDER_API_KEY"),
            classifier_provider_base_url=os.getenv("CLASSIFIER_PROVIDER_BASE_URL", "https://openrouter.ai/api/v1"),
            classifier_provider_model=os.getenv("CLASSIFIER_PROVIDER_MODEL", "openrouter/free"),
            fallback_provider=os.getenv("FALLBACK_PROVIDER", "ollama"),
            fallback_provider_api_key=os.getenv("FALLBACK_PROVIDER_API_KEY") or None,
            fallback_provider_base_url=os.getenv("FALLBACK_PROVIDER_BASE_URL", "http://localhost:11434/v1"),
            fallback_provider_model=os.getenv("FALLBACK_PROVIDER_MODEL", "qwen2.5:3b"),
            request_timeout_seconds=float(os.getenv("SUPPORT_TIMEOUT_SECONDS", "30")),
            retry_attempts=int(os.getenv("SUPPORT_RETRY_ATTEMPTS", "3")),
            history_limit=int(os.getenv("SUPPORT_HISTORY_LIMIT", "10")),
            log_path=Path(os.getenv("SUPPORT_LOG_PATH", base_dir / "assistant.log")),
            redis_host=os.getenv("REDIS_HOST", "localhost"),
            redis_port=int(os.getenv("REDIS_PORT", "6379")),
            redis_ttl=int(os.getenv("REDIS_TTL", "3600")),
        )