"""Runtime configuration (read when Settings() is created)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Settings:
    api_keys: list[str] = field(default_factory=lambda: [k for k in os.getenv("API_KEYS", "dev-key").split(",") if k])
    llm_provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "auto"))
    llm_model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", ""))
    llm_base_url: str = field(default_factory=lambda: os.getenv("LLM_BASE_URL", ""))
    llm_api_key: str = field(default_factory=lambda: os.getenv("LLM_API_KEY", os.getenv("GROQ_API_KEY", "")))
    llm_timeout_s: float = field(default_factory=lambda: float(os.getenv("LLM_TIMEOUT_S", "30")))


def get_settings() -> Settings:
    return Settings()
