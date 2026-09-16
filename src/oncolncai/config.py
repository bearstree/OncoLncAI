"""Application configuration loaded deterministically from environment variables."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, SecretStr


class Settings(BaseModel):
    """Validated runtime settings with development-safe defaults."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    environment: str = Field(default="development", min_length=1)
    log_level: str = "INFO"
    cache_dir: Path = Path(".cache/oncolncai")
    llm_provider: Literal["mock", "openai", "ollama"] = "mock"
    llm_model: str = Field(default="qwen2.5-coder:14b", min_length=1)
    llm_base_url: str | None = None
    llm_api_key: SecretStr | None = None
    ncbi_api_key: SecretStr | None = None
    llm_timeout_seconds: float = Field(default=60.0, gt=0)


def load_settings(environ: Mapping[str, str] | None = None) -> Settings:
    """Load settings from an environment mapping using the ``ONCOLNCAI_`` prefix."""

    source = os.environ if environ is None else environ
    values: dict[str, str] = {}
    for field_name in Settings.model_fields:
        key = f"ONCOLNCAI_{field_name.upper()}"
        if key in source:
            values[field_name] = source[key]
    return Settings.model_validate(values)


def configure_logging(level: str = "INFO") -> None:
    """Configure concise application logging without changing existing handlers."""

    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Unknown log level: {level}")
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
