"""Application configuration via environment variables.

Uses pydantic-settings to load from .env files and environment variables.
Every setting has a sensible default so the library works out of the box
with just an API key.
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration for rag-playbook.

    All values can be overridden via environment variables or set in a
    ``.env`` file. Variable names match the field names directly
    (e.g. ``OPENAI_API_KEY``, ``DEFAULT_LLM_MODEL``).
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # -- LLM ------------------------------------------------------------------

    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    anthropic_api_key: str = ""
    default_llm_provider: str = "openai"
    default_llm_model: str = "gpt-4o-mini"
    llm_timeout_seconds: int = Field(default=30, ge=1)
    llm_max_retries: int = Field(default=3, ge=0)

    # -- Embedding -------------------------------------------------------------

    embedding_provider: str = "openai"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimension: int = Field(default=1536, ge=1)

    # -- Vector Store ----------------------------------------------------------

    vector_store_provider: str = "memory"
    pgvector_url: str = ""
    qdrant_url: str = "http://localhost:6333"

    # -- Retrieval defaults ----------------------------------------------------

    default_top_k: int = Field(default=5, ge=1)
    hybrid_search_alpha: float = Field(default=0.5, ge=0.0, le=1.0)

    # -- Chunking defaults -----------------------------------------------------

    default_chunk_size: int = Field(default=512, ge=64)
    default_chunk_overlap: int = Field(default=50, ge=0)

    # -- Evaluation ------------------------------------------------------------

    eval_judge_model: str = "gpt-4o-mini"
