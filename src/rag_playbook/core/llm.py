"""Multi-provider async LLM client using the Strategy pattern.

Patterns call ``self.llm.generate()`` without knowing which provider
backs it. Every response includes token counts, cost, and latency —
you can't optimize what you can't measure.
"""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from typing import Any

import httpx
import structlog
from pydantic import BaseModel, ConfigDict, Field

from rag_playbook.core.config import Settings
from rag_playbook.core.cost import llm_cost
from rag_playbook.core.exceptions import ConfigurationError, GenerationError

logger = structlog.get_logger(__name__)

_RE_CODE_FENCE = None  # lazy-compiled


def _strip_code_fences(text: str) -> str:
    """Strip markdown code fences (```json ... ```) from LLM output."""
    global _RE_CODE_FENCE
    if _RE_CODE_FENCE is None:
        import re

        _RE_CODE_FENCE = re.compile(r"```(?:json)?\s*\n?(.*?)\n?\s*```", re.DOTALL)
    m = _RE_CODE_FENCE.search(text)
    return m.group(1).strip() if m else text.strip()


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------


class Message(BaseModel):
    """A single message in a conversation."""

    model_config = ConfigDict(frozen=True)

    role: str
    content: str


class LLMResponse(BaseModel):
    """Structured response from any LLM provider."""

    model_config = ConfigDict(frozen=True)

    content: str
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    model: str
    latency_ms: float = Field(ge=0.0)
    cost_usd: float = Field(ge=0.0)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class BaseLLM(ABC):
    """Abstract LLM client.

    Subclasses implement ``_call`` for the raw API interaction. The base
    class handles retries, timing, cost calculation, and logging.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._max_retries = settings.llm_max_retries
        self._timeout = settings.llm_timeout_seconds

    @property
    @abstractmethod
    def model_name(self) -> str: ...

    @abstractmethod
    async def _call(
        self,
        messages: list[Message],
        **kwargs: Any,
    ) -> tuple[str, int, int]:
        """Execute the provider-specific API call.

        Returns:
            (content, input_tokens, output_tokens)
        """

    async def generate(
        self,
        messages: list[Message],
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate a response with retry, timing, and cost tracking."""
        last_error: Exception | None = None

        for attempt in range(1, self._max_retries + 1):
            try:
                start = time.perf_counter()
                content, input_tokens, output_tokens = await self._call(messages, **kwargs)
                elapsed_ms = (time.perf_counter() - start) * 1000
                cost = llm_cost(self.model_name, input_tokens, output_tokens)

                logger.info(
                    "llm.generate",
                    model=self.model_name,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    latency_ms=round(elapsed_ms, 1),
                    cost_usd=round(cost, 6),
                    attempt=attempt,
                )

                return LLMResponse(
                    content=content,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    model=self.model_name,
                    latency_ms=elapsed_ms,
                    cost_usd=cost,
                )
            except (httpx.HTTPStatusError, httpx.TimeoutException) as exc:
                last_error = exc
                logger.warning(
                    "llm.retry",
                    model=self.model_name,
                    attempt=attempt,
                    error=str(exc),
                )
                if attempt == self._max_retries:
                    break

        raise GenerationError(
            f"LLM generation failed after {self._max_retries} attempts: {last_error}"
        )

    async def generate_json(
        self,
        messages: list[Message],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Generate and parse a JSON response.

        Strips markdown code fences if present. On parse failure, retries
        once with a JSON-nudge appended.
        """
        response = await self.generate(messages, **kwargs)
        try:
            result: dict[str, Any] = json.loads(_strip_code_fences(response.content))
            return result
        except json.JSONDecodeError:
            nudge = Message(
                role="user",
                content="Your response was not valid JSON. "
                "Please respond with raw JSON only, no markdown.",
            )
            retry_response = await self.generate([*messages, nudge], **kwargs)
            try:
                retry_result: dict[str, Any] = json.loads(
                    _strip_code_fences(retry_response.content)
                )
                return retry_result
            except json.JSONDecodeError as exc:
                raise GenerationError(
                    f"LLM failed to produce valid JSON after retry: {retry_response.content[:200]}"
                ) from exc


# ---------------------------------------------------------------------------
# OpenAI implementation
# ---------------------------------------------------------------------------


class OpenAILLM(BaseLLM):
    """OpenAI API client."""

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        if not settings.openai_api_key:
            raise ConfigurationError("OPENAI_API_KEY is required for OpenAI provider")
        self._client = httpx.AsyncClient(
            base_url=settings.openai_base_url,
            headers={"Authorization": f"Bearer {settings.openai_api_key}"},
            timeout=httpx.Timeout(self._timeout),
        )
        self._model = settings.default_llm_model

    @property
    def model_name(self) -> str:
        return self._model

    async def _call(
        self,
        messages: list[Message],
        **kwargs: Any,
    ) -> tuple[str, int, int]:
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [m.model_dump() for m in messages],
            **kwargs,
        }
        resp = await self._client.post("/chat/completions", json=payload)
        resp.raise_for_status()
        data = resp.json()
        return (
            data["choices"][0]["message"]["content"],
            data["usage"]["prompt_tokens"],
            data["usage"]["completion_tokens"],
        )


# ---------------------------------------------------------------------------
# Anthropic implementation
# ---------------------------------------------------------------------------


class AnthropicLLM(BaseLLM):
    """Anthropic API client."""

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        if not settings.anthropic_api_key:
            raise ConfigurationError("ANTHROPIC_API_KEY is required for Anthropic provider")
        self._client = httpx.AsyncClient(
            base_url="https://api.anthropic.com/v1",
            headers={
                "x-api-key": settings.anthropic_api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            timeout=httpx.Timeout(self._timeout),
        )
        self._model = settings.default_llm_model

    @property
    def model_name(self) -> str:
        return self._model

    async def _call(
        self,
        messages: list[Message],
        **kwargs: Any,
    ) -> tuple[str, int, int]:
        system_msg = ""
        api_messages = []
        for m in messages:
            if m.role == "system":
                system_msg = m.content
            else:
                api_messages.append(m.model_dump())

        payload: dict[str, Any] = {
            "model": self._model,
            "max_tokens": kwargs.pop("max_tokens", 1024),
            "messages": api_messages,
            **kwargs,
        }
        if system_msg:
            payload["system"] = system_msg

        resp = await self._client.post("/messages", json=payload)
        resp.raise_for_status()
        data = resp.json()
        return (
            data["content"][0]["text"],
            data["usage"]["input_tokens"],
            data["usage"]["output_tokens"],
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_llm(settings: Settings) -> BaseLLM:
    """Create an LLM client based on configuration."""
    match settings.default_llm_provider:
        case "openai":
            return OpenAILLM(settings)
        case "anthropic":
            return AnthropicLLM(settings)
        case provider:
            raise ConfigurationError(f"Unknown LLM provider: {provider}")
