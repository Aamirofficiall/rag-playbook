"""Tests for the LLM client layer.

Uses mock HTTP responses to verify request shaping, retry behaviour,
cost calculation, and JSON generation.
"""

import json

import pytest

from rag_playbook.core.config import Settings
from rag_playbook.core.exceptions import ConfigurationError, GenerationError
from rag_playbook.core.llm import (
    AnthropicLLM,
    LLMResponse,
    Message,
    OpenAILLM,
    create_llm,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _openai_response(content: str = "Hello", input_t: int = 10, output_t: int = 5) -> dict:
    return {
        "choices": [{"message": {"content": content}}],
        "usage": {"prompt_tokens": input_t, "completion_tokens": output_t},
    }


def _anthropic_response(content: str = "Hello", input_t: int = 10, output_t: int = 5) -> dict:
    return {
        "content": [{"text": content}],
        "usage": {"input_tokens": input_t, "output_tokens": output_t},
    }


def _settings(**overrides) -> Settings:  # type: ignore[no-untyped-def]
    defaults = {
        "openai_api_key": "sk-test",
        "anthropic_api_key": "sk-ant-test",
        "llm_max_retries": 3,
        "llm_timeout_seconds": 5,
    }
    return Settings(**{**defaults, **overrides})


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestOpenAILLM:
    async def test_generate_returns_llm_response(self, httpx_mock) -> None:  # type: ignore[no-untyped-def]
        httpx_mock.add_response(json=_openai_response("World", 20, 10))
        llm = OpenAILLM(_settings())
        messages = [Message(role="user", content="Hi")]
        result = await llm.generate(messages)

        assert isinstance(result, LLMResponse)
        assert result.content == "World"
        assert result.input_tokens == 20
        assert result.output_tokens == 10
        assert result.latency_ms > 0

    async def test_generate_calculates_cost(self, httpx_mock) -> None:  # type: ignore[no-untyped-def]
        httpx_mock.add_response(json=_openai_response("ok", 1000, 500))
        llm = OpenAILLM(_settings())
        result = await llm.generate([Message(role="user", content="Hi")])
        assert result.cost_usd > 0

    async def test_raises_on_missing_api_key(self) -> None:
        with pytest.raises(ConfigurationError, match="OPENAI_API_KEY"):
            OpenAILLM(_settings(openai_api_key=""))

    async def test_retry_on_server_error(self, httpx_mock) -> None:  # type: ignore[no-untyped-def]
        httpx_mock.add_response(status_code=500)
        httpx_mock.add_response(json=_openai_response("recovered"))
        llm = OpenAILLM(_settings(llm_max_retries=2))
        result = await llm.generate([Message(role="user", content="Hi")])
        assert result.content == "recovered"

    async def test_exhausted_retries_raises_generation_error(self, httpx_mock) -> None:  # type: ignore[no-untyped-def]
        httpx_mock.add_response(status_code=429)
        httpx_mock.add_response(status_code=429)
        httpx_mock.add_response(status_code=429)
        llm = OpenAILLM(_settings(llm_max_retries=3))
        with pytest.raises(GenerationError, match="3 attempts"):
            await llm.generate([Message(role="user", content="Hi")])


@pytest.mark.unit
class TestOpenAIGenerateJSON:
    async def test_valid_json_response(self, httpx_mock) -> None:  # type: ignore[no-untyped-def]
        payload = {"pattern": "reranking", "confidence": 0.9}
        httpx_mock.add_response(json=_openai_response(json.dumps(payload)))
        llm = OpenAILLM(_settings())
        result = await llm.generate_json([Message(role="user", content="Hi")])
        assert result["pattern"] == "reranking"

    async def test_invalid_json_retries_with_nudge(self, httpx_mock) -> None:  # type: ignore[no-untyped-def]
        httpx_mock.add_response(json=_openai_response("not json"))
        httpx_mock.add_response(json=_openai_response('{"pattern": "naive"}'))
        llm = OpenAILLM(_settings())
        result = await llm.generate_json([Message(role="user", content="Hi")])
        assert result["pattern"] == "naive"

    async def test_double_invalid_json_raises(self, httpx_mock) -> None:  # type: ignore[no-untyped-def]
        httpx_mock.add_response(json=_openai_response("bad"))
        httpx_mock.add_response(json=_openai_response("still bad"))
        llm = OpenAILLM(_settings())
        with pytest.raises(GenerationError, match="valid JSON"):
            await llm.generate_json([Message(role="user", content="Hi")])


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAnthropicLLM:
    async def test_generate_returns_llm_response(self, httpx_mock) -> None:  # type: ignore[no-untyped-def]
        httpx_mock.add_response(json=_anthropic_response("Hi there", 15, 8))
        llm = AnthropicLLM(_settings(default_llm_provider="anthropic"))
        result = await llm.generate([Message(role="user", content="Hello")])
        assert result.content == "Hi there"
        assert result.input_tokens == 15

    async def test_system_message_extracted(self, httpx_mock) -> None:  # type: ignore[no-untyped-def]
        httpx_mock.add_response(json=_anthropic_response("ok"))
        llm = AnthropicLLM(_settings(default_llm_provider="anthropic"))
        messages = [
            Message(role="system", content="You are helpful"),
            Message(role="user", content="Hi"),
        ]
        await llm.generate(messages)
        request = httpx_mock.get_request()
        body = json.loads(request.content)
        assert body["system"] == "You are helpful"
        assert len(body["messages"]) == 1

    async def test_raises_on_missing_api_key(self) -> None:
        with pytest.raises(ConfigurationError, match="ANTHROPIC_API_KEY"):
            AnthropicLLM(_settings(anthropic_api_key=""))


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCreateLLM:
    def test_creates_openai(self) -> None:
        llm = create_llm(_settings(default_llm_provider="openai"))
        assert isinstance(llm, OpenAILLM)

    def test_creates_anthropic(self) -> None:
        llm = create_llm(_settings(default_llm_provider="anthropic"))
        assert isinstance(llm, AnthropicLLM)

    def test_unknown_provider_raises(self) -> None:
        with pytest.raises(ConfigurationError, match="Unknown LLM provider"):
            create_llm(_settings(default_llm_provider="gemini"))


# ---------------------------------------------------------------------------
# LLMResponse model
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestLLMResponseModel:
    def test_frozen(self) -> None:
        resp = LLMResponse(
            content="hi",
            input_tokens=10,
            output_tokens=5,
            model="gpt-4o-mini",
            latency_ms=100.0,
            cost_usd=0.001,
        )
        with pytest.raises(Exception):  # noqa: B017
            resp.content = "changed"  # type: ignore[misc]

    def test_rejects_negative_tokens(self) -> None:
        with pytest.raises(ValueError):
            LLMResponse(
                content="hi",
                input_tokens=-1,
                output_tokens=5,
                model="gpt-4o-mini",
                latency_ms=100.0,
                cost_usd=0.001,
            )
