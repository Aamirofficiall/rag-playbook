"""Per-query cost tracking for LLM and embedding calls.

Prices are hardcoded per model — simpler and more reliable than runtime
API lookups. Updated when model pricing changes.
"""

from __future__ import annotations

# Price per 1M tokens (input, output) in USD
_LLM_PRICING: dict[str, tuple[float, float]] = {
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4-turbo": (10.00, 30.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-haiku-4-5-20251001": (0.80, 4.00),
    "claude-opus-4-6": (15.00, 75.00),
}

# Price per 1M tokens for embedding models
_EMBEDDING_PRICING: dict[str, float] = {
    "text-embedding-3-small": 0.02,
    "text-embedding-3-large": 0.13,
    "text-embedding-ada-002": 0.10,
}

_PER_MILLION = 1_000_000


def _normalize_model(model: str) -> str:
    """Strip provider prefix (e.g. 'openai/gpt-4o-mini' -> 'gpt-4o-mini')."""
    return model.rsplit("/", 1)[-1] if "/" in model else model


def llm_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Calculate USD cost for an LLM call."""
    key = _normalize_model(model)
    if key not in _LLM_PRICING:
        return 0.0
    input_price, output_price = _LLM_PRICING[key]
    return (input_tokens * input_price + output_tokens * output_price) / _PER_MILLION


def embedding_cost(model: str, token_count: int) -> float:
    """Calculate USD cost for an embedding call."""
    key = _normalize_model(model)
    if key not in _EMBEDDING_PRICING:
        return 0.0
    return token_count * _EMBEDDING_PRICING[key] / _PER_MILLION


def supported_llm_models() -> list[str]:
    """Return list of models with known pricing."""
    return list(_LLM_PRICING.keys())


def supported_embedding_models() -> list[str]:
    """Return list of embedding models with known pricing."""
    return list(_EMBEDDING_PRICING.keys())
