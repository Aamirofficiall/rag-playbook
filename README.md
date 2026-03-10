<div align="center">

# rag-playbook

**Stop guessing which RAG pattern to use. Compare them with real numbers.**

[![PyPI](https://img.shields.io/pypi/v/rag-playbook)](https://pypi.org/project/rag-playbook/)
[![Tests](https://img.shields.io/github/actions/workflow/status/Aamirofficiall/rag-playbook/ci.yml)](https://github.com/Aamirofficiall/rag-playbook/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/Aamirofficiall/rag-playbook/blob/main/LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)

[Quick Start](#quick-start) · [Patterns](#patterns) · [Decision Guide](https://github.com/Aamirofficiall/rag-playbook/blob/main/docs/WHEN_TO_USE.md) · [Architecture](https://github.com/Aamirofficiall/rag-playbook/blob/main/docs/ARCHITECTURE.md) · [CLI Reference](https://github.com/Aamirofficiall/rag-playbook/blob/main/docs/CLI.md)

</div>

---

Every RAG tutorial teaches you how to build patterns.
None of them tell you **which one to actually use**.

**rag-playbook** runs the same query against 8 production-tested RAG patterns
and shows you which one wins — with real numbers for quality, latency, and cost.

## Quick Start

```bash
pip install rag-playbook[openai]
export OPENAI_API_KEY=sk-...

# Compare all patterns on your documents
rag-playbook compare --data ./my_docs/ --query "What is the refund policy?"
```

<p align="center">
  <img src="https://raw.githubusercontent.com/Aamirofficiall/rag-playbook/main/assets/compare_screenshot.png" alt="rag-playbook compare output" width="700">
</p>

## Patterns

| # | Pattern | Best For | Latency | Cost |
|---|---------|----------|---------|------|
| 01 | [Naive](https://github.com/Aamirofficiall/rag-playbook/blob/main/docs/patterns/01-naive.md) | Simple factual queries | ~1s | $ |
| 02 | [Hybrid Search](https://github.com/Aamirofficiall/rag-playbook/blob/main/docs/patterns/02-hybrid-search.md) | Queries with codes, IDs, exact terms | ~1.1s | $ |
| 03 | [Re-ranking](https://github.com/Aamirofficiall/rag-playbook/blob/main/docs/patterns/03-reranking.md) | When top-K retrieval isn't precise enough | ~1.4s | $$ |
| 04 | [Parent-Child](https://github.com/Aamirofficiall/rag-playbook/blob/main/docs/patterns/04-parent-child.md) | Long documents with clear sections | ~1s | $ |
| 05 | [Query Decomposition](https://github.com/Aamirofficiall/rag-playbook/blob/main/docs/patterns/05-query-decomposition.md) | Complex multi-part questions | ~2.1s | $$$ |
| 06 | [HyDE](https://github.com/Aamirofficiall/rag-playbook/blob/main/docs/patterns/06-hyde.md) | Short or ambiguous queries | ~1.5s | $$ |
| 07 | [Self-Correcting](https://github.com/Aamirofficiall/rag-playbook/blob/main/docs/patterns/07-self-correcting.md) | When hallucination risk is high | ~2.8s | $$$ |
| 08 | [Agentic](https://github.com/Aamirofficiall/rag-playbook/blob/main/docs/patterns/08-agentic.md) | When query intent is unclear | ~3.2s | $$$$ |

[Which pattern should I use? (decision guide)](https://github.com/Aamirofficiall/rag-playbook/blob/main/docs/WHEN_TO_USE.md)

## How is this different from [X]?

| | NirDiamant/RAG_Techniques | FlashRAG | Ragas | **rag-playbook** |
|---|---|---|---|---|
| Format | Jupyter notebooks | Academic library | Eval metrics | **Library + CLI** |
| `pip install` | No | Complex | Yes | **Yes (simple)** |
| Benchmarks | None | Academic | N/A | **Practical comparison** |
| "Which to use?" | No | No | No | **YES** |
| License | Non-commercial | MIT | Apache-2.0 | **MIT** |

## Use as a Library

```python
import asyncio
from rag_playbook import create_pattern, Settings
from rag_playbook.core.embedder import create_embedder
from rag_playbook.core.llm import create_llm
from rag_playbook.core.vector_store import create_vector_store

async def main():
    settings = Settings()  # Reads from .env / environment
    llm = create_llm(settings)
    embedder = create_embedder(settings)
    store = create_vector_store("memory")

    pattern = create_pattern("reranking", llm=llm, embedder=embedder, store=store)

    # Index your documents first (or use the CLI: rag-playbook ingest)
    result = await pattern.query("What is the refund policy?")

    print(result.answer)
    print(f"Cost: ${result.metadata.cost_usd:.4f}")
    print(f"Latency: {result.metadata.latency_ms:.0f}ms")

asyncio.run(main())
```

See [examples/](https://github.com/Aamirofficiall/rag-playbook/tree/main/examples) for more usage patterns.

## Installation

```bash
# Minimal (in-memory store, OpenAI)
pip install rag-playbook[openai]

# Everything (includes planned vector store backends)
pip install rag-playbook[all]
```

### From Source

```bash
git clone https://github.com/Aamirofficiall/rag-playbook.git
cd rag-playbook
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -e ".[dev,chromadb,openai]"
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `rag-playbook compare` | Compare patterns side-by-side on your documents |
| `rag-playbook run` | Run a single pattern |
| `rag-playbook recommend` | Get an LLM-powered pattern recommendation |
| `rag-playbook ingest` | Load, chunk, embed, and index documents |
| `rag-playbook bench` | Run full benchmark suite |
| `rag-playbook patterns` | List all available patterns |

### Run a single pattern

```bash
rag-playbook run reranking --data ./docs/ --query "What are the laptop specs?"
```

<p align="center">
  <img src="https://raw.githubusercontent.com/Aamirofficiall/rag-playbook/main/assets/run_screenshot.png" alt="rag-playbook run output" width="700">
</p>

### List available patterns

```bash
rag-playbook patterns
```

<p align="center">
  <img src="https://raw.githubusercontent.com/Aamirofficiall/rag-playbook/main/assets/patterns_screenshot.png" alt="rag-playbook patterns output" width="700">
</p>

### Get a pattern recommendation

```bash
rag-playbook recommend --query "What is the refund policy?"
```

<p align="center">
  <img src="https://raw.githubusercontent.com/Aamirofficiall/rag-playbook/main/assets/recommend_screenshot.png" alt="rag-playbook recommend output" width="700">
</p>

### Ingest documents

```bash
rag-playbook ingest --data ./sample_docs/
```

<p align="center">
  <img src="https://raw.githubusercontent.com/Aamirofficiall/rag-playbook/main/assets/ingest_screenshot.png" alt="rag-playbook ingest output" width="700">
</p>

See [CLI Reference](https://github.com/Aamirofficiall/rag-playbook/blob/main/docs/CLI.md) for full usage.

## Configuration

Environment variables map directly to settings — no prefix needed:

```bash
# .env — Core settings
OPENAI_API_KEY=sk-...                  # Required for OpenAI provider
DEFAULT_LLM_PROVIDER=openai            # openai | anthropic
DEFAULT_LLM_MODEL=gpt-4o-mini         # Any model your provider supports
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIMENSION=1536
DEFAULT_TOP_K=5
DEFAULT_CHUNK_SIZE=512
DEFAULT_CHUNK_OVERLAP=50
```

### Using OpenRouter (or any OpenAI-compatible API)

Works with any OpenAI-compatible endpoint — OpenRouter, Azure OpenAI, Ollama, vLLM, etc:

```bash
OPENAI_API_KEY=sk-or-v1-...                     # Your OpenRouter key
OPENAI_BASE_URL=https://openrouter.ai/api/v1    # Custom base URL
DEFAULT_LLM_MODEL=openai/gpt-4o-mini            # OpenRouter model format
```

### Vector store backends

The default in-memory store works out of the box but re-embeds every session.
ChromaDB, pgvector, and Qdrant are planned as persistent backends — the
optional dependencies are installable but the store implementations are not yet
wired up. Contributions welcome.

Or pass a `Settings` object directly in code. See [.env.example](https://github.com/Aamirofficiall/rag-playbook/blob/main/.env.example) for all options.

## Architecture

```
Document → Chunk → EmbeddedChunk → RetrievedChunk → RAGResult
              │         │                │               │
          Chunker    Embedder       VectorStore        LLM
```

LLM providers, embedders, and vector stores are all swappable. Each RAG pattern extends a base class and overrides only the steps it needs. Embeddings are cached automatically so running compare across all 8 patterns doesn't re-embed your docs 8 times.

See [Architecture Guide](https://github.com/Aamirofficiall/rag-playbook/blob/main/docs/ARCHITECTURE.md) for details.

## Development

```bash
make install    # Install with dev dependencies
make test       # Run unit tests
make lint       # Lint with ruff
make format     # Auto-format with ruff
make type-check # Type check with mypy
make check      # Run all checks
```

See [CONTRIBUTING.md](https://github.com/Aamirofficiall/rag-playbook/blob/main/CONTRIBUTING.md) for the full guide.

## Author

Built by [Aamir Shahzad](https://aamirshahzad.uk) — backend engineer building data systems and AI infrastructure.

## License

[MIT](https://github.com/Aamirofficiall/rag-playbook/blob/main/LICENSE)
