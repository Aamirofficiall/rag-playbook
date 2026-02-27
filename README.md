<div align="center">

# rag-playbook

**Stop guessing which RAG pattern to use. Compare them with real numbers.**

[![PyPI](https://img.shields.io/pypi/v/rag-playbook)](https://pypi.org/project/rag-playbook/)
[![Tests](https://img.shields.io/github/actions/workflow/status/Aamirofficiall/rag-playbook/ci.yml)](https://github.com/Aamirofficiall/rag-playbook/actions)
[![Coverage](https://img.shields.io/codecov/c/github/Aamirofficiall/rag-playbook)](https://codecov.io/gh/Aamirofficiall/rag-playbook)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)

[Quick Start](#quick-start) · [Patterns](#patterns) · [Decision Guide](docs/WHEN_TO_USE.md) · [Architecture](docs/ARCHITECTURE.md) · [CLI Reference](docs/CLI.md)

</div>

---

Every RAG tutorial teaches you how to build patterns.
None of them tell you **which one to actually use**.

**rag-playbook** runs the same query against 8 production-tested RAG patterns
and shows you which one wins — with real numbers for quality, latency, and cost.

## Quick Start

```bash
pip install rag-playbook[openai,chromadb]
export OPENAI_API_KEY=sk-...

# Compare all patterns on your documents
rag-playbook compare --data ./my_docs/ --query "What is the refund policy?"
```

Output:

```
┌─────────────────────┬───────────┬────────────┬───────────┬──────────┐
│ Pattern             │ Relevance │ Faithfulness│ Latency   │ Cost     │
├─────────────────────┼───────────┼────────────┼───────────┼──────────┤
│ naive               │ ●●●○○     │ ●●●●○      │    980ms  │ $0.0030  │
│ hybrid_search       │ ●●●●○     │ ●●●●○      │  1,100ms  │ $0.0031  │
│ reranking           │ ●●●●●     │ ●●●●○      │  1,400ms  │ $0.0052  │
│ parent_child        │ ●●●●○     │ ●●●●●      │  1,020ms  │ $0.0033  │
│ query_decomposition │ ●●●●○     │ ●●●●○      │  2,100ms  │ $0.0091  │
│ hyde                │ ●●●●○     │ ●●●●○      │  1,500ms  │ $0.0048  │
│ self_correcting     │ ●●●●●     │ ●●●●●      │  2,800ms  │ $0.0095  │
│ agentic             │ ●●●●●     │ ●●●●●      │  3,200ms  │ $0.0120  │
└─────────────────────┴───────────┴────────────┴───────────┴──────────┘

Recommendation: reranking
  Best quality-to-cost ratio for straightforward factual queries.
```

## Patterns

| # | Pattern | Best For | Latency | Cost |
|---|---------|----------|---------|------|
| 01 | [Naive](docs/patterns/01-naive.md) | Simple factual queries | ~1s | $ |
| 02 | [Hybrid Search](docs/patterns/02-hybrid-search.md) | Queries with codes, IDs, exact terms | ~1.1s | $ |
| 03 | [Re-ranking](docs/patterns/03-reranking.md) | When top-K retrieval isn't precise enough | ~1.4s | $$ |
| 04 | [Parent-Child](docs/patterns/04-parent-child.md) | Long documents with clear sections | ~1s | $ |
| 05 | [Query Decomposition](docs/patterns/05-query-decomposition.md) | Complex multi-part questions | ~2.1s | $$$ |
| 06 | [HyDE](docs/patterns/06-hyde.md) | Short or ambiguous queries | ~1.5s | $$ |
| 07 | [Self-Correcting](docs/patterns/07-self-correcting.md) | When hallucination risk is high | ~2.8s | $$$ |
| 08 | [Agentic](docs/patterns/08-agentic.md) | When query intent is unclear | ~3.2s | $$$$ |

[Which pattern should I use? (decision guide)](docs/WHEN_TO_USE.md)

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

async def main():
    settings = Settings()  # Reads from .env / environment
    pattern = create_pattern("reranking", settings=settings)

    # Index your documents first (or use the CLI: rag-playbook ingest)
    result = await pattern.query("What is the refund policy?")

    print(result.answer)
    print(f"Cost: ${result.metadata.cost_usd:.4f}")
    print(f"Latency: {result.metadata.latency_ms:.0f}ms")

asyncio.run(main())
```

See [examples/](examples/) for more usage patterns.

## Installation

```bash
# Minimal (in-memory store, OpenAI)
pip install rag-playbook[openai]

# With ChromaDB
pip install rag-playbook[openai,chromadb]

# With pgvector
pip install rag-playbook[openai,pgvector]

# With re-ranking support (Pattern 03)
pip install rag-playbook[openai,chromadb,reranking]

# Everything
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
| `rag-playbook patterns` | List all available patterns |

See [CLI Reference](docs/CLI.md) for full usage.

## Configuration

All settings can be configured via environment variables with the `RAG_` prefix:

```bash
# .env
RAG_LLM_PROVIDER=openai
RAG_LLM_MODEL=gpt-4o-mini
RAG_EMBEDDING_MODEL=text-embedding-3-small
RAG_EMBEDDING_DIMENSIONS=1536
RAG_TOP_K=5
RAG_CHUNK_SIZE=512
RAG_CHUNK_OVERLAP=50
```

Or pass a `Settings` object directly in code. See [.env.example](.env.example) for all options.

## Architecture

```
Document → Chunk → EmbeddedChunk → RetrievedChunk → RAGResult
              │         │                │               │
          Chunker    Embedder       VectorStore        LLM
```

Design patterns used:
- **Strategy** — Swappable LLM and embedding providers
- **Repository** — Vector store abstraction
- **Template Method** — BaseRAGPattern with overridable pipeline steps
- **Decorator** — CachedEmbedder wrapping any embedder with SHA-256 keyed cache
- **Factory** — `create_pattern()`, `create_llm()`, `create_embedder()`, etc.

See [Architecture Guide](docs/ARCHITECTURE.md) for details.

## Development

```bash
make install    # Install with dev dependencies
make test       # Run unit tests
make lint       # Lint with ruff
make format     # Auto-format with ruff
make type-check # Type check with mypy
make check      # Run all checks
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full guide.

## Project Structure

```
src/rag_playbook/
├── core/
│   ├── llm.py           # LLM client (OpenAI, Anthropic)
│   ├── embedder.py       # Embedding with caching
│   ├── vector_store.py   # Vector store abstraction
│   ├── chunker.py        # Document chunking strategies
│   ├── evaluator.py      # LLM-as-judge evaluation
│   ├── models.py         # Immutable pipeline data models
│   ├── config.py         # Settings via pydantic-settings
│   ├── cost.py           # Per-model cost tracking
│   └── prompts.py        # All prompt templates
├── patterns/
│   ├── base.py           # Template Method base class
│   ├── naive.py          # Pattern 01: Baseline
│   ├── hybrid_search.py  # Pattern 02: BM25 + vector
│   ├── reranking.py      # Pattern 03: LLM reranking
│   ├── parent_child.py   # Pattern 04: Context expansion
│   ├── query_decomposition.py  # Pattern 05: Sub-queries
│   ├── hyde.py           # Pattern 06: Hypothetical docs
│   ├── self_correcting.py     # Pattern 07: Faithfulness check
│   └── agentic.py        # Pattern 08: Tool-calling loop
├── cli/
│   ├── app.py            # Typer CLI root
│   ├── compare_cmd.py    # The killer feature
│   ├── run_cmd.py        # Single pattern execution
│   ├── recommend_cmd.py  # LLM-powered recommendation
│   ├── ingest_cmd.py     # Document ingestion pipeline
│   ├── patterns_cmd.py   # List patterns
│   └── formatters.py     # Rich terminal output
└── utils/
    ├── timer.py          # Timing context manager
    └── tokenizer.py      # tiktoken helpers
```

## Author

Built by [Aamir Shahzad](https://aamirshahzad.uk) — backend engineer building data systems and AI infrastructure.

## License

[MIT](LICENSE)
