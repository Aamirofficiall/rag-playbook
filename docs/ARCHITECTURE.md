# Architecture

## High-Level Overview

```
                         rag-playbook
  +---------------------------------------------------------+
  |  CLI Layer (typer + rich)                                |
  |  compare | run | recommend | ingest | patterns           |
  +-------------------------+-------------------------------+
                            |
  +-------------------------v-------------------------------+
  |  Pattern Layer                                           |
  |                                                          |
  |  BaseRAGPattern (Template Method)                        |
  |    |-- NaiveRAG          (01)                            |
  |    |-- HybridSearchRAG   (02)                            |
  |    |-- RerankingRAG      (03)                            |
  |    |-- ParentChildRAG    (04)                            |
  |    |-- QueryDecompRAG    (05)                            |
  |    |-- HyDERAG           (06)                            |
  |    |-- SelfCorrectingRAG (07)                            |
  |    +-- AgenticRAG        (08)                            |
  |                                                          |
  |  PATTERN_REGISTRY (dict[str, type]) + create_pattern()   |
  +-------------------------+-------------------------------+
                            |
  +-------------------------v-------------------------------+
  |  Core Infrastructure                                     |
  |                                                          |
  |  BaseLLM -----> OpenAILLM | AnthropicLLM   (Strategy)   |
  |  BaseEmbedder -> OpenAIEmbedder             (Strategy)   |
  |  CachedEmbedder(inner)                      (Decorator)  |
  |  BaseVectorStore -> InMemoryVectorStore     (Repository) |
  |  BaseChunker -> Fixed|Recursive|Structural  (Strategy)   |
  |  Settings (pydantic-settings, .env)                      |
  |  Models: Document->Chunk->EmbeddedChunk->RetrievedChunk  |
  +---------------------------------------------------------+
```

## Design Patterns

### Strategy -- LLM and Embedding Providers

`BaseLLM` and `BaseEmbedder` define abstract interfaces. Concrete implementations
(`OpenAILLM`, `AnthropicLLM`, `OpenAIEmbedder`) are selected at runtime via
factory functions (`create_llm`, `create_embedder`). Patterns call
`self.llm.generate()` without knowing which provider backs it.

**Files:** `src/rag_playbook/core/llm.py`, `src/rag_playbook/core/embedder.py`

### Repository -- Vector Stores

`BaseVectorStore` abstracts storage with `add()`, `search()`, `hybrid_search()`,
`delete()`, `count()`, and `reset()`. The `InMemoryVectorStore` ships as the
default zero-config backend. ChromaDB, pgvector, and Qdrant are planned as
optional dependencies.

**File:** `src/rag_playbook/core/vector_store.py`

### Template Method -- BaseRAGPattern

The `query()` method in `BaseRAGPattern` is the fixed skeleton. It orchestrates
five steps in order, timing each one:

1. `preprocess_query()` -- transform or decompose the question
2. `retrieve()` -- fetch chunks from the vector store
3. `postprocess_chunks()` -- rerank, expand, or filter
4. `generate()` -- produce the answer via LLM
5. `validate()` -- check faithfulness, retry if needed

Subclasses override individual steps. For example, `HyDERAG` overrides only
`preprocess_query()`, while `RerankingRAG` overrides `retrieve()` and
`postprocess_chunks()`.

**File:** `src/rag_playbook/patterns/base.py`

### Decorator -- CachedEmbedder

`CachedEmbedder` wraps any `BaseEmbedder` with an in-memory SHA-256 cache. When
the `compare` command runs all 8 patterns against the same documents, embeddings
are computed once (pattern 1) and served from cache for patterns 2-8, cutting
embedding cost by ~8x.

**File:** `src/rag_playbook/core/embedder.py`

### Factory -- create_* Functions

Each subsystem exposes a factory: `create_llm()`, `create_embedder()`,
`create_vector_store()`, `create_chunker()`, and `create_pattern()`. These read
from `Settings` and return the appropriate concrete instance. Adding a new
provider means writing a class and adding a `case` branch to the factory.

## Pipeline Flow

Every query passes through the same data pipeline, regardless of pattern:

```
Document          Raw input (file, page, text blob)
    |
    v
Chunk             Piece of a document, pre-embedding
    |              (via BaseChunker: fixed / recursive / structural)
    v
EmbeddedChunk     Chunk + vector (enforced by Pydantic validator)
    |              (stored in BaseVectorStore)
    v
RetrievedChunk    Chunk + relevance score + retrieval method
    |              (returned from search / hybrid_search)
    v
RAGResult         Answer + sources + QueryMetadata
                   (pattern, model, tokens, latency, cost, steps)
```

All models are frozen Pydantic `BaseModel` instances. Distinct types at each
stage let the type checker catch pipeline ordering mistakes at development time.

## Directory Structure

```
src/rag_playbook/
|-- __init__.py              Public API: Document, Chunk, RAGResult, create_pattern
|-- py.typed                 PEP 561 marker
|
|-- core/
|   |-- __init__.py          Re-exports all core types
|   |-- config.py            Settings via pydantic-settings (.env / env vars)
|   |-- models.py            Pipeline data models (Document -> RAGResult)
|   |-- llm.py               BaseLLM + OpenAI/Anthropic + create_llm()
|   |-- embedder.py          BaseEmbedder + CachedEmbedder + create_embedder()
|   |-- vector_store.py      BaseVectorStore + InMemoryVectorStore + create_vector_store()
|   |-- chunker.py           BaseChunker + Fixed/Recursive/Structural + create_chunker()
|   |-- prompts.py           All LLM prompt templates in one place
|   |-- cost.py              Token cost calculations
|   |-- evaluator.py         LLM-as-judge evaluation metrics
|   +-- exceptions.py        Exception hierarchy
|
|-- patterns/
|   |-- __init__.py          PATTERN_REGISTRY, create_pattern(), all_patterns()
|   |-- base.py              BaseRAGPattern (Template Method)
|   |-- naive.py             Pattern 01
|   |-- hybrid_search.py     Pattern 02
|   |-- reranking.py         Pattern 03
|   |-- parent_child.py      Pattern 04
|   |-- query_decomposition.py  Pattern 05
|   |-- hyde.py              Pattern 06
|   |-- self_correcting.py   Pattern 07
|   +-- agentic.py           Pattern 08
|
|-- cli/
|   |-- __init__.py
|   |-- app.py               Typer app + command registration
|   |-- run_cmd.py           Single pattern execution
|   |-- compare_cmd.py       Side-by-side comparison
|   |-- recommend_cmd.py     LLM-powered pattern recommendation
|   |-- ingest_cmd.py        Document loading and indexing
|   |-- patterns_cmd.py      List registered patterns
|   +-- formatters.py        Rich terminal output helpers
|
+-- utils/
    |-- timer.py             Timing utilities
    +-- tokenizer.py         Token counting helpers
```

## Extension Points

### Adding a New Pattern

1. Create `src/rag_playbook/patterns/my_pattern.py`
2. Subclass `BaseRAGPattern`, set `_pattern_name`, implement `pattern_name` and
   `description` properties
3. Override one or more steps: `preprocess_query`, `retrieve`,
   `postprocess_chunks`, `generate`, `validate`
4. Import the class in `src/rag_playbook/patterns/__init__.py` and add it to the
   registration loop

The pattern is now automatically available in the CLI, comparison table, and
registry.

### Adding a New LLM Provider

1. Subclass `BaseLLM` in `src/rag_playbook/core/llm.py`
2. Implement `model_name` property and `_call()` method
3. Add a `case` branch to `create_llm()`
4. Add any required settings to `Settings` in `config.py`
5. Add the provider's SDK to `pyproject.toml` optional dependencies

### Adding a New Vector Store

1. Subclass `BaseVectorStore` in `src/rag_playbook/core/vector_store.py`
2. Implement all abstract methods: `add`, `search`, `hybrid_search`, `delete`,
   `count`, `reset`
3. Add a `case` branch to `create_vector_store()`
4. Add the client library to `pyproject.toml` optional dependencies

### Adding a New Chunking Strategy

1. Subclass `BaseChunker` in `src/rag_playbook/core/chunker.py`
2. Implement `strategy` property and `_split()` method
3. Add a `case` branch to `create_chunker()`
