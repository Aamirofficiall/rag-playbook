# CLI Reference

rag-playbook ships a command-line interface built on Typer with Rich terminal
output. Install the package and run:

```
rag-playbook --help
```

All commands require an LLM API key. Set `OPENAI_API_KEY` (or
`ANTHROPIC_API_KEY`) in your environment or `.env` file.

---

## `rag-playbook compare`

Run all (or selected) patterns against the same query and documents, then display
a side-by-side comparison table with latency, cost, token usage, and a
recommendation.

```
rag-playbook compare --query "What is the refund policy?" --data ./docs
```

### Options

| Flag | Short | Default | Description |
|------|-------|---------|-------------|
| `--query` | `-q` | (required) | The question to ask |
| `--data` | | `.` | Path to a document file or directory of `.txt`/`.md` files |
| `--patterns` | `-p` | all | Comma-separated pattern names to include (e.g. `naive,reranking,hyde`) |
| `--top-k` | | `5` | Number of chunks to retrieve per pattern |
| `--evaluate` | | `false` | Run LLM-as-judge quality metrics on each result |
| `--output` | `-o` | none | Save results to a JSON file |

### What it does

1. Loads and chunks all documents from `--data`
2. Embeds chunks once (cached across patterns)
3. Runs each selected pattern sequentially with a progress spinner
4. Prints a comparison table sorted by a weighted score (chunk quality 40%, cost 30%, latency 30%)
5. Highlights the recommended pattern with reasoning
6. Shows a step-by-step timing breakdown for the recommended pattern
7. Optionally saves all results as JSON

---

## `rag-playbook run`

Run a single pattern and display the result.

```
rag-playbook run reranking --query "How do I reset my password?" --data ./kb
```

### Arguments

| Argument | Description |
|----------|-------------|
| `PATTERN_NAME` | Pattern to run (positional). One of: `naive`, `hybrid_search`, `reranking`, `parent_child`, `query_decomposition`, `hyde`, `self_correcting`, `agentic` |

### Options

| Flag | Short | Default | Description |
|------|-------|---------|-------------|
| `--query` | `-q` | (required) | The question to ask |
| `--data` | | `.` | Path to documents |
| `--top-k` | | `5` | Number of chunks to retrieve |

### What it does

1. Loads, chunks, and embeds documents
2. Creates the specified pattern instance
3. Runs the query through the full pipeline
4. Prints the answer, source chunks, and metadata (latency, cost, tokens)

---

## `rag-playbook recommend`

Ask an LLM to analyze your query and suggest the best pattern without running any
retrieval.

```
rag-playbook recommend --query "Compare the pricing of product A and product B"
```

### Options

| Flag | Short | Default | Description |
|------|-------|---------|-------------|
| `--query` | `-q` | (required) | The question to analyze |

### What it does

1. Sends the query to the LLM with a structured prompt describing all 8 patterns
2. Parses the JSON response for pattern name, reasoning, and confidence
3. Prints the recommendation

This is a lightweight alternative to `compare` -- no documents or embeddings
needed.

---

## `rag-playbook ingest`

Load documents into the vector store for later querying.

```
rag-playbook ingest --data ./documents --chunker recursive --chunk-size 256
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--data` | (required) | Path to a document file or directory |
| `--chunker` | `fixed` | Chunking strategy: `fixed`, `recursive`, or `structural` |
| `--chunk-size` | `512` | Target tokens per chunk |
| `--chunk-overlap` | `50` | Overlap tokens between adjacent chunks |

### What it does

1. Scans `--data` for `.txt`, `.md`, and `.markdown` files
2. Splits documents using the selected chunking strategy
3. Embeds all chunks via the configured embedding provider
4. Stores embedded chunks in the configured vector store
5. Reports the total number of ingested chunks

---

## `rag-playbook patterns`

List all registered patterns with their descriptions.

```
rag-playbook patterns
```

No options. Outputs a table of pattern names and one-line descriptions.

---

## Environment Variables

All settings are configured via environment variables or a `.env` file.
Variable names match the settings fields directly — no prefix needed.

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | (none) | OpenAI API key |
| `ANTHROPIC_API_KEY` | (none) | Anthropic API key |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | Custom base URL (OpenRouter, Azure, etc.) |
| `DEFAULT_LLM_PROVIDER` | `openai` | LLM provider: `openai` or `anthropic` |
| `DEFAULT_LLM_MODEL` | `gpt-4o-mini` | Model name |
| `EMBEDDING_PROVIDER` | `openai` | Embedding provider |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |
| `VECTOR_STORE_PROVIDER` | `memory` | Vector store: `memory`, `chromadb`, `pgvector`, `qdrant` |
| `DEFAULT_TOP_K` | `5` | Default number of chunks to retrieve |
| `HYBRID_SEARCH_ALPHA` | `0.5` | Weight for vector vs keyword in hybrid search (0=keyword, 1=vector) |
| `DEFAULT_CHUNK_SIZE` | `512` | Default tokens per chunk |
| `DEFAULT_CHUNK_OVERLAP` | `50` | Default overlap tokens |
