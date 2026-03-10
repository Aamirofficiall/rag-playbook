# Benchmarks

How rag-playbook benchmarks are run, what datasets are used, and how to
reproduce the results.

## Datasets

| Dataset | Size | Source | Tests |
|---------|------|--------|-------|
| `squad_200` | 200 questions | SQuAD 2.0 subset | Basic factual retrieval |
| `hotpotqa_100` | 100 questions | HotpotQA subset | Multi-hop reasoning |
| `custom_docs_20` | 20 documents, 60 queries | Curated technical docs | Structural chunking, real formatting |

## Running Benchmarks

```bash
# Run all benchmarks
make bench

# Or with options
python -m benchmarks.run_all --datasets squad_200 hotpotqa_100 --patterns naive reranking hyde

# Generate charts
make charts
```

## Methodology

1. Each dataset is loaded and documents are chunked with the default `fixed`
   chunker at 512 tokens with 50-token overlap.
2. Every pattern runs against every query in the dataset.
3. Metrics are collected per query:
   - **Latency** -- wall-clock time for the full pipeline.
   - **Cost** -- total API cost (LLM + embedding calls).
   - **Relevance** -- LLM-as-judge score (0-1) using `gpt-4o-mini`.
4. Results are averaged per pattern per dataset and saved to
   `benchmarks/results/latest/comparison_table.json`.

## Configuration

All benchmarks use the same configuration to ensure fair comparison:

| Setting | Value |
|---------|-------|
| LLM | `gpt-4o-mini` |
| Embedding | `text-embedding-3-small` (1536d) |
| Vector store | In-memory |
| Chunk size | 512 tokens |
| Chunk overlap | 50 tokens |
| Top-K | 5 |

## Charts

The `benchmarks/visualize.py` script generates three charts from the results:

- **`quality_vs_cost.png`** -- Scatter plot. X = average cost per query,
  Y = average relevance. Each dot is one pattern.
- **`latency_comparison.png`** -- Horizontal bar chart sorted by latency.
- **`retrieval_relevance.png`** -- Grouped bar chart showing relevance per
  pattern per dataset.

## Estimated Cost

| Dataset | Queries | Patterns | Est. Cost |
|---------|---------|----------|-----------|
| squad_200 | 200 | 8 | ~$4 |
| hotpotqa_100 | 100 | 8 | ~$2 |
| custom_docs_20 | 60 | 8 | ~$1.20 |
| **Total** | | | **~$7-8** |

## Reproducing Results

```bash
# 1. Install with benchmark dependencies
uv pip install -e ".[bench]"

# 2. Download datasets
python -m benchmarks.datasets.download

# 3. Run benchmarks
python -m benchmarks.run_all

# 4. Generate charts
python benchmarks/visualize.py
```

Results are saved to `benchmarks/results/latest/`. The `comparison_table.json`
file contains all raw metrics.
