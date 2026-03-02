# Benchmark Datasets

## Sources

| Dataset | License | Source |
|---------|---------|--------|
| SQuAD 2.0 | CC BY-SA 4.0 | [rajpurkar/SQuAD-explorer](https://rajpurkar.github.io/SQuAD-explorer/) |
| HotpotQA | CC BY-SA 4.0 | [hotpotqa/hotpot_dev_distractor](https://hotpotqa.github.io/) |
| Custom docs | MIT | Curated technical documents |

## Download

```bash
python -m benchmarks.datasets.download
```

Downloaded files are cached in `benchmarks/datasets/cache/` (gitignored).

## Format

Each dataset is loaded via `load_dataset(name)` which returns:

```python
list[tuple[str, list[str], str]]
#          query  contexts  expected_answer
```
