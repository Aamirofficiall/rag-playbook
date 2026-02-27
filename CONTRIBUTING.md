# Contributing to rag-playbook

Thanks for your interest in contributing! This guide will help you get started.

## Development Setup

```bash
git clone https://github.com/Aamirofficiall/rag-playbook.git
cd rag-playbook
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -e ".[dev,chromadb,openai]"
```

## Running Tests

```bash
# Unit tests (no API keys needed)
make test

# All tests including integration
make test-all

# Single test file
pytest tests/core/test_models.py -v
```

## Code Quality

```bash
# Lint
make lint

# Auto-format
make format

# Type check
make type-check

# All checks
make check
```

## Adding a New RAG Pattern

1. Create `src/rag_playbook/patterns/your_pattern.py`
2. Subclass `BaseRAGPattern` and set `_pattern_name`
3. Override the relevant pipeline steps
4. Register in `patterns/__init__.py`
5. Add tests in `tests/patterns/test_your_pattern.py`
6. Add documentation in `docs/patterns/`

## Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat(core): add new embedding provider
fix(cli): handle empty document directory
docs: update decision guide with new pattern
test(patterns): add edge case for HyDE
```

## Pull Requests

- Branch from `main`
- Keep PRs focused — one feature or fix per PR
- All tests must pass
- Lint and type check must be clean
- Fill out the PR template
