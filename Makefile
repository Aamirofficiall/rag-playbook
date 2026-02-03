.PHONY: install test lint type-check bench clean

install:
	uv pip install -e ".[dev,chromadb,openai]"

test:
	uv run pytest tests/ -m unit -v --cov

test-all:
	uv run pytest tests/ -v --cov

lint:
	uv run ruff check src/ tests/ && uv run ruff format --check src/ tests/

format:
	uv run ruff check --fix src/ tests/ && uv run ruff format src/ tests/

type-check:
	uv run mypy src/

bench:
	uv run python benchmarks/run_all.py

charts:
	uv run python benchmarks/visualize.py

check: lint type-check test

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache dist build *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
