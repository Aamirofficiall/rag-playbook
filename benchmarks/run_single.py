"""Run a single RAG pattern benchmark against a dataset.

Usage:
    python -m benchmarks.run_single --pattern reranking --dataset squad_200
    python -m benchmarks.run_single --pattern naive --dataset custom_docs_20 --output-dir results/exp1
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from benchmarks.run_all import DEFAULT_OUTPUT_DIR, run_benchmarks


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a single RAG pattern against a dataset."
    )
    parser.add_argument(
        "--pattern",
        required=True,
        help="Pattern name to benchmark (e.g., naive, reranking, hyde).",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset name to benchmark against.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write results (default: benchmarks/results/latest/).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    asyncio.run(
        run_benchmarks(
            dataset_names=[args.dataset],
            pattern_names=[args.pattern],
            output_dir=args.output_dir,
        )
    )


if __name__ == "__main__":
    main()
