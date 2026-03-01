"""Generate publication-quality charts from benchmark results.

Reads ``benchmarks/results/latest/comparison_table.json`` and produces:

1. ``quality_vs_cost.png``       -- scatter: avg cost vs avg relevance per pattern
2. ``latency_comparison.png``    -- horizontal bar: patterns sorted by latency
3. ``retrieval_relevance.png``   -- grouped bar: relevance per pattern per dataset

All charts are saved alongside the JSON in the results directory.

Usage:
    python -m benchmarks.visualize
    python -m benchmarks.visualize --input benchmarks/results/experiment_1/comparison_table.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Style constants -- muted, publication-friendly palette
# ---------------------------------------------------------------------------

PALETTE = [
    "#5B8DB8",  # steel blue
    "#B85B5B",  # muted red
    "#6DB86B",  # sage green
    "#B89B5B",  # sand
    "#8B5BB8",  # muted purple
    "#5BB8B8",  # teal
    "#B85B8D",  # dusty rose
    "#8DB85B",  # olive
]

DEFAULT_INPUT = (
    Path(__file__).resolve().parent / "results" / "latest" / "comparison_table.json"
)

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#FAFAFA",
    "axes.edgecolor": "#CCCCCC",
    "axes.grid": True,
    "grid.color": "#E0E0E0",
    "grid.linewidth": 0.5,
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_results(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not data:
        raise SystemExit(f"No results found in {path}")
    return data


def _aggregate_by_pattern(
    results: list[dict[str, Any]],
) -> dict[str, dict[str, float]]:
    """Compute per-pattern averages across all datasets."""
    acc: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in results:
        p = row["pattern"]
        acc[p]["relevance"].append(row["avg_relevance"])
        acc[p]["cost"].append(row["avg_cost_usd"])
        acc[p]["latency"].append(row["avg_latency_s"])

    aggregated: dict[str, dict[str, float]] = {}
    for pattern, metrics in acc.items():
        aggregated[pattern] = {
            k: sum(v) / len(v) for k, v in metrics.items()
        }
    return aggregated


# ---------------------------------------------------------------------------
# Chart 1: Quality vs Cost scatter
# ---------------------------------------------------------------------------


def plot_quality_vs_cost(
    results: list[dict[str, Any]], output_dir: Path
) -> Path:
    agg = _aggregate_by_pattern(results)
    patterns = list(agg.keys())
    costs = [agg[p]["cost"] for p in patterns]
    relevances = [agg[p]["relevance"] for p in patterns]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for i, pattern in enumerate(patterns):
        color = PALETTE[i % len(PALETTE)]
        ax.scatter(
            costs[i],
            relevances[i],
            s=120,
            color=color,
            edgecolors="white",
            linewidths=0.8,
            zorder=3,
        )
        ax.annotate(
            pattern,
            (costs[i], relevances[i]),
            textcoords="offset points",
            xytext=(8, 4),
            fontsize=8.5,
            color="#333333",
        )

    ax.set_xlabel("Average Cost per Query (USD)")
    ax.set_ylabel("Average Relevance Score")
    ax.set_title("Quality vs Cost by RAG Pattern")

    fig.tight_layout()
    out_path = output_dir / "quality_vs_cost.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Chart 2: Latency comparison (horizontal bar)
# ---------------------------------------------------------------------------


def plot_latency_comparison(
    results: list[dict[str, Any]], output_dir: Path
) -> Path:
    agg = _aggregate_by_pattern(results)
    # Sort by latency ascending (fastest on top).
    sorted_patterns = sorted(agg, key=lambda p: agg[p]["latency"])
    latencies = [agg[p]["latency"] for p in sorted_patterns]

    fig, ax = plt.subplots(figsize=(8, max(4, len(sorted_patterns) * 0.55)))
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(sorted_patterns))]
    y_pos = np.arange(len(sorted_patterns))

    ax.barh(y_pos, latencies, color=colors, edgecolor="white", height=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_patterns)
    ax.set_xlabel("Average Latency (seconds)")
    ax.set_title("Latency Comparison by RAG Pattern")
    ax.invert_yaxis()  # fastest at top

    # Add value labels.
    for i, v in enumerate(latencies):
        ax.text(v + max(latencies) * 0.01, i, f"{v:.3f}s", va="center", fontsize=8.5)

    fig.tight_layout()
    out_path = output_dir / "latency_comparison.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Chart 3: Retrieval relevance (grouped bar per dataset)
# ---------------------------------------------------------------------------


def plot_retrieval_relevance(
    results: list[dict[str, Any]], output_dir: Path
) -> Path:
    # Organize data: {dataset: {pattern: relevance}}
    datasets_seen: list[str] = []
    patterns_seen: list[str] = []
    grid: dict[str, dict[str, float]] = defaultdict(dict)

    for row in results:
        d, p = row["dataset"], row["pattern"]
        if d not in datasets_seen:
            datasets_seen.append(d)
        if p not in patterns_seen:
            patterns_seen.append(p)
        grid[d][p] = row["avg_relevance"]

    n_datasets = len(datasets_seen)
    n_patterns = len(patterns_seen)
    x = np.arange(n_patterns)
    bar_width = 0.8 / max(n_datasets, 1)

    fig, ax = plt.subplots(figsize=(max(8, n_patterns * 1.1), 5.5))

    for idx, dataset in enumerate(datasets_seen):
        offsets = x + (idx - n_datasets / 2 + 0.5) * bar_width
        values = [grid[dataset].get(p, 0.0) for p in patterns_seen]
        ax.bar(
            offsets,
            values,
            width=bar_width * 0.9,
            label=dataset,
            color=PALETTE[idx % len(PALETTE)],
            edgecolor="white",
            linewidth=0.5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(patterns_seen, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Average Relevance Score")
    ax.set_title("Retrieval Relevance by Pattern and Dataset")
    ax.legend(frameon=True, framealpha=0.9, edgecolor="#CCCCCC")
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    out_path = output_dir / "retrieval_relevance.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def generate_all_charts(input_path: Path) -> list[Path]:
    """Generate all benchmark charts and return their file paths."""
    results = _load_results(input_path)
    output_dir = input_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = [
        plot_quality_vs_cost(results, output_dir),
        plot_latency_comparison(results, output_dir),
        plot_retrieval_relevance(results, output_dir),
    ]
    for p in paths:
        print(f"[saved] {p}")
    return paths


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate benchmark visualizations from comparison results."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to comparison_table.json (default: benchmarks/results/latest/).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    generate_all_charts(args.input)
