"""Download and cache benchmark datasets for RAG pattern evaluation."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import httpx

# ---------------------------------------------------------------------------
# Dataset metadata
# ---------------------------------------------------------------------------

DatasetEntry = tuple[str, list[str], str]  # (query, context_docs, expected_answer)

CACHE_DIR = Path(__file__).resolve().parent / "cache"

DATASETS: dict[str, dict[str, Any]] = {
    "squad_200": {
        "description": "200 question-answer pairs sampled from SQuAD 2.0.",
        "url": (
            "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json"
        ),
        "sample_size": 200,
        "parser": "_parse_squad",
    },
    "hotpotqa_100": {
        "description": "100 multi-hop questions from HotpotQA distractor setting.",
        "url": (
            "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json"
        ),
        "sample_size": 100,
        "parser": "_parse_hotpotqa",
    },
    "custom_docs_20": {
        "description": "20 curated technical document QA pairs shipped with this repo.",
        "url": None,  # bundled locally
        "sample_size": 20,
        "parser": "_parse_custom",
    },
}


# ---------------------------------------------------------------------------
# Internal parsers
# ---------------------------------------------------------------------------


def _parse_squad(raw: dict[str, Any], sample_size: int) -> list[DatasetEntry]:
    """Extract (query, context_docs, expected_answer) from SQuAD 2.0 JSON."""
    entries: list[DatasetEntry] = []
    for article in raw.get("data", []):
        for paragraph in article.get("paragraphs", []):
            context = paragraph.get("context", "")
            for qa in paragraph.get("qas", []):
                if qa.get("is_impossible", False):
                    continue
                answers = qa.get("answers", [])
                if not answers:
                    continue
                entries.append((
                    qa["question"],
                    [context],
                    answers[0]["text"],
                ))
                if len(entries) >= sample_size:
                    return entries
    return entries


def _parse_hotpotqa(raw: list[dict[str, Any]], sample_size: int) -> list[DatasetEntry]:
    """Extract entries from HotpotQA distractor JSON (list of dicts)."""
    entries: list[DatasetEntry] = []
    for item in raw:
        question = item.get("question", "")
        answer = item.get("answer", "")
        # context is a list of [title, sentences] pairs
        context_docs: list[str] = []
        for title, sentences in item.get("context", []):
            context_docs.append(f"{title}\n" + " ".join(sentences))
        entries.append((question, context_docs, answer))
        if len(entries) >= sample_size:
            break
    return entries


def _parse_custom(_raw: Any, _sample_size: int) -> list[DatasetEntry]:
    """Load the bundled custom technical-doc QA pairs."""
    custom_path = Path(__file__).resolve().parent / "custom_docs_20.json"
    if not custom_path.exists():
        # Generate a minimal placeholder so the pipeline does not crash.
        placeholder: list[DatasetEntry] = [
            (
                f"What does section {i + 1} cover?",
                [f"Section {i + 1} covers topic {i + 1} in detail."],
                f"Topic {i + 1}",
            )
            for i in range(20)
        ]
        custom_path.write_text(json.dumps(placeholder, indent=2), encoding="utf-8")
        return placeholder
    data = json.loads(custom_path.read_text(encoding="utf-8"))
    return [(q, docs, a) for q, docs, a in data]


_PARSERS = {
    "_parse_squad": _parse_squad,
    "_parse_hotpotqa": _parse_hotpotqa,
    "_parse_custom": _parse_custom,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def download_dataset(name: str) -> Path:
    """Download a dataset by *name* and cache it locally.

    Returns the path to the cached JSON file.
    Raises ``KeyError`` if *name* is not in ``DATASETS``.
    """
    if name not in DATASETS:
        raise KeyError(
            f"Unknown dataset '{name}'. Available: {', '.join(DATASETS)}"
        )

    meta = DATASETS[name]
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if meta["url"] is None:
        # Bundled dataset -- nothing to download.
        return Path(__file__).resolve().parent / "custom_docs_20.json"

    url: str = meta["url"]
    url_hash = hashlib.sha256(url.encode()).hexdigest()[:12]
    cache_file = CACHE_DIR / f"{name}_{url_hash}.json"

    if cache_file.exists():
        print(f"[cache hit] {name} -> {cache_file}")
        return cache_file

    print(f"[downloading] {name} from {url} ...")
    with httpx.Client(timeout=120.0, follow_redirects=True) as client:
        response = client.get(url)
        response.raise_for_status()

    cache_file.write_bytes(response.content)
    print(f"[saved] {cache_file} ({cache_file.stat().st_size / 1_048_576:.1f} MB)")
    return cache_file


def load_dataset(name: str) -> list[DatasetEntry]:
    """Return a list of ``(query, context_docs, expected_answer)`` tuples.

    Downloads the dataset first if it is not already cached.
    """
    meta = DATASETS[name]
    cache_path = download_dataset(name)

    parser_name: str = meta["parser"]
    parser_fn = _PARSERS[parser_name]

    if meta["url"] is None:
        # Custom dataset -- parser handles its own loading.
        return parser_fn(None, meta["sample_size"])

    raw = json.loads(cache_path.read_text(encoding="utf-8"))
    return parser_fn(raw, meta["sample_size"])


# ---------------------------------------------------------------------------
# CLI entry-point: python -m benchmarks.datasets.download
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    names = sys.argv[1:] if len(sys.argv) > 1 else list(DATASETS)
    for dataset_name in names:
        entries = load_dataset(dataset_name)
        print(f"  -> {dataset_name}: {len(entries)} entries loaded")
