"""Process Nemotron terminal data for augmentation.

Nemotron-Cascade-2-SFT-Data (swe + terminal subsets):
- 439K SWE agent + 822K terminal agent samples
- Selective augmentation for tester (shell execution) and debugger (terminal diagnosis)

Nemotron-Terminal-Corpus (366K):
- Augmentation for tester and debugger
"""

from __future__ import annotations

import logging
from typing import Any

from datasets import Dataset, load_dataset

logger = logging.getLogger(__name__)

CASCADE2_DATASET = "nvidia/Nemotron-Cascade-2-SFT-Data"
TERMINAL_DATASET = "nvidia/Nemotron-Terminal-Corpus"


def load_cascade2_swe(
    cache_dir: str | None = None,
    max_samples: int | None = None,
) -> Dataset:
    """Load SWE subset of Nemotron-Cascade-2-SFT-Data."""
    logger.info(f"Loading {CASCADE2_DATASET} (swe subset)...")
    try:
        ds = load_dataset(CASCADE2_DATASET, "swe", split="train", cache_dir=cache_dir)
    except Exception:
        # Try loading without subset name
        ds = load_dataset(CASCADE2_DATASET, split="train", cache_dir=cache_dir)
    if max_samples and len(ds) > max_samples:
        ds = ds.select(range(max_samples))
    logger.info(f"Loaded {len(ds)} cascade2 SWE samples")
    return ds


def load_terminal_corpus(
    cache_dir: str | None = None,
    max_samples: int | None = None,
) -> Dataset:
    """Load Nemotron-Terminal-Corpus."""
    logger.info(f"Loading {TERMINAL_DATASET}...")
    ds = load_dataset(TERMINAL_DATASET, split="train", cache_dir=cache_dir)
    if max_samples and len(ds) > max_samples:
        ds = ds.select(range(max_samples))
    logger.info(f"Loaded {len(ds)} terminal samples")
    return ds


def filter_for_tester(ds: Dataset) -> list[dict[str, Any]]:
    """Filter terminal/SWE data for tester augmentation (shell execution patterns)."""
    samples = []
    for row in ds:
        text = row.get("text") or row.get("content") or row.get("output", "")
        # Heuristic: samples that contain test execution patterns
        if any(kw in text.lower() for kw in ["pytest", "test_", "assert", "unittest", "failed"]):
            samples.append({
                "role": "tester",
                "input_text": row.get("input", "") or row.get("prompt", ""),
                "output_text": text,
                "source": "terminal_augmentation",
                "metadata": {},
            })
    return samples


def filter_for_debugger(ds: Dataset) -> list[dict[str, Any]]:
    """Filter terminal/SWE data for debugger augmentation (terminal diagnosis)."""
    samples = []
    for row in ds:
        text = row.get("text") or row.get("content") or row.get("output", "")
        # Heuristic: samples that contain debugging patterns
        if any(kw in text.lower() for kw in [
            "traceback", "error", "exception", "debug", "stacktrace",
            "segfault", "core dump", "import error",
        ]):
            samples.append({
                "role": "debugger",
                "input_text": row.get("input", "") or row.get("prompt", ""),
                "output_text": text,
                "source": "terminal_augmentation",
                "metadata": {},
            })
    return samples
