"""Unified role dataset: combines all sources into per-role training datasets.

Final counts (approximate):
- Planner:  ~5-8K    (teacher-generated)
- Coder:    ~87K     (Nemotron SWE repair)
- Tester:   ~32K     (Nemotron SWE test-gen) + optional augmentation
- Debugger: ~92K     (Nemotron SWE localization) + optional augmentation
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict

logger = logging.getLogger(__name__)


def build_role_datasets(
    samples_by_role: dict[str, list[dict[str, Any]]],
    output_dir: str | Path | None = None,
    val_fraction: float = 0.05,
) -> DatasetDict:
    """Build per-role datasets with train/val splits.

    Args:
        samples_by_role: {"planner": [...], "coder": [...], ...}
        output_dir: if provided, save to disk
        val_fraction: fraction for validation split
    """
    role_datasets = {}

    for role, samples in samples_by_role.items():
        if not samples:
            logger.warning(f"No samples for role {role}")
            continue

        ds = Dataset.from_list(samples)
        split = ds.train_test_split(test_size=val_fraction, seed=42)

        role_datasets[f"{role}_train"] = split["train"]
        role_datasets[f"{role}_val"] = split["test"]

        logger.info(
            f"{role}: {len(split['train'])} train, {len(split['test'])} val"
        )

    dd = DatasetDict(role_datasets)

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        dd.save_to_disk(str(out))
        logger.info(f"Saved role datasets to {out}")

    return dd


def load_role_dataset(
    data_dir: str | Path,
    role: str,
    split: str = "train",
) -> Dataset:
    """Load a single role's dataset."""
    dd = DatasetDict.load_from_disk(str(data_dir))
    key = f"{role}_{split}"
    if key not in dd:
        raise KeyError(f"Dataset key {key} not found. Available: {list(dd.keys())}")
    return dd[key]


def verify_counts(dd: DatasetDict) -> dict[str, int]:
    """Verify dataset counts and return summary."""
    counts = {}
    for key, ds in dd.items():
        counts[key] = len(ds)
        logger.info(f"  {key}: {len(ds)}")
    return counts
