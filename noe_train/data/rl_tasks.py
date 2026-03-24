"""RL task pools for Stage B and Stage C.

Stage B: Nemotron-Cascade-RL-SWE subset (short fixes, <=1 file) + SWE-bench train
Stage C: Full Nemotron-Cascade-RL-SWE (110K) + SWE-bench train
Eval: SWE-bench Verified (500) — NEVER train
"""

from __future__ import annotations

import logging
from typing import Any

from datasets import Dataset, load_dataset

logger = logging.getLogger(__name__)

RL_DATASET = "nvidia/Nemotron-Cascade-RL-SWE"
SWEBENCH_DATASET = "princeton-nlp/SWE-bench"
SWEBENCH_VERIFIED = "princeton-nlp/SWE-bench_Verified"


def load_rl_swe(cache_dir: str | None = None) -> Dataset:
    """Load full Nemotron-Cascade-RL-SWE (~110K)."""
    logger.info(f"Loading {RL_DATASET}...")
    ds = load_dataset(RL_DATASET, split="train", cache_dir=cache_dir)
    logger.info(f"Loaded {len(ds)} RL-SWE samples")
    return ds


def filter_stage_b(ds: Dataset, max_files_changed: int = 1) -> Dataset:
    """Filter RL dataset for Stage B: short fixes (<=1 file)."""
    def _is_short(row):
        # Heuristic: count files in patch
        patch = row.get("patch") or row.get("gold_patch", "")
        files = set()
        for line in patch.split("\n"):
            if line.startswith("+++ b/"):
                files.add(line[6:].strip())
        return len(files) <= max_files_changed

    filtered = ds.filter(_is_short)
    logger.info(f"Stage B filter: {len(ds)} → {len(filtered)} (max {max_files_changed} file)")
    return filtered


def load_swebench_train(cache_dir: str | None = None) -> Dataset:
    """Load SWE-bench train split."""
    logger.info(f"Loading {SWEBENCH_DATASET} train...")
    ds = load_dataset(SWEBENCH_DATASET, split="train", cache_dir=cache_dir)
    logger.info(f"Loaded {len(ds)} SWE-bench train samples")
    return ds


def load_swebench_verified(cache_dir: str | None = None) -> Dataset:
    """Load SWE-bench Verified for evaluation only. NEVER TRAIN ON THIS."""
    logger.info(f"Loading {SWEBENCH_VERIFIED} (EVAL ONLY)...")
    ds = load_dataset(SWEBENCH_VERIFIED, split="test", cache_dir=cache_dir)
    logger.info(f"Loaded {len(ds)} SWE-bench Verified samples (EVAL ONLY)")
    return ds


def build_rl_task_pool(
    rl_ds: Dataset,
    swebench_train: Dataset | None = None,
    stage: str = "c",
) -> list[dict[str, Any]]:
    """Build RL task pool for a given stage.

    Args:
        rl_ds: Nemotron RL dataset (filtered for B, full for C)
        swebench_train: optional SWE-bench train for augmentation
        stage: "b" or "c"
    """
    tasks = []

    for row in rl_ds:
        task = {
            "instance_id": row.get("instance_id", ""),
            "repo": row.get("repo", ""),
            "issue_text": row.get("problem_statement") or row.get("issue", ""),
            "gold_patch": row.get("patch") or row.get("gold_patch", ""),
            "fail_to_pass": row.get("FAIL_TO_PASS") or row.get("fail_to_pass", []),
            "pass_to_pass": row.get("PASS_TO_PASS") or row.get("pass_to_pass", []),
            "source": "nemotron_rl_swe",
        }
        tasks.append(task)

    if swebench_train is not None:
        for row in swebench_train:
            task = {
                "instance_id": row.get("instance_id", ""),
                "repo": row.get("repo", ""),
                "issue_text": row.get("problem_statement", ""),
                "gold_patch": row.get("patch", ""),
                "fail_to_pass": row.get("FAIL_TO_PASS", []),
                "pass_to_pass": row.get("PASS_TO_PASS", []),
                "source": "swebench_train",
            }
            tasks.append(task)

    logger.info(f"Stage {stage} task pool: {len(tasks)} tasks")
    return tasks
