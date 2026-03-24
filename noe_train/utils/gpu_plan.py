"""GPU allocation planning.

GPU 0: Planner (~6.2GB)
GPU 1: Coder (~6.2GB)
GPU 2: Tester (~6.2GB)
GPU 3: Debugger (~6.2GB)
GPU 4: Router + Critic (~1.2GB)
GPU 5-7: Training compute
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch

from noe_train.schema.messages import ExpertRole

logger = logging.getLogger(__name__)


@dataclass()
class GPUAssignment:
    role: str
    device: int
    estimated_memory_gb: float


DEFAULT_GPU_MAP: dict[str, int] = {
    "planner": 0,
    "coder": 1,
    "tester": 2,
    "debugger": 3,
    "router": 4,
    "critic": 4,
    "training": 5,  # 5-7
}

MEMORY_ESTIMATES: dict[str, float] = {
    "planner": 6.2,
    "coder": 6.2,
    "tester": 6.2,
    "debugger": 6.2,
    "router": 0.6,
    "critic": 0.6,
}


def get_device(role: str, gpu_map: dict[str, int] | None = None) -> torch.device:
    """Get the torch device for a given role."""
    gmap = gpu_map or DEFAULT_GPU_MAP
    if role in gmap and torch.cuda.is_available():
        return torch.device(f"cuda:{gmap[role]}")
    return torch.device("cpu")


def get_gpu_assignments(
    n_gpus: int | None = None,
    gpu_map: dict[str, int] | None = None,
) -> list[GPUAssignment]:
    """Get GPU assignments for all components."""
    gmap = gpu_map or DEFAULT_GPU_MAP
    if n_gpus is None:
        n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0

    assignments = []
    for role, gpu_id in gmap.items():
        if gpu_id < n_gpus:
            assignments.append(GPUAssignment(
                role=role,
                device=gpu_id,
                estimated_memory_gb=MEMORY_ESTIMATES.get(role, 0.0),
            ))

    return assignments


def log_gpu_status() -> None:
    """Log current GPU memory usage."""
    if not torch.cuda.is_available():
        logger.info("No GPUs available")
        return

    for i in range(torch.cuda.device_count()):
        mem = torch.cuda.get_device_properties(i)
        allocated = torch.cuda.memory_allocated(i) / 1e9
        reserved = torch.cuda.memory_reserved(i) / 1e9
        total = mem.total_mem / 1e9
        logger.info(
            f"GPU {i} ({mem.name}): {allocated:.1f}GB allocated, "
            f"{reserved:.1f}GB reserved, {total:.1f}GB total"
        )
