#!/usr/bin/env python3
"""Train Stage A: Individual Role SFT.

Modes:
  --sequential    Train all roles one-by-one, sharing base model (1 GPU, safest)
  --groups N      Split roles into N groups, run groups in parallel (N GPUs)
  --parallel      One role per GPU (4 GPUs)

Single GPU timing estimates (A100 80GB):
  sequential: ~60h (planner 2h + coder 20h + tester 8h + debugger 20h + overhead)
  groups 2:   ~30h (group1: coder+planner on GPU0, group2: debugger+tester on GPU1)

Single GPU grouping (--groups 1):
  Loads base model once (~6GB), trains roles sequentially with LoRA swap.
  Peak VRAM: ~18GB (model 6GB + LoRA 150MB + optimizer ~8GB + activations ~4GB).
  Saves ~8 min of model reload time across 4 roles.
"""

import argparse
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

from noe_train.utils.logging import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


def _setup_wandb(project: str, entity: str | None = None):
    """Set wandb env vars so HF Trainer picks them up in all subprocesses."""
    os.environ["WANDB_PROJECT"] = project
    if entity:
        os.environ["WANDB_ENTITY"] = entity
    os.environ["WANDB_LOG_MODEL"] = "false"
    logger.info(f"wandb: project={project}, entity={entity or '(default)'}")

# Group roles by dataset size so parallel groups finish around the same time.
# Group 0 (heavy): coder (87K) + planner (5K) ≈ 92K samples
# Group 1 (heavy): debugger (92K) + tester (32K) ≈ 124K samples
# Not perfectly balanced, but debugger r=32 trains slower per-sample than coder r=16,
# so effective time is close.
DEFAULT_GROUPS = [
    ["coder", "planner"],     # GPU 0: ~22h on A100
    ["debugger", "tester"],   # GPU 1: ~28h on A100
]


def train_group(roles: list[str], data_dir: str, output_dir: str, gpu_id: int):
    """Train a group of roles sequentially on one GPU, sharing the base model."""
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    from noe_train.data.role_dataset import load_role_dataset
    from noe_train.schema.messages import ExpertRole
    from noe_train.training.stage_a import StageAConfig, train_role_group

    expert_roles = [ExpertRole(r) for r in roles]
    datasets = {}
    for role_name in roles:
        train_ds = load_role_dataset(data_dir, role_name, "train")
        val_ds = load_role_dataset(data_dir, role_name, "val")
        datasets[role_name] = (train_ds, val_ds)

    config = StageAConfig()
    train_role_group(expert_roles, datasets, output_dir=output_dir, config=config)


def train_single_role(role_name: str, data_dir: str, output_dir: str, gpu_id: int):
    """Train one role on one GPU."""
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    from noe_train.data.role_dataset import load_role_dataset
    from noe_train.schema.messages import ExpertRole
    from noe_train.training.stage_a import StageAConfig, train_role

    role = ExpertRole(role_name)
    train_ds = load_role_dataset(data_dir, role_name, "train")
    val_ds = load_role_dataset(data_dir, role_name, "val")
    train_role(role, train_ds, val_ds, output_dir=output_dir, config=StageAConfig())


def main():
    parser = argparse.ArgumentParser(description="Stage A: Role SFT")
    parser.add_argument("--data-dir", type=str, default="data/processed")
    parser.add_argument("--output-dir", type=str, default="checkpoints/stage_a")
    parser.add_argument("--roles", nargs="+", default=["planner", "coder", "tester", "debugger"])

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--sequential", action="store_true",
        help="Train all roles sequentially on GPU 0, sharing base model",
    )
    mode.add_argument(
        "--groups", type=int, metavar="N",
        help="Split into N parallel groups (default grouping balances by dataset size)",
    )
    mode.add_argument(
        "--parallel", action="store_true",
        help="One role per GPU (requires 4 GPUs)",
    )

    parser.add_argument("--gpu-offset", type=int, default=0, help="First GPU ID to use")
    parser.add_argument("--wandb-project", type=str, default="noe-train", help="wandb project name")
    parser.add_argument("--wandb-entity", type=str, default=None, help="wandb team/entity")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    args = parser.parse_args()

    if args.no_wandb:
        os.environ["WANDB_DISABLED"] = "true"
        logger.info("wandb disabled")
    else:
        _setup_wandb(args.wandb_project, args.wandb_entity)

    if args.sequential or (not args.groups and not args.parallel):
        # Default: sequential on single GPU
        logger.info("Mode: sequential (single GPU, shared base model)")
        logger.info(f"  Roles: {args.roles}")
        train_group(args.roles, args.data_dir, args.output_dir, args.gpu_offset)

    elif args.groups:
        n = args.groups
        if n == 1:
            # Same as sequential
            train_group(args.roles, args.data_dir, args.output_dir, args.gpu_offset)
        else:
            # Split roles into N groups
            if n == 2:
                groups = DEFAULT_GROUPS
            else:
                # Round-robin assignment
                groups = [[] for _ in range(n)]
                for i, role in enumerate(args.roles):
                    groups[i % n].append(role)

            logger.info(f"Mode: {n} parallel groups")
            for i, g in enumerate(groups):
                logger.info(f"  GPU {args.gpu_offset + i}: {g}")

            with ProcessPoolExecutor(max_workers=n) as executor:
                futures = {}
                for i, group_roles in enumerate(groups):
                    if not group_roles:
                        continue
                    gpu_id = args.gpu_offset + i
                    future = executor.submit(
                        train_group, group_roles, args.data_dir, args.output_dir, gpu_id
                    )
                    futures[future] = (i, group_roles)

                for future in as_completed(futures):
                    idx, roles = futures[future]
                    try:
                        future.result()
                        logger.info(f"Group {idx} completed: {roles}")
                    except Exception as e:
                        logger.error(f"Group {idx} failed ({roles}): {e}")

    elif args.parallel:
        logger.info("Mode: parallel (one role per GPU)")
        with ProcessPoolExecutor(max_workers=len(args.roles)) as executor:
            futures = {}
            for i, role_name in enumerate(args.roles):
                gpu_id = args.gpu_offset + i
                future = executor.submit(
                    train_single_role, role_name, args.data_dir, args.output_dir, gpu_id
                )
                futures[future] = role_name

            for future in as_completed(futures):
                role_name = futures[future]
                try:
                    future.result()
                    logger.info(f"Completed: {role_name}")
                except Exception as e:
                    logger.error(f"Failed: {role_name}: {e}")

    logger.info("Stage A complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
