#!/usr/bin/env python3
"""Train Stage A: Individual Role SFT.

Modes:
  --sequential    Train all roles one-by-one, sharing base model (1 GPU, safest)
  --groups N      Split roles into N groups, run groups in parallel (N GPUs)
  --parallel      One role per GPU (4 GPUs)

Single GPU timing estimates (A100 80GB):
  sequential: ~60h (planner 2h + coder 20h + tester 8h + debugger 20h + overhead)
  groups 2:   ~30h (group1: coder+planner on GPU0, group2: debugger+tester on GPU1)
"""

import argparse
import logging
import multiprocessing as mp
import os
import sys

from noe_train.utils.logging import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

# Group roles by dataset size so parallel groups finish around the same time.
DEFAULT_GROUPS = [
    ["coder", "planner"],     # GPU 0: ~22h on A100
    ["debugger", "tester"],   # GPU 1: ~28h on A100
]


def _worker_train_group(gpu_id: int, roles: list, data_dir: str, output_dir: str, env_vars: dict):
    """Spawned subprocess entry point. Sets CUDA_VISIBLE_DEVICES BEFORE torch import."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    # Propagate wandb + other env vars from parent
    for k, v in env_vars.items():
        os.environ[k] = v

    # Now safe to import torch — it will see only the assigned GPU
    from noe_train.data.role_dataset import load_role_dataset
    from noe_train.schema.messages import ExpertRole
    from noe_train.training.stage_a import StageAConfig, train_role_group
    from noe_train.utils.logging import setup_logging as _setup

    _setup()
    _logger = logging.getLogger(f"stage_a.gpu{gpu_id}")
    _logger.info(f"Worker started: GPU {gpu_id}, roles={roles}, CUDA_VISIBLE_DEVICES={gpu_id}")

    expert_roles = [ExpertRole(r) for r in roles]
    datasets = {}
    for role_name in roles:
        train_ds = load_role_dataset(data_dir, role_name, "train")
        val_ds = load_role_dataset(data_dir, role_name, "val")
        datasets[role_name] = (train_ds, val_ds)
        _logger.info(f"  {role_name}: {len(train_ds)} train, {len(val_ds)} val")

    config = StageAConfig()
    train_role_group(expert_roles, datasets, output_dir=output_dir, config=config)
    _logger.info(f"Worker done: GPU {gpu_id}, roles={roles}")


def _worker_train_single(gpu_id: int, role_name: str, data_dir: str, output_dir: str, env_vars: dict):
    """Spawned subprocess entry point for single role."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    for k, v in env_vars.items():
        os.environ[k] = v

    from noe_train.data.role_dataset import load_role_dataset
    from noe_train.schema.messages import ExpertRole
    from noe_train.training.stage_a import StageAConfig, train_role
    from noe_train.utils.logging import setup_logging as _setup

    _setup()
    _logger = logging.getLogger(f"stage_a.gpu{gpu_id}")
    _logger.info(f"Worker started: GPU {gpu_id}, role={role_name}")

    role = ExpertRole(role_name)
    train_ds = load_role_dataset(data_dir, role_name, "train")
    val_ds = load_role_dataset(data_dir, role_name, "val")
    train_role(role, train_ds, val_ds, output_dir=output_dir, config=StageAConfig())
    _logger.info(f"Worker done: GPU {gpu_id}, role={role_name}")


def _collect_env_vars() -> dict:
    """Collect env vars that need to propagate to spawned workers."""
    keys = [
        "WANDB_PROJECT", "WANDB_ENTITY", "WANDB_API_KEY", "WANDB_DISABLED",
        "WANDB_LOG_MODEL", "WANDB_MODE", "HF_HOME", "TRANSFORMERS_CACHE",
        "HF_TOKEN",
    ]
    return {k: os.environ[k] for k in keys if k in os.environ}


def _setup_wandb(project: str, entity: str | None = None):
    """Set wandb env vars so workers pick them up."""
    os.environ["WANDB_PROJECT"] = project
    if entity:
        os.environ["WANDB_ENTITY"] = entity
    os.environ["WANDB_LOG_MODEL"] = "false"
    logger.info(f"wandb: project={project}, entity={entity or '(default)'}")


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

    # Use spawn context — each child gets a fresh Python with no inherited CUDA state
    ctx = mp.get_context("spawn")
    env_vars = _collect_env_vars()

    if args.sequential or (not args.groups and not args.parallel):
        # Sequential: run in a spawned subprocess for clean CUDA isolation
        logger.info("Mode: sequential (single GPU, shared base model)")
        logger.info(f"  Roles: {args.roles}")
        p = ctx.Process(
            target=_worker_train_group,
            args=(args.gpu_offset, args.roles, args.data_dir, args.output_dir, env_vars),
        )
        p.start()
        p.join()
        if p.exitcode != 0:
            logger.error(f"Sequential training failed (exit code {p.exitcode})")
            return 1

    elif args.groups:
        n = args.groups
        if n == 1:
            p = ctx.Process(
                target=_worker_train_group,
                args=(args.gpu_offset, args.roles, args.data_dir, args.output_dir, env_vars),
            )
            p.start()
            p.join()
            if p.exitcode != 0:
                return 1
        else:
            if n == 2:
                groups = DEFAULT_GROUPS
            else:
                groups = [[] for _ in range(n)]
                for i, role in enumerate(args.roles):
                    groups[i % n].append(role)

            logger.info(f"Mode: {n} parallel groups")
            for i, g in enumerate(groups):
                logger.info(f"  GPU {args.gpu_offset + i}: {g}")

            # Launch one spawned process per group
            processes = []
            for i, group_roles in enumerate(groups):
                if not group_roles:
                    continue
                gpu_id = args.gpu_offset + i
                p = ctx.Process(
                    target=_worker_train_group,
                    args=(gpu_id, group_roles, args.data_dir, args.output_dir, env_vars),
                )
                p.start()
                processes.append((p, i, group_roles))
                logger.info(f"  Launched group {i} (pid={p.pid}) on GPU {gpu_id}: {group_roles}")

            # Wait for all processes
            failed = False
            for p, idx, roles in processes:
                p.join()
                if p.exitcode == 0:
                    logger.info(f"Group {idx} completed: {roles}")
                else:
                    logger.error(f"Group {idx} failed (exit code {p.exitcode}): {roles}")
                    failed = True

            if failed:
                return 1

    elif args.parallel:
        logger.info("Mode: parallel (one role per GPU)")
        processes = []
        for i, role_name in enumerate(args.roles):
            gpu_id = args.gpu_offset + i
            p = ctx.Process(
                target=_worker_train_single,
                args=(gpu_id, role_name, args.data_dir, args.output_dir, env_vars),
            )
            p.start()
            processes.append((p, role_name))
            logger.info(f"  Launched {role_name} (pid={p.pid}) on GPU {gpu_id}")

        failed = False
        for p, role_name in processes:
            p.join()
            if p.exitcode == 0:
                logger.info(f"Completed: {role_name}")
            else:
                logger.error(f"Failed (exit code {p.exitcode}): {role_name}")
                failed = True

        if failed:
            return 1

    logger.info("Stage A complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
