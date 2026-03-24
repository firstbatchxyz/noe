#!/usr/bin/env python3
"""Train Stage B: Router GRPO."""

import argparse
import logging
import sys

import torch

from noe_train.data.rl_tasks import build_rl_task_pool, filter_stage_b, load_rl_swe, load_swebench_train
from noe_train.router.model import RouterModel
from noe_train.training.stage_b import StageBConfig, train_stage_b
from noe_train.utils.logging import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Stage B: Router GRPO")
    parser.add_argument("--router-checkpoint", type=str, default=None)
    parser.add_argument("--expert-dir", type=str, default="checkpoints/stage_a")
    parser.add_argument("--output-dir", type=str, default="checkpoints/stage_b")
    parser.add_argument("--num-episodes", type=int, default=100_000)
    parser.add_argument("--device", type=str, default="cuda:4")
    args = parser.parse_args()

    # Load router
    router = RouterModel()
    if args.router_checkpoint:
        router.load_state_dict(torch.load(args.router_checkpoint))
    router.to(args.device)

    # Load task pool
    logger.info("Loading RL task pool...")
    rl_ds = load_rl_swe()
    stage_b_ds = filter_stage_b(rl_ds)
    swebench_train = load_swebench_train()
    task_pool = build_rl_task_pool(stage_b_ds, swebench_train, stage="b")

    # Load experts (frozen)
    # In practice, load from stage_a checkpoints
    experts = {}  # Placeholder: loaded from checkpoints

    config = StageBConfig(num_episodes=args.num_episodes)

    train_stage_b(
        router=router,
        experts=experts,
        task_pool=task_pool,
        orchestrator_factory=None,  # Needs to be wired up with sandbox
        output_dir=args.output_dir,
        config=config,
        device=args.device,
    )

    logger.info("Stage B complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
