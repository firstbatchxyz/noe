#!/usr/bin/env python3
"""Train Stage C: Team RL."""

import argparse
import logging
import sys

import torch

from noe_train.data.rl_tasks import build_rl_task_pool, load_rl_swe, load_swebench_train
from noe_train.router.model import RouterModel
from noe_train.training.stage_c import StageCConfig, train_stage_c
from noe_train.utils.logging import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Stage C: Team RL")
    parser.add_argument("--router-checkpoint", type=str, default="checkpoints/stage_b/router_final/model.pt")
    parser.add_argument("--expert-dir", type=str, default="checkpoints/stage_a")
    parser.add_argument("--output-dir", type=str, default="checkpoints/stage_c")
    parser.add_argument("--num-episodes", type=int, default=30_000)
    parser.add_argument("--device", type=str, default="cuda:4")
    args = parser.parse_args()

    # Load router
    router = RouterModel()
    router.load_state_dict(torch.load(args.router_checkpoint))
    router.to(args.device)

    # Load experts from Stage A
    experts = {}  # Placeholder: loaded from checkpoints

    # Load full task pool
    logger.info("Loading RL task pool...")
    rl_ds = load_rl_swe()
    swebench_train = load_swebench_train()
    task_pool = build_rl_task_pool(rl_ds, swebench_train, stage="c")

    config = StageCConfig(num_episodes=args.num_episodes)

    train_stage_c(
        router=router,
        experts=experts,
        task_pool=task_pool,
        orchestrator_factory=None,  # Needs sandbox wiring
        output_dir=args.output_dir,
        config=config,
        device=args.device,
    )

    logger.info("Stage C complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
