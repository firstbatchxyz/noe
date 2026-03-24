#!/usr/bin/env python3
"""Run a single episode for testing/debugging the orchestrator loop."""

import argparse
import json
import logging
import sys
from pathlib import Path

import torch

from noe_train.artifact_store.store import ArtifactStore
from noe_train.orchestrator.episode import create_episode, episode_summary
from noe_train.orchestrator.loop import Orchestrator
from noe_train.retrieval.chunk_candidates import ChunkBuilder
from noe_train.router.model import RouterModel
from noe_train.utils.logging import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run a single NoE episode")
    parser.add_argument("--task-file", type=str, required=True, help="JSON task file")
    parser.add_argument("--expert-dir", type=str, default="checkpoints/stage_a")
    parser.add_argument("--router-checkpoint", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="episodes")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max-rounds", type=int, default=5)
    args = parser.parse_args()

    # Load task
    with open(args.task_file) as f:
        task = json.load(f)

    logger.info(f"Running episode for: {task.get('instance_id', 'unknown')}")

    # Create episode
    episode = create_episode(
        task_id=task.get("instance_id", "test"),
        repo=task.get("repo", ""),
        instance_id=task.get("instance_id", "test"),
        issue_text=task.get("problem_statement", task.get("issue", "")),
        repo_map=task.get("repo_map", ""),
        max_rounds=args.max_rounds,
    )

    # Initialize components
    router = RouterModel()
    if args.router_checkpoint:
        router.load_state_dict(torch.load(args.router_checkpoint, map_location=args.device))
    router.to(args.device)
    router.eval()

    artifact_store = ArtifactStore(Path(args.output_dir) / "artifacts")
    chunk_builder = ChunkBuilder()

    # Load experts (placeholder — would load from checkpoints in production)
    experts = {}

    orchestrator = Orchestrator(
        experts=experts,
        router=router,
        chunk_builder=chunk_builder,
        artifact_store=artifact_store,
        device=args.device,
    )

    # Note: In a real run, we'd also need:
    # - RepoState (git worktree)
    # - TestHarness
    # - PatchAssembler
    # For now, log the episode setup
    logger.info(f"Episode created: {episode.episode_id}")
    logger.info(f"Task: {episode.instance_id}")
    logger.info(f"Max rounds: {episode.max_rounds}")

    # Save episode info
    output_path = Path(args.output_dir) / f"{episode.episode_id}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(episode_summary(episode), f, indent=2)

    logger.info(f"Episode info saved to {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
