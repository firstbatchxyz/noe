"""Episode management: create, track, and finalize episodes."""

from __future__ import annotations

import uuid
from typing import Any

from noe_train.schema.episode import EnvState, EpisodeState, EpisodeStatus
from noe_train.schema.messages import FinalVerdict, MessageHistory


def create_episode(
    task_id: str,
    repo: str,
    instance_id: str,
    issue_text: str,
    repo_map: str = "",
    max_rounds: int = 5,
    max_repairs: int = 2,
) -> EpisodeState:
    """Create a new episode state."""
    return EpisodeState(
        episode_id=str(uuid.uuid4()),
        task_id=task_id,
        repo=repo,
        instance_id=instance_id,
        issue_text=issue_text,
        repo_map=repo_map,
        max_rounds=max_rounds,
        max_repairs=max_repairs,
    )


def finalize_episode(
    episode: EpisodeState,
    verdict: FinalVerdict,
    reason: str = "",
) -> None:
    """Finalize an episode with a verdict."""
    if verdict == FinalVerdict.ACCEPT:
        episode.status = EpisodeStatus.ACCEPTED
    elif verdict == FinalVerdict.ROLLBACK:
        episode.status = EpisodeStatus.ROLLED_BACK
    else:
        episode.status = EpisodeStatus.ERROR


def timeout_episode(episode: EpisodeState) -> None:
    """Mark episode as timed out."""
    episode.status = EpisodeStatus.TIMEOUT


def episode_summary(episode: EpisodeState) -> dict[str, Any]:
    """Produce a summary dict for logging/checkpointing."""
    return {
        "episode_id": episode.episode_id,
        "task_id": episode.task_id,
        "instance_id": episode.instance_id,
        "status": episode.status.value,
        "rounds": episode.round_idx,
        "expert_calls": [e.value for e in episode.expert_calls],
        "total_input_tokens": episode.total_input_tokens,
        "total_gen_tokens": episode.total_gen_tokens,
        "total_tool_calls": episode.total_tool_calls,
        "repair_count": episode.repair_count,
        "compile_ok": episode.env.compile_ok,
        "tests_passed": episode.env.tests_passed,
        "tests_failed": episode.env.tests_failed,
    }
