"""Orchestrator module."""

from noe_train.orchestrator.episode import create_episode, episode_summary, finalize_episode
from noe_train.orchestrator.state_compiler import compile_state_text


def __getattr__(name):
    if name == "Orchestrator":
        from noe_train.orchestrator.loop import Orchestrator
        return Orchestrator
    raise AttributeError(f"module 'noe_train.orchestrator' has no attribute {name!r}")


__all__ = [
    "Orchestrator",
    "compile_state_text",
    "create_episode",
    "episode_summary",
    "finalize_episode",
]
