"""Reward function for RL stages.

Phi(s) = 0.50 * test_pass_rate + 0.20 * compile_ok
       + 0.10 * lint_ok + 0.10 * coverage_gain + 0.10 * verifier_confidence

r_t = Phi(s_{t+1}) - Phi(s_t) - cost_penalties
Terminal: +1.0 if hidden suite passes.

Cost annealing: 0 → target over 20K episodes.
lambda_call=0.02, lambda_round=0.01, lambda_msg=5e-5/token
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass()
class RewardConfig:
    # Phi weights
    w_test_pass: float = 0.50
    w_compile: float = 0.20
    w_lint: float = 0.10
    w_coverage: float = 0.10
    w_verifier: float = 0.10

    # Cost penalties
    lambda_call: float = 0.02
    lambda_round: float = 0.01
    lambda_msg: float = 5e-5

    # Terminal bonus
    terminal_bonus: float = 1.0

    # Cost annealing
    anneal_episodes: int = 20_000

    # Reward split: shared vs role
    shared_fraction: float = 0.90
    role_bonus_fraction: float = 0.10


def compute_phi(
    state: dict[str, Any],
    config: RewardConfig | None = None,
) -> float:
    """Compute state value Phi(s)."""
    cfg = config or RewardConfig()

    total_tests = state.get("tests_passed", 0) + state.get("tests_failed", 0)
    test_pass_rate = state["tests_passed"] / total_tests if total_tests > 0 else 0.0
    compile_ok = float(state.get("compile_ok", False))
    lint_ok = float(state.get("lint_ok", False))
    coverage_gain = state.get("coverage_gain", 0.0)
    verifier_confidence = state.get("verifier_confidence", 0.0)

    return (
        cfg.w_test_pass * test_pass_rate
        + cfg.w_compile * compile_ok
        + cfg.w_lint * lint_ok
        + cfg.w_coverage * coverage_gain
        + cfg.w_verifier * verifier_confidence
    )


def compute_step_reward(
    state_before: dict[str, Any],
    state_after: dict[str, Any],
    n_calls: int = 0,
    n_msg_tokens: int = 0,
    episode_idx: int = 0,
    config: RewardConfig | None = None,
) -> float:
    """Compute step reward r_t = Phi(s') - Phi(s) - costs."""
    cfg = config or RewardConfig()

    phi_before = compute_phi(state_before, cfg)
    phi_after = compute_phi(state_after, cfg)
    delta_phi = phi_after - phi_before

    # Cost annealing: scale from 0 → 1 over anneal_episodes
    anneal_factor = min(episode_idx / cfg.anneal_episodes, 1.0) if cfg.anneal_episodes > 0 else 1.0

    cost = anneal_factor * (
        cfg.lambda_call * n_calls
        + cfg.lambda_round
        + cfg.lambda_msg * n_msg_tokens
    )

    return delta_phi - cost


def compute_terminal_reward(
    hidden_suite_passes: bool,
    config: RewardConfig | None = None,
) -> float:
    """Terminal reward: +1.0 if hidden test suite passes."""
    cfg = config or RewardConfig()
    return cfg.terminal_bonus if hidden_suite_passes else 0.0


def compute_role_bonus(
    role: str,
    episode_info: dict[str, Any],
    config: RewardConfig | None = None,
) -> float:
    """Per-role bonus (training only, uses gold metadata).

    Tester: discriminative test (fails buggy, passes gold, covers changed line)
    Debugger: file/symbol hit@3 vs gold diff
    Router: fewer calls, same success
    """
    cfg = config or RewardConfig()

    if role == "tester":
        # Tester bonus: did generated test discriminate?
        if episode_info.get("test_discriminative", False):
            return 0.1
        return 0.0

    elif role == "debugger":
        # Debugger bonus: file localization accuracy
        gold_files = set(episode_info.get("gold_files", []))
        predicted = episode_info.get("predicted_files", [])[:3]
        hits = sum(1 for f in predicted if f in gold_files)
        return 0.1 * (hits / max(len(gold_files), 1))

    elif role == "router":
        # Router bonus: efficiency
        n_calls = episode_info.get("total_calls", 5)
        return 0.05 * max(0, (5 - n_calls) / 5)

    return 0.0
