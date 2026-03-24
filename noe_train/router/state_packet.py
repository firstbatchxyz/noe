"""State packet: compact representation of episode state for router/critic input."""

from __future__ import annotations

from dataclasses import dataclass

from noe_train.schema.episode import EpisodeState


@dataclass()
class StatePacket:
    """Compact state representation for router and critic (<=512 tokens)."""

    text: str
    round_idx: int
    budget_pct: float
    compile_ok: bool
    tests_passed: int
    tests_failed: int
    repair_count: int
    num_expert_calls: int


def compile_state(episode: EpisodeState) -> StatePacket:
    """Build a state packet from current episode state."""
    lines = []

    # Header
    lines.append(f"[Episode {episode.episode_id[:8]}] R{episode.round_idx}/{episode.max_rounds}")

    # Env state
    env = episode.env
    compile_str = "OK" if env.compile_ok else "FAIL"
    lint_str = "OK" if env.lint_ok else "FAIL"
    lines.append(
        f"compile={compile_str} lint={lint_str} "
        f"pass={env.tests_passed} fail={env.tests_failed}"
    )

    if env.failing_tests:
        tests_str = ", ".join(env.failing_tests[:3])
        if len(env.failing_tests) > 3:
            tests_str += f" (+{len(env.failing_tests) - 3})"
        lines.append(f"failing: {tests_str}")

    if env.files_changed:
        files_str = ", ".join(env.files_changed[:5])
        if len(env.files_changed) > 5:
            files_str += f" (+{len(env.files_changed) - 5})"
        lines.append(f"changed: {files_str}")

    lines.append(f"repairs={episode.repair_count}/{episode.max_repairs}")

    # Expert call history
    if episode.expert_calls:
        calls = [e.value for e in episode.expert_calls]
        lines.append(f"calls: {' → '.join(calls)}")

    # Message summaries (last 3)
    recent = episode.history.messages[-3:]
    for msg in recent:
        conf_str = ""
        if "confidence" in (msg.content or {}):
            conf_str = f" conf={msg.content['confidence']:.2f}"
        lines.append(
            f"[R{msg.round_idx}] {msg.sender.value}{conf_str} | {msg.summary[:80]}"
        )

    # Budget usage
    total_budget = episode.max_rounds * 6144  # rough max
    used = episode.total_input_tokens + episode.total_gen_tokens
    budget_pct = min(used / total_budget, 1.0) if total_budget > 0 else 0.0
    lines.append(f"budget={budget_pct:.0%}")

    text = "\n".join(lines)
    return StatePacket(
        text=text,
        round_idx=episode.round_idx,
        budget_pct=budget_pct,
        compile_ok=env.compile_ok,
        tests_passed=env.tests_passed,
        tests_failed=env.tests_failed,
        repair_count=episode.repair_count,
        num_expert_calls=len(episode.expert_calls),
    )
