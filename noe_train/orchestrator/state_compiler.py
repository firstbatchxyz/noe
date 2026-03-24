"""State compiler: deterministic template for router/critic input."""

from __future__ import annotations

from noe_train.schema.episode import EpisodeState
from noe_train.schema.messages import ExpertRole


def compile_state_text(episode: EpisodeState, max_tokens: int = 512) -> str:
    """Compile episode state into a deterministic text summary for router/critic.

    Format: [R{round}] {expert} conf={conf:.2f} | {key_stats} | budget={pct}%
    Capped at max_tokens (rough word count).
    """
    lines = []

    # Header
    lines.append(f"task={episode.instance_id} round={episode.round_idx}/{episode.max_rounds}")

    # Environment state
    env = episode.env
    c = "OK" if env.compile_ok else "FAIL"
    l = "OK" if env.lint_ok else "FAIL"
    lines.append(f"compile={c} lint={l} pass={env.tests_passed} fail={env.tests_failed}")

    # Failing tests (truncated)
    if env.failing_tests:
        tests = env.failing_tests[:3]
        extra = f" +{len(env.failing_tests) - 3}" if len(env.failing_tests) > 3 else ""
        lines.append(f"failing=[{', '.join(tests)}{extra}]")

    # Files changed
    if env.files_changed:
        files = env.files_changed[:5]
        extra = f" +{len(env.files_changed) - 5}" if len(env.files_changed) > 5 else ""
        lines.append(f"changed=[{', '.join(files)}{extra}]")

    # Repair status
    lines.append(f"repairs={episode.repair_count}/{episode.max_repairs}")

    # Expert call trace
    if episode.expert_calls:
        trace = " → ".join(e.value for e in episode.expert_calls[-6:])
        lines.append(f"trace=[{trace}]")

    # Message summaries (compact, recent first)
    for msg in episode.history.messages[-4:]:
        conf_str = ""
        if isinstance(msg.content, dict) and "confidence" in msg.content:
            conf_str = f" conf={msg.content['confidence']:.2f}"
        lines.append(
            f"[R{msg.round_idx}] {msg.sender.value}{conf_str} | {msg.summary[:60]}"
        )

    # Budget usage
    total_budget = episode.max_rounds * 6144
    used = episode.total_input_tokens + episode.total_gen_tokens
    budget_pct = min(used / total_budget * 100, 100) if total_budget > 0 else 0
    lines.append(f"budget={budget_pct:.0f}%")

    # Truncate to max_tokens
    text = "\n".join(lines)
    words = text.split()
    if len(words) > max_tokens:
        text = " ".join(words[:max_tokens])

    return text
