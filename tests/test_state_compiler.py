"""Tests for state compiler."""

from noe_train.orchestrator.state_compiler import compile_state_text
from noe_train.schema.episode import EnvState, EpisodeState
from noe_train.schema.messages import ExpertRole, MessageHistory


def test_compile_basic():
    episode = EpisodeState(
        episode_id="test-001",
        task_id="task-1",
        repo="django/django",
        instance_id="django__django-12345",
        issue_text="Bug in views",
        repo_map="",
    )

    text = compile_state_text(episode)
    assert "django__django-12345" in text
    assert "round=0" in text
    assert "compile=FAIL" in text  # default is False


def test_compile_with_progress():
    episode = EpisodeState(
        episode_id="test-002",
        task_id="task-2",
        repo="django/django",
        instance_id="django__django-99999",
        issue_text="Bug",
        repo_map="",
    )
    episode.round_idx = 2
    episode.env.compile_ok = True
    episode.env.lint_ok = True
    episode.env.tests_passed = 5
    episode.env.tests_failed = 2
    episode.env.failing_tests = ["test_a", "test_b"]
    episode.expert_calls = [ExpertRole.PLANNER, ExpertRole.CODER]
    episode.repair_count = 1

    text = compile_state_text(episode)
    assert "compile=OK" in text
    assert "pass=5" in text
    assert "fail=2" in text
    assert "repairs=1" in text
    assert "planner" in text


def test_compile_max_tokens():
    episode = EpisodeState(
        episode_id="test-003",
        task_id="task-3",
        repo="repo",
        instance_id="inst",
        issue_text="x" * 10000,
        repo_map="",
    )
    text = compile_state_text(episode, max_tokens=50)
    words = text.split()
    assert len(words) <= 50
