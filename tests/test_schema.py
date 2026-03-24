"""Tests for schema module."""

from noe_train.schema.budget import BUDGET_TABLE, BudgetLevel, BudgetTracker, get_budget
from noe_train.schema.messages import (
    ExpertRole,
    FinalVerdict,
    MessageHistory,
    MessageType,
    PlanContent,
    TypedMessage,
)
from noe_train.schema.episode import EnvState, EpisodeState, EpisodeStatus
from noe_train.schema.artifacts import Artifact, ArtifactType


def test_budget_table_complete():
    assert len(BUDGET_TABLE) == 5
    for level in BudgetLevel:
        budget = get_budget(level)
        assert budget.level == level
        assert budget.max_input_tokens > 0
        assert budget.max_gen_tokens > 0


def test_budget_b4_coder_only():
    b4 = get_budget(BudgetLevel.B4)
    assert b4.max_input_tokens == 6144
    assert b4.max_gen_tokens == 2048
    assert b4.max_tools == 3


def test_budget_tracker():
    budget = get_budget(BudgetLevel.B2)
    tracker = BudgetTracker(budget)

    assert not tracker.soft_exceeded
    assert not tracker.hard_exceeded
    assert tracker.can_generate(100)
    assert tracker.can_use_tool()

    tracker.gen_tokens_used = budget.max_gen_tokens
    assert tracker.soft_exceeded
    assert not tracker.hard_exceeded

    tracker.gen_tokens_used = 2 * budget.max_gen_tokens
    assert tracker.hard_exceeded


def test_message_types():
    assert len(MessageType) == 8
    assert MessageType.PLAN.value == "PLAN"
    assert MessageType.PATCH_HUNK.value == "PATCH_HUNK"


def test_plan_content():
    plan = PlanContent(
        files_to_touch=["a.py", "b.py"],
        invariants=["tests pass"],
        risks=["might break c.py"],
        strategy="Fix the bug in a.py line 10",
    )
    d = plan.to_dict()
    assert d["files_to_touch"] == ["a.py", "b.py"]
    assert d["strategy"] == "Fix the bug in a.py line 10"


def test_message_history():
    history = MessageHistory()
    msg = TypedMessage(
        msg_type=MessageType.PLAN,
        sender=ExpertRole.PLANNER,
        round_idx=1,
        content={"strategy": "fix it"},
        summary="Plan to fix a.py",
    )
    history.add(msg)

    assert len(history.messages) == 1
    assert history.last() is msg
    assert len(history.by_type(MessageType.PLAN)) == 1
    assert len(history.by_sender(ExpertRole.PLANNER)) == 1
    assert len(history.by_round(1)) == 1
    assert len(history.by_round(2)) == 0


def test_episode_state():
    episode = EpisodeState(
        episode_id="test-123",
        task_id="task-1",
        repo="django/django",
        instance_id="django__django-12345",
        issue_text="Some bug",
        repo_map="",
    )

    assert not episode.is_terminal()
    assert episode.can_repair()
    assert episode.round_idx == 0

    episode.advance_round()
    assert episode.round_idx == 1

    episode.record_call(ExpertRole.PLANNER, 100, 50)
    assert len(episode.expert_calls) == 1
    assert episode.total_input_tokens == 100


def test_episode_terminal_states():
    episode = EpisodeState(
        episode_id="test",
        task_id="t",
        repo="r",
        instance_id="i",
        issue_text="i",
        repo_map="",
    )
    episode.status = EpisodeStatus.ACCEPTED
    assert episode.is_terminal()


def test_artifact():
    art = Artifact(
        artifact_type=ArtifactType.PATCH,
        content="--- a/file.py\n+++ b/file.py\n@@ -1,1 +1,1 @@\n-old\n+new",
    )
    assert art.artifact_type == ArtifactType.PATCH
    assert art.sha256 is None


def test_env_state_defaults():
    env = EnvState()
    assert not env.compile_ok
    assert env.tests_passed == 0
    assert env.failing_tests == []
