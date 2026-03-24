"""Tests for reward function."""

from noe_train.training.reward import (
    RewardConfig,
    compute_phi,
    compute_role_bonus,
    compute_step_reward,
    compute_terminal_reward,
)


def test_phi_all_pass():
    state = {
        "tests_passed": 10,
        "tests_failed": 0,
        "compile_ok": True,
        "lint_ok": True,
        "coverage_gain": 0.5,
        "verifier_confidence": 0.8,
    }
    phi = compute_phi(state)
    expected = 0.50 * 1.0 + 0.20 * 1.0 + 0.10 * 1.0 + 0.10 * 0.5 + 0.10 * 0.8
    assert abs(phi - expected) < 1e-6


def test_phi_all_fail():
    state = {
        "tests_passed": 0,
        "tests_failed": 10,
        "compile_ok": False,
        "lint_ok": False,
        "coverage_gain": 0.0,
        "verifier_confidence": 0.0,
    }
    phi = compute_phi(state)
    assert phi == 0.0


def test_step_reward_improvement():
    before = {"tests_passed": 0, "tests_failed": 5, "compile_ok": False, "lint_ok": False}
    after = {"tests_passed": 3, "tests_failed": 2, "compile_ok": True, "lint_ok": True}

    reward = compute_step_reward(before, after, episode_idx=0)
    assert reward > 0  # improvement should give positive reward


def test_cost_annealing():
    before = {"tests_passed": 5, "tests_failed": 0, "compile_ok": True, "lint_ok": True}
    after = {"tests_passed": 5, "tests_failed": 0, "compile_ok": True, "lint_ok": True}

    # Early episode: no cost
    r_early = compute_step_reward(before, after, n_calls=3, episode_idx=0)
    # Late episode: full cost
    r_late = compute_step_reward(before, after, n_calls=3, episode_idx=50000)

    assert r_early >= r_late  # costs should reduce reward


def test_terminal_reward():
    assert compute_terminal_reward(True) == 1.0
    assert compute_terminal_reward(False) == 0.0


def test_role_bonus_tester():
    bonus = compute_role_bonus("tester", {"test_discriminative": True})
    assert bonus == 0.1

    bonus = compute_role_bonus("tester", {"test_discriminative": False})
    assert bonus == 0.0


def test_role_bonus_debugger():
    bonus = compute_role_bonus(
        "debugger",
        {"gold_files": ["a.py", "b.py"], "predicted_files": ["a.py", "c.py", "d.py"]},
    )
    assert bonus > 0  # at least one hit
