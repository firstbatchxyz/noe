"""Tests for checkpoint manager."""

import tempfile

import torch
import torch.nn as nn

from noe_train.utils.checkpoint import CheckpointManager


def test_save_and_load():
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = CheckpointManager(tmpdir)

        model = nn.Linear(10, 5)
        path = mgr.save(
            step=100,
            stage="stage_a",
            models={"test_model": model},
            metrics={"loss": 0.5},
        )

        meta = mgr.load(path)
        assert meta["step"] == 100
        assert meta["stage"] == "stage_a"
        assert meta["metrics"]["loss"] == 0.5


def test_best_tracking():
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = CheckpointManager(tmpdir)
        model = nn.Linear(10, 5)

        mgr.save(step=100, stage="a", models={"m": model}, metrics={"resolve_rate": 0.10})
        mgr.save(step=200, stage="a", models={"m": model}, metrics={"resolve_rate": 0.25})
        mgr.save(step=300, stage="a", models={"m": model}, metrics={"resolve_rate": 0.15})

        assert mgr.best_path is not None
        assert "step_000200" in mgr.best_path
