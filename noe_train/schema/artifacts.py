"""Artifact definitions for episode outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ArtifactType(str, Enum):
    PLAN = "plan"
    PATCH = "patch"
    TEST_SUITE = "test_suite"
    EXEC_TRACE = "exec_trace"
    BUG_REPORT = "bug_report"
    CODE_SLICE = "code_slice"


@dataclass()
class Artifact:
    artifact_type: ArtifactType
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    sha256: str | None = None  # set by artifact store on save
    episode_id: str | None = None
    round_idx: int = 0
