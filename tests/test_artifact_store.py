"""Tests for artifact store."""

import tempfile
from pathlib import Path

from noe_train.artifact_store.store import ArtifactStore
from noe_train.schema.artifacts import Artifact, ArtifactType


def test_save_and_load():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = ArtifactStore(tmpdir)
        artifact = Artifact(
            artifact_type=ArtifactType.PATCH,
            content="--- a/f.py\n+++ b/f.py\n@@ -1 +1 @@\n-x\n+y",
        )

        sha = store.save(artifact, "ep-001")
        assert sha is not None
        assert len(sha) == 64  # SHA-256 hex

        content = store.load("ep-001", sha)
        assert content == artifact.content


def test_manifest():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = ArtifactStore(tmpdir)

        a1 = Artifact(artifact_type=ArtifactType.PLAN, content="plan content")
        a2 = Artifact(artifact_type=ArtifactType.PATCH, content="patch content")

        sha1 = store.save(a1, "ep-002")
        sha2 = store.save(a2, "ep-002")

        artifacts = store.list_artifacts("ep-002")
        assert sha1 in artifacts
        assert sha2 in artifacts


def test_dedup():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = ArtifactStore(tmpdir)
        content = "same content"

        a1 = Artifact(artifact_type=ArtifactType.PLAN, content=content)
        a2 = Artifact(artifact_type=ArtifactType.PLAN, content=content)

        sha1 = store.save(a1, "ep-003")
        sha2 = store.save(a2, "ep-003")

        assert sha1 == sha2  # same content → same hash


def test_metadata():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = ArtifactStore(tmpdir)
        artifact = Artifact(
            artifact_type=ArtifactType.PATCH,
            content="patch",
            metadata={"files": ["a.py"]},
        )

        sha = store.save(artifact, "ep-004")
        meta = store.load_metadata("ep-004", sha)

        assert meta is not None
        assert meta["type"] == "patch"
        assert meta["metadata"]["files"] == ["a.py"]
