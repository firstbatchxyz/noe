"""Content-addressed artifact store with per-episode directories."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from noe_train.schema.artifacts import Artifact


class ArtifactStore:
    """SHA-256 content-addressed artifact store."""

    def __init__(self, root_dir: str | Path):
        self.root = Path(root_dir)
        self.root.mkdir(parents=True, exist_ok=True)

    def _episode_dir(self, episode_id: str) -> Path:
        d = self.root / episode_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def save(self, artifact: Artifact, episode_id: str) -> str:
        """Save artifact, return SHA-256 hash."""
        content_bytes = artifact.content.encode("utf-8")
        sha = hashlib.sha256(content_bytes).hexdigest()
        artifact.sha256 = sha
        artifact.episode_id = episode_id

        ep_dir = self._episode_dir(episode_id)
        artifact_path = ep_dir / f"{sha}.artifact"

        if not artifact_path.exists():
            artifact_path.write_text(artifact.content, encoding="utf-8")
            # Write metadata sidecar
            meta_path = ep_dir / f"{sha}.meta.json"
            meta = {
                "sha256": sha,
                "type": artifact.artifact_type.value,
                "round_idx": artifact.round_idx,
                "metadata": artifact.metadata,
            }
            meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

        # Update manifest
        self._update_manifest(episode_id, sha, artifact.artifact_type.value)
        return sha

    def load(self, episode_id: str, sha: str) -> str | None:
        """Load artifact content by SHA-256."""
        artifact_path = self._episode_dir(episode_id) / f"{sha}.artifact"
        if artifact_path.exists():
            return artifact_path.read_text(encoding="utf-8")
        return None

    def load_metadata(self, episode_id: str, sha: str) -> dict[str, Any] | None:
        meta_path = self._episode_dir(episode_id) / f"{sha}.meta.json"
        if meta_path.exists():
            return json.loads(meta_path.read_text(encoding="utf-8"))
        return None

    def list_artifacts(self, episode_id: str) -> list[str]:
        """List all artifact SHAs for an episode."""
        manifest = self._load_manifest(episode_id)
        return [entry["sha256"] for entry in manifest.get("artifacts", [])]

    def _update_manifest(self, episode_id: str, sha: str, artifact_type: str) -> None:
        manifest = self._load_manifest(episode_id)
        artifacts = manifest.get("artifacts", [])
        if not any(a["sha256"] == sha for a in artifacts):
            artifacts.append({"sha256": sha, "type": artifact_type})
        manifest["artifacts"] = artifacts
        manifest_path = self._episode_dir(episode_id) / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    def _load_manifest(self, episode_id: str) -> dict:
        manifest_path = self._episode_dir(episode_id) / "manifest.json"
        if manifest_path.exists():
            return json.loads(manifest_path.read_text(encoding="utf-8"))
        return {"episode_id": episode_id, "artifacts": []}
