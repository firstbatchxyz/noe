"""Git-based repository state management for sandboxes."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


class RepoState:
    """Manages git worktree for a single episode."""

    def __init__(self, repo_dir: str | Path, worktree_dir: str | Path):
        self.repo_dir = Path(repo_dir)
        self.worktree_dir = Path(worktree_dir)
        self._initialized = False

    def init_worktree(self, commit: str = "HEAD") -> None:
        """Create a git worktree for this episode."""
        self.worktree_dir.mkdir(parents=True, exist_ok=True)
        self._run_git(
            ["worktree", "add", "--detach", str(self.worktree_dir), commit],
            cwd=self.repo_dir,
        )
        self._initialized = True
        logger.info(f"Created worktree at {self.worktree_dir} from {commit}")

    def apply_patch(self, diff: str) -> tuple[bool, str]:
        """Apply a unified diff. Returns (success, message)."""
        if not self._initialized:
            raise RuntimeError("Worktree not initialized")

        # First verify with --check
        check_result = self._run_git(
            ["apply", "--check", "-"],
            input_data=diff,
            cwd=self.worktree_dir,
        )
        if check_result.returncode != 0:
            return False, f"Patch check failed: {check_result.stderr}"

        # Apply
        apply_result = self._run_git(
            ["apply", "-"],
            input_data=diff,
            cwd=self.worktree_dir,
        )
        if apply_result.returncode != 0:
            return False, f"Patch apply failed: {apply_result.stderr}"

        return True, "Patch applied successfully"

    def rollback(self) -> None:
        """Reset worktree to clean state."""
        if self._initialized:
            self._run_git(["checkout", "--", "."], cwd=self.worktree_dir)
            self._run_git(["clean", "-fd"], cwd=self.worktree_dir)

    def get_diff(self) -> str:
        """Get current diff from base."""
        result = self._run_git(["diff"], cwd=self.worktree_dir)
        return result.stdout

    def cleanup(self) -> None:
        """Remove the worktree."""
        if self._initialized:
            self._run_git(
                ["worktree", "remove", "--force", str(self.worktree_dir)],
                cwd=self.repo_dir,
            )
            self._initialized = False

    def _run_git(
        self,
        args: list[str],
        cwd: Path | None = None,
        input_data: str | None = None,
    ) -> subprocess.CompletedProcess:
        cmd = ["git"] + args
        return subprocess.run(
            cmd,
            cwd=cwd or self.worktree_dir,
            capture_output=True,
            text=True,
            input=input_data,
            timeout=30,
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
        return False
