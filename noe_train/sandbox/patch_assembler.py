"""Transactional patch assembler: PATCH_PLAN → PATCH_HUNK[] → PATCH_DONE.

Invalid hunk → reject entire file, emit NACK. No silent partial application.
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass()
class PatchNACK:
    """Structured rejection for invalid patches."""

    file_path: str
    hunk_idx: int
    error: str
    suggestion: str


@dataclass()
class PatchAssembly:
    """Accumulates hunks and produces a unified diff."""

    files: list[str] = field(default_factory=list)
    hunks: dict[str, list[str]] = field(default_factory=dict)  # file → list of hunk diffs
    finalized: bool = False
    nacks: list[PatchNACK] = field(default_factory=list)

    def add_hunk(self, file_path: str, hunk_idx: int, diff: str) -> None:
        if self.finalized:
            raise RuntimeError("Patch already finalized")
        if file_path not in self.hunks:
            self.hunks[file_path] = []
            self.files.append(file_path)
        self.hunks[file_path].append(diff)

    def to_unified_diff(self) -> str:
        """Combine all hunks into a single unified diff."""
        parts = []
        for file_path in self.files:
            file_hunks = self.hunks.get(file_path, [])
            for hunk in file_hunks:
                parts.append(hunk)
        return "\n".join(parts)


class PatchAssembler:
    """Transactional patch assembly + verification."""

    def __init__(self, worktree_dir: str | Path):
        self.worktree_dir = Path(worktree_dir)

    def assemble_and_verify(self, assembly: PatchAssembly) -> tuple[bool, str, list[PatchNACK]]:
        """Verify the assembled patch. Returns (ok, unified_diff, nacks).

        Transactional: if any file's patch is invalid, reject the entire file
        and produce a NACK with error details.
        """
        nacks = []
        valid_files = []

        for file_path in assembly.files:
            file_hunks = assembly.hunks.get(file_path, [])
            file_diff = "\n".join(file_hunks)

            # Verify this file's patch
            ok, error = self._verify_patch(file_diff)
            if ok:
                valid_files.append(file_path)
            else:
                nack = PatchNACK(
                    file_path=file_path,
                    hunk_idx=-1,  # whole file rejected
                    error=error,
                    suggestion=f"Regenerate patch for {file_path}: {error}",
                )
                nacks.append(nack)
                logger.warning(f"NACK for {file_path}: {error}")

        if nacks:
            # Transactional: reject entire patch if any file fails
            assembly.nacks = nacks
            return False, "", nacks

        unified = assembly.to_unified_diff()

        # Final verification of combined diff
        ok, error = self._verify_patch(unified)
        if not ok:
            nack = PatchNACK(
                file_path="<combined>",
                hunk_idx=-1,
                error=error,
                suggestion="Combined patch verification failed, regenerate all hunks",
            )
            return False, unified, [nack]

        assembly.finalized = True
        return True, unified, []

    def apply_single_shot(self, diff: str) -> tuple[bool, str, list[PatchNACK]]:
        """Accept a single-shot unified diff (small patches)."""
        ok, error = self._verify_patch(diff)
        if not ok:
            nack = PatchNACK(
                file_path="<single_shot>",
                hunk_idx=-1,
                error=error,
                suggestion=f"Patch verification failed: {error}",
            )
            return False, diff, [nack]
        return True, diff, []

    def _verify_patch(self, diff: str) -> tuple[bool, str]:
        """Run git apply --check to verify a patch."""
        try:
            result = subprocess.run(
                ["git", "apply", "--check", "-"],
                input=diff,
                cwd=self.worktree_dir,
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                return True, ""
            return False, result.stderr.strip()
        except subprocess.TimeoutExpired:
            return False, "Patch verification timed out"
        except Exception as e:
            return False, str(e)
