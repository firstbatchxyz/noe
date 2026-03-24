"""Test harness: compile → lint → pytest → normalized results."""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

STEP_TIMEOUT = 60  # per-step timeout in seconds


@dataclass()
class HarnessResult:
    compile_ok: bool
    lint_ok: bool
    tests_passed: int
    tests_failed: int
    failing_tests: list[str]
    trace_ref: str | None = None  # artifact ref if stored
    stdout: str = ""
    stderr: str = ""


class TestHarness:
    """Runs compile → lint → pytest in a repo worktree."""

    def __init__(self, worktree_dir: str | Path, timeout: int = STEP_TIMEOUT):
        self.worktree_dir = Path(worktree_dir)
        self.timeout = timeout

    def run(self) -> HarnessResult:
        """Run full harness pipeline."""
        # Step 1: Compile check (syntax validation)
        compile_ok, compile_out = self._compile_check()

        # Step 2: Lint
        lint_ok, lint_out = self._lint_check()

        # Step 3: Pytest
        tests_passed, tests_failed, failing_tests, test_out = self._run_tests()

        return HarnessResult(
            compile_ok=compile_ok,
            lint_ok=lint_ok,
            tests_passed=tests_passed,
            tests_failed=tests_failed,
            failing_tests=failing_tests,
            stdout=test_out,
        )

    def _compile_check(self) -> tuple[bool, str]:
        """Check Python syntax validity."""
        result = self._run_cmd(
            ["python", "-m", "py_compile"]
            + [str(p) for p in self.worktree_dir.rglob("*.py") if "test" not in str(p)][:50]
        )
        return result.returncode == 0, result.stderr

    def _lint_check(self) -> tuple[bool, str]:
        """Run ruff linter."""
        result = self._run_cmd(
            ["python", "-m", "ruff", "check", "--select", "E,F", "--no-fix", str(self.worktree_dir)]
        )
        return result.returncode == 0, result.stdout

    def _run_tests(self) -> tuple[int, int, list[str], str]:
        """Run pytest, parse results."""
        result = self._run_cmd(
            ["python", "-m", "pytest", str(self.worktree_dir), "-x", "--tb=short", "-q"]
        )

        output = result.stdout + result.stderr
        passed, failed, failing = self._parse_pytest_output(output)
        return passed, failed, failing, output

    def _parse_pytest_output(self, output: str) -> tuple[int, int, list[str]]:
        """Parse pytest output for pass/fail counts and failing test names."""
        passed = 0
        failed = 0
        failing_tests = []

        for line in output.split("\n"):
            line = line.strip()
            # "X passed, Y failed" or "X passed"
            if "passed" in line or "failed" in line:
                parts = line.split(",")
                for part in parts:
                    part = part.strip()
                    if "passed" in part:
                        try:
                            passed = int(part.split()[0])
                        except (ValueError, IndexError):
                            pass
                    elif "failed" in part:
                        try:
                            failed = int(part.split()[0])
                        except (ValueError, IndexError):
                            pass
            # "FAILED test_file::test_name"
            if line.startswith("FAILED "):
                test_name = line[7:].strip()
                failing_tests.append(test_name)

        return passed, failed, failing_tests

    def _run_cmd(self, cmd: list[str]) -> subprocess.CompletedProcess:
        try:
            return subprocess.run(
                cmd,
                cwd=self.worktree_dir,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
        except subprocess.TimeoutExpired:
            return subprocess.CompletedProcess(
                cmd, returncode=-1, stdout="", stderr="timeout"
            )
        except FileNotFoundError:
            return subprocess.CompletedProcess(
                cmd, returncode=-1, stdout="", stderr="command not found"
            )
