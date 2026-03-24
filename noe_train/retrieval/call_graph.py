"""Approximate call graph from Python AST."""

from __future__ import annotations

import ast
import logging
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)


class CallGraph:
    """Approximate function-level call graph from AST analysis."""

    def __init__(self):
        # caller → set of callees
        self.calls: dict[str, set[str]] = defaultdict(set)
        # callee → set of callers
        self.callers: dict[str, set[str]] = defaultdict(set)

    def build(self, repo_dir: str | Path) -> int:
        """Build call graph from Python files. Returns edge count."""
        repo = Path(repo_dir)
        edge_count = 0

        for py_file in sorted(repo.rglob("*.py")):
            if any(part.startswith(".") for part in py_file.parts):
                continue
            try:
                source = py_file.read_text(encoding="utf-8", errors="replace")
                tree = ast.parse(source, filename=str(py_file))
            except (SyntaxError, OSError):
                continue

            rel_path = str(py_file.relative_to(repo))
            edge_count += self._extract_calls(tree, rel_path)

        return edge_count

    def callees_of(self, func_key: str) -> set[str]:
        """Get functions called by func_key."""
        return self.calls.get(func_key, set())

    def callers_of(self, func_key: str) -> set[str]:
        """Get functions that call func_key."""
        return self.callers.get(func_key, set())

    def _extract_calls(self, tree: ast.AST, file_path: str) -> int:
        edges = 0
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                caller_key = f"{file_path}:{node.name}"
                for child in ast.walk(node):
                    if isinstance(child, ast.Call):
                        callee_name = self._resolve_call_name(child)
                        if callee_name:
                            self.calls[caller_key].add(callee_name)
                            self.callers[callee_name].add(caller_key)
                            edges += 1
        return edges

    def _resolve_call_name(self, node: ast.Call) -> str | None:
        """Extract the name being called."""
        if isinstance(node.func, ast.Name):
            return node.func.id
        elif isinstance(node.func, ast.Attribute):
            return node.func.attr
        return None
