"""AST-based symbol index for Python repositories."""

from __future__ import annotations

import ast
import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass()
class Symbol:
    name: str
    kind: str  # "class", "function", "method", "import"
    file_path: str
    line: int
    end_line: int
    parent: str | None = None  # class name for methods


@dataclass()
class ImportEdge:
    source_file: str
    imported_module: str
    imported_names: list[str]


class SymbolIndex:
    """Python AST-based symbol table and import graph."""

    def __init__(self):
        self.symbols: dict[str, list[Symbol]] = {}  # name → symbols
        self.file_symbols: dict[str, list[Symbol]] = {}  # file → symbols
        self.imports: list[ImportEdge] = []

    def build(self, repo_dir: str | Path) -> int:
        """Index all Python files. Returns number of symbols found."""
        repo = Path(repo_dir)
        count = 0

        for py_file in sorted(repo.rglob("*.py")):
            if any(part.startswith(".") for part in py_file.parts):
                continue
            try:
                source = py_file.read_text(encoding="utf-8", errors="replace")
                tree = ast.parse(source, filename=str(py_file))
            except (SyntaxError, OSError):
                continue

            rel_path = str(py_file.relative_to(repo))
            file_syms = self._extract_symbols(tree, rel_path)
            self.file_symbols[rel_path] = file_syms

            for sym in file_syms:
                self.symbols.setdefault(sym.name, []).append(sym)
                count += 1

            # Extract imports
            self._extract_imports(tree, rel_path)

        return count

    def lookup(self, name: str) -> list[Symbol]:
        """Find all symbols matching a name."""
        return self.symbols.get(name, [])

    def symbols_in_file(self, file_path: str) -> list[Symbol]:
        return self.file_symbols.get(file_path, [])

    def expand_1hop(self, file_path: str) -> list[str]:
        """Get files 1-hop connected via imports."""
        connected = set()
        for edge in self.imports:
            if edge.source_file == file_path:
                # Find files that define the imported names
                for name in edge.imported_names:
                    for sym in self.symbols.get(name, []):
                        connected.add(sym.file_path)
            # Also files that import from this file
            for name in edge.imported_names:
                file_syms = self.file_symbols.get(file_path, [])
                if any(s.name == name for s in file_syms):
                    connected.add(edge.source_file)
        connected.discard(file_path)
        return sorted(connected)

    def _extract_symbols(self, tree: ast.AST, file_path: str) -> list[Symbol]:
        symbols = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                symbols.append(Symbol(
                    name=node.name,
                    kind="class",
                    file_path=file_path,
                    line=node.lineno,
                    end_line=node.end_lineno or node.lineno,
                ))
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        symbols.append(Symbol(
                            name=item.name,
                            kind="method",
                            file_path=file_path,
                            line=item.lineno,
                            end_line=item.end_lineno or item.lineno,
                            parent=node.name,
                        ))
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Only top-level functions (not methods)
                if not any(
                    isinstance(parent, ast.ClassDef)
                    for parent in ast.walk(tree)
                    if hasattr(parent, "body") and node in getattr(parent, "body", [])
                ):
                    symbols.append(Symbol(
                        name=node.name,
                        kind="function",
                        file_path=file_path,
                        line=node.lineno,
                        end_line=node.end_lineno or node.lineno,
                    ))
        return symbols

    def _extract_imports(self, tree: ast.AST, file_path: str) -> None:
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    self.imports.append(ImportEdge(
                        source_file=file_path,
                        imported_module=alias.name,
                        imported_names=[alias.asname or alias.name.split(".")[-1]],
                    ))
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                names = [alias.asname or alias.name for alias in node.names]
                self.imports.append(ImportEdge(
                    source_file=file_path,
                    imported_module=module,
                    imported_names=names,
                ))
