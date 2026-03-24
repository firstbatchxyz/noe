"""BM25 retrieval over repository Python files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

@dataclass()
class RetrievedDoc:
    file_path: str
    score: float
    content: str
    start_line: int = 0
    end_line: int = 0


class BM25Index:
    """BM25 index over Python files in a repository."""

    def __init__(self):
        self._corpus: list[str] = []
        self._file_paths: list[str] = []
        self._contents: list[str] = []
        self._index = None

    def build(self, repo_dir: str | Path) -> int:
        """Index all Python files in repo. Returns number of documents."""
        repo = Path(repo_dir)
        self._corpus = []
        self._file_paths = []
        self._contents = []

        for py_file in sorted(repo.rglob("*.py")):
            if any(part.startswith(".") for part in py_file.parts):
                continue
            try:
                content = py_file.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            rel_path = str(py_file.relative_to(repo))
            tokens = content.lower().split()
            self._corpus.append(tokens)
            self._file_paths.append(rel_path)
            self._contents.append(content)

        if self._corpus:
            from rank_bm25 import BM25Okapi
            self._index = BM25Okapi(self._corpus)
        return len(self._corpus)

    def search(self, query: str, top_k: int = 10) -> list[RetrievedDoc]:
        """Search for relevant files."""
        if self._index is None or not self._corpus:
            return []

        tokens = query.lower().split()
        scores = self._index.get_scores(tokens)

        # Get top-k indices
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results = []
        for idx in ranked:
            if scores[idx] <= 0:
                break
            results.append(
                RetrievedDoc(
                    file_path=self._file_paths[idx],
                    score=float(scores[idx]),
                    content=self._contents[idx],
                )
            )
        return results
