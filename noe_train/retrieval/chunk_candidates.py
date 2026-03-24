"""Chunk candidate builder: produces ~10 chunk types for expert context packing."""

from __future__ import annotations

from dataclasses import dataclass

from noe_train.experts.base import Chunk
from noe_train.schema.episode import EpisodeState


@dataclass()
class ChunkBuilder:
    """Builds chunk candidates from episode state and retrieval results."""

    def build_candidates(
        self,
        episode: EpisodeState,
        retrieved_files: list[tuple[str, str, float]] | None = None,
        repo_map: str = "",
        file_summaries: list[tuple[str, str]] | None = None,
        code_slices: list[tuple[str, str]] | None = None,
        api_docs: list[str] | None = None,
    ) -> list[Chunk]:
        """Build all available chunk candidates.

        Chunk types:
        1. issue_summary        6. previous_patch_diff
        2. repo_map             7. compile_log_summary
        3. top_file_summaries   8. failing_test_summary
        4. selected_code_slices 9. last_round_message_summaries
        5. retrieved_docs_apis
        """
        chunks = []
        idx = 0

        # 1. Issue summary
        if episode.issue_text:
            chunks.append(Chunk(
                id=f"chunk_{idx}",
                chunk_type="issue_summary",
                content=episode.issue_text[:2000],
                token_count=len(episode.issue_text.split()),
            ))
            idx += 1

        # 2. Repo map
        if repo_map:
            chunks.append(Chunk(
                id=f"chunk_{idx}",
                chunk_type="repo_map",
                content=repo_map[:1500],
                token_count=len(repo_map.split()),
            ))
            idx += 1

        # 3. Top file summaries
        if file_summaries:
            for path, summary in file_summaries[:5]:
                chunks.append(Chunk(
                    id=f"chunk_{idx}",
                    chunk_type="top_file_summaries",
                    content=f"# {path}\n{summary}",
                    token_count=len(summary.split()) + 5,
                ))
                idx += 1

        # 4. Selected code slices
        if code_slices:
            for path, code in code_slices[:10]:
                chunks.append(Chunk(
                    id=f"chunk_{idx}",
                    chunk_type="selected_code_slices",
                    content=f"# {path}\n{code}",
                    token_count=len(code.split()) + 5,
                ))
                idx += 1

        # 5. Retrieved docs/APIs
        if retrieved_files:
            for path, content, score in retrieved_files[:5]:
                chunks.append(Chunk(
                    id=f"chunk_{idx}",
                    chunk_type="retrieved_docs_apis",
                    content=f"# {path} (score={score:.2f})\n{content[:1000]}",
                    token_count=len(content[:1000].split()) + 5,
                ))
                idx += 1

        if api_docs:
            for doc in api_docs[:3]:
                chunks.append(Chunk(
                    id=f"chunk_{idx}",
                    chunk_type="retrieved_docs_apis",
                    content=doc[:500],
                    token_count=len(doc[:500].split()),
                ))
                idx += 1

        # 6. Previous patch diff
        if episode.env.current_patch_diff:
            diff = episode.env.current_patch_diff
            chunks.append(Chunk(
                id=f"chunk_{idx}",
                chunk_type="previous_patch_diff",
                content=diff[:2000],
                token_count=len(diff[:2000].split()),
            ))
            idx += 1

        # 7. Compile log summary
        # Populated from harness results stored in messages
        exec_msgs = episode.history.by_type_name("EXEC_REPORT") if hasattr(episode.history, "by_type_name") else []
        # Fall back to checking env state
        if not episode.env.compile_ok and episode.env.files_changed:
            chunks.append(Chunk(
                id=f"chunk_{idx}",
                chunk_type="compile_log_summary",
                content=f"Compile failed. Changed files: {', '.join(episode.env.files_changed)}",
                token_count=10 + len(episode.env.files_changed),
            ))
            idx += 1

        # 8. Failing test summary
        if episode.env.failing_tests:
            test_list = "\n".join(f"- {t}" for t in episode.env.failing_tests[:10])
            content = f"Failing tests ({episode.env.tests_failed}):\n{test_list}"
            chunks.append(Chunk(
                id=f"chunk_{idx}",
                chunk_type="failing_test_summary",
                content=content,
                token_count=len(content.split()),
            ))
            idx += 1

        # 9. Last round message summaries
        recent_msgs = episode.history.messages[-3:]
        if recent_msgs:
            summaries = "\n".join(
                f"[R{m.round_idx}] {m.sender.value}: {m.summary}"
                for m in recent_msgs
            )
            chunks.append(Chunk(
                id=f"chunk_{idx}",
                chunk_type="last_round_message_summaries",
                content=summaries,
                token_count=len(summaries.split()),
            ))
            idx += 1

        return chunks
