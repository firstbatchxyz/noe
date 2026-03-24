"""Orchestrator loop: R1→R2→R3→R4 with failure handling.

R1: Router → Planner → PLAN
R2: Router → Coder → PATCH
R3: Router → Tester + Debugger (parallel) → EXEC_REPORT + BUG_REPORT
R4: Router → accept / Coder repair / rollback
(R5: optional second repair)
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from noe_train.artifact_store.store import ArtifactStore
from noe_train.experts.base import BaseExpert, Chunk, ExpertOutput
from noe_train.orchestrator.episode import finalize_episode, timeout_episode
from noe_train.orchestrator.state_compiler import compile_state_text
from noe_train.retrieval.chunk_candidates import ChunkBuilder
from noe_train.router.model import RouterModel
from noe_train.router.packer import greedy_knapsack
from noe_train.sandbox.harness import HarnessResult, TestHarness
from noe_train.sandbox.patch_assembler import PatchAssembler
from noe_train.sandbox.repo_state import RepoState
from noe_train.schema.budget import BudgetLevel, get_budget
from noe_train.schema.episode import EpisodeState, EpisodeStatus
from noe_train.schema.messages import (
    ExecReportContent,
    ExpertRole,
    FinalContent,
    FinalVerdict,
    MessageType,
    TypedMessage,
)

logger = logging.getLogger(__name__)

# Thresholds
STOP_THRESHOLD = 0.5
MIN_EXPERT_CONFIDENCE = 0.3
MAX_SCHEMA_RETRIES = 1


class Orchestrator:
    """Manages the expert routing loop for a single episode."""

    def __init__(
        self,
        experts: dict[ExpertRole, BaseExpert],
        router: RouterModel,
        chunk_builder: ChunkBuilder,
        artifact_store: ArtifactStore,
        device: torch.device | str = "cuda",
    ):
        self.experts = experts
        self.router = router
        self.chunk_builder = chunk_builder
        self.artifact_store = artifact_store
        self.device = torch.device(device) if isinstance(device, str) else device

    def run_episode(
        self,
        episode: EpisodeState,
        repo_state: RepoState,
        harness: TestHarness,
        patch_assembler: PatchAssembler,
        repo_map: str = "",
    ) -> EpisodeState:
        """Run a full episode loop. Returns the finalized episode state."""

        try:
            while not episode.is_terminal() and episode.round_idx < episode.max_rounds:
                episode.advance_round()
                logger.info(
                    f"Episode {episode.episode_id[:8]} round {episode.round_idx}"
                )

                # Compile state for router
                state_text = compile_state_text(episode)

                # Build chunk candidates
                chunks = self.chunk_builder.build_candidates(
                    episode=episode,
                    repo_map=repo_map,
                )

                # Router decision
                router_out = self.router.decide(
                    state_text,
                    n_chunks=len(chunks),
                    device=self.device,
                )

                # Check stop signal
                stop_prob = torch.sigmoid(router_out.stop_logit).item()
                if stop_prob > STOP_THRESHOLD and episode.round_idx > 1:
                    if episode.env.compile_ok and episode.env.tests_failed == 0:
                        finalize_episode(episode, FinalVerdict.ACCEPT, "Router stop: tests pass")
                        break
                    elif not episode.can_repair():
                        finalize_episode(episode, FinalVerdict.ROLLBACK, "Router stop: no repairs left")
                        break

                # Select experts
                expert_probs = torch.sigmoid(router_out.expert_logits)
                selected_experts = self._select_experts(expert_probs, episode)

                # Select budget
                budget_level = BudgetLevel(torch.argmax(router_out.budget_logits).item())
                budget = get_budget(budget_level)

                # Pack chunks
                chunk_scores = torch.sigmoid(router_out.chunk_logits[:len(chunks)])
                selected_chunks = greedy_knapsack(chunks, chunk_scores, budget)

                # Run selected experts
                for expert_role in selected_experts:
                    if episode.is_terminal():
                        break

                    expert = self.experts.get(expert_role)
                    if expert is None:
                        continue

                    output = self._run_expert(
                        expert, selected_chunks, episode, budget_level
                    )
                    if output is None:
                        continue

                    # Store artifact
                    sha = self.artifact_store.save(output.artifact, episode.episode_id)
                    output.message.artifact_ref = sha
                    output.message.round_idx = episode.round_idx

                    # Record
                    episode.history.add(output.message)
                    episode.record_call(expert_role, output.input_tokens, output.gen_tokens)

                    # Handle coder output → apply patch → run harness
                    if expert_role == ExpertRole.CODER:
                        self._handle_patch(
                            episode, output, repo_state, harness, patch_assembler
                        )

                    # Handle test/debug results
                    if expert_role == ExpertRole.TESTER:
                        self._handle_test_result(episode, harness)

                # Check if we should repair
                if (
                    episode.env.tests_failed > 0
                    and episode.can_repair()
                    and not episode.is_terminal()
                ):
                    episode.repair_count += 1
                    logger.info(
                        f"Repair loop {episode.repair_count}/{episode.max_repairs}"
                    )
                    continue

                # Check success
                if episode.env.compile_ok and episode.env.tests_failed == 0 and episode.env.patch_applied:
                    finalize_episode(episode, FinalVerdict.ACCEPT, "All tests pass")
                    break

            # Timeout if still running
            if episode.status == EpisodeStatus.RUNNING:
                if episode.env.compile_ok and episode.env.tests_failed == 0 and episode.env.patch_applied:
                    finalize_episode(episode, FinalVerdict.ACCEPT, "Max rounds reached, tests pass")
                else:
                    timeout_episode(episode)

        except Exception as e:
            logger.error(f"Episode error: {e}", exc_info=True)
            episode.status = EpisodeStatus.ERROR

        return episode

    def _select_experts(
        self,
        expert_probs: torch.Tensor,
        episode: EpisodeState,
    ) -> list[ExpertRole]:
        """Select experts based on router probabilities and episode context."""
        roles = [ExpertRole.PLANNER, ExpertRole.CODER, ExpertRole.TESTER, ExpertRole.DEBUGGER]
        selected = []

        for i, role in enumerate(roles):
            if expert_probs[i].item() > 0.5:
                selected.append(role)

        # Fallback: if nothing selected, use default workflow
        if not selected:
            if episode.round_idx == 1:
                selected = [ExpertRole.PLANNER]
            elif not episode.env.patch_applied:
                selected = [ExpertRole.CODER]
            else:
                selected = [ExpertRole.TESTER]

        return selected

    def _run_expert(
        self,
        expert: BaseExpert,
        chunks: list[Chunk],
        episode: EpisodeState,
        budget_level: BudgetLevel,
    ) -> ExpertOutput | None:
        """Run a single expert with retry on schema failure."""
        budget = get_budget(budget_level)
        task_context = {
            "issue_text": episode.issue_text,
            "repo_map": episode.repo_map,
        }

        for attempt in range(MAX_SCHEMA_RETRIES + 1):
            try:
                prompt = expert.build_prompt(
                    chunks, episode.history.messages, episode.env, task_context
                )
                output = expert.generate(prompt, budget)
                return output
            except Exception as e:
                logger.warning(
                    f"Expert {expert.role.value} attempt {attempt + 1} failed: {e}"
                )
                if attempt == MAX_SCHEMA_RETRIES:
                    return None

    def _handle_patch(
        self,
        episode: EpisodeState,
        output: ExpertOutput,
        repo_state: RepoState,
        harness: TestHarness,
        patch_assembler: PatchAssembler,
    ) -> None:
        """Apply coder's patch and run harness."""
        diff = output.artifact.content

        # Try single-shot apply
        ok, verified_diff, nacks = patch_assembler.apply_single_shot(diff)
        if not ok:
            logger.warning(f"Patch NACK: {nacks}")
            # Record NACK in message for next round
            return

        # Apply to repo
        success, msg = repo_state.apply_patch(verified_diff)
        if success:
            episode.env.patch_applied = True
            episode.env.current_patch_diff = verified_diff
            episode.env.files_changed = output.artifact.metadata.get("files", [])

            # Run harness
            result = harness.run()
            self._update_env_from_harness(episode, result)
        else:
            logger.warning(f"Patch apply failed: {msg}")

    def _handle_test_result(self, episode: EpisodeState, harness: TestHarness) -> None:
        """Run harness after tester generates tests."""
        if episode.env.patch_applied:
            result = harness.run()
            self._update_env_from_harness(episode, result)

    def _update_env_from_harness(self, episode: EpisodeState, result: HarnessResult) -> None:
        """Update episode env from harness result."""
        episode.env.compile_ok = result.compile_ok
        episode.env.lint_ok = result.lint_ok
        episode.env.tests_passed = result.tests_passed
        episode.env.tests_failed = result.tests_failed
        episode.env.failing_tests = result.failing_tests
