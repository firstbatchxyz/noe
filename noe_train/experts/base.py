"""Base expert class — shared interface for all 4 experts."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, PreTrainedTokenizerBase

from noe_train.schema.artifacts import Artifact
from noe_train.schema.budget import BudgetClass, BudgetTracker
from noe_train.schema.episode import EnvState
from noe_train.schema.messages import ExpertRole, TypedMessage


@dataclass()
class Chunk:
    id: str
    chunk_type: str
    content: str
    token_count: int


@dataclass()
class ExpertOutput:
    artifact: Artifact
    message: TypedMessage
    raw_text: str
    input_tokens: int
    gen_tokens: int
    confidence: float


class BaseExpert(ABC):
    """Abstract base for all NoE experts."""

    def __init__(
        self,
        role: ExpertRole,
        model: Any,
        tokenizer: Any,
        device: Any = "cuda",
    ):
        import torch

        self.role = role
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(device) if isinstance(device, str) else device

    @abstractmethod
    def build_prompt(
        self,
        chunks: list[Chunk],
        messages: list[TypedMessage],
        env: EnvState,
        task_context: dict[str, Any],
    ) -> str:
        """Build the input prompt from chunks, message history, and env state."""
        ...

    @abstractmethod
    def parse_output(self, raw: str) -> tuple[Artifact, TypedMessage]:
        """Parse raw model output into a typed artifact + message."""
        ...

    def generate(
        self,
        prompt: str,
        budget: BudgetClass,
        temperature: float = 0.6,
        top_p: float = 0.95,
    ) -> ExpertOutput:
        """Generate output within budget constraints."""
        import torch

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=budget.max_input_tokens,
        ).to(self.device)

        input_tokens = inputs["input_ids"].shape[1]
        tracker = BudgetTracker(budget, input_tokens_used=input_tokens)

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=min(budget.max_gen_tokens, budget.max_gen_tokens),
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                return_dict_in_generate=True,
                output_scores=True,
            )

        generated_ids = outputs.sequences[0, input_tokens:]
        raw_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        gen_tokens = len(generated_ids)

        tracker.gen_tokens_used = gen_tokens
        confidence = self._estimate_confidence(outputs.scores)

        artifact, message = self.parse_output(raw_text)
        return ExpertOutput(
            artifact=artifact,
            message=message,
            raw_text=raw_text,
            input_tokens=input_tokens,
            gen_tokens=gen_tokens,
            confidence=confidence,
        )

    def _estimate_confidence(self, scores: tuple) -> float:
        """Estimate confidence from output token entropy (lower entropy = higher confidence)."""
        import torch

        if not scores:
            return 0.5
        entropies = []
        for logits in scores:
            probs = torch.softmax(logits[0], dim=-1)
            log_probs = torch.log(probs + 1e-10)
            entropy = -(probs * log_probs).sum().item()
            entropies.append(entropy)
        mean_entropy = sum(entropies) / len(entropies)
        # Normalize: assume max entropy ~10 (vocab ~150K)
        confidence = max(0.0, min(1.0, 1.0 - mean_entropy / 10.0))
        return confidence

    def _truncate_chunks(self, chunks: list[Chunk], max_tokens: int) -> list[Chunk]:
        """Select chunks that fit within token budget."""
        selected = []
        total = 0
        for chunk in chunks:
            if total + chunk.token_count > max_tokens:
                break
            selected.append(chunk)
            total += chunk.token_count
        return selected

    def _format_messages(self, messages: list[TypedMessage], max_tokens: int) -> str:
        """Format message history as compact summaries."""
        parts = []
        total = 0
        for msg in messages:
            line = f"[R{msg.round_idx}] {msg.sender.value}: {msg.summary}"
            tokens = len(line.split())  # rough estimate
            if total + tokens > max_tokens:
                break
            parts.append(line)
            total += tokens
        return "\n".join(parts)
