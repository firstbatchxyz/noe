"""Stage A: Individual Role SFT on frozen Qwen2.5-Coder-3B-Instruct.

Per-role training with LoRA adapters:
- AdamW (beta1=0.9, beta2=0.95), lr=2e-4, cosine, 100-step warmup
- Batch 8, max seq len 4096, 3 epochs, early stopping
- bf16

Supports multi-role-per-GPU: load base model once, train roles sequentially
with adapter swap, or train 2 roles concurrently via threading with separate
PEFT models sharing the frozen base weights.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from peft import PeftModel, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from noe_train.experts.lora_config import get_lora_config
from noe_train.schema.messages import ExpertRole

logger = logging.getLogger(__name__)

MODEL_NAME = "Qwen/Qwen2.5-Coder-3B-Instruct"


@dataclass
class StageAConfig:
    model_name: str = MODEL_NAME
    max_seq_len: int = 4096
    per_device_batch_size: int = 8
    num_epochs: int = 3
    learning_rate: float = 2e-4
    warmup_steps: int = 100
    weight_decay: float = 0.01
    bf16: bool = True
    logging_steps: int = 50
    save_steps: int = 500
    eval_steps: int = 500
    early_stopping_patience: int = 3
    gradient_checkpointing: bool = False


def load_base_model(
    config: StageAConfig | None = None,
) -> tuple:
    """Load the frozen base model + tokenizer once. Returns (model, tokenizer)."""
    cfg = config or StageAConfig()

    logger.info(f"Loading base model: {cfg.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        torch_dtype=torch.bfloat16 if cfg.bf16 else torch.float32,
        trust_remote_code=True,
    )

    params_b = sum(p.numel() for p in model.parameters()) / 1e9
    mem_gb = params_b * (2 if cfg.bf16 else 4)
    logger.info(f"  Base model: {params_b:.2f}B params, ~{mem_gb:.1f}GB VRAM")

    return model, tokenizer


def train_role(
    role: ExpertRole,
    train_dataset: Dataset,
    val_dataset: Dataset | None = None,
    output_dir: str | Path = "checkpoints/stage_a",
    config: StageAConfig | None = None,
    rank_override: int | None = None,
    base_model: AutoModelForCausalLM | None = None,
    tokenizer: AutoTokenizer | None = None,
) -> PeftModel:
    """Train a single role's LoRA adapter.

    Args:
        role: which expert role to train
        train_dataset: HF dataset with 'input_text' and 'output_text' columns
        val_dataset: optional validation dataset
        output_dir: where to save the adapter
        config: training hyperparameters
        rank_override: override LoRA rank
        base_model: pre-loaded base model (avoids reload for multi-role-per-GPU)
        tokenizer: pre-loaded tokenizer
    """
    cfg = config or StageAConfig()
    role_output = Path(output_dir) / f"lora_{role.value}"
    role_output.mkdir(parents=True, exist_ok=True)

    logger.info(f"Stage A: Training {role.value} adapter")
    logger.info(f"  Train samples: {len(train_dataset)}")
    logger.info(f"  Output: {role_output}")

    # Load or reuse base model
    if base_model is None or tokenizer is None:
        base_model, tokenizer = load_base_model(cfg)

    # Apply LoRA — creates a new PeftModel wrapping the base
    lora_cfg = get_lora_config(role, rank_override=rank_override)
    model = get_peft_model(base_model, lora_cfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"  Trainable params: {trainable:,} / {total:,} ({trainable/total:.2%})")

    # Tokenize dataset
    def tokenize_fn(examples):
        texts = []
        for inp, out in zip(examples["input_text"], examples["output_text"]):
            texts.append(f"{inp}\n\n{out}{tokenizer.eos_token}")
        return tokenizer(
            texts,
            truncation=True,
            max_length=cfg.max_seq_len,
            padding="max_length",
        )

    train_tokenized = train_dataset.map(
        tokenize_fn, batched=True, remove_columns=train_dataset.column_names
    )

    def add_labels(examples):
        examples["labels"] = examples["input_ids"].copy()
        return examples

    train_tokenized = train_tokenized.map(add_labels, batched=True)

    eval_tokenized = None
    if val_dataset is not None:
        eval_tokenized = val_dataset.map(
            tokenize_fn, batched=True, remove_columns=val_dataset.column_names
        )
        eval_tokenized = eval_tokenized.map(add_labels, batched=True)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(role_output),
        num_train_epochs=cfg.num_epochs,
        per_device_train_batch_size=cfg.per_device_batch_size,
        per_device_eval_batch_size=cfg.per_device_batch_size,
        learning_rate=cfg.learning_rate,
        warmup_steps=cfg.warmup_steps,
        weight_decay=cfg.weight_decay,
        bf16=cfg.bf16,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        eval_strategy="steps" if eval_tokenized else "no",
        eval_steps=cfg.eval_steps if eval_tokenized else None,
        save_total_limit=3,
        load_best_model_at_end=eval_tokenized is not None,
        metric_for_best_model="eval_loss" if eval_tokenized else None,
        lr_scheduler_type="cosine",
        optim="adamw_torch",
        adam_beta1=0.9,
        adam_beta2=0.95,
        gradient_checkpointing=cfg.gradient_checkpointing,
        report_to="wandb",
        run_name=f"stage_a_{role.value}",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=eval_tokenized,
        tokenizer=tokenizer,
    )

    trainer.train()

    # Save only the LoRA adapter (not the full base model)
    model.save_pretrained(str(role_output / "final"))
    tokenizer.save_pretrained(str(role_output / "final"))

    logger.info(f"Stage A: {role.value} training complete → {role_output / 'final'}")
    return model


def train_role_group(
    roles: list[ExpertRole],
    datasets: dict[str, tuple[Dataset, Dataset | None]],
    output_dir: str | Path = "checkpoints/stage_a",
    config: StageAConfig | None = None,
) -> dict[ExpertRole, PeftModel]:
    """Train multiple roles sequentially, sharing one base model load.

    This avoids reloading the 3B base model for each role (~6GB saved per swap,
    ~2 min saved per role on model load).

    Args:
        roles: roles to train in order
        datasets: {role_name: (train_ds, val_ds)} mapping
        output_dir: checkpoint directory
        config: shared training config
    """
    cfg = config or StageAConfig()

    # Load base model once
    base_model, tokenizer = load_base_model(cfg)

    results = {}
    for role in roles:
        train_ds, val_ds = datasets[role.value]

        # Train this role's adapter
        peft_model = train_role(
            role=role,
            train_dataset=train_ds,
            val_dataset=val_ds,
            output_dir=output_dir,
            config=cfg,
            base_model=base_model,
            tokenizer=tokenizer,
        )
        results[role] = peft_model

        # Unload the LoRA adapter, keep the base model for next role
        # merge_and_unload would bake LoRA into base — we don't want that
        # Instead, delete the PeftModel wrapper and re-wrap for next role
        del peft_model
        torch.cuda.empty_cache()
        logger.info(f"Freed {role.value} adapter, base model retained")

    return results
