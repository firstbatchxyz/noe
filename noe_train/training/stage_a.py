"""Stage A: Individual Role SFT on frozen Qwen3.5-4B (DeltaNet hybrid).

Uses Unsloth for optimized DeltaNet Triton kernels (1.5x faster, 50% less VRAM).

Per-role training with LoRA adapters:
- AdamW-8bit, lr=2e-4, cosine, 100-step warmup
- Batch 8, max seq len 8192, 1 epoch
- bf16, loss masked on input tokens (output-only supervision)

Supports multi-role-per-GPU: load base model once, train roles sequentially
with adapter reload between roles.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import torch
from datasets import Dataset
from transformers import DataCollatorForSeq2Seq
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig

from noe_train.experts.lora_config import LORA_TARGET_MODULES, ROLE_RANKS
from noe_train.schema.messages import ExpertRole

logger = logging.getLogger(__name__)

MODEL_NAME = "Qwen/Qwen3.5-4B"


@dataclass
class StageAConfig:
    model_name: str = MODEL_NAME
    max_seq_len: int = 8192
    per_device_batch_size: int = 4
    gradient_accumulation_steps: int = 2  # effective batch = 4 * 2 = 8
    num_epochs: int = 1
    learning_rate: float = 2e-4
    warmup_steps: int = 100
    weight_decay: float = 0.01
    bf16: bool = True
    logging_steps: int = 50
    save_steps: int = 500
    eval_steps: int = 500
    early_stopping_patience: int = 3
    lora_alpha: int = 64
    lora_dropout: float = 0.0  # Unsloth recommends 0 for LoRA


def load_base_model(
    config: StageAConfig | None = None,
) -> tuple:
    """Load the frozen base model + tokenizer via Unsloth.

    Unsloth handles VLM → text model extraction, DeltaNet kernel patching,
    and attention implementation selection automatically.
    """
    cfg = config or StageAConfig()

    logger.info(f"Loading base model via Unsloth: {cfg.model_name}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.model_name,
        max_seq_length=cfg.max_seq_len,
        load_in_16bit=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    params_b = sum(p.numel() for p in model.parameters()) / 1e9
    logger.info(f"  Base model: {params_b:.2f}B params")
    logger.info(f"  Architecture: {type(model).__name__}")

    return model, tokenizer


def _tokenize_sft(
    dataset: Dataset,
    tokenizer,
    max_seq_len: int,
) -> Dataset:
    """Tokenize dataset with proper input masking for SFT.

    Loss is computed only on output tokens — input tokens get label=-100.
    Output is always fully preserved; input is truncated from the left if needed.
    No padding during tokenization; DataCollatorForSeq2Seq handles per-batch padding.
    """

    def _tokenize_row(examples):
        all_input_ids = []
        all_labels = []
        all_attention_mask = []

        for inp, out in zip(examples["input_text"], examples["output_text"]):
            # Tokenize prompt and response separately to find the boundary
            prompt_text = f"{inp}\n\n"
            response_text = f"{out}{tokenizer.eos_token}"

            prompt_enc = tokenizer(prompt_text, add_special_tokens=True)
            response_enc = tokenizer(response_text, add_special_tokens=False)

            prompt_ids = prompt_enc["input_ids"]
            response_ids = response_enc["input_ids"]

            # Always preserve the full output — truncate the input if needed.
            # The output is the entire training signal (input is masked to -100).
            max_prompt_len = max_seq_len - len(response_ids)
            if max_prompt_len < 1:
                # Output alone exceeds max_seq_len — truncate output (rare)
                response_ids = response_ids[:max_seq_len - 1]
                prompt_ids = prompt_ids[-1:]  # keep at least 1 token
            elif max_prompt_len < len(prompt_ids):
                # Truncate prompt from the left — keep the end (closest to output)
                prompt_ids = prompt_ids[-max_prompt_len:]

            input_ids = prompt_ids + response_ids

            # Labels: -100 for prompt, real ids for response
            prompt_len = len(prompt_ids)
            labels = [-100] * prompt_len + response_ids

            attention_mask = [1] * len(input_ids)

            all_input_ids.append(input_ids)
            all_labels.append(labels)
            all_attention_mask.append(attention_mask)

        return {
            "input_ids": all_input_ids,
            "labels": all_labels,
            "attention_mask": all_attention_mask,
        }

    return dataset.map(
        _tokenize_row,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing",
    )


def train_role(
    role: ExpertRole,
    train_dataset: Dataset,
    val_dataset: Dataset | None = None,
    output_dir: str | Path = "checkpoints/stage_a",
    config: StageAConfig | None = None,
    rank_override: int | None = None,
    base_model=None,
    tokenizer=None,
):
    """Train a single role's LoRA adapter via Unsloth.

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

    # Apply LoRA via Unsloth — handles gradient checkpointing + kernel patching
    r = rank_override if rank_override is not None else ROLE_RANKS[role]
    model = FastLanguageModel.get_peft_model(
        base_model,
        r=r,
        target_modules=LORA_TARGET_MODULES,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"  LoRA r={r}, trainable: {trainable:,} / {total:,} ({trainable/total:.2%})")

    # Tokenize with input masking (output-preserving truncation)
    train_tokenized = _tokenize_sft(train_dataset, tokenizer, cfg.max_seq_len)

    eval_tokenized = None
    if val_dataset is not None:
        eval_tokenized = _tokenize_sft(val_dataset, tokenizer, cfg.max_seq_len)

    # Dynamic padding collator — pads to longest in batch, labels padded with -100
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        pad_to_multiple_of=8,
    )

    # Training config
    training_args = SFTConfig(
        output_dir=str(role_output),
        num_train_epochs=cfg.num_epochs,
        per_device_train_batch_size=cfg.per_device_batch_size,
        per_device_eval_batch_size=cfg.per_device_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
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
        optim="adamw_8bit",
        adam_beta1=0.9,
        adam_beta2=0.95,
        max_seq_length=cfg.max_seq_len,
        report_to="wandb",
        run_name=f"stage_a_{role.value}",
        dataloader_pin_memory=True,
        dataloader_num_workers=2,
        remove_unused_columns=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=eval_tokenized,
        tokenizer=tokenizer,
        data_collator=data_collator,
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
) -> dict[ExpertRole, object]:
    """Train multiple roles sequentially, sharing one base model load.

    Loads the base model once at the start. Between roles, the LoRA adapter
    is properly unloaded and the base model is re-wrapped with the next
    role's adapter config.

    Args:
        roles: roles to train in order
        datasets: {role_name: (train_ds, val_ds)} mapping
        output_dir: checkpoint directory
        config: shared training config
    """
    cfg = config or StageAConfig()

    results = {}
    base_model = None
    tokenizer = None

    for i, role in enumerate(roles):
        train_ds, val_ds = datasets[role.value]

        if i == 0:
            base_model, tokenizer = load_base_model(cfg)
        else:
            # Reload base model — get_peft_model modifies in-place
            logger.info(f"Reloading base model for {role.value}...")
            del base_model
            torch.cuda.empty_cache()
            base_model, tokenizer = load_base_model(cfg)

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

        del peft_model
        torch.cuda.empty_cache()
        logger.info(f"Freed {role.value} adapter")

    return results
