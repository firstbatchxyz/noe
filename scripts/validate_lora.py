#!/usr/bin/env python3
"""Quick LoRA sanity check on Qwen2.5-Coder-3B-Instruct.

- Load model, apply PEFT LoraConfig, run 50-step SFT on toy data
- Confirm loss decreases, verify trainable param count
- ~30 min on 1 GPU
"""

import logging
import sys

import torch
from datasets import Dataset
from peft import get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from noe_train.experts.lora_config import get_lora_config
from noe_train.schema.messages import ExpertRole
from noe_train.utils.logging import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

MODEL_NAME = "Qwen/Qwen2.5-Coder-3B-Instruct"
NUM_STEPS = 50
BATCH_SIZE = 2


def make_toy_dataset(tokenizer, n=100, max_len=256):
    """Create a toy dataset for validation."""
    texts = [
        f"Fix the bug in function_{i}: "
        f"The issue is that the return value is incorrect. "
        f"def function_{i}(x):\n    return x + {i}\n"
        for i in range(n)
    ]
    tokenized = tokenizer(texts, truncation=True, max_length=max_len, padding="max_length")
    tokenized["labels"] = tokenized["input_ids"].copy()
    return Dataset.from_dict(tokenized)


def main():
    logger.info(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    # Apply LoRA for planner (r=32)
    lora_config = get_lora_config(ExpertRole.PLANNER)
    model = get_peft_model(model, lora_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable: {trainable:,} / {total:,} ({trainable / total:.2%})")

    # Toy dataset
    dataset = make_toy_dataset(tokenizer)
    logger.info(f"Toy dataset: {len(dataset)} samples")

    # Train
    args = TrainingArguments(
        output_dir="/tmp/noe_lora_validate",
        max_steps=NUM_STEPS,
        per_device_train_batch_size=BATCH_SIZE,
        learning_rate=2e-4,
        bf16=True,
        logging_steps=10,
        remove_unused_columns=False,
        report_to="none",
    )

    trainer = Trainer(model=model, args=args, train_dataset=dataset, tokenizer=tokenizer)

    logger.info(f"Training {NUM_STEPS} steps...")
    result = trainer.train()

    # Validate
    final_loss = result.training_loss
    log_history = trainer.state.log_history
    losses = [h["loss"] for h in log_history if "loss" in h]

    if len(losses) >= 2:
        first_loss = losses[0]
        last_loss = losses[-1]
        decreased = last_loss < first_loss
        logger.info(f"Loss: {first_loss:.4f} → {last_loss:.4f} (decreased={decreased})")

        if decreased:
            logger.info("PASS: LoRA validation successful")
            return 0
        else:
            logger.error("FAIL: Loss did not decrease")
            return 1
    else:
        logger.warning("Not enough loss data points to validate")
        return 1


if __name__ == "__main__":
    sys.exit(main())
