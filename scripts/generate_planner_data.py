#!/usr/bin/env python3
"""Generate planner traces using Qwen2.5-Coder-32B-Instruct teacher.

~5-8K samples from (issue, repo_map) pairs.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from noe_train.data.teacher_gen import build_teacher_prompt, format_as_training_sample, validate_plan
from noe_train.utils.logging import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Generate planner training data")
    parser.add_argument("--input-tasks", type=str, required=True, help="JSON file with tasks")
    parser.add_argument("--output-dir", type=str, default="data/planner_traces")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-Coder-32B-Instruct")
    parser.add_argument("--max-samples", type=int, default=8000)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load tasks
    with open(args.input_tasks) as f:
        tasks = json.load(f)
    logger.info(f"Loaded {len(tasks)} tasks")

    # Load teacher model
    logger.info(f"Loading teacher model: {args.model}")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Generate plans
    samples = []
    valid = 0
    total = 0

    for task in tasks[: args.max_samples]:
        issue = task.get("problem_statement") or task.get("issue", "")
        repo_map = task.get("repo_map", "")
        instance_id = task.get("instance_id", "")

        prompt = build_teacher_prompt(issue, repo_map)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.3,
                top_p=0.95,
                do_sample=True,
            )

        generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        # Validate
        ok, plan_data = validate_plan(generated)
        total += 1

        if ok:
            sample = format_as_training_sample(issue, repo_map, plan_data, instance_id)
            samples.append(sample)
            valid += 1

        if total % 100 == 0:
            logger.info(f"Progress: {total}/{len(tasks)}, valid: {valid}/{total}")

    # Save
    output_file = output_dir / "planner_traces.json"
    with open(output_file, "w") as f:
        json.dump(samples, f, indent=2)

    logger.info(f"Generated {valid}/{total} valid planner traces → {output_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
