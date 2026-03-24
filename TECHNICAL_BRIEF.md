# Network of Experts (NoE): Bounded Repo-Repair via Coordinated Small Models

## Claims

This project tests two separable hypotheses on SWE-bench Verified (Python, <=3 files, <=80 changed lines):

**Claim A — Role specialization**: A team of 4 role-specialized 3B models (shared backbone, per-role LoRA) coordinated by a learned workflow controller outperforms a single 3B model with equivalent total compute on bounded repo-repair tasks.

**Claim B — Latent compression**: Augmenting the text-only NoE with learned latent vectors (512 bytes per message) between selected expert pairs further improves resolve rate over structured text alone, while reducing inter-expert token traffic.

Claim A is the primary contribution. Claim B is tested only if Claim A succeeds — it is an optimization on a working text-only system, not a prerequisite.

**Scope boundary**: We target bounded repo-repair — single-repo Python bugs requiring <=3 file edits and <=80 changed lines. This covers ~70% of SWE-bench Verified. We do not claim generality to multi-repo, cross-language, or greenfield coding tasks.

**Decentralized inference angle**: If both claims hold, the system demonstrates that 4 nodes each running a 3B model (10-16 GB VRAM with 4-bit quantized serving) and exchanging kilobyte-scale messages can match a single 13B+ model requiring 40+ GB VRAM — splitting the workload across commodity GPUs.

---

## Architecture

### Expert Network

| Component | Model | Parameters | VRAM (bf16 training) | VRAM (4-bit serving) |
|-----------|-------|-----------|---------------------|---------------------|
| Planner | Qwen2.5-Coder-3B-Instruct + LoRA (r=32) | 3.09B + ~150M | ~6.2 GB | ~2.5 GB |
| Coder | Qwen2.5-Coder-3B-Instruct + LoRA (r=16) | 3.09B + ~75M | ~6.2 GB | ~2.5 GB |
| Tester | Qwen2.5-Coder-3B-Instruct + LoRA (r=16) | 3.09B + ~75M | ~6.2 GB | ~2.5 GB |
| Debugger | Qwen2.5-Coder-3B-Instruct + LoRA (r=32) | 3.09B + ~150M | ~6.2 GB | ~2.5 GB |
| Router/Controller | UniXcoder-base + 4 MLP heads | 125M | ~0.5 GB | ~0.3 GB |
| Critic | UniXcoder-base + value head | 125M | ~0.5 GB | ~0.3 GB |
| **Total unique params** | | **~3.5B** (shared backbone + role adapters) | | |

All 4 experts share the same frozen Qwen2.5-Coder-3B-Instruct backbone. Only the LoRA adapters are role-specific. In a decentralized setting, each node holds the full backbone plus its own adapter (~150 MB). With 4-bit quantization (GPTQ/AWQ) + KV cache, each node requires **10-16 GB VRAM** depending on sequence length — not 6 GB, since KV cache for 4K+ sequences adds significant overhead.

**LoRA configuration**: Target all attention + MLP projections (q, k, v, o, gate, up, down). Alpha=64, dropout=0.05. Planner and debugger use higher rank (r=32) because their tasks require more nuanced reasoning; coder and tester use r=16 since their outputs are more structured (diffs, test code).

### Workflow Controller (Router)

The router is a 125M-parameter model (UniXcoder-base) with 4 decision heads. In practice, for bounded repo-repair, the workflow is largely fixed (plan → code → test → debug → repair loop), so the router functions primarily as a **workflow controller** — its learned decisions are:

1. **Stop/continue** (binary): whether the episode is done (sigmoid). This is the most impactful decision.
2. **Budget class** (categorical): how much compute to allocate per call, B0-B4 (softmax over 5).
3. **Chunk scoring** (multi-label): which context chunks to include (sigmoid over up to 64 chunks).
4. **Expert selection** (multi-label): which experts to call this round (sigmoid over 4 logits). In practice this follows the default ordering ~90% of the time.

The router sees a compressed state packet (512 tokens max) summarizing the episode so far: which experts have run, what they produced, current test/compile status, and budget consumed.

We are honest that for bounded tasks the expert ordering is mostly deterministic. The router's real value is in budget allocation, chunk selection, and stop timing — not in discovering novel expert orderings. For larger tasks (multi-file, multi-round), expert selection may become more meaningful, but that is outside v1 scope.

### Communication Channels

Experts communicate through **structured text messages** (v1) with an optional **latent channel** (v1.5):

**1. Typed text messages (primary)**: JSON messages routed through a star topology. Each message has a type (PLAN, PATCH_HUNK, EXEC_REPORT, BUG_REPORT, etc.), a compact summary (64-96 tokens), and a reference to the full artifact in storage. Human-readable and debuggable.

**2. Latent vectors (optional, trained in Stage C)**: Each expert has a projector (Linear 2048→256 + LayerNorm, ~525K params) that compresses its hidden states into a 256-dim latent vector, and a receiver (~2.1M params) that injects incoming latent vectors as 4 virtual tokens prepended to input embeddings.

The receiver has a **learned gate** initialized nearly closed (sigmoid(-5) ≈ 0.007). The system starts operating on text messages only. During Stage C RL, the gate opens if and only if latent communication improves the episode reward. If it doesn't help, the gate stays closed — zero degradation by construction.

**Warm-start strategy**: Rather than wiring latent channels between all expert pairs (which would be 12 directed channels), we warm-start with only **planner→coder** (the highest-bandwidth link — the plan determines what the coder does). Additional channels are added only if the planner→coder channel shows measurable improvement in Stage C.

**Bandwidth comparison:**

| Channel | Size per message | Requires decoding |
|---------|-----------------|-------------------|
| Text message | ~400-800 bytes (100-200 tokens) | Yes (autoregressive) |
| Latent vector | 512 bytes (256 × bf16) | No (single matmul) |

### Budget System

The router assigns a compute budget class to each expert call:

| Class | Max Input | Max Generation | Max Message | Tools |
|-------|-----------|---------------|-------------|-------|
| B0 | 768 | 192 | 64 | 0 |
| B1 | 1,536 | 384 | 96 | 1 |
| B2 | 3,072 | 768 | 128 | 2 |
| B3 | 4,608 | 1,024 | 192 | 3 |
| B4 | 6,144 | 2,048 | 256 | 3 |

Default assignments: Planner=B2, Coder=B4, Tester=B2, Debugger=B2. The router learns to adjust — easy bugs might only need B1 for the coder, while complex multi-file issues might warrant B4.

---

## Episode Flow

A typical episode for resolving a bounded repo-repair task:

```
Round 1: Controller → Planner
         Planner receives: issue description + repo map + retrieved files
         Planner produces: PLAN (files_to_touch, invariants, risks, strategy)

Round 2: Controller → Coder
         Coder receives: issue + plan + selected code slices
         Coder produces: PATCH (unified diff via PATCH_PLAN → HUNK → DONE)
         Sandbox: applies patch, runs compile + lint + tests

Round 3: Controller → Tester + Debugger (parallel)
         Tester: designs new test cases + triages existing failures
           Input: issue + patch diff + touched symbols + test results
           Output: EXEC_REPORT (new tests + failure triage + regression analysis)
         Debugger: localizes root cause from failures
           Input: issue + patch + failing tests + traces
           Output: BUG_REPORT (root cause + suspect files + fix hint)

Round 4: Controller decision
         If tests pass → ACCEPT
         If fixable → Coder repair (up to 2 repair loops)
         If unfixable → ROLLBACK
```

**Tester role**: The tester is a test designer and failure triager, not a test executor. It generates targeted test cases that exercise the patch, triages failures into "patch regression" vs "pre-existing" vs "flaky", and provides structured feedback. Test execution happens in the sandbox.

Patch assembly is **transactional**: if any hunk in a file is invalid, the entire file's patch is rejected with a structured error, and the coder gets a retry with the error context. No silent partial applies.

---

## Training Data

### Stage A: Individual Role SFT

**Primary corpus: NVIDIA Nemotron-Cascade-SFT-SWE** (~210K samples total)

Note: The HuggingFace dataset viewer reports ~141K rows because it samples; the dataset card and actual download confirm ~210K total across all categories.

This dataset was created by NVIDIA for their Cascade agentic coding system. It contains structured traces from an Agentless pipeline (localization → repair → test generation) processed through DeepSeek-R1 for chain-of-thought reasoning. The data is directly role-aligned:

| Role | Source Category | Train | Val | Content |
|------|----------------|-------|-----|---------|
| Debugger | SWE Localization | 87,651 | 4,614 | Issue + repo context → localized files + diagnosis |
| Coder | SWE Repair | 82,561 | 4,346 | Issue + localized files → unified diff patch |
| Tester | SWE TestGen | 30,068 | 1,583 | Issue + patch → test cases + expected results |
| Planner | Derived from Localization | 7,600 | 400 | Issue + repo map → plan JSON (files, strategy, risks) |

**Data processing**: Each sample is a 2-turn chat (user prompt, assistant response). The assistant responses contain `<think>...</think>` reasoning blocks from DeepSeek-R1 which we strip — the model should learn to produce direct outputs, not chain-of-thought (the LoRA capacity at 3B params is better spent on task execution than reasoning traces).

**Planner data**: The Nemotron dataset doesn't have a planner category. We derive planner training data from the localization subset by extracting file paths mentioned in the localization output and reformatting into our PLAN JSON schema (files_to_touch, invariants, risks, strategy). 92,130 out of 92,265 localization samples had extractable file paths; we sample 8,000 for the planner.

**Why this dataset**: Nemotron-Cascade-SFT-SWE is the best available SWE-specific dataset. It's role-aligned by design (localization/repair/test-gen maps directly to our debugger/coder/tester), it excludes repos in SWE-bench Verified (our eval stays clean), and it was generated by a strong pipeline (Agentless + DeepSeek-R1 reasoning).

### Stage B: Router RL Task Pool

**NVIDIA Nemotron-Cascade-RL-SWE** (~110K samples), filtered to short fixes (<=1 file changed) for router training. Each sample provides:
- `prompt`: the issue/problem statement
- `golden_patch`: the reference fix (unified diff)
- `relevant_file_contents`: list of {file_path, content} for context
- `instance_id` and `source` for tracking

Note: The RL dataset is ~110K samples (not 141K — confirmed from actual download). Supplemented with SWE-bench train (small patches) to provide diversity in repo structure and issue types.

### Stage C: Team RL Task Pool

Full Nemotron-Cascade-RL-SWE (~110K) + SWE-bench train. Repo-level tasks with golden patches and FAIL_TO_PASS test specifications.

### Evaluation (never trained on)

**SWE-bench Verified** (500 instances): Curated subset of SWE-bench with human-verified solutions. Standard benchmark for evaluating automated software engineering systems.

---

## Training Pipeline

### Stage A: Individual Role SFT with SDFT

Each expert's LoRA adapter is trained independently on its role-specific data using Self-Distillation Fine-Tuning (Shenfeld et al., 2026, arXiv:2601.19897):

```
L_total = L_SFT + α × T² × KL(softmax(z_teacher/T) || softmax(z_student/T))
```

The frozen base model (LoRA adapters disabled) serves as teacher. The KL term prevents the LoRA adapter from drifting too far from the base distribution, preserving general capabilities while learning role-specific skills. Extra cost: one forward pass per batch (no gradients), ~1.5x total compute.

- **Optimizer**: AdamW (β1=0.9, β2=0.95), lr=2e-4, cosine schedule, 100-step warmup
- **Batch**: effective batch 8 (per_device=4, gradient_accumulation=2)
- **Precision**: bf16 with gradient checkpointing
- **Epochs**: 3, early stopping on validation loss (patience=3)
- **Loss**: causal LM loss on output tokens only (input tokens masked with label=-100) + SDFT KL (α=0.5, T=2.0)
- **Hardware**: 2x A100 80GB, 2 parallel groups (GPU 0: coder+planner, GPU 1: debugger+tester)

**Validation gates** (must pass before proceeding):
- Schema validity >98% (outputs parse as expected type)
- Execution validity >95% (patches apply, tests run)
- Each role's metric above untrained baseline

### Stage B: Router RL via GRPO

Group Relative Policy Optimization (validated by NVIDIA's Cascade RL work):
- **Group size 6**: same task, 6 different routing trajectories
- **Experts frozen** (Stage A checkpoints). Only router updated.
- **Signal**: episode outcome (test pass/fail, compile status, patch quality)
- lr=2e-5, clip=0.2, KL penalty to supervised router=0.02
- ~80-100K episodes

**Gate**: Team must beat single-expert baseline by >=5% on held-out tasks. **Counterfactual utility test**: re-run successful episodes with each expert removed — if removing an expert degrades the outcome, that expert provided genuine value. Require >30% of successes to have at least one expert with positive counterfactual utility (not just ">1 expert was called").

### Stage C: Team RL with Optional Latent Communication

End-to-end fine-tuning of the full system. Latent communication is introduced here as an optional channel — the system must already work with text only from Stages A+B.

**Unfreezing schedule**:
- Episodes 0-5K: router only (+ planner→coder latent projector/receiver if enabled)
- Episodes 5K+: + coder LoRA + debugger LoRA
- Episodes 15K+: + tester LoRA (if test quality is the bottleneck)
- Planner: stays frozen throughout (small dataset, high risk of forgetting)

**Reward function**:
```
Phi(s) = 0.50 x test_pass_rate + 0.20 x compile_ok
       + 0.10 x lint_ok + 0.10 x coverage_gain + 0.10 x verifier_confidence

Step reward: r_t = Phi(s_{t+1}) - Phi(s_t) - cost_penalties
Terminal:    +1.0 if hidden test suite passes

Cost penalties (annealed over 20K episodes):
  0.02 per tool call, 0.01 per round, 5e-5 per message token

Attribution: 90% shared, 10% role bonuses
```

The cost penalties force the router to learn **efficiency**: call only the experts that help, at the minimum budget that works.

**Latent gate learning**: The planner→coder projector and receiver are unfrozen from episode 0. The gate starts at ~0.007. As RL explores, if the latent vector improves patch quality, the gradient pushes the gate open. This is conditioned on episode return, not reconstruction loss. Additional expert-pair channels are added only if planner→coder shows measurable gain.

---

## Baselines and Evaluation Plan

### Baselines

To isolate what actually helps, we compare against:

| Baseline | Description | Purpose |
|----------|-------------|---------|
| **Single 3B** | Qwen2.5-Coder-3B-Instruct, no LoRA, standard prompting | Floor: what can one 3B do? |
| **Single 3B + LoRA** | Same backbone, single general-purpose LoRA on all SFT data | Is role specialization better than one adapter? |
| **Best-of-N 3B** | Single 3B + LoRA, generate N patches, pick best by test pass | Does naive compute scaling close the gap? |
| **Self-repair 3B** | Single 3B + LoRA with iterative fix loop (no role split) | Does iterative refinement need role separation? |
| **Single 7B** | Qwen2.5-Coder-7B-Instruct, standard prompting | Compute-matched: is NoE 4x3B better than 1x7B? |
| **Single 14B** | Qwen2.5-Coder-14B-Instruct, standard prompting | Stretch target: can NoE 4x3B reach 14B level? |
| **NoE text-only** | Full 4-expert team, structured text messages, no latent | Isolates Claim B: does latent add to text? |
| **NoE + latent** | Full team + planner→coder latent channel | Full system for Claim B |

The critical comparison for **Claim A** is: NoE text-only vs Single 3B + LoRA vs Best-of-N 3B. If NoE doesn't beat best-of-N, the routing/specialization overhead isn't justified.

The critical comparison for **Claim B** is: NoE + latent vs NoE text-only. The latent channel must show statistically significant improvement on the same task set.

### Primary: SWE-bench Verified (500 instances)

| Metric | What it measures |
|--------|-----------------|
| Resolve rate | % of instances where generated patch passes all tests |
| Efficiency | Resolved instances per 1K tokens consumed |

### Per-Role Metrics

| Role | Metric |
|------|--------|
| Planner | File recall@3 vs gold changed files |
| Coder | Syntax-valid rate, compile rate, patch acceptance |
| Tester | Generated tests that run + discriminate (fail buggy, pass gold) |
| Debugger | Correct file in top-3 localizations |
| Router | Budget accuracy, stop accuracy, counterfactual expert utility |

### Ablations (8 experiments)

Each ablation removes one component to measure its contribution:

1. **Fixed workflow** vs learned router (is adaptive routing worth learning?)
2. **Single LoRA** vs role-specific adapters (is specialization worth it?) — most important for Claim A
3. **Free text** vs typed messages (does structure help?)
4. **No tester** (is test design/triage needed?)
5. **No debugger** (is localization needed?)
6. **No retrieval** (is BM25/symbol search needed?)
7. **No critic** (is the value function needed for RL?)
8. **No cost penalty** (does efficiency pressure help?)

---

## Decentralized Inference Architecture

The trained NoE system maps naturally to decentralized deployment:

```
Node A (10-16 GB)          Node B (10-16 GB)
┌──────────────┐           ┌──────────────┐
│  Qwen2.5-3B  │           │  Qwen2.5-3B  │
│  (4-bit)     │   512B    │  (4-bit)     │
│  + Planner   │ ───────── │  + Coder     │
│    LoRA      │  latent   │    LoRA      │
│  + Projector │  (opt.)   │  + Receiver  │
└──────────────┘           └──────────────┘
       │                          │
       │        Controller        │
       │    ┌────────────┐        │
       └────│ UniXcoder  │────────┘
            │  125M      │
            └────────────┘
                 │
       ┌─────────┴─────────┐
Node C (10-16 GB)    Node D (10-16 GB)
┌──────────────┐    ┌──────────────┐
│  Qwen2.5-3B  │    │  Qwen2.5-3B  │
│  (4-bit)     │    │  (4-bit)     │
│  + Tester    │    │  + Debugger  │
│    LoRA      │    │    LoRA      │
└──────────────┘    └──────────────┘
```

VRAM per node: ~2.5 GB (4-bit model) + ~150 MB (LoRA adapter) + 4-12 GB (KV cache depending on sequence length) = **10-16 GB**. This is honest about KV cache overhead — bf16 training at 6 GB and 4-bit inference at 10-16 GB are different numbers.

**Network cost per expert call**: 512 bytes (latent, optional) + ~200 bytes (routing decision) + ~1 KB (text message summary). Under 2 KB per round-trip. At 4 rounds per episode, that's ~8 KB total network traffic to resolve a software bug.

**Latency**: Each expert runs full autoregressive generation locally. The network overhead is a single latent projection (one matmul) per expert call. The bottleneck is generation time, not communication.

**Scaling**: Adding more experts requires only training a new LoRA adapter (+ optional latent channel). The backbone is shared. The router learns to incorporate new experts via additional RL.

---

## Current Status

- **Stage A training**: Running on 2x A100 80GB with SDFT (2 parallel groups)
- **Latent communication module**: Built, tested, wired into expert framework — dormant until Stage C
- **Data pipeline**: Complete (~210K SFT samples processed, ~110K RL samples ready)
- **Infrastructure**: Sandbox, retrieval (BM25 + symbol index), patch assembler, artifact store — all implemented
- **Next**: Stage A validation → Stage B router RL → Stage C team RL (text-only first, then +latent)

---

## Key References

- **SDFT**: Shenfeld et al., "Self-Distillation Enables Continual Learning" (2026), arXiv:2601.19897
- **Nemotron-Cascade**: NVIDIA's agentic coding system using cascaded RL (source of our training data)
- **GRPO**: Group Relative Policy Optimization (DeepSeek), used for router and team RL
- **SWE-bench**: Standard benchmark for automated software engineering (Princeton NLP)
- **LoRA**: Low-Rank Adaptation for parameter-efficient fine-tuning (Hu et al.)
- **Qwen2.5-Coder**: Code-specialized language model family (Alibaba)
