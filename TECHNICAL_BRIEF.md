# Network of Experts (NoE): Coordinated Small Models via Learned Latent Communication

## Core Thesis

A network of 4 specialized 3B-parameter models, coordinating through learned latent communication, can match or exceed the performance of a single 13-32B model on complex software engineering tasks — while being distributable across commodity hardware.

This has direct implications for **decentralized inference**: instead of requiring a single node with 80GB+ VRAM to host a 70B model, the workload is distributed across 4 nodes each running a 3B model (~6GB VRAM), exchanging 512-byte latent vectors over the network. The learned router decides which expert speaks next, what context it receives, and how much compute budget it gets.

---

## Architecture

### Expert Network

| Component | Model | Parameters | VRAM (bf16) |
|-----------|-------|-----------|-------------|
| Planner | Qwen2.5-Coder-3B-Instruct + LoRA (r=32) | 3.09B + ~150M | ~6.2 GB |
| Coder | Qwen2.5-Coder-3B-Instruct + LoRA (r=16) | 3.09B + ~75M | ~6.2 GB |
| Tester | Qwen2.5-Coder-3B-Instruct + LoRA (r=16) | 3.09B + ~75M | ~6.2 GB |
| Debugger | Qwen2.5-Coder-3B-Instruct + LoRA (r=32) | 3.09B + ~150M | ~6.2 GB |
| Router | UniXcoder-base + 4 MLP heads | 125M | ~0.5 GB |
| Critic | UniXcoder-base + value head | 125M | ~0.5 GB |
| **Total unique params** | | **~3.5B** (shared backbone + role adapters) | **~25 GB** |

All 4 experts share the same frozen Qwen2.5-Coder-3B-Instruct backbone. Only the LoRA adapters are role-specific. In a decentralized setting, each node holds the full backbone plus its own adapter (~150MB) and latent channel (~5MB).

**LoRA configuration**: Target all attention + MLP projections (q, k, v, o, gate, up, down). Alpha=64, dropout=0.05. Planner and debugger use higher rank (r=32) because their tasks require more nuanced reasoning; coder and tester use r=16 since their outputs are more structured (diffs, test code).

### Router

The router is a separate 125M-parameter model (UniXcoder-base) with 4 decision heads:

1. **Expert selection** (multi-label): which experts to call this round (sigmoid over 4 logits)
2. **Stop/continue** (binary): whether the episode is done (sigmoid)
3. **Budget class** (categorical): how much compute to allocate, B0-B4 (softmax over 5)
4. **Chunk scoring** (multi-label): which context chunks to include (sigmoid over up to 64 chunks)

The router sees a compressed state packet (512 tokens max) summarizing the episode so far: which experts have run, what they produced, current test/compile status, and budget consumed.

### Communication Channels

Experts communicate through two channels:

**1. Text messages (v1, active now)**: Typed JSON messages routed through a star topology. Each message has a type (PLAN, PATCH_HUNK, EXEC_REPORT, BUG_REPORT, etc.), a compact summary (64-96 tokens), and a reference to the full artifact in storage. Human-readable and debuggable.

**2. Latent vectors (v1.5, architecture built, trained in Stage C)**: Each expert has a projector (Linear 2048→256 + LayerNorm, ~525K params) that compresses its hidden states into a 256-dim latent vector, and a receiver (~2.1M params) that injects incoming latent vectors as 4 virtual tokens prepended to input embeddings.

The receiver has a **learned gate** initialized nearly closed (sigmoid(-5) ≈ 0.007). The system starts operating on text messages only. During Stage C reinforcement learning, the gate opens if and only if latent communication improves the episode reward. If it doesn't help, the gate stays closed — zero degradation.

**Bandwidth comparison:**

| Channel | Size per message | Requires decoding |
|---------|-----------------|-------------------|
| Text message | ~400-800 bytes (100-200 tokens) | Yes (autoregressive) |
| Latent vector | 512 bytes (256 × bf16) | No (single matmul) |

For decentralized inference, latent vectors are cheaper in both bandwidth and compute — no autoregressive generation required, just a linear projection.

### Budget System

The router assigns a compute budget class to each expert call:

| Class | Max Input | Max Generation | Max Message | Tools |
|-------|-----------|---------------|-------------|-------|
| B0 | 768 | 192 | 64 | 0 |
| B1 | 1,536 | 384 | 96 | 1 |
| B2 | 3,072 | 768 | 128 | 2 |
| B3 | 4,608 | 1,024 | 192 | 3 |
| B4 | 6,144 | 2,048 | 256 | 3 |

Default assignments: Planner=B2, Coder=B4, Tester=B2, Debugger=B2. The router learns to adjust these — easy bugs might only need B1 for the coder, while complex multi-file issues might warrant B4.

---

## Episode Flow

A typical episode for resolving a software engineering task:

```
Round 1: Router → Planner
         Planner receives: issue description + repo map + retrieved files
         Planner produces: PLAN (files_to_touch, invariants, risks, strategy)

Round 2: Router → Coder
         Coder receives: issue + plan + selected code slices
         Coder produces: PATCH (unified diff via PATCH_PLAN → HUNK → DONE)
         Sandbox: applies patch, runs compile + lint + tests

Round 3: Router → Tester + Debugger (parallel)
         Tester receives: issue + patch diff + touched symbols
         Tester produces: EXEC_REPORT (new test cases + results)
         Debugger receives: issue + patch + failing tests + traces
         Debugger produces: BUG_REPORT (root cause + suspect files + fix hint)

Round 4: Router decision
         If tests pass → ACCEPT
         If fixable → Coder repair (up to 2 repair loops)
         If unfixable → ROLLBACK
```

Patch assembly is **transactional**: if any hunk in a file is invalid, the entire file's patch is rejected with a structured error, and the coder gets a retry with the error context. No silent partial applies.

---

## Training Data

### Stage A: Individual Role SFT

**Primary corpus: NVIDIA Nemotron-Cascade-SFT-SWE** (210,823 samples total)

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

**NVIDIA Nemotron-Cascade-RL-SWE** (~141K samples), filtered to short fixes (≤1 file changed) for router training. Each sample provides:
- `prompt`: the issue/problem statement
- `golden_patch`: the reference fix (unified diff)
- `relevant_file_contents`: list of {file_path, content} for context
- `instance_id` and `source` for tracking

Supplemented with SWE-bench train (small patches) to provide diversity in repo structure and issue types. These give the router real routing decisions to learn from — which expert to call, what budget to assign, which chunks to include.

### Stage C: Team RL Task Pool

Full Nemotron-Cascade-RL-SWE (~141K) + SWE-bench train. Repo-level tasks with golden patches and FAIL_TO_PASS test specifications.

### Evaluation (never trained on)

**SWE-bench Verified** (500 instances): Curated subset of SWE-bench with human-verified solutions. This is the standard benchmark for evaluating automated software engineering systems.

---

## Training Pipeline

### Stage A: Individual Role SFT (Current — running)

Each expert's LoRA adapter is trained independently on its role-specific data:
- **Optimizer**: AdamW (β1=0.9, β2=0.95), lr=2e-4, cosine schedule, 100-step warmup
- **Batch**: effective batch 8 (per_device=2, gradient_accumulation=4)
- **Precision**: bf16 with gradient checkpointing
- **Epochs**: 3, early stopping on validation loss (patience=3)
- **Loss**: causal LM loss on output tokens only (input tokens masked with label=-100)
- **Hardware**: 2× A100 80GB, 2 parallel groups (GPU 0: coder+planner, GPU 1: debugger+tester)

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

**Gate**: Team must beat single-expert baseline by ≥5% on held-out tasks. At least 30% of successes must use >1 expert (proving routing adds value).

### Stage C: Team RL with Latent Communication

End-to-end fine-tuning of the full system, including latent communication channels:

**Unfreezing schedule**:
- Episodes 0-5K: router only (latent projectors/receivers included)
- Episodes 5K+: + coder LoRA + debugger LoRA
- Episodes 15K+: + tester LoRA (if test quality is the bottleneck)
- Planner: stays frozen throughout (small dataset, high risk of forgetting)

**Reward function**:
```
Phi(s) = 0.50 × test_pass_rate + 0.20 × compile_ok
       + 0.10 × lint_ok + 0.10 × coverage_gain + 0.10 × verifier_confidence

Step reward: r_t = Phi(s_{t+1}) - Phi(s_t) - cost_penalties
Terminal:    +1.0 if hidden test suite passes

Cost penalties (annealed over 20K episodes):
  0.02 per tool call, 0.01 per round, 5e-5 per message token

Attribution: 90% shared, 10% role bonuses
```

The cost penalties are critical — without them, the system learns to maximize accuracy by running every expert at max budget every round. The penalties force the router to learn **efficiency**: call only the experts that help, at the minimum budget that works.

**Latent gate learning**: The projector and receiver parameters are unfrozen from episode 0. Since the gate starts at ~0.007, the latent channel initially contributes almost nothing. As RL explores, if sending a latent vector from the planner to the coder improves patch quality, the gradient will push the gate open. This is conditioned on episode return, not on reconstruction loss — the system only learns to use latent communication when it actually helps the task.

---

## Evaluation Plan

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
| Router | Expert recall@2, budget accuracy, stop accuracy, ECE |

### Ablations (8 experiments)

Each ablation removes one component to measure its contribution:

1. **Fixed workflow** vs learned router (is routing worth learning?)
2. **Single LoRA** vs role-specific adapters (is specialization worth it?)
3. **Free text** vs typed messages (does structure help?)
4. **No tester** (is test generation needed?)
5. **No debugger** (is localization needed?)
6. **No retrieval** (is BM25/symbol search needed?)
7. **No critic** (is the value function needed for RL?)
8. **No cost penalty** (does efficiency pressure help?)

The most important ablation for the thesis is **#2**: if 4 specialized models with routing significantly beat a single model with the same total compute, that validates the network-of-experts approach. If latent communication further improves over text-only, that validates the decentralized inference thesis.

---

## Decentralized Inference Architecture

The trained NoE system maps naturally to decentralized deployment:

```
Node A (6GB VRAM)          Node B (6GB VRAM)
┌──────────────┐           ┌──────────────┐
│  Qwen2.5-3B  │           │  Qwen2.5-3B  │
│  + Planner   │   512B    │  + Coder     │
│    LoRA      │ ───────── │    LoRA      │
│  + Projector │  latent   │  + Receiver  │
└──────────────┘           └──────────────┘
       │                          │
       │        Router            │
       │    ┌────────────┐        │
       └────│ UniXcoder  │────────┘
            │  125M      │
            └────────────┘
                 │
       ┌─────────┴─────────┐
Node C (6GB VRAM)    Node D (6GB VRAM)
┌──────────────┐    ┌──────────────┐
│  Qwen2.5-3B  │    │  Qwen2.5-3B  │
│  + Tester    │    │  + Debugger  │
│    LoRA      │    │    LoRA      │
└──────────────┘    └──────────────┘
```

**Network cost per expert call**: 512 bytes (latent) + ~200 bytes (routing decision) + ~1KB (text message summary). Under 2KB per round-trip. At 4 rounds per episode, that's ~8KB total network traffic to resolve a software bug.

**Latency**: Each expert runs full autoregressive generation locally. The network overhead is a single latent projection (one matmul) per expert call. The bottleneck is generation time, not communication.

**Scaling**: Adding more experts (e.g., a security reviewer, a documentation writer) requires only training a new LoRA adapter + latent channel. The backbone is shared. The router learns to incorporate new experts via additional RL.

---

## Current Status

- **Stage A training**: Running on 2× A100 80GB (2 parallel groups)
- **Latent communication module**: Built, tested (46 unit tests passing), wired into expert framework
- **Data pipeline**: Complete (210K SFT samples processed, 141K RL samples ready)
- **Infrastructure**: Sandbox, retrieval (BM25 + symbol index), patch assembler, artifact store — all implemented
- **Next**: Stage A validation → Stage B router RL → Stage C team RL with latent

---

## Key References

- **Nemotron-Cascade**: NVIDIA's agentic coding system using cascaded RL (source of our training data)
- **GRPO**: Group Relative Policy Optimization (DeepSeek), used for router and team RL
- **SWE-bench**: Standard benchmark for automated software engineering (Princeton NLP)
- **LoRA**: Low-Rank Adaptation for parameter-efficient fine-tuning (Hu et al.)
- **Qwen2.5-Coder**: Code-specialized language model family (Alibaba)
