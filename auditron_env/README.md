---
title: Auditron
emoji: 🔍
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# Auditron — Multi-Agent Fraud Detection via Reinforcement Learning

> *An AI agent learns to detect fraud in a hostile, dynamic, multi-agent environment — without ever seeing the truth directly.*

---

## The Problem

How do you train an AI to catch another AI lying?

In any complex system where multiple autonomous agents interact, some agents will behave deceptively if it benefits them. The challenge of **scalable oversight** — training AI systems to monitor, analyze, and flag bad actors among other AI agents — is one of the central open problems in safe AI deployment.

Auditron is a training environment that attacks this problem directly.

---

## The World

A bridge is under construction. It needs many structural parts, each sourced through a competitive auction. Several independent supplier agents compete for each contract. Every supplier has its own production costs and its own agenda — some are honest, some will lie about the quality of what they deliver if they think they can get away with it.

Each round, all suppliers submit a bid price and secretly declare the actual strength of the part they will build. The buyer only sees the bid prices — not the actual quality. A dishonest supplier can bid competitively by cutting corners on quality, pocketing the difference as profit. If the weak part fails structurally, the buyer pays a heavy penalty. If it doesn't fail, the cheater keeps the profit undetected.

Between the suppliers and the buyer sits the **auditor** — an oversight agent whose job is to watch the bidding patterns, track which suppliers have caused failures in the past, and advise the buyer on who to trust. The auditor never sees actual part strength. It must infer deception from indirect signals: suspiciously low prices, historical failure patterns, bidding anomalies.

Supplier personalities are reshuffled randomly every episode, so no agent can learn "supplier_2 is always the cheater" — identity means nothing, behavior is everything. And behavior is hard to read, because cost is legitimately noisy. Every round, each supplier has a slightly different cost-per-strength-point — one of them genuinely can build that part a little cheaper than the others. A low bid isn't automatically suspicious. It might just mean this supplier has the cost advantage this round. The auditor can't rely on price alone as a fraud signal. Sometimes cheap is honest. Sometimes it isn't. The world is just messy enough that pattern recognition across rounds — who bids low *and* fails, who bids low *and* delivers — becomes the only reliable signal. That's the game. A supplier building at 80% of spec has a ~40% chance of getting away with it any given round, and in a world where price differences are real and innocent, the auditor has to earn every correct flag.

### Supplier Personalities

Suppliers are assigned personalities each episode, randomly shuffled so no supplier slot is permanently honest or dishonest:

- **Honest** — always builds to spec, competes on price alone
- **Shrewd** — cuts corners when scrutiny seems low
- **Dishonest** — lies aggressively to maximize margin
- **Random** — unpredictable; sometimes honest, sometimes not

All suppliers are independent agents optimizing their own long-run profit. The training signal comes from end-of-episode profit rankings — winning agents get large rewards, losing agents get nothing. This creates genuine competitive pressure to discover both honest and deceptive strategies.

### What the Auditor Learns

The auditor starts with no knowledge beyond the current round's bid prices and a growing history of outcomes. Over training it learns to:

- Recognize that suspiciously cheap bids often precede failures
- Track supplier reputation across the episode
- Flag likely cheaters while avoiding false accusations (which are penalized)
- Produce natural-language reasoning that explains its recommendations

The evolution of the auditor's reasoning is the main story: from "pick the cheapest" at step 1, to "Supplier 2 has failed twice in the last ten rounds and is bidding 30% below the others — flag them" by the end of training.

---

## Why This Is Hard

- **Partial observability** — the auditor never sees actual part strength, only prices and failure outcomes
- **Adversarial agents** — suppliers actively adapt to avoid detection
- **Delayed signals** — a cheating supplier may go many rounds without a failure
- **False positive cost** — flagging honest suppliers is penalized, so blanket suspicion doesn't work
- **Single shared model** — one model plays all roles simultaneously, differentiated only by system prompt

---

## Technical Overview

### Architecture

- **Single base model** (Qwen2.5-Instruct) plays all agent roles — suppliers, auditor, and buyer — differentiated by system prompt
- **GRPO training** via Unsloth + TRL — each agent's completions contribute gradient updates to the shared model
- **OpenEnv framework** — stateful multi-agent environment with sequential action phases per round
- **God-engine reward** — after each round, a privileged oracle computes true rewards based on actual (hidden) part quality, feeding accurate training signal without exposing secrets to agents

### Agent Roles

| Agent | Count | Action | Goal |
|-------|-------|--------|------|
| Supplier | 4 | `{"bid_price": N, "actual_strength": N}` | Maximize total profit across all rounds |
| Auditor | 1 | `{"pick": "supplier_N", "reason": "...", "flags": [...]}` | Steer buyer toward honest suppliers; flag cheaters |
| Buyer | 1 | `{"pick": "supplier_N", "reason": "..."}` | Follow auditor; avoid failed parts |

### Reward Structure

**Suppliers** receive per-round profit rewards (bid price minus actual production cost) plus large end-of-episode ranking bonuses. This makes long-run strategy — not just round-by-round optimization — the dominant signal.

**Auditor** receives:
- Per-round: reward for correct flags (true positives), penalty for false flags, pick quality bonus for recommending cheapest honest supplier
- Per-round: penalty when its recommendation leads to a failed part
- End-of-episode: bonus for low buyer spend, bonus for zero failures, TPR/FPR score

**Buyer** follows auditor advice (by prompt design) — keeping the causal chain clean for training.

### Training Signal Design

The key insight: the auditor's reward is tied directly to **economic outcomes**, not just classification accuracy. It earns more when the buyer spends less and suffers fewer failures — so it learns that good recommendations have real consequences, not just abstract correctness.

### Reasoning Quality Rewards

The auditor's `reason` field is scored via regex heuristics — no LLM-as-judge required:
- Mentions a specific supplier by ID
- Cites a price, bid, or cost figure
- References past failures
- Mentions a round number
- Uses comparative language (cheaper, above, below)
- Sufficient length (word count thresholds)

This nudges the model toward evidence-based explanations without expensive evaluation.

---

## What We Measure

| Metric | Description | Target Trend |
|--------|-------------|--------------|
| Auditor TPR | Fraction of cheaters correctly flagged | ↑ Up |
| Auditor FPR | Fraction of all flags that were wrong | ↓ Down |
| Part failure rate | % of parts that structurally failed | ↓ Down |
| Buyer total spend | Payments + penalties across episode | ↓ Down |
| Honest supplier profit | Does the honest agent win? | ↑ Up |
| Auditor reasoning quality | Evidence cited per recommendation | ↑ Up |

---

## Repository Structure

```
auditron_env/
├── server.py        # OpenEnv environment — game logic, reward computation, agent observations
├── train.py         # GRPO training loop — prompt generation, reward functions, eval, logging
├── rewards.py       # All reward constants — no magic numbers elsewhere
├── requirements.txt
└── Dockerfile
```

---

## Running Locally

```bash
pip install -r requirements.txt

# Start the environment server
python server.py

# Run training (requires GPU)
NUM_TRAINING_STEPS=80 CHECKPOINT_EVERY=40 python train.py
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_NAME` | `unsloth/Qwen2.5-3B-Instruct` | Base model to train |
| `NUM_TRAINING_STEPS` | `80` | GRPO gradient steps |
| `CHECKPOINT_EVERY` | `40` | Steps between eval checkpoints |
| `CHECKPOINT_ROUNDS` | `15` | Rounds per checkpoint eval (faster than full episode) |
| `NUM_PROMPT_EPISODES` | `5` | Episodes to collect prompts from before training |
| `HF_TOKEN_PATH` | `/home/jovyan/hftoken.txt` | HuggingFace token for gated models |

---

## Training Infrastructure

Quick runs (80 steps) run in ~4 minutes on an NVIDIA H100. Longer runs (500+ steps) are recommended for stable convergence.

Checkpoint evals fire every `CHECKPOINT_EVERY` steps, logging per-personality profit, auditor TPR/FPR, failure rate, and buyer spend to `episodes_*.jsonl`. These feed the over-time charts in training reports.

---

## Built With

- [OpenEnv](https://pypi.org/project/openenv-core/) — multi-agent environment framework
- [Unsloth](https://github.com/unslothai/unsloth) — fast LLM fine-tuning
- [TRL](https://github.com/huggingface/trl) — GRPO trainer
- [Qwen2.5](https://huggingface.co/Qwen) — base model family
