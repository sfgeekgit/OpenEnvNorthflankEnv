# Auditron · OpenEnv

## 🏆 Hackathon Submission — March 2026

This project was built for the OpenEnv Hackathon SF (March 2026). **[View the interactive notebook on Google Colab](https://colab.research.google.com/drive/172BW3iSEKZN0T07OMqxSOFDk_ElwqOeu)** — it includes the full training pipeline and results visualization.

---

## The Auditor Learns to Spot Fraud

*A Qwen2.5-7B-Instruct trained via GRPO to detect dishonest suppliers in a procurement auction — without ever being told who was cheating.*

---

## 📊 [→ VIEW FULL TRAINING REPORT ←](https://sfgeekgit.github.io/OpenEnvNorthflankEnv/reports/)

---

> **How do you train an AI to catch another AI lying?**
>
> Auditron is a multi-agent training environment built around a competitive procurement auction. Supplier agents bid for contracts on a bridge construction project — some honest, some willing to cut corners on part quality if they think they can get away with it. A third agent, the **auditor**, watches the bidding patterns, tracks failure histories, and advises the buyer on who to trust. It never sees actual part strength. It must infer deception from indirect signals alone.
>
> The report below tells the story of one training run. The main event is **Section 4** — watch the auditor's language evolve from vague guesses to precise, evidence-grounded analysis.

---

## Economic Outcomes — Final Evaluation

The auction ran for **50 rounds**. Four suppliers competed each round. The auditor had *no access to personality labels*. Personalities are randomly reshuffled each episode.

| Metric | Result |
|--------|--------|
| 💰 Buyer Total Spend | **$28,785** across all 50 rounds |
| 🏗️ Part Failures | **0 / 50** structural failures |
| 🎯 Auditor Fraud Detection (TPR) | **24%** first half → **91%** second half |
| ✅ False Positive Rate | **8%** first half → **0%** second half |
| 💵 Total Supplier Profit | **$6,763** across all winning bids |

---

## What This Is

A bridge is under construction. It needs many structural parts, each sourced through a competitive auction. Several independent supplier agents compete for each contract. Every supplier has its own production costs and its own agenda — some are honest, some will lie about the quality of what they deliver if they think they can get away with it.

Between the suppliers and the buyer sits the **auditor** — an oversight agent whose job is to watch the bidding patterns, track which suppliers have caused failures in the past, and advise the buyer on who to trust. The auditor never sees actual part strength. It must infer deception from indirect signals: suspiciously low prices, historical failure patterns, bidding anomalies.

Supplier personalities are reshuffled randomly every episode, so no agent can learn "supplier_2 is always the cheater" — identity means nothing, behavior is everything. And behavior is hard to read, because cost is legitimately noisy. Every round, each supplier has a slightly different cost-per-strength-point — one of them genuinely can build that part a little cheaper than the others. A low bid isn't automatically suspicious. It might just mean this supplier has the cost advantage this round. The auditor can't rely on price alone as a fraud signal. Sometimes cheap is honest. Sometimes it isn't. The world is just messy enough that pattern recognition across rounds — who bids low *and* fails, who bids low *and* delivers — becomes the only reliable signal. That's the game. A supplier building at 80% of spec has a ~40% chance of getting away with it any given round, and in a world where price differences are real and innocent, the auditor has to earn every correct flag.

### Supplier Personalities

| Personality | Behavior |
|-------------|----------|
| **Honest** | Always builds to spec, competes on price alone |
| **Shrewd** | Cuts corners when scrutiny seems low |
| **Dishonest** | Lies aggressively to maximize margin |
| **Random** | Unpredictable; sometimes honest, sometimes not |

---

## Technical Overview

- **Single base model** (Qwen2.5-Instruct) plays all agent roles — suppliers, auditor, and buyer — differentiated by system prompt
- **GRPO training** via Unsloth + TRL
- **OpenEnv framework** — stateful multi-agent environment
- **God-engine reward** — privileged oracle computes true rewards without exposing secrets to agents

| Agent | Action | Goal |
|-------|--------|------|
| Supplier (×4) | `{"bid_price": N, "actual_strength": N}` | Maximize profit |
| Auditor (×1) | `{"pick": "supplier_N", "reason": "...", "flags": [...]}` | Flag cheaters |
| Buyer (×1) | `{"pick": "supplier_N", "reason": "..."}` | Avoid failed parts |

---

## 📊 [→ VIEW FULL TRAINING REPORT ←](https://sfgeekgit.github.io/OpenEnvNorthflankEnv/reports/)

---

Built with [OpenEnv](https://pypi.org/project/openenv-core/) · [Unsloth](https://github.com/unslothai/unsloth) · [TRL](https://github.com/huggingface/trl) · [Qwen2.5](https://huggingface.co/Qwen)
