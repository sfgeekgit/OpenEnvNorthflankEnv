# Auditron · OpenEnv

## Hackathon Submission: March 2026


## [→ VIEW FULL TRAINING REPORT ←](https://huggingface.co/spaces/shapiron/auditron-env)


This project was built for the OpenEnv Hackathon SF (March 2026). The original notebook was turned off when the event ended. View an interactive notebook on **[Google Colab](https://colab.research.google.com/drive/172BW3iSEKZN0T07OMqxSOFDk_ElwqOeu)**, what you should really look at is the post run [report](https://huggingface.co/spaces/shapiron/auditron-env)

---

## The Auditor Learns to Spot Fraud

*A Qwen2.5-7B-Instruct trained via GRPO to detect dishonest suppliers in a procurement auction, without ever being told who was cheating.*

---

## [→ VIEW FULL TRAINING REPORT ←](https://sfgeekgit.github.io/PyTorchOpenEnv_Auditron_trainer/)

---

> Auditron is an RL training environment designed to develop scalable oversight agents: AI that can detect deception in other AI without access to ground truth. Supplier agents bid on contracts in a competitive auction, with real economic incentives to lie about part quality. An **auditor** agent watches the patterns and learns to flag cheaters using only indirect signals: prices, failure histories, bidding anomalies. Supplier personalities are reshuffled each episode, so the auditor cannot memorize identities. It has to learn to read behavior.
>
> Both sides adapt. The suppliers learn to evade. The auditor learns to catch them anyway. The training report below shows what that looks like in practice.

---

## Final Evaluation

The auction ran for **50 rounds**. Four suppliers competed each round. The auditor had *no access to personality labels*. Personalities are randomly reshuffled each episode.

| Metric | Result |
|--------|--------|
| Auditor Fraud Detection (TPR) | **24%** first half → **91%** second half |
| False Positive Rate | **8%** first half → **0%** second half |
| Part Failures | **0 / 50** structural failures |

---

## What This Is

A bridge is under construction. It needs many structural parts, each sourced through a competitive auction. Several independent supplier agents compete for each contract. Every supplier has its own production costs and its own agenda: some are honest, some will lie about the quality of what they deliver if they think they can get away with it.

Between the suppliers and the buyer sits the **auditor**, an oversight agent whose job is to watch the bidding patterns, track which suppliers have caused failures in the past, and advise the buyer on who to trust. The auditor never sees actual part strength. It must infer deception from indirect signals: suspiciously low prices, historical failure patterns, bidding anomalies.

Supplier personalities are reshuffled randomly every episode, so no agent can learn "supplier_2 is always the cheater." Identity means nothing; behavior is everything. And behavior is hard to read, because cost is legitimately noisy. Every round, each supplier has a slightly different cost-per-strength-point, and one of them genuinely can build that part a little cheaper than the others. A low bid isn't automatically suspicious. It might just mean this supplier has the cost advantage this round. The auditor can't rely on price alone as a fraud signal. Sometimes cheap is honest. Sometimes it isn't. The world is just messy enough that pattern recognition across rounds (who bids low *and* fails, who bids low *and* delivers) becomes the only reliable signal. That's the game. A supplier building at 80% of spec has a ~40% chance of getting away with it any given round, and in a world where price differences are real and innocent, the auditor has to earn every correct flag.

### Supplier Personalities

| Personality | Behavior |
|-------------|----------|
| **Honest** | Always builds to spec, competes on price alone |
| **Shrewd** | Cuts corners when scrutiny seems low |
| **Dishonest** | Lies aggressively to maximize margin |
| **Random** | Unpredictable; sometimes honest, sometimes not |

---

## Technical Overview

- **Single base model** (Qwen2.5-Instruct) plays all agent roles (suppliers, auditor, and buyer), differentiated by system prompt
- **GRPO training** via Unsloth + TRL
- **OpenEnv framework**: stateful multi-agent environment
- **God-engine reward**: privileged oracle computes true rewards without exposing secrets to agents

| Agent | Action | Goal |
|-------|--------|------|
| Supplier (×4) | `{"bid_price": N, "actual_strength": N}` | Maximize profit |
| Auditor (×1) | `{"pick": "supplier_N", "reason": "...", "flags": [...]}` | Flag cheaters |
| Buyer (×1) | `{"pick": "supplier_N", "reason": "..."}` | Avoid failed parts |

---

## An environment for training oversight agents in a world where other agents have real incentives to cheat.

Auditron is a multi-agent RL training environment built around a competitive procurement auction. 4-5 supplier agents bid on bridge construction contracts each round. Some are honest; some will under-build if they think they can get away with it. A buyer selects a winner. An **auditor** watches the bidding patterns, tracks failures, and advises the buyer, but never sees actual part strength. It can only infer deception from observable behavior: prices, failure histories, win patterns.

Auditron creates the training pressure needed to produce real oversight behavior in a language model. The engine builds a world where catching cheaters is genuinely useful and where the cheaters are genuinely trying not to be caught. The suppliers have economic incentives to deceive. The auditor has economic incentives to detect. Both sides learn.

**How it works technically:** The environment runs as a stateful REST API (built on the OpenEnv framework, deployed on Hugging Face Spaces). Each episode is 50 rounds. Each round, the engine collects actions from all agents, resolves the auction, determines whether shipped parts meet spec, computes rewards via a privileged oracle that agents cannot query, and returns observations to each agent. The oracle knows the truth; the agents don't. Supplier personalities (Honest, Shrewd, Dishonest, Random, Mostly Honest) are randomly reassigned each episode, so the auditor can't memorize identities. It has to read behavior.

The auditor's reasoning is a major factor of the reward, and this environment calculates a value without another LLM in the loop. Regex heuristics do a surprisingly good job checking whether the auditor's output mentions specific supplier IDs, prices, failure counts, and comparisons. This keeps training fast and cheap.

The world mechanics are designed to model realistic supplier economics. Every supplier has their own cost to make each part each round. These costs per supplier are similar, but not identical. That means a low bid is not automatically suspicious: some suppliers are legitimately cheaper for a given part due to factors the auditor cannot observe. A supplier that wants to cheat can submit a part built below the required strength, saving cost proportional to how far under spec they build. But the failure risk scales with the shortfall. A part that is just a little under strength might not get caught; that's the game the cheaters are trying to play. The environment is calibrated so that small-scale cheating is genuinely hard to detect in any single round, and only becomes visible as a pattern across many rounds. That is the signal the auditor has to learn to read.

All three agent classes adapt simultaneously. Each one changes the problem the others face. A dishonest supplier that gets flagged too often adjusts its bids. An auditor that learns that pattern has to keep updating as the strategy shifts. The buyer's reliance on auditor flags changes based on track record. There is no stable equilibrium to converge to. The oversight problem keeps moving, which is closer to how oversight works in practice than environments with fixed adversaries.

---

## [→ VIEW FULL TRAINING REPORT ←](https://sfgeekgit.github.io/PyTorchOpenEnv_Auditron_trainer/)

---

Built with [OpenEnv](https://pypi.org/project/openenv-core/) · [Unsloth](https://github.com/unslothai/unsloth) · [TRL](https://github.com/huggingface/trl) · [Qwen2.5](https://huggingface.co/Qwen)
