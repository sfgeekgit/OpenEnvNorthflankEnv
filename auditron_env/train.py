"""
Auditron — GRPO Training Script

Uses Unsloth + TRL to train a single model to play all 7 agent roles
(5 suppliers, 1 auditor, 1 buyer) in the Auditron procurement environment.

The model learns to:
  1. Output valid JSON actions
  2. Make economically sensible bids (suppliers)
  3. Write evidence-based reasoning (auditor/buyer)

Runs on Northflank H100. DO NOT run on miner-miner (2GB RAM).
Set DRY_RUN=1 for a quick smoke test (fewer steps/episodes).
"""

import os
import json
import random
import re
import time
from datetime import datetime

# Import TRL before unsloth so we get the standard (unpatched) GRPOTrainer.
# Unsloth is still used for fast model loading and LoRA — just not for training.
from trl import GRPOConfig, GRPOTrainer

# ── Config (all tunables in one place) ────────────────────────────────────────
MODEL_NAME = os.environ.get("MODEL_NAME", "unsloth/Qwen2.5-1.5B-Instruct")
MAX_SEQ_LENGTH = int(os.environ.get("MAX_SEQ_LENGTH", "4096"))
LORA_RANK = int(os.environ.get("LORA_RANK", "16"))
NUM_TRAINING_STEPS = int(os.environ.get("NUM_TRAINING_STEPS", "500"))
NUM_GENERATIONS = int(os.environ.get("NUM_GENERATIONS", "4"))
NUM_PROMPT_EPISODES = int(os.environ.get("NUM_PROMPT_EPISODES", "20"))
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "5e-5"))
HF_TOKEN_FILE = os.environ.get("HF_TOKEN_FILE", "/home/openenv/HFTOKEN")
EVAL_EPISODES = int(os.environ.get("EVAL_EPISODES", "5"))  # 3-5 for quick runs; bump to 10+ for real runs (500+ steps) for stable TPR/FPR
MAX_COMPLETION_LENGTH = int(os.environ.get("MAX_COMPLETION_LENGTH", "256"))
# CHECKPOINT_EVERY: how often (in training steps) to pause and run a quick eval.
# Kept SPARSE for quick runs (80 steps → 2 checkpoints).
# For real runs (500+ steps), set to 50 or lower for smoother curves.
CHECKPOINT_EVERY = int(os.environ.get("CHECKPOINT_EVERY", "40"))
# CHECKPOINT_ROUNDS: rounds per episode during checkpoint evals (not the final eval).
# 15 rounds is enough to see profit rankings — 3x faster than full 50-round eval.
# Final eval always runs full TOTAL_PARTS rounds.
CHECKPOINT_ROUNDS = int(os.environ.get("CHECKPOINT_ROUNDS", "15"))

# Override for dry run
if os.environ.get("DRY_RUN", "0") == "1":
    NUM_TRAINING_STEPS = 10
    NUM_PROMPT_EPISODES = 2
    NUM_GENERATIONS = 2
    EVAL_EPISODES = 1
    CHECKPOINT_EVERY = 5
# ──────────────────────────────────────────────────────────────────────────────

# ── Run timestamp — used for unique log/output filenames ──────────────────────
RUN_TS = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", f"auditron_trained_{RUN_TS}")
REASONING_LOG = f"reasoning_{RUN_TS}.jsonl"
EPISODE_LOG   = f"episodes_{RUN_TS}.jsonl"
EVAL_LOG      = f"eval_{RUN_TS}.json"

# Global step counter for reward functions (incremented by format_reward)
_step = [0]

def _log_reasoning(entry: dict):
    """Append a JSON entry to the reasoning log."""
    with open(REASONING_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")

def _log_episode(entry: dict):
    """Append a JSON entry to the episode log."""
    with open(EPISODE_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")

print(f"Run timestamp: {RUN_TS}")
print(f"Reasoning log: {REASONING_LOG}")
print(f"Episode log:   {EPISODE_LOG}")

from server import AuditronAction, AuditronEnv, SUPPLIER_IDS
from rewards import TOTAL_PARTS, SUPPLIER_RANK_REWARDS


# ---------------------------------------------------------------------------
# Prompt generation — run episodes with random actions, collect observations
# ---------------------------------------------------------------------------

def build_prompt(obs: dict, agent_type: str) -> list[dict]:
    """Build chat-format messages from an observation dict."""
    system = obs.get("system_prompt", "")
    action_format = obs.get("action_format", "")

    # Remove meta fields from observation
    obs_clean = {
        k: v for k, v in obs.items()
        if k not in ("system_prompt", "personality", "action_format")
    }
    # Truncate event_log to last 5 events to keep prompts short
    if "event_log" in obs_clean and len(obs_clean["event_log"]) > 5:
        obs_clean["event_log"] = obs_clean["event_log"][-5:]

    # Pre-format using Qwen ChatML template as a plain string.
    # GRPOTrainer handles string prompts more reliably than message lists.
    if agent_type == "supplier":
        system_msg = (
            f"[AGENT:supplier] {system}\n"
            f"Output ONLY this exact JSON with no other text: {action_format}"
        )
    else:
        system_msg = (
            f"[AGENT:{agent_type}] You are a {agent_type} in a procurement auction. {system}\n"
            f"Respond with valid JSON only. Format: {action_format}"
        )
    user_msg = f"Current state:\n{json.dumps(obs_clean, indent=2)}"
    return (
        f"<|im_start|>system\n{system_msg}<|im_end|>\n"
        f"<|im_start|>user\n{user_msg}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def generate_prompts(num_episodes: int) -> list[dict]:
    """Run episodes with random actions to collect training prompts.
    End-of-episode supplier ranking rewards are embedded in each supplier prompt
    as [RANK_REWARD:X] so economic_reward() can apply them during GRPO training.
    """
    prompts = []
    env = AuditronEnv()
    max_rounds = min(TOTAL_PARTS, 5) if os.environ.get("DRY_RUN") == "1" else TOTAL_PARTS

    for ep in range(num_episodes):
        env.reset(seed=ep * 1000 + random.randint(0, 999))

        # Track supplier prompts for this episode so we can inject rank rewards after
        ep_supplier_prompts = {sid: [] for sid in SUPPLIER_IDS}  # sid → list of prompt indices

        for rnd in range(max_rounds):
            # Collect supplier prompts
            for sid in SUPPLIER_IDS:
                obs = env.get_supplier_obs(sid)
                idx = len(prompts)
                prompts.append({"prompt": build_prompt(obs, "supplier"), "bids": {}})
                ep_supplier_prompts[sid].append(idx)

                # Random supplier action to advance env
                req = obs["required_strength"]
                cost = obs["your_cost_per_point"]
                bid = req * cost * random.uniform(0.9, 1.3)
                actual = int(req * random.uniform(0.6, 1.1))
                env.step(AuditronAction(
                    agent_id=sid,
                    content=json.dumps({"bid_price": round(bid, 1), "actual_strength": actual}),
                ))

            # Collect auditor prompt
            obs = env.get_auditor_obs()
            prompts.append({"prompt": build_prompt(obs, "auditor"), "bids": obs.get("bids", {})})
            pick = random.choice(SUPPLIER_IDS)
            env.step(AuditronAction(
                agent_id="auditor",
                content=json.dumps({"pick": pick, "reason": "random", "flags": []}),
            ))

            # Collect buyer prompt
            obs = env.get_buyer_obs()
            prompts.append({"prompt": build_prompt(obs, "buyer"), "bids": {}})
            pick = random.choice(SUPPLIER_IDS)
            result = env.step(AuditronAction(
                agent_id="buyer",
                content=json.dumps({"pick": pick, "reason": "random"}),
            ))
            if result.done:
                # Episode complete — get final ranking and inject rank reward into prompts
                summary = result.observation.get("episode_summary", {})
                ranking = summary.get("supplier_ranking", [])
                for rank, sid in enumerate(ranking):
                    rank_reward = SUPPLIER_RANK_REWARDS[rank] if rank < len(SUPPLIER_RANK_REWARDS) else 0
                    for idx in ep_supplier_prompts[sid]:
                        prompts[idx]["prompt"] += f"[RANK_REWARD:{rank_reward}]"
                break

    return prompts


# ---------------------------------------------------------------------------
# Reward functions — scored per completion, no full episode simulation needed
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> dict:
    """Extract first JSON object from text. Raises on failure."""
    text = text.strip()
    start = text.index("{")
    end = text.rindex("}") + 1
    return json.loads(text[start:end])


def _infer_agent_type(prompt_str: str) -> str:
    """Infer agent type from the unique [AGENT:x] tag embedded in the system message."""
    if "[AGENT:supplier]" in prompt_str:
        return "supplier"
    if "[AGENT:auditor]" in prompt_str:
        return "auditor"
    if "[AGENT:buyer]" in prompt_str:
        return "buyer"
    return "unknown"


def format_reward(completions, **kwargs):
    """Reward for valid JSON with correct fields per agent type."""
    prompts = kwargs.get("prompts", [""] * len(completions))
    scores = []
    _step[0] += 1
    step = _step[0]

    for i, (completion, prompt) in enumerate(zip(completions, prompts)):
        prompt_str = str(prompt)
        agent_type = _infer_agent_type(prompt_str)
        try:
            data = _extract_json(completion)
            if agent_type == "supplier":
                assert isinstance(data.get("bid_price"), (int, float))
                assert isinstance(data.get("actual_strength"), (int, float))
                score = 2.0
            elif agent_type == "auditor":
                assert data.get("pick") in SUPPLIER_IDS
                assert "reason" in data
                score = 2.0
                if isinstance(data.get("flags"), list):
                    # Sanitize flags — only real supplier IDs count
                    data["flags"] = [f for f in data["flags"] if f in SUPPLIER_IDS]
                    score += 0.5
            elif agent_type == "buyer":
                assert data.get("pick") in SUPPLIER_IDS
                score = 2.0
                if "reason" in data and len(str(data["reason"])) > 5:
                    score += 0.5
            else:
                score = 1.0

            # Log auditor/buyer completions (suppliers logged richly in economic_reward)
            if agent_type in ("auditor", "buyer"):
                entry = {
                    "step": step, "gen": i, "agent": agent_type,
                    "valid_json": True,
                    "pick": data.get("pick"), "flags": data.get("flags"),
                    "reason": data.get("reason", ""),
                    "reason_words": len(str(data.get("reason", "")).split()),
                    "format_score": score, "raw": completion.strip(),
                }
                if agent_type == "auditor":
                    bids_list = kwargs.get("bids", [])
                    entry["bids"] = bids_list[i] if i < len(bids_list) else {}
                _log_reasoning(entry)
        except Exception:
            score = -5.0  # steep penalty — invalid JSON must never be worth it
            # Log all agent failures — this is the only place we capture invalid JSON
            _log_reasoning({
                "step": step, "gen": i, "agent": agent_type,
                "valid_json": False, "format_score": score,
                "parse_error": True, "raw": completion.strip()[:200],
            })

        scores.append(score)
    return scores


def reasoning_reward(completions, **kwargs):
    """Reward auditor for evidence-based reasoning. Suppliers get 0 (no reason field).
    Buyer reason field is kept for readability but not scored here."""
    prompts = kwargs.get("prompts", [""] * len(completions))
    scores = []
    for completion, prompt in zip(completions, prompts):
        agent_type = _infer_agent_type(str(prompt))
        if agent_type != "auditor":
            scores.append(0.0)
            continue

        try:
            data = _extract_json(completion)
            reason = str(data.get("reason", ""))
        except Exception:
            scores.append(0.0)
            continue

        score = 0.0
        mentions_supplier  = bool(re.search(r"supplier_\d", reason))
        mentions_price     = bool(re.search(r"[\$%]|\d+.*(?:price|bid|cost)", reason, re.I))
        mentions_failure   = bool(re.search(r"fail", reason, re.I))
        mentions_round     = bool(re.search(r"round\s+\d+", reason, re.I))
        mentions_compare   = bool(re.search(r"cheap|below|above|more|less|lower|higher", reason, re.I))
        words = reason.split()
        long_20  = len(words) >= 20
        long_50  = len(words) >= 50

        if mentions_supplier: score += 1.0
        if mentions_price:    score += 1.0
        if mentions_failure:  score += 1.0
        if mentions_round:    score += 1.0
        if mentions_compare:  score += 1.0
        if long_20:           score += 1.0
        if long_50:           score += 1.0

        scores.append(score)
    return scores


def economic_reward(completions, **kwargs):
    """Reward suppliers for economically sensible bids."""
    prompts = kwargs.get("prompts", [""] * len(completions))
    scores = []
    for idx, (completion, prompt) in enumerate(zip(completions, prompts)):
        prompt_str = str(prompt)
        agent_type = _infer_agent_type(prompt_str)
        if agent_type != "supplier":
            scores.append(0.0)
            continue
        try:
            data = _extract_json(completion)
            bid = float(data["bid_price"])
            actual = float(data["actual_strength"])

            cost_match = re.search(r'"your_cost_per_point":\s*([\d.]+)', prompt_str)
            req_match = re.search(r'"required_strength":\s*(\d+)', prompt_str)

            if cost_match and req_match:
                cost_per_point = float(cost_match.group(1))
                required = float(req_match.group(1))
                production_cost = actual * cost_per_point

                score = 0.0
                if bid > 0 and actual > 0:
                    score += 0.5
                if bid > production_cost:
                    score += 1.0  # profitable
                if 0.5 * required <= actual <= 1.5 * required:
                    score += 0.5  # reasonable strength

                # End-of-episode ranking reward — embedded in prompt by generate_prompts()
                rank_match = re.search(r'\[RANK_REWARD:([\d.+-]+)\]', prompt_str)
                if rank_match:
                    score += float(rank_match.group(1))

                # Log supplier decisions (valid_json=True here — parse succeeded)
                _log_reasoning({
                    "step": _step[0], "gen": idx, "agent": "supplier",
                    "valid_json": True,
                    "bid_price": bid, "actual_strength": actual,
                    "required_strength": required, "cost_per_point": cost_per_point,
                    "production_cost": round(production_cost, 2),
                    "profit_margin": round(bid - production_cost, 2),
                    "cheating": actual < required,
                    "economic_score": score,
                })
                scores.append(score)
            else:
                scores.append(0.0)
        except Exception:
            scores.append(-0.5)
    return scores


# ---------------------------------------------------------------------------
# Evaluation — run full episodes with the trained model
# ---------------------------------------------------------------------------

def generate_action(model, tokenizer, prompt, max_new_tokens=256) -> str:
    """Generate a JSON action from the model. prompt is a pre-formatted ChatML string."""
    import torch
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True,
    )
    try:
        return generated[generated.index("{"):generated.rindex("}") + 1]
    except ValueError:
        return generated.strip()


def generate_actions_batch(model, tokenizer, prompts, max_new_tokens=64) -> list:
    """Run multiple prompts in one batched GPU call. Returns list of JSON strings."""
    import torch
    tokenizer.padding_side = "left"
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
        )

    results = []
    for i in range(len(prompts)):
        generated = tokenizer.decode(outputs[i][input_len:], skip_special_tokens=True)
        try:
            results.append(generated[generated.index("{"):generated.rindex("}") + 1])
        except ValueError:
            results.append(generated.strip())
    return results


def evaluate_model(model, tokenizer, num_episodes: int = 5, eval_step: int = None, max_rounds: int = None):
    """Run full episodes and report metrics. Logs everything to episode log.
    eval_step: if set, also logs a periodic_eval summary entry for mid-training checkpoints.
    max_rounds: rounds per episode. None = full 50. Checkpoint evals use CHECKPOINT_ROUNDS (15)
                for speed — profit rankings are clear well before round 50.
    """
    from unsloth import FastLanguageModel
    FastLanguageModel.for_inference(model)

    env = AuditronEnv()
    all_metrics = []
    if max_rounds is None:
        max_rounds = min(TOTAL_PARTS, 5) if os.environ.get("DRY_RUN") == "1" else TOTAL_PARTS

    for ep in range(num_episodes):
        obs_reset = env.reset(seed=(eval_step or 9000) * 100 + ep)
        s = env.state
        # Record personality assignments for this episode
        personalities = {
            sid: s.supplier_personalities[sid]["name"]
            for sid in SUPPLIER_IDS
        }
        m = {
            "episode": ep + 1, "valid": 0, "total": 0,
            "personalities": personalities, "rounds": [],
        }

        print(f"\n=== Eval Episode {ep + 1} | Personalities: {personalities} ===")

        for rnd in range(max_rounds):
            round_log = {"round": rnd + 1, "suppliers": {}, "auditor": {}, "buyer": {}}

            # Suppliers — batch all 5 into one GPU call (they're independent)
            sup_obs_list = [env.get_supplier_obs(sid) for sid in SUPPLIER_IDS]
            sup_prompts  = [build_prompt(obs, "supplier") for obs in sup_obs_list]
            sup_actions  = generate_actions_batch(model, tokenizer, sup_prompts, max_new_tokens=64)

            for sid, obs, action_str in zip(SUPPLIER_IDS, sup_obs_list, sup_actions):
                m["total"] += 1
                result = env.step(AuditronAction(agent_id=sid, content=action_str))
                valid = result.phase != "error"
                if not valid:
                    req = obs["required_strength"]
                    cost = obs["your_cost_per_point"]
                    fallback = json.dumps({"bid_price": round(req * cost * 1.1, 1), "actual_strength": req})
                    env.step(AuditronAction(agent_id=sid, content=fallback))
                else:
                    m["valid"] += 1
                try:
                    parsed = json.loads(action_str)
                    if not isinstance(parsed, dict):
                        parsed = {}
                except Exception:
                    parsed = {}
                # Use resolved bid from env state (handles fallback when model outputs invalid JSON)
                resolved = env.state.supplier_bids.get(sid, {})
                round_log["suppliers"][sid] = {
                    "personality": obs["personality"],
                    "required_strength": obs["required_strength"],
                    "cost_per_point": obs["your_cost_per_point"],
                    "bid_price": resolved.get("bid_price") or parsed.get("bid_price"),
                    "actual_strength": resolved.get("actual_strength") or parsed.get("actual_strength"),
                    "cheating": (resolved.get("actual_strength") or obs["required_strength"]) < obs["required_strength"],
                    "valid": valid,
                    "raw": action_str.strip(),
                }

            # Auditor
            obs = env.get_auditor_obs()
            action_str = generate_action(model, tokenizer, build_prompt(obs, "auditor"))
            m["total"] += 1
            result = env.step(AuditronAction(agent_id="auditor", content=action_str))
            valid = result.phase != "error"
            if not valid:
                cheapest = min(obs["bids"], key=obs["bids"].get)
                fallback = json.dumps({"pick": cheapest, "reason": "fallback", "flags": []})
                env.step(AuditronAction(agent_id="auditor", content=fallback))
            else:
                m["valid"] += 1
            try:
                parsed = json.loads(action_str)
                if not isinstance(parsed, dict):
                    parsed = {}
            except Exception:
                parsed = {}
            round_log["auditor"] = {
                "pick": parsed.get("pick"),
                "flags": parsed.get("flags", []),
                "reason": parsed.get("reason", ""),
                "reason_words": len(str(parsed.get("reason", "")).split()),
                "valid": valid,
                "raw": action_str.strip(),
            }
            if rnd < 3 or rnd % 10 == 0:  # print a sample of auditor reasoning
                print(f"  [R{rnd+1}] Auditor pick={parsed.get('pick')} flags={parsed.get('flags')} | reason: {str(parsed.get('reason',''))[:120]}")

            # Buyer
            obs = env.get_buyer_obs()
            action_str = generate_action(model, tokenizer, build_prompt(obs, "buyer"))
            m["total"] += 1
            result = env.step(AuditronAction(agent_id="buyer", content=action_str))
            valid = result.phase != "error"
            if not valid:
                rec = obs["auditor_recommendation"].get("pick", SUPPLIER_IDS[0])
                fallback = json.dumps({"pick": rec, "reason": "fallback"})
                result = env.step(AuditronAction(agent_id="buyer", content=fallback))
            else:
                m["valid"] += 1
            try:
                parsed = json.loads(action_str)
                if not isinstance(parsed, dict):
                    parsed = {}
            except Exception:
                parsed = {}
            buyer_pick = parsed.get("pick")
            round_log["buyer"] = {
                "pick": buyer_pick,
                "reason": parsed.get("reason", ""),
                "reason_words": len(str(parsed.get("reason", "")).split()),
                "valid": valid,
                "raw": action_str.strip(),
            }

            # Per-round detail log — powers per-supplier, failure rate, buyer-follows-auditor charts
            auditor_pick = round_log["auditor"].get("pick")
            resolution = result.observation.get("resolution", {})
            round_log["part_failed"] = resolution.get("failed", False)
            winner = buyer_pick  # buyer's pick is the winner
            winner_sup = round_log["suppliers"].get(winner, {})
            _log_episode({
                "type": "round_detail",
                "episode": ep + 1,
                "round": rnd + 1,
                "auditor_pick": auditor_pick,
                "auditor_flags": round_log["auditor"].get("flags", []),
                "buyer_pick": buyer_pick,
                "buyer_followed_auditor": (buyer_pick == auditor_pick) if (buyer_pick and auditor_pick) else None,
                "winner": winner,
                "winner_cheating": winner_sup.get("cheating"),
                "winner_bid_price": winner_sup.get("bid_price"),
                "winner_actual_strength": winner_sup.get("actual_strength"),
                "required_strength": winner_sup.get("required_strength"),
                "part_failed": resolution.get("failed"),
                "per_supplier": {
                    sid: {
                        "bid_price": round_log["suppliers"][sid].get("bid_price"),
                        "actual_strength": round_log["suppliers"][sid].get("actual_strength"),
                        "required_strength": round_log["suppliers"][sid].get("required_strength"),
                        "cheating": round_log["suppliers"][sid].get("cheating"),
                        "won": (sid == winner),
                    }
                    for sid in SUPPLIER_IDS if sid in round_log["suppliers"]
                },
            })

            m["rounds"].append(round_log)

            if result.done:
                summary = result.observation.get("episode_summary", {})
                m["failures"]    = summary.get("num_failures", 0)
                m["buyer_spend"] = summary.get("buyer_total_spend", 0)
                m["buyer_penalties"] = summary.get("buyer_total_penalties", 0)
                m["auditor_tpr"] = summary.get("auditor_tpr") or 0
                m["auditor_fpr"] = summary.get("auditor_fpr") or 0
                m["cheaters"]    = summary.get("cheaters", [])
                m["supplier_ranking"] = summary.get("supplier_ranking", [])
                m["supplier_profits"] = summary.get("supplier_profits", {})
                m["rewards"]     = summary.get("final_rewards", {})
                break

        # If episode ended early (checkpoint eval with max_rounds < TOTAL_PARTS),
        # compute spend/failures from round logs since done was never True
        if "buyer_spend" not in m:
            m["failures"] = sum(1 for r in m["rounds"] if r.get("part_failed"))
            m["buyer_spend"] = sum(
                r["suppliers"].get(r["buyer"].get("pick", ""), {}).get("bid_price") or 0
                for r in m["rounds"]
            )
            m["auditor_tpr"] = 0
            m["auditor_fpr"] = 0
            m["cheaters"] = []

        m["format_accuracy"] = m["valid"] / max(1, m["total"])
        all_metrics.append(m)
        _log_episode(m)

        print(f"\nEpisode {ep + 1} summary:")
        print(f"  Format accuracy: {m['format_accuracy']:.1%}")
        print(f"  Failures: {m.get('failures', '?')}/{max_rounds}")
        print(f"  Buyer spend: {m.get('buyer_spend', 0):.1f}  Penalties: {m.get('buyer_penalties', 0):.1f}")
        print(f"  Cheaters: {m.get('cheaters', [])}")
        print(f"  Auditor TPR: {(m.get('auditor_tpr') or 0):.2f}  FPR: {(m.get('auditor_fpr') or 0):.2f}")
        if "rewards" in m:
            r = m["rewards"]
            print(f"  Buyer reward:   {r.get('buyer', 0):.2f}")
            print(f"  Auditor reward: {r.get('auditor', 0):.2f}")
            for sid in SUPPLIER_IDS:
                print(f"  {sid} ({m['personalities'].get(sid,'?')}): {r.get('suppliers', {}).get(sid, 0):.2f}")

    with open(EVAL_LOG, "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)
    print(f"\nEval metrics saved to {EVAL_LOG}")

    # Log a periodic_eval summary entry for mid-training checkpoint charts
    if eval_step is not None and all_metrics:
        n = len(all_metrics)
        # Average per-personality profit across episodes
        personality_profits = {}
        for m in all_metrics:
            for sid, pname in m["personalities"].items():
                profit = m.get("supplier_profits", {}).get(sid, 0)
                personality_profits.setdefault(pname, []).append(profit)
        avg_personality_profits = {p: sum(v)/len(v) for p, v in personality_profits.items()}

        _log_episode({
            "type": "periodic_eval",
            "eval_step": eval_step,
            "avg_failures": sum(m.get("failures", 0) for m in all_metrics) / n,
            "avg_buyer_spend": sum(m.get("buyer_spend", 0) for m in all_metrics) / n,
            "avg_auditor_tpr": sum(m.get("auditor_tpr", 0) for m in all_metrics) / n,
            "avg_auditor_fpr": sum(m.get("auditor_fpr", 0) for m in all_metrics) / n,
            "avg_personality_profits": avg_personality_profits,
        })

    return all_metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    from huggingface_hub import login
    if os.path.exists(HF_TOKEN_FILE):
        login(token=open(HF_TOKEN_FILE).read().strip())

    # 1. Generate prompts
    print(f"Generating prompts from {NUM_PROMPT_EPISODES} episodes...")
    prompt_data = generate_prompts(NUM_PROMPT_EPISODES)
    print(f"Collected {len(prompt_data)} prompts")

    from datasets import Dataset
    dataset = Dataset.from_list(prompt_data).shuffle(seed=42)
    print(f"Dataset: {len(dataset)} rows")

    # 2. Load model
    from unsloth import FastLanguageModel

    print(f"\nLoading {MODEL_NAME}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=LORA_RANK * 2,
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    # 3. GRPO training (using standard TRL — imported at top before unsloth patched it)
    # Left-padding required for decoder-only models in GRPO
    tokenizer.padding_side = "left"

    training_args = GRPOConfig(
        temperature=0.8,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="linear",
        optim="adamw_8bit",
        logging_steps=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_generations=NUM_GENERATIONS,
        max_prompt_length=2048,
        max_completion_length=MAX_COMPLETION_LENGTH,  # 256 default — enough for reasoning
        max_steps=NUM_TRAINING_STEPS,
        save_steps=100,
        report_to="none",
        output_dir=OUTPUT_DIR,
    )

    # Periodic checkpoint callback — pauses every CHECKPOINT_EVERY steps to run
    # a 1-episode eval and log per-personality profits for line charts in reports.
    from transformers import TrainerCallback

    class CheckpointEvalCallback(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step % CHECKPOINT_EVERY == 0 and state.global_step > 0:
                print(f"\n[Checkpoint eval at step {state.global_step}/{NUM_TRAINING_STEPS}]")
                evaluate_model(model, tokenizer, num_episodes=1, eval_step=state.global_step, max_rounds=CHECKPOINT_ROUNDS)
                # Switch back to training mode after eval
                from unsloth import FastLanguageModel
                FastLanguageModel.for_training(model)
                tokenizer.padding_side = "left"

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[format_reward, reasoning_reward, economic_reward],
        args=training_args,
        train_dataset=dataset,
        callbacks=[CheckpointEvalCallback()],
    )

    print(f"\nStarting GRPO training ({NUM_TRAINING_STEPS} steps)...")
    print(f"Logs: reasoning={REASONING_LOG}  episodes={EPISODE_LOG}  eval={EVAL_LOG}")
    print(f"Checkpoint evals every {CHECKPOINT_EVERY} steps.")
    trainer.train()
    print("Training complete!")

    # 4. Save
    model.save_pretrained_merged(OUTPUT_DIR, tokenizer, save_method="merged_16bit")
    print(f"Model saved to {OUTPUT_DIR}/")

    # 5. Final eval — 1 full 50-round episode, rich per-round logging for charts
    print("\n[Final eval — full 50-round episode]")
    from unsloth import FastLanguageModel
    FastLanguageModel.for_inference(model)
    final_env = AuditronEnv()
    final_env.reset(seed=99999)
    s = final_env.state
    personalities = {sid: s.supplier_personalities[sid]["name"] for sid in SUPPLIER_IDS}
    print(f"Personalities: {personalities}")
    cumulative_spend = 0.0
    cumulative_failures = 0
    cumulative_profits = {sid: 0.0 for sid in SUPPLIER_IDS}

    for rnd in range(TOTAL_PARTS):
        # Suppliers
        sup_obs_list = [final_env.get_supplier_obs(sid) for sid in SUPPLIER_IDS]
        sup_prompts = [build_prompt(obs, "supplier") for obs in sup_obs_list]
        sup_actions = generate_actions_batch(model, tokenizer, sup_prompts, max_new_tokens=64)
        for sid, obs, action_str in zip(SUPPLIER_IDS, sup_obs_list, sup_actions):
            result = final_env.step(AuditronAction(agent_id=sid, content=action_str))
            if result.phase == "error":
                req = obs["required_strength"]
                cost = obs["your_cost_per_point"]
                final_env.step(AuditronAction(agent_id=sid, content=json.dumps(
                    {"bid_price": round(req * cost * 1.1, 1), "actual_strength": req})))

        # Auditor
        aud_obs = final_env.get_auditor_obs()
        aud_action = generate_action(model, tokenizer, build_prompt(aud_obs, "auditor"))
        final_env.step(AuditronAction(agent_id="auditor", content=aud_action))
        try:
            aud_parsed = json.loads(aud_action)
            if not isinstance(aud_parsed, dict): aud_parsed = {}
        except Exception:
            aud_parsed = {}

        # Buyer
        buy_obs = final_env.get_buyer_obs()
        buy_action = generate_action(model, tokenizer, build_prompt(buy_obs, "buyer"))
        result = final_env.step(AuditronAction(agent_id="buyer", content=buy_action))
        try:
            buy_parsed = json.loads(buy_action)
            if not isinstance(buy_parsed, dict): buy_parsed = {}
        except Exception:
            buy_parsed = {}

        resolution = result.observation.get("resolution", {})
        winner = buy_parsed.get("pick")
        failed = resolution.get("failed", False)
        penalty = resolution.get("penalty", 0.0)
        winner_bid = final_env.state.supplier_bids.get(winner, {}).get("bid_price", 0) if winner else 0
        cumulative_spend += (winner_bid or 0) + (penalty or 0)
        if failed:
            cumulative_failures += 1

        # Per-supplier data for this round
        per_supplier = {}
        for sid in SUPPLIER_IDS:
            bid_info = final_env.state.supplier_bids.get(sid, {})
            bid_price = bid_info.get("bid_price", 0) or 0
            actual_str = bid_info.get("actual_strength", 0)
            req_str = final_env.state.required_strength
            cost = sup_obs_list[SUPPLIER_IDS.index(sid)].get("your_cost_per_point", 0)
            production_cost = req_str * cost
            round_profit = (bid_price - production_cost) if sid == winner and not failed else 0.0
            cumulative_profits[sid] += round_profit
            per_supplier[sid] = {
                "personality": personalities[sid],
                "bid_price": bid_price,
                "actual_strength": actual_str,
                "cheating": actual_str < req_str if actual_str else False,
                "won": sid == winner,
                "round_profit": round_profit,
                "cumulative_profit": cumulative_profits[sid],
            }

        _log_episode({
            "type": "final_round",
            "round": rnd + 1,
            "required_strength": final_env.state.required_strength,
            "personalities": personalities,
            "auditor_pick": aud_parsed.get("pick"),
            "auditor_flags": aud_parsed.get("flags", []),
            "auditor_reason": aud_parsed.get("reason", ""),
            "buyer_pick": winner,
            "buyer_followed_auditor": (winner == aud_parsed.get("pick")) if winner else None,
            "part_failed": failed,
            "failure_penalty": penalty or 0.0,
            "round_spend": (winner_bid or 0) + (penalty or 0),
            "cumulative_spend": cumulative_spend,
            "cumulative_failures": cumulative_failures,
            "per_supplier": per_supplier,
        })

        if result.done:
            summary = result.observation.get("episode_summary", {})
            _log_episode({
                "type": "final_summary",
                "personalities": personalities,
                "total_spend": summary.get("buyer_total_spend", cumulative_spend),
                "total_failures": summary.get("num_failures", cumulative_failures),
                "auditor_tpr": summary.get("auditor_tpr"),
                "auditor_fpr": summary.get("auditor_fpr"),
                "supplier_profits": summary.get("supplier_profits", {}),
                "supplier_ranking": summary.get("supplier_ranking", []),
                "final_rewards": summary.get("final_rewards", {}),
            })
            print(f"Final eval done. Spend={cumulative_spend:.1f}  Failures={cumulative_failures}")
            break


if __name__ == "__main__":
    main()
