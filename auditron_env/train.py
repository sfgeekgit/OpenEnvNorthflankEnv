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

# Import TRL before unsloth so we get the standard (unpatched) GRPOTrainer.
# Unsloth is still used for fast model loading and LoRA — just not for training.
from trl import GRPOConfig, GRPOTrainer

# ── Config (all tunables in one place) ────────────────────────────────────────
MODEL_NAME = os.environ.get("MODEL_NAME", "unsloth/Qwen2.5-1.5B-Instruct")
MAX_SEQ_LENGTH = int(os.environ.get("MAX_SEQ_LENGTH", "2048"))
LORA_RANK = int(os.environ.get("LORA_RANK", "16"))
NUM_TRAINING_STEPS = int(os.environ.get("NUM_TRAINING_STEPS", "500"))
NUM_GENERATIONS = int(os.environ.get("NUM_GENERATIONS", "4"))
NUM_PROMPT_EPISODES = int(os.environ.get("NUM_PROMPT_EPISODES", "20"))
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "5e-5"))
HF_TOKEN_FILE = os.environ.get("HF_TOKEN_FILE", "/home/openenv/HFTOKEN")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "auditron_trained")
EVAL_EPISODES = int(os.environ.get("EVAL_EPISODES", "5"))

# Override for dry run
if os.environ.get("DRY_RUN", "0") == "1":
    NUM_TRAINING_STEPS = 10
    NUM_PROMPT_EPISODES = 2
    NUM_GENERATIONS = 2
    EVAL_EPISODES = 1
# ──────────────────────────────────────────────────────────────────────────────

from server import AuditronAction, AuditronEnv, SUPPLIER_IDS
from rewards import TOTAL_PARTS


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
    system_msg = (
        f"You are a {agent_type} in a procurement auction. {system}\n"
        f"Respond with valid JSON only. Format: {action_format}"
    )
    user_msg = f"Current state:\n{json.dumps(obs_clean, indent=2)}"
    return (
        f"<|im_start|>system\n{system_msg}<|im_end|>\n"
        f"<|im_start|>user\n{user_msg}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def generate_prompts(num_episodes: int) -> list[dict]:
    """Run episodes with random actions to collect training prompts."""
    prompts = []
    env = AuditronEnv()
    max_rounds = min(TOTAL_PARTS, 5) if os.environ.get("DRY_RUN") == "1" else TOTAL_PARTS

    for ep in range(num_episodes):
        env.reset(seed=ep * 1000 + random.randint(0, 999))

        for rnd in range(max_rounds):
            # Collect supplier prompts
            for sid in SUPPLIER_IDS:
                obs = env.get_supplier_obs(sid)
                prompts.append({"prompt": build_prompt(obs, "supplier")})

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
            prompts.append({"prompt": build_prompt(obs, "auditor")})
            pick = random.choice(SUPPLIER_IDS)
            env.step(AuditronAction(
                agent_id="auditor",
                content=json.dumps({"pick": pick, "reason": "random", "flags": []}),
            ))

            # Collect buyer prompt
            obs = env.get_buyer_obs()
            prompts.append({"prompt": build_prompt(obs, "buyer")})
            pick = random.choice(SUPPLIER_IDS)
            result = env.step(AuditronAction(
                agent_id="buyer",
                content=json.dumps({"pick": pick, "reason": "random"}),
            ))
            if result.done:
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
    """Infer agent type from the system message in the prompt."""
    if "supplier" in prompt_str.lower() and "bid_price" in prompt_str:
        return "supplier"
    if "oversight" in prompt_str.lower() or "auditor" in prompt_str.lower():
        return "auditor"
    if "buyer" in prompt_str.lower():
        return "buyer"
    return "unknown"


def format_reward(completions, **kwargs):
    """Reward for valid JSON with correct fields per agent type."""
    prompts = kwargs.get("prompts", [""] * len(completions))
    scores = []
    for completion, prompt in zip(completions, prompts):
        prompt_str = str(prompt)
        agent_type = _infer_agent_type(prompt_str)
        try:
            data = _extract_json(completion)
            if agent_type == "supplier":
                assert isinstance(data.get("bid_price"), (int, float))
                assert isinstance(data.get("actual_strength"), (int, float))
                scores.append(2.0)
            elif agent_type == "auditor":
                assert data.get("pick") in SUPPLIER_IDS
                assert "reason" in data
                score = 2.0
                if isinstance(data.get("flags"), list):
                    score += 0.5
                scores.append(score)
            elif agent_type == "buyer":
                assert data.get("pick") in SUPPLIER_IDS
                score = 2.0
                if "reason" in data and len(str(data["reason"])) > 5:
                    score += 0.5
                scores.append(score)
            else:
                scores.append(1.0)
        except Exception:
            scores.append(-1.0)
    return scores


def reasoning_reward(completions, **kwargs):
    """Reward for evidence-based reasoning (auditor & buyer)."""
    scores = []
    for completion in completions:
        try:
            data = _extract_json(completion)
            reason = str(data.get("reason", ""))
        except Exception:
            scores.append(0.0)
            continue

        score = 0.0
        if re.search(r"supplier_\d", reason):
            score += 1.0
        if re.search(r"[\$%]|\d+.*(?:price|bid|cost)", reason, re.I):
            score += 1.0
        if re.search(r"fail", reason, re.I):
            score += 1.0
        if re.search(r"round\s+\d+", reason, re.I):
            score += 1.0
        if re.search(r"cheap|below|above|more|less|lower|higher", reason, re.I):
            score += 1.0
        words = reason.split()
        if len(words) >= 20:
            score += 1.0
        if len(words) >= 50:
            score += 1.0
        scores.append(score)
    return scores


def economic_reward(completions, **kwargs):
    """Reward suppliers for economically sensible bids."""
    prompts = kwargs.get("prompts", [""] * len(completions))
    scores = []
    for completion, prompt in zip(completions, prompts):
        prompt_str = str(prompt)
        if _infer_agent_type(prompt_str) != "supplier":
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
                scores.append(score)
            else:
                scores.append(0.0)
        except Exception:
            scores.append(-0.5)
    return scores


# ---------------------------------------------------------------------------
# Evaluation — run full episodes with the trained model
# ---------------------------------------------------------------------------

def generate_action(model, tokenizer, prompt_messages: list[dict]) -> str:
    """Generate a JSON action from the model given chat messages."""
    import torch
    text = tokenizer.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
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


def evaluate_model(model, tokenizer, num_episodes: int = 5):
    """Run full episodes and report metrics."""
    from unsloth import FastLanguageModel
    FastLanguageModel.for_inference(model)

    env = AuditronEnv()
    all_metrics = []
    max_rounds = min(TOTAL_PARTS, 5) if os.environ.get("DRY_RUN") == "1" else TOTAL_PARTS

    for ep in range(num_episodes):
        env.reset(seed=9000 + ep)
        m = {"episode": ep + 1, "valid": 0, "total": 0}

        for rnd in range(max_rounds):
            # Suppliers — use fallback if model output is invalid
            for sid in SUPPLIER_IDS:
                obs = env.get_supplier_obs(sid)
                action = generate_action(model, tokenizer, build_prompt(obs, "supplier"))
                m["total"] += 1
                result = env.step(AuditronAction(agent_id=sid, content=action))
                if result.phase == "error":
                    # Submit valid fallback so env can proceed
                    req = obs["required_strength"]
                    cost = obs["your_cost_per_point"]
                    fallback = json.dumps({"bid_price": round(req * cost * 1.1, 1), "actual_strength": req})
                    env.step(AuditronAction(agent_id=sid, content=fallback))
                else:
                    m["valid"] += 1

            # Auditor
            obs = env.get_auditor_obs()
            action = generate_action(model, tokenizer, build_prompt(obs, "auditor"))
            m["total"] += 1
            result = env.step(AuditronAction(agent_id="auditor", content=action))
            if result.phase == "error":
                cheapest = min(obs["bids"], key=obs["bids"].get)
                fallback = json.dumps({"pick": cheapest, "reason": "fallback", "flags": []})
                env.step(AuditronAction(agent_id="auditor", content=fallback))
            else:
                m["valid"] += 1

            # Buyer
            obs = env.get_buyer_obs()
            action = generate_action(model, tokenizer, build_prompt(obs, "buyer"))
            m["total"] += 1
            result = env.step(AuditronAction(agent_id="buyer", content=action))
            if result.phase == "error":
                rec = obs["auditor_recommendation"].get("pick", SUPPLIER_IDS[0])
                fallback = json.dumps({"pick": rec, "reason": "fallback"})
                result = env.step(AuditronAction(agent_id="buyer", content=fallback))
            else:
                m["valid"] += 1

            if result.done:
                summary = result.observation.get("episode_summary", {})
                m["failures"] = summary.get("num_failures", 0)
                m["buyer_spend"] = summary.get("buyer_total_spend", 0)
                m["auditor_tpr"] = summary.get("auditor_tpr", 0)
                m["auditor_fpr"] = summary.get("auditor_fpr", 0)
                m["rewards"] = summary.get("final_rewards", {})
                break

        m["format_accuracy"] = m["valid"] / max(1, m["total"])
        all_metrics.append(m)

        print(f"\nEpisode {ep + 1}:")
        print(f"  Format accuracy: {m['format_accuracy']:.1%}")
        print(f"  Failures: {m.get('failures', '?')}/{max_rounds}")
        if "auditor_tpr" in m:
            print(f"  Auditor TPR: {m['auditor_tpr']:.2f}")
            print(f"  Auditor FPR: {m['auditor_fpr']:.2f}")
        if "rewards" in m:
            r = m["rewards"]
            print(f"  Buyer reward:   {r.get('buyer', 0):.2f}")
            print(f"  Auditor reward: {r.get('auditor', 0):.2f}")

    with open("eval_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)
    print(f"\nMetrics saved to eval_metrics.json")
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
        max_prompt_length=1024,
        max_completion_length=128,   # JSON actions are short
        max_steps=NUM_TRAINING_STEPS,
        save_steps=100,
        report_to="none",
        output_dir=OUTPUT_DIR,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[format_reward, reasoning_reward, economic_reward],
        args=training_args,
        train_dataset=dataset,
    )

    print(f"\nStarting GRPO training ({NUM_TRAINING_STEPS} steps)...")
    trainer.train()
    print("Training complete!")

    # 4. Save
    model.save_pretrained_merged(OUTPUT_DIR, tokenizer, save_method="merged_16bit")
    print(f"Model saved to {OUTPUT_DIR}/")

    # 5. Evaluate
    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)
    evaluate_model(model, tokenizer, num_episodes=EVAL_EPISODES)


if __name__ == "__main__":
    main()
