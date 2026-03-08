"""
Minimal training script for the Letter Guessing Environment.
Uses HF TRL with a simple RL loop — no GRPO complexity, just proof-of-concept.
Model must learn to output 'a' then 'z' to get max reward.

Model name is a config variable so it's easy to swap dev (0.5B) for prod (7-8B).
"""

import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login

from server import LetterAction, LetterEnv

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME = "Qwen/Qwen2-0.5B"
NUM_EPISODES = 10
MAX_STEPS_PER_EPISODE = 10
HF_TOKEN_FILE = "/home/openenv/HFTOKEN"
# ─────────────────────────────────────────────────────────────────────────────


def extract_letter(text: str) -> str:
    """Pull the first alphabetic character out of model output."""
    match = re.search(r"[a-zA-Z]", text)
    return match.group(0).lower() if match else "?"


def run_episode(model, tokenizer, env, verbose=True) -> float:
    """Run one episode, return total reward."""
    obs = env.reset()
    total_reward = 0.0

    for step in range(MAX_STEPS_PER_EPISODE):
        prompt = f"Environment: {obs.message}\nYour letter guess: "
        inputs = tokenizer(prompt, return_tensors="pt")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=5,
                do_sample=True,
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        letter = extract_letter(generated)

        obs = env.step(LetterAction(letter=letter))
        total_reward += obs.reward or 0.0

        if verbose:
            print(f"  step {step+1}: guessed '{letter}' → {obs.message} (reward={obs.reward})")

        if obs.done:
            break

    return total_reward


def main():
    login(token=open(HF_TOKEN_FILE).read().strip())

    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.eval()

    env = LetterEnv()

    print(f"\nRunning {NUM_EPISODES} episodes...\n")
    rewards = []
    for ep in range(NUM_EPISODES):
        print(f"Episode {ep+1}/{NUM_EPISODES}")
        r = run_episode(model, tokenizer, env)
        rewards.append(r)
        print(f"  → total reward: {r:.2f}\n")

    print(f"Mean reward over {NUM_EPISODES} episodes: {sum(rewards)/len(rewards):.3f}")
    print(f"Reward curve: {[round(r, 2) for r in rewards]}")


if __name__ == "__main__":
    main()
