"""
Terminal play script for the Letter Guessing Environment.
You are the agent. Type a letter and press Enter.
Goal: guess 'a' first, then 'z' to win.
"""

from server import LetterAction, LetterEnv


def main():
    env = LetterEnv()
    obs = env.reset()
    print(f"\n{obs.message}\n")

    while not obs.done:
        raw = input("Your guess > ").strip()
        if not raw:
            continue

        action = LetterAction(letter=raw[0])
        obs = env.step(action)

        reward_str = f"  [reward: {obs.reward}]" if obs.reward else ""
        print(f"\n{obs.message}{reward_str}\n")

    print(f"Episode done in {env.state.step_count} steps.")


if __name__ == "__main__":
    main()
