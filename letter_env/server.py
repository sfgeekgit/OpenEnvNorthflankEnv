"""
Letter Guessing Environment

The agent must guess "a" then "z" in order to win.
"""

import os
from typing import Optional
from openenv.core import Action, Environment, Observation, State, create_app

REWARD_WRONG = 0
REWARD_CORRECT_A = 0.5
REWARD_WIN = 1.0


class LetterAction(Action):
    letter: str


class LetterObservation(Observation):
    message: str


class LetterState(State):
    stage: int  # 0 = need 'a', 1 = need 'z', 2 = done


class LetterEnv(Environment[LetterAction, LetterObservation, LetterState]):
    def __init__(self):
        super().__init__()
        self._state = LetterState(stage=0)

    def reset(self, seed: Optional[int] = None, episode_id: Optional[str] = None, **kwargs) -> LetterObservation:
        self._state = LetterState(episode_id=episode_id, stage=0)
        return LetterObservation(
            message="Welcome! Guess a letter. You need to guess 'a' first.",
            done=False,
            reward=REWARD_WRONG,
        )

    def step(self, action: LetterAction, timeout_s: Optional[float] = None, **kwargs) -> LetterObservation:
        self._state.step_count += 1
        guess = action.letter.strip().lower()

        if self._state.stage == 0:
            if guess == "a":
                self._state.stage = 1
                return LetterObservation(
                    message="Correct! Now guess 'z' to win.",
                    done=False,
                    reward=REWARD_CORRECT_A,
                )
            else:
                return LetterObservation(
                    message=f"Wrong! You guessed '{guess}'. You need to guess 'a' first.",
                    done=False,
                    reward=REWARD_WRONG,
                )

        elif self._state.stage == 1:
            if guess == "z":
                self._state.stage = 2
                return LetterObservation(
                    message="You win! You guessed 'a' then 'z'. Episode complete.",
                    done=True,
                    reward=REWARD_WIN,
                )
            else:
                return LetterObservation(
                    message=f"Wrong! You guessed '{guess}'. You need to guess 'z' now.",
                    done=False,
                    reward=REWARD_WRONG,
                )

        else:
            return LetterObservation(
                message="Episode already complete. Call reset() to start over.",
                done=True,
                reward=REWARD_WRONG,
            )

    @property
    def state(self) -> LetterState:
        return self._state


app = create_app(
    env=LetterEnv,
    action_cls=LetterAction,
    observation_cls=LetterObservation,
)

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
