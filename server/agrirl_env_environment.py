import random
from uuid import uuid4
from typing import cast, Literal

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

from .models import AgrirlAction, AgrirlObservation 


class AgrirlEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self.max_days = 30
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self.task = "easy"
        self.reset()

    
    def reset(self, task: str | None = None) -> AgrirlObservation:
        self._state = State(episode_id=str(uuid4()), step_count=0)

                 
        self.task = task if task is not None else "easy"        
        self.soil_moisture = 50.0
        self.crop_growth = 0.0
        self.day = 1
        self.done = False

        return AgrirlObservation(
            soil_moisture=self.soil_moisture,
            crop_growth=self.crop_growth,
            day=self.day,
            weather="sunny",
            done=False,
            reward=0.0,
            score=None,
            task=self.task 
        )

    def step(self, action: AgrirlAction) -> AgrirlObservation:
        self._state.step_count += 1

        # If already finished
        if self.done:
            return AgrirlObservation(
                soil_moisture=self.soil_moisture,
                crop_growth=self.crop_growth,
                day=self.day,
                weather="sunny",
                done=True,
                reward=0.0,
                score=self._compute_score(),
                task=self.task
            )

        reward = 0.0

        
        weather_val = "sunny"   # safe default, overwritten below for non-easy tasks
        if self.task == "medium":
            weather_val = random.choice(["sunny", "rainy"])
        elif self.task == "hard":
            weather_val = random.choice(["sunny", "rainy", "cloudy"])

        # Natural effects
        if weather_val == "sunny":
            self.soil_moisture -= 5
        elif weather_val == "rainy":
            self.soil_moisture += 10

        # Agent actions
        if action.action == "irrigate":
            self.soil_moisture += 15
            reward -= 2

        elif action.action == "fertilize":
            self.crop_growth += 5
            reward -= 3

        elif action.action == "harvest":
            self.done = True
            reward += self.crop_growth * 2

        
        if not self.done:
            if 30 <= self.soil_moisture <= 80:
                growth = random.uniform(2, 5)
                self.crop_growth += growth
                reward += growth
            else:
                reward -= 2

            # Realism penalties
            if self.task == "hard":
                if self.soil_moisture > 85:
                    reward -= 4
                if self.soil_moisture < 25:
                    reward -= 4

        # Clamp values
        self.soil_moisture = max(0, min(100, self.soil_moisture))
        self.crop_growth = max(0, min(100, self.crop_growth))

        self.day += 1

        # End condition
        if self.day > self.max_days:
            self.done = True

        # Compute score if done
        score = None
        if self.done:
            score = self._compute_score()

        return AgrirlObservation(
            soil_moisture=self.soil_moisture,
            crop_growth=self.crop_growth,
            day=self.day,
            weather=cast(Literal['sunny', 'rainy', 'cloudy'], weather_val),
            done=self.done,
            reward=reward,
            score=score,
            task=self.task
        )

    def _compute_score(self):
        if self.task == "easy":
            return min(1.0, self.crop_growth / 100)

        elif self.task == "medium":
            return max(0.0, (self.crop_growth - self.day) / 100)

        elif self.task == "hard":
            efficiency = self.crop_growth / (self.day + 1)
            return max(0.0, min(1.0, efficiency / 5))

    @property
    def state(self) -> State:
        return self._state