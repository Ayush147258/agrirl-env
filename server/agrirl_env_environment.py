import random
import sys
import os
from uuid import uuid4
from typing import List, Optional ,cast


from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State


# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from models import Crop, AgrirlAction, AgrirlObservation, WeatherType

class AgriCoreEnv(Environment):
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        self.max_days = 30
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self.task = "easy"
        self.reset()

    def reset(self, task: Optional[str] = None) -> AgrirlObservation:
        self._state = State(episode_id=str(uuid4()), step_count=0)

        # ✅ Task difficulty from 2nd file
        self.task = task if task is not None else "easy"

        self.day = 1
        self.done = False

        self.crops: List[Crop] = [
            Crop(id=i, moisture=50, growth=0, stage="seed",
                 wait_days=0, fertilized_times=0, pest_level=0)
            for i in range(4)
        ]

        self.water = 120
        self.fertilizer = 60
        self.energy = 200
        self.pesticide = 40

        self.market_price = 1.0
        self.soil_health = 1.0
        self.fertilizer_used = 0

        self.total_harvest_today = 0
        self.forecast = self._random_weather()

        return self._obs("sunny", 0)

    def _random_weather(self) -> WeatherType:
        # ✅ Task-aware weather from 2nd file + heatwave/frost from 1st file
        if self.task == "easy":
            return random.choice(["sunny", "sunny", "rainy"])
        elif self.task == "medium":
            return random.choice(["sunny", "rainy", "cloudy"])
        else:  # hard
            roll = random.random()
            if roll < 0.05:
                return "heatwave"
            elif roll < 0.1:
                return "frost"
            return random.choice(["sunny", "rainy", "cloudy"])
            

    def step(self, action: AgrirlAction) -> AgrirlObservation:
        self._state.step_count += 1

        # ✅ Guard from 2nd file — return immediately if already done
        if self.done:
            return self._obs("sunny", 0.0, done=True, score=self._compute_score())

        reward = 0.0
        weather = self.forecast
        self.forecast = self._random_weather()

        # 🌡️ WEATHER EFFECTS (1st file — full heatwave/frost support)
        for c in self.crops:
            c.wait_days += 1

            if weather == "sunny":
                c.moisture -= 5
            elif weather == "rainy":
                c.moisture += 10
            elif weather == "cloudy":
                pass  # neutral
            elif weather == "heatwave":
                c.moisture -= 20
                reward -= 3
            elif weather == "frost":
                c.growth -= 2

        # ✅ Hard mode extra penalties from 2nd file
        if self.task == "hard":
            for c in self.crops:
                if c.moisture > 85:
                    reward -= 4
                if c.moisture < 25:
                    reward -= 4

        # ⚡ ENERGY + ACTION (1st file — resource-aware actions)
        c = self.crops[action.crop_id]

        if action.action == "irrigate":
            if self.water >= 10 and self.energy >= 5:
                c.moisture += 15
                self.water -= 10
                self.energy -= 5
                reward -= 1
            else:
                reward -= 2  # penalty for invalid action

        elif action.action == "fertilize":
            if self.fertilizer >= 5 and self.energy >= 3:
                c.fertilized_times += 1
                self.fertilizer_used += 5
                self.fertilizer -= 5
                self.energy -= 3

                # 📉 Diminishing returns (1st file)
                gain = 5 / c.fertilized_times
                c.growth += gain
                reward += gain
            else:
                reward -= 3  # penalty for invalid action
        elif action.action == "pesticide":
            if self.pesticide >= 5 and self.energy >= 2:
               c.pest_level = max(0, c.pest_level - 3)  # ✅ reduces pest level
               self.pesticide -= 5
               self.energy -= 2
               reward += 2  # small reward for pest control
            else:
               reward -= 1  # penalty for invalid action
        elif action.action == "harvest":
            if c.stage == "mature":
                # ✅ Market price system from 1st file
                self.total_harvest_today += c.growth
                price = max(0.5, 1.5 - self.total_harvest_today / 100)
                self.market_price = price
                reward += c.growth * 2 * price
                c.growth = 0
                c.stage = "seed"
            else:
                # Penalty for harvesting non-mature crops
                reward -= 5

        elif action.action == "wait":
            # ✅ Wait action from 2nd file — no cost, small penalty to discourage overuse
            reward -= 0.5

        # 🌱 SOIL HEALTH (1st file)
        if self.fertilizer_used > 50:
            self.soil_health = 0.7
        else:
            self.soil_health = 1.0

        # 🐜 PEST SYSTEM (1st file — unique feature)
        for c in self.crops:
            if c.moisture > 80:
                c.pest_level += 1
            else:
                c.pest_level = max(0, c.pest_level - 1)

            if c.pest_level > 2:
                c.growth -= 2
                reward -= 1

        # 🌾 GROWTH (1st file — soil health aware)
        for c in self.crops:
            if 30 <= c.moisture <= 80:
                growth = random.uniform(1, 4) * self.soil_health
                c.growth += growth
                reward += growth
            else:
                reward -= 1

            if c.wait_days > 3:
                reward -= 2

            # Stage progression
            if c.growth > 20:
                c.stage = "vegetative"
            if c.growth > 50:
                c.stage = "flowering"
            if c.growth > 80:
                c.stage = "mature"

            c.moisture = max(0, min(100, c.moisture))
            c.growth = max(0, min(100, c.growth))

        self.day += 1
        self.total_harvest_today = 0

        if self.day > self.max_days:
            self.done = True

        # ✅ _compute_score as separate method from 2nd file
        score = None
        if self.done:
            score = self._compute_score()

        return self._obs(weather, reward, self.done, score)

    def _compute_score(self) -> float:
        # ✅ Task-aware scoring from 2nd file + multi-crop from 1st file
        total_growth = sum(c.growth for c in self.crops)

        if self.task == "easy":
            return min(1.0, total_growth / 300)

        elif self.task == "medium":
            return max(0.0, (total_growth - self.day) / 300)

        elif self.task == "hard":
            efficiency = total_growth / (self.day + 1)
            return max(0.0, min(1.0, efficiency / 10))

        return min(1.0, total_growth / 300)

    def _obs(self, weather: str, reward: float, done: bool = False, score=None) -> AgrirlObservation:
        return AgrirlObservation(
            crops=self.crops,
            water=self.water,
            fertilizer=self.fertilizer,
            energy=self.energy,
            pesticide=self.pesticide,
            day=self.day,
            weather=cast(WeatherType, weather),
            forecast=cast(WeatherType, self.forecast),
            market_price=self.market_price,
            soil_health=self.soil_health,
            reward=reward,
            done=done,
            score=score,
        )

    @property
    def state(self) -> State:
        return self._state