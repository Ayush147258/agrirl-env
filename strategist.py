# strategist.py
"""
Strategist Agent — Hierarchical Multi-Agent Layer
--------------------------------------------------
Uses Claude to set high-level strategic directives every N days.

Resilience features:
  - Graceful degradation: API failure retains last known strategy silently
  - Local heuristic fallback: deterministic expert-system strategy (no API)
  - Mock mode: USE_MOCK_AI = True reads from mock_responses.json
  - Low-token model: claude-haiku by default (fast + cheap)
  - Review interval: only calls API every 5 days (not every step)
"""

import json
import os
import re
from dataclasses import dataclass
from typing import Optional

# ── Config ─────────────────────────────────────────────────────────────────────
USE_MOCK_AI    = os.getenv("USE_MOCK_AI", "false").lower() == "true"
MOCK_FILE      = "mock_responses.json"
DEFAULT_MODEL = "gemini-2.0-flash"
api_key = os.getenv("GEMINI_API_KEY")

# ── Strategy Directives ────────────────────────────────────────────────────────

@dataclass
class Strategy:
    """
    High-level directives issued by the Strategist to the Executor.
    All thresholds override the Executor's built-in defaults.
    """
    priority: str            = "balanced"   # soil_health | profit | balanced | survival
    moisture_threshold: float = 35.0        # irrigate below this
    harvest_price_floor: float = 1.0        # only harvest above this market price
    fertilize_max_day: int    = 20          # stop fertilizing after this day
    pest_threshold: float     = 2.0         # spray above this pest level
    water_reserve_pct: float  = 0.25        # keep this fraction of water in reserve
    reasoning: str            = ""          # explanation of this strategy

    def summary(self) -> str:
        return (
            f"[Strategy: {self.priority.upper()}] "
            f"Irrigate<{self.moisture_threshold:.1f} | "
            f"Harvest>${self.harvest_price_floor:.2f} | "
            f"Fertilize<=Day{self.fertilize_max_day} | "
            f"Spray>{self.pest_threshold:.1f} | "
            f"WaterReserve={self.water_reserve_pct*100:.0f}%"
        )


# ── Local Heuristic Fallback ───────────────────────────────────────────────────

def _heuristic_strategy(obs, current: Strategy) -> Strategy:
    """
    Deterministic expert-system strategy — no API required.
    Applied when the LLM is unavailable or USE_MOCK_AI is False.

    Rules mirror what a human agronomist would decide:
      - Low water + late game       → conserve water aggressively
      - Low soil health             → prioritise soil recovery
      - High market price + mature  → switch to profit mode
      - Many dry crops              → survival mode
    """
    s = Strategy(
        priority            = current.priority,
        moisture_threshold  = current.moisture_threshold,
        harvest_price_floor = current.harvest_price_floor,
        fertilize_max_day   = current.fertilize_max_day,
        pest_threshold      = current.pest_threshold,
        water_reserve_pct   = current.water_reserve_pct,
        reasoning           = "[Heuristic Fallback]",
    )

    water_pct  = obs.water / 120 if obs.water else 0
    day_ratio  = obs.day / 30
    dry_count  = sum(1 for c in obs.crops if c.moisture < 25)
    mature     = [c for c in obs.crops if c.stage == "mature"]

    # Survival: many crops critically dry
    if dry_count >= len(obs.crops) // 2:
        s.priority           = "survival"
        s.moisture_threshold = 50.0
        s.water_reserve_pct  = 0.15
        s.reasoning          = "[Heuristic] Survival mode — mass drought detected"

    # Soil health emergency
    elif obs.soil_health < 40:
        s.priority       = "soil_health"
        s.pest_threshold = 1.5
        s.reasoning      = "[Heuristic] Soil health critical — reducing pest tolerance"

    # Profit opportunity
    elif mature and obs.market_price > 1.5:
        s.priority            = "profit"
        s.harvest_price_floor = 0.9
        s.reasoning           = "[Heuristic] High market price + mature crops — harvest aggressively"

    # Water conservation
    elif water_pct < 0.3 and day_ratio > 0.5:
        s.moisture_threshold = min(s.moisture_threshold + 10, 55.0)
        s.water_reserve_pct  = 0.35
        s.reasoning          = "[Heuristic] Low water reserve — raising thresholds"

    # Heatwave adjustment
    elif obs.weather == "heatwave":
        s.moisture_threshold = min(s.moisture_threshold + 8, 55.0)
        s.reasoning          = "[Heuristic] Heatwave — boosting moisture priority"

    else:
        s.reasoning = "[Heuristic] Stable conditions — maintaining balanced strategy"

    return s


# ── Strategist Agent ───────────────────────────────────────────────────────────

class StrategistAgent:
    """
    LLM-powered Strategist that sets high-level directives every N days.

    Resilience hierarchy (in order):
      1. Mock mode (USE_MOCK_AI=true)  → reads from mock_responses.json
      2. Live API (claude-haiku)       → fast, low-cost LLM call
      3. Graceful degradation          → retains previous strategy on failure
      4. Heuristic fallback            → deterministic expert-system rules

    Usage:
        strategist = StrategistAgent(review_interval=5)
        strategy   = strategist.advise(obs, reward_history)
    """

    SYSTEM_PROMPT = """You are an expert agricultural AI strategist.
You receive a snapshot of a simulated farm and its recent reward history.
Output ONLY a valid JSON object (no markdown, no extra text):
{
  "priority":            "<soil_health|profit|balanced|survival>",
  "moisture_threshold":  <float 20-60>,
  "harvest_price_floor": <float 0.5-2.0>,
  "fertilize_max_day":   <int 10-25>,
  "pest_threshold":      <float 1.0-5.0>,
  "water_reserve_pct":   <float 0.1-0.5>,
  "reasoning":           "<one sentence max>"
}
Rules:
- Water < 30% and day > 15 → raise moisture_threshold and water_reserve_pct.
- soil_health < 40 → set priority=soil_health, lower pest_threshold.
- market_price > 1.5 and mature crops present → set priority=profit.
- More than half crops with moisture < 25 → set priority=survival.
- Always return valid JSON. No markdown fences."""

    def __init__(
        self,
        review_interval: int = 5,
        model: str = DEFAULT_MODEL,
    ):
        self.review_interval = review_interval
        self.model           = model
        self._current        = Strategy()
        self._last_review    = -review_interval     # trigger immediately on day 1
        self._mock_pool      = self._load_mock()
        self._mock_index     = 0

        # Lazy import gemini — graceful if not installed
        self._client = None
        if not USE_MOCK_AI:
            try:
                from google import genai
                self._client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
            except Exception as e:
                print(f"  [PostMortem] Gemini unavailable ({e}) — statistical mode active.")
    # ── Public ─────────────────────────────────────────────────────────────────

    @property
    def current_strategy(self) -> Strategy:
        return self._current

    def advise(self, obs, reward_history: list) -> Strategy:
        """
        Return current Strategy. Calls LLM only every `review_interval` days.
        """
        if obs.day - self._last_review >= self.review_interval:
            self._current     = self._get_strategy(obs, reward_history)
            self._last_review = obs.day
        return self._current

    # ── Internal ───────────────────────────────────────────────────────────────

    def _get_strategy(self, obs, reward_history: list) -> Strategy:
        """Route to mock, live API, or heuristic — in that order."""

        # ── Path 1: Mock mode ──────────────────────────────────────────────────
        if USE_MOCK_AI and self._mock_pool:
            entry = self._mock_pool[self._mock_index % len(self._mock_pool)]
            self._mock_index += 1
            print(f"  [Strategist] Mock response #{self._mock_index}")
            return self._parse_strategy(entry)

        # ── Path 2: Live LLM ───────────────────────────────────────────────────
        if self._client:
            try:
                prompt   = self._build_prompt(obs, reward_history)
                resp = self._client.models.generate_content(
                model="gemini-2.0-flash",
                contents=self.SYSTEM_PROMPT + "\n\n" + prompt
                )
                raw = resp.text.strip()
                raw = re.sub(r"^```[a-z]*\n?", "", raw).rstrip("```").strip()
                return self._parse_strategy(json.loads(raw))
            except Exception as e:
                print(f"  [Strategist] API failed ({type(e).__name__}) — switching to heuristic.")

        # ── Path 3: Local heuristic (always works) ─────────────────────────────
        return _heuristic_strategy(obs, self._current)

    def _parse_strategy(self, data: dict) -> Strategy:
        return Strategy(
            priority            = data.get("priority",            self._current.priority),
            moisture_threshold  = float(data.get("moisture_threshold",  self._current.moisture_threshold)),
            harvest_price_floor = float(data.get("harvest_price_floor", self._current.harvest_price_floor)),
            fertilize_max_day   = int(data.get("fertilize_max_day",     self._current.fertilize_max_day)),
            pest_threshold      = float(data.get("pest_threshold",       self._current.pest_threshold)),
            water_reserve_pct   = float(data.get("water_reserve_pct",   self._current.water_reserve_pct)),
            reasoning           = data.get("reasoning", ""),
        )

    def _build_prompt(self, obs, reward_history: list) -> str:
        crops = [
            {"id": c.id, "stage": c.stage, "moisture": round(c.moisture, 1),
             "growth": round(c.growth, 1), "pest": round(c.pest_level, 1)}
            for c in obs.crops
        ]
        state = {
            "day": obs.day, "water": round(obs.water, 1),
            "fertilizer": round(obs.fertilizer, 1), "pesticide": round(obs.pesticide, 1),
            "energy": round(obs.energy, 1), "weather": obs.weather,
            "forecast": obs.forecast, "market_price": round(obs.market_price, 2),
            "soil_health": round(obs.soil_health, 1), "crops": crops,
            "recent_rewards": reward_history[-5:],
        }
        return f"Farm state:\n{json.dumps(state, indent=2)}"

    def _load_mock(self) -> list:
        if not os.path.exists(MOCK_FILE):
            return []
        try:
            with open(MOCK_FILE) as f:
                data = json.load(f)
                return data.get("strategist", [])
        except Exception:
            return []