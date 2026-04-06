# post_mortem.py
"""
Post-Mortem Analyst — Self-Refining Reflection Loop
-----------------------------------------------------
At the end of each episode, performs causal analysis and emits
actionable strategy adjustments for the next run.

OpenEnv compliance:
  - Reads API_BASE_URL, MODEL_NAME, HF_TOKEN env variables
  - OpenAI client for all LLM calls
  - Warnings emitted as structured JSON {"type": "WARN", ...}

Resilience hierarchy:
  1. Mock mode (USE_MOCK_AI=true)  → reads from mock_responses.json
  2. Live API (OpenAI → Gemini)    → full causal analysis
  3. Statistical fallback          → pure-Python reward/water/soil analysis

Differentiator features preserved:
  - Full ReflectionReport dataclass with display() table
  - Statistical fallback with 4 scenario rules
  - patch_strategy() with 70/30 blend — prevents overcorrection
  - Detailed SYSTEM_PROMPT with all 9 JSON fields + rules
  - Smart episode log sampling for large episodes
"""

import json
import os
import re
import statistics
from dataclasses import dataclass
from typing import Optional

from openai import OpenAI

# ── OpenEnv env variables (matches inference.py) ──────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL",  "https://generativelanguage.googleapis.com/v1beta/openai/")
MODEL_NAME   = os.getenv("MODEL_NAME",    "gemini-2.0-flash")
HF_TOKEN     = os.getenv("HF_TOKEN")

USE_MOCK_AI = os.getenv("USE_MOCK_AI", "false").lower() == "true"
MOCK_FILE   = "mock_responses.json"


# ── Reflection Report ──────────────────────────────────────────────────────────

@dataclass
class ReflectionReport:
    """
    Full causal analysis of a completed episode.
    Stored in AgriculturalLedger for cross-task transfer.
    """
    total_reward:                    float
    final_score:                     float
    failure_day:                     Optional[int]
    root_cause:                      str
    key_mistake:                     str
    recommended_moisture_threshold:  float
    recommended_harvest_price_floor: float
    recommended_pest_threshold:      float
    recommended_water_reserve_pct:   float
    next_run_directive:              str
    confidence:                      float

    def display(self) -> str:
        lines = [
            "",
            "  +============================================================+",
            "  |            REFLECTION REPORT (Post-Mortem)                |",
            "  +============================================================+",
            f"  |  Total Reward    : {self.total_reward:>8.2f}                          |",
            f"  |  Final Score     : {self.final_score:>8.2f}                          |",
            f"  |  Failure Day     : {str(self.failure_day or 'N/A'):>8}                          |",
            "  +------------------------------------------------------------+",
            f"  |  Root Cause   : {self.root_cause[:44]:<44} |",
            f"  |  Key Mistake  : {self.key_mistake[:44]:<44} |",
            "  +------------------------------------------------------------+",
            "  |  RECOMMENDED THRESHOLDS FOR NEXT RUN:                     |",
            f"  |    moisture_threshold    -> {self.recommended_moisture_threshold:<6.1f}                      |",
            f"  |    harvest_price_floor  -> {self.recommended_harvest_price_floor:<6.2f}                      |",
            f"  |    pest_threshold       -> {self.recommended_pest_threshold:<6.1f}                      |",
            f"  |    water_reserve_pct    -> {self.recommended_water_reserve_pct:<6.2f}                      |",
            "  +------------------------------------------------------------+",
            f"  |  Directive : {self.next_run_directive[:48]:<48} |",
            f"  |  Confidence: {self.confidence:.0%}                                         |",
            "  +============================================================+",
        ]
        return "\n".join(lines)


# ── Statistical Fallback Analyser ──────────────────────────────────────────────

def _statistical_analysis(
    episode_log:  list,
    total_reward: float,
    final_score:  float,
) -> ReflectionReport:
    """
    Pure-Python statistical post-mortem — no API required.
    Analyses reward curve, water usage, and soil health drops
    to produce deterministic recommendations.
    """
    rewards      = [e["reward"]      for e in episode_log]
    water        = [e["water"]       for e in episode_log]
    soil         = [e["soil_health"] for e in episode_log]

    # Find the day with the worst reward drop
    failure_day = None
    if len(rewards) > 1:
        drops      = [rewards[i] - rewards[i - 1] for i in range(1, len(rewards))]
        worst_drop = min(drops)
        if worst_drop < -3.0:
            failure_day = episode_log[drops.index(worst_drop) + 1]["day"]

    avg_water    = statistics.mean(water) if water else 80
    min_soil     = min(soil)              if soil  else 80
    pct_negative = sum(1 for r in rewards if r < 0) / max(len(rewards), 1)
    rec_pest     = 2.0  # safe default

    if avg_water < 30:
        root_cause  = "Insufficient water reserve — ran out too early"
        key_mistake = "Did not conserve water in mid-game"
        rec_moisture = 45.0
        rec_reserve  = 0.35

    elif min_soil < 40:
        root_cause  = "Soil health degraded — pest or over-fertilization"
        key_mistake = "Did not treat pests early enough"
        rec_moisture = 35.0
        rec_reserve  = 0.25
        rec_pest     = 1.5

    elif pct_negative > 0.3:
        root_cause  = "High proportion of negative rewards"
        key_mistake = "Over-irrigated or acted when waiting was better"
        rec_moisture = 40.0
        rec_reserve  = 0.30

    else:
        root_cause  = "Performance within expected range"
        key_mistake = "Minor resource inefficiency"
        rec_moisture = 35.0
        rec_reserve  = 0.25

    directive = (
        f"Raise moisture_threshold to {rec_moisture:.0f} and "
        f"keep {rec_reserve*100:.0f}% water reserve from Day 1."
    )

    return ReflectionReport(
        total_reward                    = total_reward,
        final_score                     = final_score,
        failure_day                     = failure_day,
        root_cause                      = root_cause,
        key_mistake                     = key_mistake,
        recommended_moisture_threshold  = rec_moisture,
        recommended_harvest_price_floor = 1.0,
        recommended_pest_threshold      = rec_pest,
        recommended_water_reserve_pct   = rec_reserve,
        next_run_directive              = directive,
        confidence                      = 0.6,
    )


# ── Post-Mortem Analyst ────────────────────────────────────────────────────────

class PostMortemAnalyst:
    """
    LLM-powered post-episode reflection agent with full fallback chain.

    Resilience hierarchy:
      1. Mock mode     → pre-written responses from mock_responses.json
      2. Live API      → OpenAI-compatible client → Gemini
      3. Statistical   → pure-Python reward/water/soil analysis (always works)
    """

    SYSTEM_PROMPT = """You are a precision agricultural AI analyst.
You receive an episode log and must identify failure causes and improvements.

Output ONLY a valid JSON object (no markdown, no extra text):
{
  "failure_day":                     <int or null>,
  "root_cause":                      "<max 55 chars>",
  "key_mistake":                     "<max 55 chars>",
  "recommended_moisture_threshold":  <float 20-60>,
  "recommended_harvest_price_floor": <float 0.5-2.0>,
  "recommended_pest_threshold":      <float 1.0-5.0>,
  "recommended_water_reserve_pct":   <float 0.1-0.5>,
  "next_run_directive":              "<max 80 chars, concrete actionable sentence>",
  "confidence":                      <float 0.0-1.0>
}
Rules:
- Sharp reward drop on a day = failure_day.
- Crops dried out → mention "insufficient irrigation" in root_cause.
- soil_health < 40 → mention "soil degradation".
- Water exhausted before day 20 → raise recommended_water_reserve_pct.
- Pests caused loss → lower recommended_pest_threshold."""

    def __init__(self):
        self._mock_pool  = self._load_mock()
        self._mock_index = 0
        self._client     = None

        if not USE_MOCK_AI:
            try:
                self._client = OpenAI(
                    base_url = API_BASE_URL,
                    api_key  = os.getenv("GEMINI_API_KEY", HF_TOKEN or "no-key"),
                )
            except Exception as e:
                print(json.dumps({"type": "WARN", "msg": f"PostMortem client failed: {e}"}))

    def analyse(
        self,
        episode_log:  list,
        total_reward: float,
        final_score:  float,
        task:         str = "easy",
    ) -> ReflectionReport:
        """Run post-mortem via mock → live API → statistical fallback."""

        # Path 1: Mock
        if USE_MOCK_AI and self._mock_pool:
            entry = self._mock_pool[self._mock_index % len(self._mock_pool)]
            self._mock_index += 1
            print(json.dumps({"type": "WARN", "msg": f"PostMortem mock response #{self._mock_index}"}))
            return self._parse(entry, total_reward, final_score)

        # Path 2: Live OpenAI client → Gemini
        if self._client:
            try:
                prompt   = self._build_prompt(episode_log, total_reward, final_score, task)
                response = self._client.chat.completions.create(
                    model    = MODEL_NAME,
                    messages = [
                        {"role": "system", "content": self.SYSTEM_PROMPT},
                        {"role": "user",   "content": prompt},
                    ],
                    max_tokens = 350,
                )
                raw = (response.choices[0].message.content or "").strip()
                raw = re.sub(r"^```[a-z]*\n?", "", raw).rstrip("```").strip()
                return self._parse(json.loads(raw), total_reward, final_score)
            except Exception as e:
                print(json.dumps({"type": "WARN", "msg": f"PostMortem API failed: {type(e).__name__} — statistical"}))

        # Path 3: Statistical fallback (always works)
        return _statistical_analysis(episode_log, total_reward, final_score)

    def patch_strategy(self, strategy, report: ReflectionReport):
        """
        Blend report recommendations into strategy (70/30 — old/new)
        to prevent overcorrection across episodes.
        """
        BLEND, NEW = 0.7, 0.3
        strategy.moisture_threshold  = BLEND * strategy.moisture_threshold  + NEW * report.recommended_moisture_threshold
        strategy.harvest_price_floor = BLEND * strategy.harvest_price_floor + NEW * report.recommended_harvest_price_floor
        strategy.pest_threshold      = BLEND * strategy.pest_threshold      + NEW * report.recommended_pest_threshold
        strategy.water_reserve_pct   = BLEND * strategy.water_reserve_pct  + NEW * report.recommended_water_reserve_pct
        strategy.reasoning           = f"[Post-Mortem Adjusted] {report.next_run_directive}"
        return strategy

    def _parse(self, data: dict, total_reward: float, final_score: float) -> ReflectionReport:
        return ReflectionReport(
            total_reward                    = total_reward,
            final_score                     = final_score,
            failure_day                     = data.get("failure_day"),
            root_cause                      = data.get("root_cause",   "Unknown"),
            key_mistake                     = data.get("key_mistake",  "Unknown"),
            recommended_moisture_threshold  = float(data.get("recommended_moisture_threshold",  35.0)),
            recommended_harvest_price_floor = float(data.get("recommended_harvest_price_floor",  1.0)),
            recommended_pest_threshold      = float(data.get("recommended_pest_threshold",        2.0)),
            recommended_water_reserve_pct   = float(data.get("recommended_water_reserve_pct",    0.25)),
            next_run_directive              = data.get("next_run_directive", "Maintain current strategy."),
            confidence                      = float(data.get("confidence", 0.7)),
        )

    def _build_prompt(
        self,
        episode_log:  list,
        total_reward: float,
        final_score:  float,
        task:         str,
    ) -> str:
        # Smart sampling — keep first 5, middle 4, last 5 for large episodes
        sample = (
            episode_log if len(episode_log) <= 30 else
            episode_log[:5] +
            episode_log[len(episode_log)//2-2 : len(episode_log)//2+2] +
            episode_log[-5:]
        )
        return (
            f"Task: {task} | Reward: {total_reward:.2f} | Score: {final_score:.2f}\n\n"
            f"Episode log ({len(sample)} of {len(episode_log)} steps):\n"
            f"{json.dumps(sample, indent=2)}"
        )

    def _load_mock(self) -> list:
        if not os.path.exists(MOCK_FILE):
            return []
        try:
            with open(MOCK_FILE) as f:
                return json.load(f).get("post_mortem", [])
        except Exception:
            return []