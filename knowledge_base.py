# knowledge_base.py
"""
Agricultural Ledger — Cross-Task Transfer Learning
---------------------------------------------------
Persists ReflectionReport data to a local knowledge_base.json file.
When starting a new task (Medium/Hard), the agent reads lessons from
previous tasks and applies them immediately — zero warm-up needed.

Pattern: Cross-Task Transfer Learning
  Easy episode ends  → lessons saved  → Medium episode starts
  Medium loads Easy lessons → pre-tuned thresholds from Day 1
"""

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Optional

LEDGER_PATH = "knowledge_base.json"


# ── Ledger Entry ───────────────────────────────────────────────────────────────

@dataclass
class LedgerEntry:
    task: str
    timestamp: str
    total_reward: float
    final_score: float
    failure_day: Optional[int]
    root_cause: str
    key_mistake: str
    next_run_directive: str
    recommended_moisture_threshold: float
    recommended_harvest_price_floor: float
    recommended_pest_threshold: float
    recommended_water_reserve_pct: float
    confidence: float
    real_weather_region: str = "unknown"
    real_temp_c: float = 25.0


# ── Agricultural Ledger ────────────────────────────────────────────────────────

class AgriculturalLedger:
    """
    Persistent cross-episode memory system.

    Usage:
        ledger = AgriculturalLedger()
        ledger.save(task="easy", report=report, region="punjab", temp=32.0)
        strategy = ledger.load_lessons_into(strategy, from_task="easy")
    """

    def __init__(self, path: str = LEDGER_PATH):
        self.path = path
        self._data: list[dict] = self._load_raw()

    # ── Public API ─────────────────────────────────────────────────────────────

    def save(self, task: str, report, region: str = "unknown", temp_c: float = 25.0):
        """
        Persist a ReflectionReport as a LedgerEntry.

        Args:
            task:    "easy" | "medium" | "hard"
            report:  PostMortemAnalyst ReflectionReport
            region:  Digital Twin region name
            temp_c:  Real-world temperature at time of run
        """
        entry = LedgerEntry(
            task                            = task,
            timestamp                       = datetime.now().isoformat(timespec="seconds"),
            total_reward                    = round(report.total_reward, 3),
            final_score                     = round(report.final_score, 3),
            failure_day                     = report.failure_day,
            root_cause                      = report.root_cause,
            key_mistake                     = report.key_mistake,
            next_run_directive              = report.next_run_directive,
            recommended_moisture_threshold  = round(report.recommended_moisture_threshold, 2),
            recommended_harvest_price_floor = round(report.recommended_harvest_price_floor, 2),
            recommended_pest_threshold      = round(report.recommended_pest_threshold, 2),
            recommended_water_reserve_pct   = round(report.recommended_water_reserve_pct, 3),
            confidence                      = round(report.confidence, 2),
            real_weather_region             = region,
            real_temp_c                     = round(temp_c, 1),
        )
        self._data.append(asdict(entry))
        self._flush()
        print(f"  [Ledger] Entry saved for task='{task}' → {self.path}")

    def load_lessons_into(self, strategy, from_task: str = "easy"):
        """
        Find the most recent high-confidence entry for `from_task` and
        blend its recommended thresholds into the given Strategy.

        Uses a 60/40 blend (new lesson / current strategy) so lessons
        from Easy don't completely override Medium defaults.

        Args:
            strategy:  A strategist.Strategy instance to patch.
            from_task: Which task's lessons to load ("easy" | "medium").

        Returns:
            The patched Strategy (mutated in-place).
        """
        entry = self._best_entry(from_task)
        if entry is None:
            print(f"  [Ledger] No lessons found for task='{from_task}', using defaults.")
            return strategy

        LESSON = 0.6    # weight for the lesson
        CURR   = 0.4    # weight for current strategy

        strategy.moisture_threshold  = LESSON * entry["recommended_moisture_threshold"]  + CURR * strategy.moisture_threshold
        strategy.harvest_price_floor = LESSON * entry["recommended_harvest_price_floor"] + CURR * strategy.harvest_price_floor
        strategy.pest_threshold      = LESSON * entry["recommended_pest_threshold"]      + CURR * strategy.pest_threshold
        strategy.water_reserve_pct   = LESSON * entry["recommended_water_reserve_pct"]   + CURR * strategy.water_reserve_pct
        strategy.reasoning           = (
            f"[Cross-Task Transfer from {from_task.upper()}] {entry['next_run_directive']}"
        )

        print(
            f"  [Ledger] Loaded lessons from task='{from_task}' "
            f"(score={entry['final_score']:.2f}, confidence={entry['confidence']:.0%}) "
            f"→ strategy pre-tuned."
        )
        return strategy

    def print_summary(self):
        """Print a formatted history table of all ledger entries."""
        if not self._data:
            print("  [Ledger] No entries yet.")
            return

        print("\n  ╔══════════════════════════════════════════════════════════════════╗")
        print("  ║                   📖  AGRICULTURAL LEDGER                      ║")
        print("  ╠══════╦═══════╦══════════╦═════════════════════════════════════╣")
        print("  ║ Task ║ Score ║ Fail Day ║ Root Cause                          ║")
        print("  ╠══════╬═══════╬══════════╬═════════════════════════════════════╣")
        for e in self._data[-10:]:      # show last 10 entries
            task      = e["task"][:4].upper()
            score     = f"{e['final_score']:.2f}"
            fail_day  = str(e["failure_day"] or "N/A")
            cause     = e["root_cause"][:35]
            print(f"  ║ {task:<4} ║ {score:<5} ║ {fail_day:<8} ║ {cause:<35} ║")
        print("  ╚══════╩═══════╩══════════╩═════════════════════════════════════╝")

    # ── Internal ───────────────────────────────────────────────────────────────

    def _best_entry(self, task: str) -> Optional[dict]:
        """Return the highest-confidence entry for a given task."""
        candidates = [e for e in self._data if e["task"] == task]
        if not candidates:
            return None
        return max(candidates, key=lambda e: e["confidence"])

    def _load_raw(self) -> list[dict]:
        if not os.path.exists(self.path):
            return []
        try:
            with open(self.path, "r") as f:
                data = json.load(f)
                return data if isinstance(data, list) else []
        except (json.JSONDecodeError, IOError):
            return []

    def _flush(self):
        with open(self.path, "w") as f:
            json.dump(self._data, f, indent=2)