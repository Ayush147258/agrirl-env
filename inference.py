# inference.py
"""
AgriRL — OpenEnv-Compliant Hierarchical Multi-Agent Inference Engine
=====================================================================
OpenEnv validator requirements (DO NOT REMOVE):
  - API_BASE_URL, MODEL_NAME, HF_TOKEN env variables
  - OpenAI-compatible client for LLM calls
  - START / STEP / END / FINAL structured JSON stdout logging
  - if __name__ == "__main__": entry point

Five-layer differentiator architecture:
  Layer 1 — DigitalTwin         Real-world weather grounds simulation physics
  Layer 2 — StrategistAgent     LLM sets high-level directives every 5 days
  Layer 3 — Executor (act)      RL agent acts under Strategist directives
  Layer 4 — PostMortemAnalyst   LLM/statistical reflection patches next Strategy
  Layer 5 — AgriculturalLedger  Cross-task transfer: Easy lessons → Medium/Hard

Resilience:
  - All LLM calls have graceful degradation + heuristic/statistical fallbacks
  - USE_MOCK_AI=true enables full demo without any API key
  - Strategy thresholds visualised as a timeline table after each episode
  - Matplotlib charts saved as agri_dashboard_{task}.png

Run:
    python inference.py
    USE_MOCK_AI=true python inference.py
"""

import os
import json

from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI

# ── OpenEnv required env variables ────────────────────────────────────────────
API_BASE_URL     = os.getenv("API_BASE_URL",  "https://generativelanguage.googleapis.com/v1beta/openai/")
MODEL_NAME       = os.getenv("MODEL_NAME",    "gemini-2.0-flash")
HF_TOKEN         = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

# ── OpenAI-compatible client → Gemini ─────────────────────────────────────────
client = OpenAI(
    base_url = API_BASE_URL,
    api_key  = os.getenv("GEMINI_API_KEY", HF_TOKEN or "no-key"),
)

# ── Project imports ────────────────────────────────────────────────────────────
try:
    from server.agrirl_env_environment import AgriCoreEnv
except ModuleNotFoundError:
    from agrirl_env.server.agrirl_env_environment import AgriCoreEnv

try:
    try:
        from .models import AgrirlAction, AgrirlObservation
    except ImportError:
        from models import AgrirlAction, AgrirlObservation
    from grader import _get_score
    from strategist import StrategistAgent, Strategy
    from digital_twin import DigitalTwin
    from post_mortem import PostMortemAnalyst
    from knowledge_base import AgriculturalLedger
    from visualizer import StrategyTimeline, save_episode_charts
except ModuleNotFoundError:
    from agrirl_env.models import AgrirlAction as Action
    from agrirl_env.grader import _get_score
    from agrirl_env.strategist import StrategistAgent, Strategy
    from agrirl_env.digital_twin import DigitalTwin
    from agrirl_env.post_mortem import PostMortemAnalyst
    from agrirl_env.knowledge_base import AgriculturalLedger
    from agrirl_env.visualizer import StrategyTimeline, save_episode_charts


# ── Executor (RL Agent) ────────────────────────────────────────────────────────

def act(obs, strategy: "Strategy | None" = None) -> Action:
    """
    RL Executor — 8-priority decision tree driven by Strategy thresholds.
    Falls back to hardcoded defaults if no strategy provided.
    """
    if not obs.crops:
        return Action(crop_id=0, action="wait")

    # Use strategy thresholds or safe defaults
    moisture_threshold  = strategy.moisture_threshold  if strategy else 35.0
    pest_threshold      = strategy.pest_threshold      if strategy else 2.0
    harvest_price_floor = strategy.harvest_price_floor if strategy else 1.0
    water_reserve_pct   = strategy.water_reserve_pct   if strategy else 0.25
    fertilize_max_day   = strategy.fertilize_max_day   if strategy else 20

    # 1. EMERGENCY irrigation
    critical = [c for c in obs.crops if c.moisture < 20]
    if critical and obs.water >= 10 and obs.energy >= 5:
        return Action(crop_id=min(critical, key=lambda c: c.moisture).id, action="irrigate")

    # 2. PESTICIDE
    if obs.pesticide >= 5 and obs.energy >= 2:
        infested = [c for c in obs.crops if c.pest_level > pest_threshold]
        if infested:
            return Action(crop_id=max(infested, key=lambda c: c.pest_level).id, action="pesticide")

    # 3. HARVEST mature crops
    mature = [c for c in obs.crops if c.stage == "mature"]
    if mature and obs.market_price >= harvest_price_floor:
        return Action(crop_id=mature[0].id, action="harvest")

    # 4. WATER CONSERVATION
    if obs.forecast == "rainy" and not critical:
        if all(c.moisture >= moisture_threshold for c in obs.crops):
            return Action(crop_id=0, action="wait")

    # 5. IRRIGATE — water-budget aware
    dry_crops     = [c for c in obs.crops if c.moisture < moisture_threshold]
    water_reserve = obs.water * water_reserve_pct
    if dry_crops and obs.water - water_reserve >= 10 and obs.energy >= 5:
        water_ratio = obs.water / 120
        day_ratio   = obs.day   / 30
        if water_ratio < water_reserve_pct + 0.05 and day_ratio > 0.5:
            critical_only = [c for c in dry_crops if c.moisture < 30]
            if critical_only:
                return Action(crop_id=min(critical_only, key=lambda c: c.moisture).id, action="irrigate")
        else:
            return Action(crop_id=min(dry_crops, key=lambda c: c.moisture).id, action="irrigate")

    # 6. FERTILIZE — rationed
    if obs.fertilizer >= 5 and obs.energy >= 3 and obs.day <= fertilize_max_day:
        growing = [c for c in obs.crops if c.stage in ("vegetative", "flowering") and c.fertilized_times < 2]
        if growing:
            return Action(crop_id=growing[0].id, action="fertilize")
        if obs.fertilizer >= 20:
            early = [c for c in obs.crops if c.stage == "seed" and c.growth < 20 and c.fertilized_times < 2]
            if early:
                return Action(crop_id=early[0].id, action="fertilize")

    # 7. LATE GAME harvest
    if obs.day > 20 and mature:
        return Action(crop_id=mature[0].id, action="harvest")

    return Action(crop_id=0, action="wait")


def explain(obs, strategy: "Strategy | None" = None) -> str:
    """One-line human-readable justification of the chosen action."""
    if not obs.crops:
        return "No crops -> wait"

    moisture_threshold  = strategy.moisture_threshold  if strategy else 35.0
    pest_threshold      = strategy.pest_threshold      if strategy else 2.0
    harvest_price_floor = strategy.harvest_price_floor if strategy else 1.0
    water_reserve_pct   = strategy.water_reserve_pct   if strategy else 0.25

    critical = [c for c in obs.crops if c.moisture < 20]
    if critical and obs.water >= 10:
        c = min(critical, key=lambda c: c.moisture)
        return f"EMERGENCY irrigate crop {c.id} (moisture {c.moisture:.1f})"
    if obs.pesticide >= 5 and obs.energy >= 2:
        infested = [c for c in obs.crops if c.pest_level > pest_threshold]
        if infested:
            w = max(infested, key=lambda c: c.pest_level)
            return f"Spray crop {w.id} pest={w.pest_level:.1f} > {pest_threshold:.1f}"
    mature = [c for c in obs.crops if c.stage == "mature"]
    if mature and obs.market_price >= harvest_price_floor:
        return f"Harvest crop {mature[0].id} price={obs.market_price:.2f}"
    if obs.forecast == "rainy" and all(c.moisture >= moisture_threshold for c in obs.crops):
        return "Rain forecast -> conserve water"
    dry = [c for c in obs.crops if c.moisture < moisture_threshold]
    if dry and obs.water >= 10:
        water_ratio = obs.water / 120
        day_ratio   = obs.day   / 30
        if water_ratio < water_reserve_pct + 0.05 and day_ratio > 0.5:
            return f"Water scarce ({obs.water:.0f}) — critical crops only"
        c = min(dry, key=lambda c: c.moisture)
        return f"Irrigate crop {c.id} moisture={c.moisture:.1f} < {moisture_threshold:.1f}"
    if obs.fertilizer >= 5:
        growing = [c for c in obs.crops if c.stage in ("vegetative", "flowering") and c.fertilized_times < 2]
        if growing:
            return f"Fertilize crop {growing[0].id} stage={growing[0].stage}"
    if obs.day > 20 and mature:
        return f"Late harvest crop {mature[0].id}"
    return "All stable -> wait"


# ── LLM Episode Directive (OpenAI client → Gemini) ────────────────────────────

def get_llm_directive(obs) -> str:
    """
    One-shot LLM tactical directive at episode start.
    Uses OpenAI-compatible client pointing to Gemini.
    Falls back gracefully if API unavailable.
    """
    try:
        crops_summary = [
            {"id": c.id, "stage": c.stage, "moisture": round(c.moisture, 1)}
            for c in obs.crops
        ]
        prompt = (
            f"Farm state: day={obs.day}, water={obs.water:.0f}, "
            f"soil_health={obs.soil_health:.1f}, weather={obs.weather}, "
            f"market_price={obs.market_price:.2f}, crops={crops_summary}. "
            f"Give one short tactical directive for this episode (max 15 words)."
        )
        response = client.chat.completions.create(
            model      = MODEL_NAME,
            messages   = [{"role": "user", "content": prompt}],
            max_tokens = 40,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Heuristic mode (API unavailable: {type(e).__name__})"


# ── Episode Runner ─────────────────────────────────────────────────────────────

def run_episode(
    task:             str = "easy",
    verbose:          bool = True,
    region:           str = "punjab",
    use_real_weather: bool = True,
    use_strategist:   bool = True,
    use_post_mortem:  bool = True,
    carry_strategy:   "Strategy | None" = None,
    ledger:           "AgriculturalLedger | None" = None,
) -> tuple:
    """
    Run one full episode with all five layers + OpenEnv structured logging.

    Returns:
        (final_score: float, final_strategy: Strategy, report)
    """
    # ── Layer init ─────────────────────────────────────────────────────────────
    env        = AgriCoreEnv()
    strategist = StrategistAgent(review_interval=5) if use_strategist  else None
    twin       = DigitalTwin(region=region) if (
                     use_real_weather and
                     region in ("punjab", "california", "midwest", "maharashtra")
                 ) else None
    analyst    = PostMortemAnalyst()                if use_post_mortem  else None
    timeline   = StrategyTimeline()

    # Layer 5: pre-load cross-task lessons from ledger
    strategy = carry_strategy if carry_strategy else Strategy()
    if ledger and task in ("medium", "hard"):
        prev_task = "easy" if task == "medium" else "medium"
        strategy  = ledger.load_lessons_into(strategy, from_task=prev_task)

    obs            = env.reset(task=task)
    total_reward   = 0.0
    step           = 0
    episode_log    = []
    reward_history = []

    # Layer 1: real-world weather snapshot
    weather_snapshot = None
    real_temp        = 25.0
    if twin:
        weather_snapshot = twin.fetch()
        real_temp        = weather_snapshot.temperature_c

    # ── OpenEnv START log ──────────────────────────────────────────────────────
    llm_directive = get_llm_directive(obs)
    print(json.dumps({
        "type":      "START",
        "task":      task,
        "model":     MODEL_NAME,
        "directive": llm_directive,
        "region":    region,
        "real_temp": real_temp,
    }))

    timeline.record(day=1, strategy=strategy, trigger="episode start")

    # ── Episode loop ───────────────────────────────────────────────────────────
    while not obs.done:

        # Layer 1: Digital Twin weather grounding
        if twin and weather_snapshot:
            obs = twin.apply(obs, weather_snapshot)

        # Layer 2: Strategist — LLM directives every 5 days
        prev_priority = strategy.priority
        if strategist:
            strategy = strategist.advise(obs, reward_history)
            if obs.day % 5 == 1 or strategy.priority != prev_priority:
                trigger = (
                    f"{obs.weather} day {obs.day}"
                    if strategy.priority != prev_priority
                    else f"day {obs.day} review"
                )
                timeline.record(day=obs.day, strategy=strategy, trigger=trigger)

        # Layer 3: Executor
        action = act(obs, strategy)
        reason = explain(obs, strategy)
        obs    = env.step(action)
        total_reward += obs.reward
        step         += 1
        reward_history.append(round(obs.reward, 3))

        # Step log
        avg_moisture = sum(c.moisture for c in obs.crops) / max(len(obs.crops), 1)
        episode_log.append({
            "day":          obs.day,
            "action":       action.action,
            "crop_id":      action.crop_id,
            "reward":       round(obs.reward, 3),
            "water":        round(obs.water, 1),
            "energy":       round(obs.energy, 1),
            "soil_health":  round(obs.soil_health, 1),
            "weather":      obs.weather,
            "avg_moisture": round(avg_moisture, 1),
            "reason":       reason,
        })

        # ── OpenEnv STEP log ───────────────────────────────────────────────────
        print(json.dumps({
            "type":      "STEP",
            "day":       obs.day,
            "action":    action.action,
            "crop_id":   action.crop_id,
            "reward":    round(obs.reward, 3),
            "reason":    reason,
            "strategy":  strategy.priority,
            "water":     round(obs.water, 1),
        }))

    # ── Episode end ────────────────────────────────────────────────────────────
    final_score = _get_score(obs)

    # Layer 4: Post-Mortem
    report = None
    if analyst:
        report   = analyst.analyse(episode_log, total_reward, final_score, task)
        strategy = analyst.patch_strategy(strategy, report)

        # Layer 5: Ledger — persist lessons for next task
        if ledger and report:
            ledger.save(task=task, report=report, region=region, temp_c=real_temp)

    # Strategy timeline table to stdout
    timeline.print_table()

    # Charts
    save_episode_charts(
        episode_log = episode_log,
        task        = task,
        final_score = final_score,
        output_path = f"agri_dashboard_{task}.png",
    )

    # ── OpenEnv END log ────────────────────────────────────────────────────────
    print(json.dumps({
        "type":         "END",
        "task":         task,
        "days_run":     step,
        "total_reward": round(total_reward, 3),
        "final_score":  round(final_score, 3),
        "water_left":   round(env.water, 1),
        "soil_health":  round(env.soil_health, 1),
        "root_cause":   report.root_cause       if report else "N/A",
        "directive":    report.next_run_directive if report else "N/A",
    }))

    return final_score, strategy, report


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ledger   = AgriculturalLedger()
    scores   = []
    strategy = None

    for task in ["easy", "medium", "hard"]:
        score, strategy, _ = run_episode(
            task             = task,
            verbose          = True,
            region           = "punjab",
            use_real_weather = True,
            use_strategist   = True,
            use_post_mortem  = True,
            carry_strategy   = strategy,
            ledger           = ledger,
        )
        scores.append(score)

    ledger.print_summary()

    avg = sum(scores) / len(scores)

    # ── OpenEnv FINAL log ──────────────────────────────────────────────────────
    print(json.dumps({
        "type":         "FINAL",
        "easy_score":   round(scores[0], 3),
        "medium_score": round(scores[1], 3),
        "hard_score":   round(scores[2], 3),
        "average":      round(avg, 3),
    }))


if __name__ == "__main__":
    main()