# inference.py
"""
AgriRL — Hierarchical Multi-Agent Inference Engine
===================================================
Five-layer architecture:

  Layer 1 — DigitalTwin       Real-world weather grounds simulation physics
  Layer 2 — StrategistAgent   LLM sets high-level directives every 5 days
  Layer 3 — Executor (act)    RL agent acts under Strategist directives
  Layer 4 — PostMortemAnalyst LLM/statistical reflection patches next Strategy
  Layer 5 — AgriculturalLedger Cross-task transfer: Easy lessons → Medium/Hard

Resilience:
  - All LLM calls have graceful degradation + heuristic/statistical fallbacks
  - USE_MOCK_AI=true enables full demo without any API key
  - Strategy thresholds visualised as a timeline table after each episode
  - Matplotlib charts saved as agri_dashboard_{task}.png

Run:
    python inference.py
    USE_MOCK_AI=true python inference.py   # demo / no API key needed
"""

import os

from grader import grade_episode, evaluate, _get_score
from strategist import StrategistAgent, Strategy
from digital_twin import DigitalTwin
from post_mortem import PostMortemAnalyst
from knowledge_base import AgriculturalLedger
from visualizer import StrategyTimeline, save_episode_charts

from server.agrirl_env_environment import AgriCoreEnv  # type: ignore
from models import AgrirlAction as Action


# ── Executor (RL Agent) ───────────────────────────────────────────────────────

def act(obs, strategy: Strategy) -> Action:
    """
    RL Executor — logic-heavy rule system driven entirely by Strategy thresholds.
    Every threshold comes from the Strategist; nothing is hardcoded here.

    Priority order:
      1. Emergency irrigation  (moisture < 20 — always override)
      2. Pesticide             (pest_level > strategy.pest_threshold)
      3. Harvest               (mature + price >= strategy.harvest_price_floor)
      4. Water conservation    (rain forecast + all crops above threshold)
      5. Irrigate              (below strategy.moisture_threshold, budget-aware)
      6. Fertilize             (within strategy.fertilize_max_day, rationed)
      7. Late-game harvest     (day > 20 regardless of price)
      8. Wait
    """
    if not obs.crops:
        return Action(crop_id=0, action="wait")

    # ── 1. EMERGENCY ──────────────────────────────────────────────────────────
    critical = [c for c in obs.crops if c.moisture < 20]
    if critical and obs.water >= 10 and obs.energy >= 5:
        return Action(crop_id=min(critical, key=lambda c: c.moisture).id, action="irrigate")

    # ── 2. PESTICIDE ──────────────────────────────────────────────────────────
    if obs.pesticide >= 5 and obs.energy >= 2:
        infested = [c for c in obs.crops if c.pest_level > strategy.pest_threshold]
        if infested:
            return Action(crop_id=max(infested, key=lambda c: c.pest_level).id, action="pesticide")

    # ── 3. HARVEST ────────────────────────────────────────────────────────────
    mature = [c for c in obs.crops if c.stage == "mature"]
    if mature and obs.market_price >= strategy.harvest_price_floor:
        return Action(crop_id=mature[0].id, action="harvest")

    # ── 4. WATER CONSERVATION ────────────────────────────────────────────────
    if obs.forecast == "rainy" and not critical:
        if all(c.moisture >= strategy.moisture_threshold for c in obs.crops):
            return Action(crop_id=0, action="wait")

    # ── 5. IRRIGATE ───────────────────────────────────────────────────────────
    dry_crops     = [c for c in obs.crops if c.moisture < strategy.moisture_threshold]
    water_reserve = obs.water * strategy.water_reserve_pct
    if dry_crops and obs.water - water_reserve >= 10 and obs.energy >= 5:
        water_ratio = obs.water / 120
        day_ratio   = obs.day   / 30
        if water_ratio < strategy.water_reserve_pct + 0.05 and day_ratio > 0.5:
            critical_only = [c for c in dry_crops if c.moisture < 30]
            if critical_only:
                return Action(crop_id=min(critical_only, key=lambda c: c.moisture).id, action="irrigate")
        else:
            return Action(crop_id=min(dry_crops, key=lambda c: c.moisture).id, action="irrigate")

    # ── 6. FERTILIZE ──────────────────────────────────────────────────────────
    if obs.fertilizer >= 5 and obs.energy >= 3 and obs.day <= strategy.fertilize_max_day:
        growing = [c for c in obs.crops if c.stage in ("vegetative", "flowering") and c.fertilized_times < 2]
        if growing:
            return Action(crop_id=growing[0].id, action="fertilize")
        if obs.fertilizer >= 20:
            early = [c for c in obs.crops if c.stage == "seed" and c.growth < 20 and c.fertilized_times < 2]
            if early:
                return Action(crop_id=early[0].id, action="fertilize")

    # ── 7. LATE GAME HARVEST ──────────────────────────────────────────────────
    if obs.day > 20 and mature:
        return Action(crop_id=mature[0].id, action="harvest")

    return Action(crop_id=0, action="wait")


def explain(obs, strategy: Strategy) -> str:
    """One-line human-readable justification of the chosen action."""
    if not obs.crops:
        return "No crops -> wait"
    critical = [c for c in obs.crops if c.moisture < 20]
    if critical and obs.water >= 10:
        c = min(critical, key=lambda c: c.moisture)
        return f"EMERGENCY irrigate crop {c.id} (moisture {c.moisture:.1f})"
    if obs.pesticide >= 5 and obs.energy >= 2:
        infested = [c for c in obs.crops if c.pest_level > strategy.pest_threshold]
        if infested:
            w = max(infested, key=lambda c: c.pest_level)
            return f"Spray crop {w.id} — pest {w.pest_level:.1f} > threshold {strategy.pest_threshold:.1f}"
    mature = [c for c in obs.crops if c.stage == "mature"]
    if mature and obs.market_price >= strategy.harvest_price_floor:
        return f"Harvest crop {mature[0].id} — price {obs.market_price:.2f} >= floor {strategy.harvest_price_floor:.2f}"
    if obs.forecast == "rainy" and all(c.moisture >= strategy.moisture_threshold for c in obs.crops):
        return "Rain forecast + moisture OK -> conserve water"
    dry = [c for c in obs.crops if c.moisture < strategy.moisture_threshold]
    if dry and obs.water >= 10:
        water_ratio = obs.water / 120
        day_ratio   = obs.day   / 30
        if water_ratio < strategy.water_reserve_pct + 0.05 and day_ratio > 0.5:
            return f"Water scarce ({obs.water:.0f}) — critical crops only"
        c = min(dry, key=lambda c: c.moisture)
        return f"Irrigate crop {c.id} — moisture {c.moisture:.1f} < {strategy.moisture_threshold:.1f}"
    if obs.fertilizer >= 5 and obs.day <= strategy.fertilize_max_day:
        growing = [c for c in obs.crops if c.stage in ("vegetative", "flowering") and c.fertilized_times < 2]
        if growing:
            return f"Fertilize crop {growing[0].id} — {growing[0].stage} stage"
    if obs.day > 20 and mature:
        return f"Late-game harvest crop {mature[0].id}"
    return "All stable -> wait"


# ── Episode Runner ─────────────────────────────────────────────────────────────

def run_episode(
    task: str = "easy",
    verbose: bool = True,
    region: str = "punjab",
    use_real_weather: bool = True,
    use_strategist: bool = True,
    use_post_mortem: bool = True,
    carry_strategy:"Strategy | None" = None,
    ledger: "AgriculturalLedger|None" = None,
) -> tuple:
    """
    Run one full episode with all five layers active.

    Returns:
        (final_score: float, final_strategy: Strategy, report: ReflectionReport)
    """
    # ── Init ───────────────────────────────────────────────────────────────────
    env        = AgriCoreEnv()
    strategist = StrategistAgent(review_interval=5) if use_strategist  else None
    twin = DigitalTwin(region=region) if (use_real_weather and region in ("punjab", "california", "midwest", "maharashtra")) else None
    analyst    = PostMortemAnalyst()                if use_post_mortem  else None
    timeline   = StrategyTimeline()

    # Load cross-task lessons from ledger before episode starts
    strategy = carry_strategy if carry_strategy else Strategy()
    if ledger and task in ("medium", "hard"):
        prev_task = "easy" if task == "medium" else "medium"
        strategy  = ledger.load_lessons_into(strategy, from_task=prev_task)

    obs            = env.reset(task=task)
    total_reward   = 0.0
    step           = 0
    episode_log    = []
    reward_history = []

    # ── Real-world weather snapshot ────────────────────────────────────────────
    weather_snapshot = None
    real_temp        = 25.0
    if twin:
        weather_snapshot = twin.fetch()
        real_temp        = weather_snapshot.temperature_c
        print(f"\n  {weather_snapshot.summary()}")

    print(f"\n  Running Episode | Task: {task.upper()} | Region: {region.upper()}")
    if os.getenv("USE_MOCK_AI", "false").lower() == "true":
        print("  [Mode: MOCK AI — using cached responses]")
    print("  " + "=" * 70)

    # Record initial strategy
    timeline.record(day=1, strategy=strategy, trigger="episode start")

    # ── Episode loop ───────────────────────────────────────────────────────────
    while not obs.done:

        # Layer 1: Digital Twin
        if twin and weather_snapshot:
            obs = twin.apply(obs, weather_snapshot)

        # Layer 2: Strategist
        prev_priority = strategy.priority
        if strategist:
            strategy = strategist.advise(obs, reward_history)
            # Record snapshot on update days or when priority changes
            if obs.day % 5 == 1 or strategy.priority != prev_priority:
                trigger = (
                    f"{obs.weather} day {obs.day}"
                    if strategy.priority != prev_priority
                    else f"day {obs.day} review"
                )
                timeline.record(day=obs.day, strategy=strategy, trigger=trigger)
                print(f"\n  [Strategist] {strategy.summary()}")
                if strategy.reasoning:
                    print(f"               {strategy.reasoning}")

        # Layer 3: Executor
        action = act(obs, strategy)
        reason = explain(obs, strategy)
        obs    = env.step(action)
        total_reward += obs.reward
        step         += 1
        reward_history.append(round(obs.reward, 3))

        # Log step
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

        if verbose:
            print(
                f"  Day {obs.day:>2} | "
                f"Act: {action.action:<10} Crop:{action.crop_id} | "
                f"Reward: {obs.reward:>6.2f} | {reason}"
            )

    # ── Episode end ────────────────────────────────────────────────────────────
    final_score = _get_score(obs)

    print(f"\n  {'='*70}")
    print(f"  Episode Summary | Task: {task} | Score: {final_score:.2f} | Reward: {total_reward:.2f}")
    print(f"  Water: {env.water:.1f} | Fertilizer: {env.fertilizer:.1f} | "
          f"Pesticide: {env.pesticide:.1f} | Soil: {env.soil_health:.1f}")
    print(f"  Crops: {[c.stage for c in env.crops]}")

    # ── Strategy Timeline Table ────────────────────────────────────────────────
    timeline.print_table()

    # ── Layer 4: Post-Mortem ───────────────────────────────────────────────────
    report = None
    if analyst:
        report   = analyst.analyse(episode_log, total_reward, final_score, task)
        print(report.display())
        strategy = analyst.patch_strategy(strategy, report)
        print(f"\n  [PostMortem] Strategy patched: {strategy.summary()}")

        # ── Layer 5: Ledger — persist lessons ─────────────────────────────────
        if ledger and report:
            ledger.save(task=task, report=report, region=region, temp_c=real_temp)

    # ── Charts ─────────────────────────────────────────────────────────────────
    save_episode_charts(
        episode_log  = episode_log,
        task         = task,
        final_score  = final_score,
        output_path  = f"agri_dashboard_{task}.png",
    )

    return final_score, strategy, report


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("\n  AgriRL — Hierarchical Multi-Agent Farm Evaluation")
    print("  " + "=" * 70)

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

    # Print ledger history
    ledger.print_summary()

    avg = sum(scores) / len(scores)
    print("\n  " + "=" * 70)
    print("  FINAL RESULTS")
    print(f"  Easy Score   : {scores[0]:.2f}")
    print(f"  Medium Score : {scores[1]:.2f}")
    print(f"  Hard Score   : {scores[2]:.2f}")
    print(f"  Average      : {avg:.2f}")
    print("  " + "=" * 70)
    print("\n  Charts saved: agri_dashboard_easy.png, agri_dashboard_medium.png, agri_dashboard_hard.png")
    print(f"  Lessons saved: knowledge_base.json")


if __name__ == "__main__":
    main()