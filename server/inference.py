"""
AgriRL — OpenEnv-Compliant Hierarchical Multi-Agent Inference Engine
=====================================================================
OpenEnv validator requirements (DO NOT REMOVE):
  - API_BASE_URL, MODEL_NAME, HF_TOKEN env variables
  - OpenAI-compatible client for LLM calls
  - [START] / [STEP] / [END] / [FINAL] structured stdout logging with flush=True
  - if __name__ == "__main__": entry point

Five-layer differentiator architecture:
  Layer 1 — DigitalTwin         Real-world weather grounds simulation physics
  Layer 2 — StrategistAgent     LLM sets high-level directives every 5 days
  Layer 3 — Executor (act)      Hybrid PyTorch + Heuristic RL agent
  Layer 4 — PostMortemAnalyst   LLM/statistical reflection patches next Strategy
  Layer 5 — AgriculturalLedger  Cross-task transfer: Easy lessons → Medium/Hard

Agent modes (set via env var USE_TORCH):
  USE_TORCH=true   → PyTorch trained policy (AgriPolicy MLP)
  USE_TORCH=false  → Smart heuristic policy (default)

Resilience:
  - All LLM calls have graceful degradation + heuristic/statistical fallbacks
  - USE_MOCK_AI=true enables full demo without any API key
  - Strategy thresholds visualised as a timeline table after each episode
  - Matplotlib charts saved as agri_dashboard_{task}.png

Run:
    python inference.py
    USE_TORCH=true python inference.py
    USE_MOCK_AI=true python inference.py
"""

import os
import sys

from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI

# ── OpenEnv required env variables ────────────────────────────────────────────
API_BASE_URL     = os.getenv("API_BASE_URL",  "https://generativelanguage.googleapis.com/v1beta/openai/")
MODEL_NAME       = os.getenv("MODEL_NAME",    "gemini-2.0-flash")
HF_TOKEN         = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

# ── Agent mode ────────────────────────────────────────────────────────────────
USE_TORCH   = os.getenv("USE_TORCH",   "false").lower() == "true"
POLICY_PATH = os.getenv("POLICY_PATH", "agri_policy.pt")

# ── OpenAI-compatible client → Gemini ─────────────────────────────────────────
client = OpenAI(
    base_url = API_BASE_URL,
    api_key  = os.getenv("GEMINI_API_KEY", HF_TOKEN or "no-key"),
)

# ── Project imports ────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from server.agrirl_env_environment import AgriCoreEnv
except ModuleNotFoundError:
    from agrirl_env.server.agrirl_env_environment import AgriCoreEnv

from models import AgrirlAction as Action
from grader import _get_score
from strategist import StrategistAgent, Strategy
from digital_twin import DigitalTwin
from post_mortem import PostMortemAnalyst
from knowledge_base import AgriculturalLedger
from visualizer import StrategyTimeline, save_episode_charts

# ── PyTorch policy (optional) ─────────────────────────────────────────────────
_torch_policy = None
if USE_TORCH:
    try:
        from policy import load_policy, AgriPolicy
        _torch_policy = load_policy(POLICY_PATH)
        if _torch_policy is None:
            _torch_policy = AgriPolicy()
            print("[WARN] No trained weights found — using untrained AgriPolicy", flush=True)
        print(f"[INFO] PyTorch policy loaded from {POLICY_PATH}", flush=True)
    except ImportError:
        print("[WARN] torch not installed — falling back to heuristic", flush=True)
        USE_TORCH = False


# ── Structured stdout helpers (flush=True — required by OpenEnv) ──────────────

def log_start(task, mode, directive, region, real_temp):
    print(
        f"[START] task={task} model={MODEL_NAME} mode={mode} "
        f"region={region} temp={real_temp:.1f}C directive={directive!r}",
        flush=True,
    )

def log_step(step, day, action, crop_id, reward, reason, strategy_priority, water, mode):
    print(
        f"[STEP]  step={step} day={day} action={action} crop_id={crop_id} "
        f"reward={reward:.4f} strategy={strategy_priority} "
        f"water={water:.1f} mode={mode} reason={reason!r}",
        flush=True,
    )

def log_end(task, score, steps, total_reward, water_left, soil,
            mode, root_cause, directive):
    print(
        f"[END]   task={task} score={score:.4f} steps={steps} "
        f"total_reward={total_reward:.4f} water_left={water_left:.1f} "
        f"soil_health={soil:.1f} mode={mode} "
        f"root_cause={root_cause!r} next_directive={directive!r}",
        flush=True,
    )

def log_final(scores, mode):
    avg = sum(scores) / len(scores)
    print(
        f"[FINAL] easy={scores[0]:.4f} medium={scores[1]:.4f} "
        f"hard={scores[2]:.4f} average={avg:.4f} mode={mode}",
        flush=True,
    )

def log_info(msg: str):
    print(f"[INFO]  {msg}", flush=True)

def log_warn(msg: str):
    print(f"[WARN]  {msg}", flush=True)


# ── Heuristic policy ──────────────────────────────────────────────────────────

def _heuristic_act(obs, strategy=None):
    """Smart rule-based policy — 8-priority decision tree."""
    if not obs.crops:
        return Action(crop_id=0, action="wait")

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


def _explain(obs, strategy=None) -> str:
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
        return f"EMERGENCY irrigate crop {c.id} moisture={c.moisture:.1f}"
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


# ── Hybrid act — PyTorch or heuristic ────────────────────────────────────────

def act(obs, strategy=None) -> tuple:
    """
    Returns (Action, reason, mode).
    Tries PyTorch policy first, falls back to strategy-driven heuristic.
    """
    if USE_TORCH and _torch_policy is not None:
        try:
            action_str, crop_id, conf = _torch_policy.predict(obs)
            VALID = {"irrigate", "fertilize", "wait", "harvest", "pesticide"}
            action_str = action_str if action_str in VALID else "wait"
            valid_actions = {"irrigate", "fertilize", "wait", "harvest", "pesticide"}
            if action_str not in valid_actions:
                action_str = "wait"
            action = Action(crop_id=int(crop_id), action=action_str)   # type: ignore[arg-type]
            reason = f"[torch] {action_str} crop={crop_id} conf={conf:.2f}"
            return action, reason, "torch"
        except Exception:
            pass  # fall through to heuristic

    action = _heuristic_act(obs, strategy)
    reason = _explain(obs, strategy)
    return action, reason, "heuristic"


# ── LLM Episode Directive ─────────────────────────────────────────────────────

def get_llm_directive(obs) -> str:
    """One-shot LLM directive at episode start. Graceful fallback."""
    try:
        crops = [
            {"id": c.id, "stage": c.stage, "moisture": round(c.moisture, 1)}
            for c in obs.crops
        ]
        prompt = (
            f"Farm state: day={obs.day}, water={obs.water:.0f}, "
            f"soil_health={obs.soil_health:.1f}, weather={obs.weather}, "
            f"market_price={obs.market_price:.2f}, crops={crops}. "
            f"Give one short tactical directive for this episode (max 15 words)."
        )
        response = client.chat.completions.create(
            model      = MODEL_NAME,
            messages   = [{"role": "user", "content": prompt}],
            max_tokens = 40,
        )
        return (response.choices[0].message.content or "").strip()
    except Exception as e:
        return f"Heuristic mode ({type(e).__name__})"


# ── Episode Runner ─────────────────────────────────────────────────────────────

def run_episode(
    task:             str  = "easy",
    region:           str  = "punjab",
    use_real_weather: bool = True,
    use_strategist:   bool = True,
    use_post_mortem:  bool = True,
    carry_strategy         = None,
    ledger                 = None,
) -> tuple:
    """
    Run one full episode with all five layers.
    Stdout: [START] / [STEP] / [END] with flush=True (OpenEnv standard).

    Returns:
        (final_score: float, final_strategy: Strategy, report)
    """
    mode = "torch" if (USE_TORCH and _torch_policy) else "heuristic"

    # ── Layer init ─────────────────────────────────────────────────────────────
    env        = AgriCoreEnv()
    strategist = StrategistAgent(review_interval=5) if use_strategist  else None
    twin       = DigitalTwin(region=region) if (
                     use_real_weather and
                     region in ("punjab", "california", "midwest", "maharashtra")
                 ) else None
    analyst    = PostMortemAnalyst()                if use_post_mortem  else None
    timeline   = StrategyTimeline()

    # Layer 5: pre-load cross-task lessons
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

    # ── [START] ────────────────────────────────────────────────────────────────
    directive = get_llm_directive(obs)
    log_start(task, mode, directive, region, real_temp)
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

        # Layer 3: Hybrid Executor (torch or heuristic)
        action, reason, used_mode = act(obs, strategy)
        obs          = env.step(action)
        total_reward += obs.reward
        step         += 1
        reward_history.append(round(obs.reward, 3))

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
            "mode":         used_mode,
        })

        # ── [STEP] ─────────────────────────────────────────────────────────────
        log_step(
            step              = step,
            day               = obs.day,
            action            = action.action,
            crop_id           = action.crop_id,
            reward            = obs.reward,
            reason            = reason,
            strategy_priority = strategy.priority,
            water             = obs.water,
            mode              = used_mode,
        )

    # ── Episode end ────────────────────────────────────────────────────────────
    final_score = _get_score(obs)

    # Layer 4: Post-Mortem
    report = None
    if analyst:
        report   = analyst.analyse(episode_log, total_reward, final_score, task)
        print(report.display(), flush=True)
        strategy = analyst.patch_strategy(strategy, report)
        log_info(f"Strategy patched: {strategy.summary()}")

        # Layer 5: Ledger — persist lessons for next task
        if ledger and report:
            ledger.save(task=task, report=report, region=region, temp_c=real_temp)

    # Strategy timeline + charts
    timeline.print_table()
    save_episode_charts(
        episode_log = episode_log,
        task        = task,
        final_score = final_score,
        output_path = f"agri_dashboard_{task}.png",
    )

    # ── [END] ──────────────────────────────────────────────────────────────────
    log_end(
        task         = task,
        score        = final_score,
        steps        = step,
        total_reward = total_reward,
        water_left   = env.water,
        soil         = env.soil_health,
        mode         = mode,
        root_cause   = report.root_cause        if report else "N/A",
        directive    = report.next_run_directive if report else "N/A",
    )

    return final_score, strategy, report


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ledger   = AgriculturalLedger()
    scores   = []
    strategy = None

    for task in ["easy", "medium", "hard"]:
        score, strategy, _ = run_episode(
            task             = task,
            region           = "punjab",
            use_real_weather = True,
            use_strategist   = True,
            use_post_mortem  = True,
            carry_strategy   = strategy or Strategy(),
            ledger           = ledger,
        )
        scores.append(score)

    ledger.print_summary()

    # ── [FINAL] ────────────────────────────────────────────────────────────────
    log_final(scores, mode="torch" if USE_TORCH else "heuristic")


if __name__ == "__main__":
    main()