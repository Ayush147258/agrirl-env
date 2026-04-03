import random
from grader import grade_episode, evaluate

from server.agrirl_env_environment import AgriCoreEnv  # type: ignore
from models import AgrirlAction as Action


# ── Policy ────────────────────────────────────────────────────────────────────

def act(obs) -> Action:
    """
    Improved smart policy for AgriCoreEnv.
    Fixes: water management, harvest timing, fertilizer rationing, crop rotation.
    """
    if not obs.crops:
        return Action(crop_id=0, action="wait")

    # ──────────────────────────────────────────
    # 1. EMERGENCY — irrigate any crop near death
    # ──────────────────────────────────────────
    critical = [c for c in obs.crops if c.moisture < 20]
    if critical and obs.water >= 10 and obs.energy >= 5:
        crop = min(critical, key=lambda c: c.moisture)
        return Action(crop_id=crop.id, action="irrigate")

    # ──────────────────────────────────────────
    # 2. HARVEST mature crops when price is good
    # ──────────────────────────────────────────
    mature = [c for c in obs.crops if c.stage == "mature"]
    if mature and obs.market_price >= 1.0:
        return Action(crop_id=mature[0].id, action="harvest")

    # ──────────────────────────────────────────
    # 3. WATER CONSERVATION — skip if rain coming
    #    and no crop is critically dry
    # ──────────────────────────────────────────
    if obs.forecast == "rainy" and not critical:
        if all(c.moisture >= 35 for c in obs.crops):
            return Action(crop_id=0, action="wait")

    # ──────────────────────────────────────────
    # 4. IRRIGATE — rotate across all dry crops
    #    with water budget awareness
    # ──────────────────────────────────────────
    dry_crops = [c for c in obs.crops if c.moisture < 40]
    if dry_crops and obs.water >= 10 and obs.energy >= 5:
        water_ratio = obs.water / 120
        day_ratio = obs.day / 30

        # conserve water if scarce and past halfway
        if water_ratio < 0.3 and day_ratio > 0.5:
            critical_only = [c for c in dry_crops if c.moisture < 30]
            if critical_only:
                crop = min(critical_only, key=lambda c: c.moisture)
                return Action(crop_id=crop.id, action="irrigate")
        else:
            crop = min(dry_crops, key=lambda c: c.moisture)
            return Action(crop_id=crop.id, action="irrigate")

    # ──────────────────────────────────────────
    # 5. FERTILIZE — rationed carefully
    #    max 2x per crop, stop after day 20
    # ──────────────────────────────────────────
    if obs.fertilizer >= 5 and obs.energy >= 3 and obs.day <= 20:
        growing = [
            c for c in obs.crops
            if c.stage in ("vegetative", "flowering")
            and c.fertilized_times < 2
        ]
        if growing:
            return Action(crop_id=growing[0].id, action="fertilize")

        if obs.fertilizer >= 20:
            early = [
                c for c in obs.crops
                if c.stage == "seed"
                and c.growth < 20
                and c.fertilized_times < 2
            ]
            if early:
                return Action(crop_id=early[0].id, action="fertilize")

    # ──────────────────────────────────────────
    # 6. LATE GAME harvest regardless of price
    # ──────────────────────────────────────────
    if obs.day > 20 and mature:
        return Action(crop_id=mature[0].id, action="harvest")

    return Action(crop_id=0, action="wait")


def explain(obs) -> str:
    """Explains current action decision."""
    if not obs.crops:
        return "No crops → wait"

    critical = [c for c in obs.crops if c.moisture < 20]
    if critical and obs.water >= 10:
        crop = min(critical, key=lambda c: c.moisture)
        return f"EMERGENCY irrigate crop {crop.id} — moisture critical ({crop.moisture:.1f})"

    mature = [c for c in obs.crops if c.stage == "mature"]
    if mature and obs.market_price >= 1.0:
        return f"Harvest crop {mature[0].id} — mature, price {obs.market_price:.2f}"

    if obs.forecast == "rainy" and all(c.moisture >= 35 for c in obs.crops):
        return "Rain forecast, moisture okay → wait to save water"

    dry_crops = [c for c in obs.crops if c.moisture < 40]
    if dry_crops and obs.water >= 10:
        water_ratio = obs.water / 120
        day_ratio = obs.day / 30
        if water_ratio < 0.3 and day_ratio > 0.5:
            return f"Water scarce ({obs.water} left) — conserving, only critical irrigation"
        crop = min(dry_crops, key=lambda c: c.moisture)
        return f"Irrigate crop {crop.id} — moisture low ({crop.moisture:.1f})"

    if obs.fertilizer >= 5 and obs.day <= 20:
        growing = [c for c in obs.crops if c.stage in ("vegetative", "flowering") and c.fertilized_times < 2]
        if growing:
            return f"Fertilize crop {growing[0].id} — {growing[0].stage} stage"
        if obs.fertilizer >= 20:
            early = [c for c in obs.crops if c.stage == "seed" and c.growth < 20 and c.fertilized_times < 2]
            if early:
                return f"Fertilize crop {early[0].id} — early boost (growth {early[0].growth:.1f})"

    if obs.day > 20 and mature:
        return f"Late game harvest crop {mature[0].id}"

    return "All stable → wait"


# ── Episode Runner ─────────────────────────────────────────────────────────────

def run_episode(task: str = "easy", verbose: bool = True) -> float:
    """Run one full episode with smart policy."""
    env = AgriCoreEnv()
    obs = env.reset(task=task)

    total_reward = 0.0
    step = 0

    print(f"\n🚀 Running Episode | Task: {task.upper()}")
    print("=" * 60)

    while not obs.done:
        action = act(obs)
        reason = explain(obs)
        obs = env.step(action)
        total_reward += obs.reward
        step += 1

        if verbose:
            print(
                f"Day {obs.day:>2} | "
                f"Action: {action.action:<10} Crop: {action.crop_id} | "
                f"Reward: {obs.reward:>6.2f} | "
                f"Reason: {reason}"
            )

    final_score = obs.score if obs.score is not None else 0.0

    print(f"\n📊 --- Episode Report ---")
    print(f"Task Level     : {task}")
    print(f"Days Run       : {step}")
    print(f"Total Reward   : {total_reward:.2f}")
    print(f"Final Score    : {final_score:.2f}")
    print(f"Water Left     : {env.water}")
    print(f"Fertilizer Left: {env.fertilizer}")
    print(f"Soil Health    : {env.soil_health}")
    print(f"Market Price   : {env.market_price:.2f}")
    print(f"Crops Growth   : {[round(c.growth, 1) for c in env.crops]}")
    print(f"Crops Stage    : {[c.stage for c in env.crops]}")

    return final_score


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("\n🔥 Starting AgriCore Evaluation")
    print("=" * 60)

    scores = []
    for task in ["easy", "medium", "hard"]:
        score = run_episode(task=task, verbose=True)
        scores.append(score)

    avg = sum(scores) / len(scores)

    print("\n" + "=" * 60)
    print("🏆 FINAL RESULTS")
    print(f"Easy Score  : {scores[0]:.2f}")
    print(f"Medium Score: {scores[1]:.2f}")
    print(f"Hard Score  : {scores[2]:.2f}")
    print(f"Average     : {avg:.2f}")
    print("=" * 60)


if __name__ == "__main__":
    main()