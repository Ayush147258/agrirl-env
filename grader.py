from models import AgrirlAction as Action


def greedy_policy(obs) -> Action:
    """Always irrigate the driest crop."""
    if not obs.crops:
        return Action(crop_id=0, action="wait")

    crop = min(obs.crops, key=lambda c: c.moisture)
    return Action(crop_id=crop.id, action="irrigate")


def smart_policy(obs) -> Action:
    """
    Smart policy:
    - Harvest mature crops when market price is good
    - Wait if rain is forecast (save water)
    - Fertilize crops that have grown enough but not over-fertilized
    - Irrigate the driest crop otherwise
    """
    if not obs.crops:
        return Action(crop_id=0, action="wait")

    # 🌾 Harvest mature crops when price is favorable
    mature = [c for c in obs.crops if c.stage == "mature"]
    if mature and obs.market_price > 1.0:
        return Action(crop_id=mature[0].id, action="harvest")

    # 🌧️ Wait if rain forecast — no need to irrigate
    if obs.forecast == "rainy":
        dry = min(obs.crops, key=lambda c: c.moisture)
        if dry.moisture > 40:
            return Action(crop_id=0, action="wait")

    # 💊 Fertilize if resources allow and crop is growing
    if obs.fertilizer >= 5 and obs.energy >= 3:
        growing = [
            c for c in obs.crops
            if c.stage in ("vegetative", "flowering")
            and c.fertilized_times < 3  # avoid diminishing returns
        ]
        if growing:
            return Action(crop_id=growing[0].id, action="fertilize")

    # 💧 Irrigate driest crop by default
    crop = min(obs.crops, key=lambda c: c.moisture)
    return Action(crop_id=crop.id, action="irrigate")


def run(env, policy, task: str = "easy") -> tuple:
    """
    Run one full episode with given policy.
    Returns (total_reward, final_score)
    """
    obs = env.reset(task=task)
    total_reward = 0.0

    while not obs.done:
        action = policy(obs)
        obs = env.step(action)
        total_reward += obs.reward

    # ✅ Safe score fallback if None
    final_score = obs.score if obs.score is not None else 0.0

    return total_reward, final_score


def grade_episode(env) -> float:
    """
    Grade a completed episode by comparing smart vs greedy policy.
    Returns a normalized score between 0.0 and 1.0.
    """
    task = getattr(env, "task", "easy")

    # ✅ Use reset instead of deepcopy — safer with complex env state
    greedy_reward, greedy_score = run(env, greedy_policy, task=task)
    smart_reward, smart_score = run(env, smart_policy, task=task)

    # ✅ Clamp intelligence ratio — handles negative rewards safely
    if greedy_reward <= 0:
        intelligence = 1.0 if smart_reward > greedy_reward else 0.0
    else:
        intelligence = min(1.0, max(0.0, smart_reward / greedy_reward))

    result = {
        "greedy": {"reward": greedy_reward, "score": greedy_score},
        "smart":  {"reward": smart_reward,  "score": smart_score},
        "intelligence": round(intelligence, 4),
        "final_score": round(smart_score, 4),
    }

    print("\n📊 --- Grader Report ---")
    print(f"Task Level      : {task}")
    print(f"Greedy Reward   : {greedy_reward:.2f} | Score: {greedy_score:.2f}")
    print(f"Smart Reward    : {smart_reward:.2f}  | Score: {smart_score:.2f}")
    print(f"Intelligence    : {intelligence:.2f}")
    print(f"Final Score     : {smart_score:.2f}")

    return smart_score


def evaluate(env) -> dict:
    """
    Full evaluation across all difficulty levels.
    Returns per-task results and overall average score.
    """
    results = {}
    scores = []

    for task in ["easy", "medium", "hard"]:
        greedy_reward, greedy_score = run(env, greedy_policy, task=task)
        smart_reward, smart_score = run(env, smart_policy, task=task)

        if greedy_reward <= 0:
            intelligence = 1.0 if smart_reward > greedy_reward else 0.0
        else:
            intelligence = min(1.0, max(0.0, smart_reward / greedy_reward))

        results[task] = {
            "greedy": {"reward": greedy_reward, "score": greedy_score},
            "smart":  {"reward": smart_reward,  "score": smart_score},
            "intelligence": round(intelligence, 4),
        }
        scores.append(smart_score)

    results["average_score"] = round(sum(scores) / len(scores), 4)
    return results