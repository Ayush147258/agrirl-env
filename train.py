# train.py
"""
AgriCore Policy Training — PyTorch
====================================
Two-phase training:

  Phase 1 — Imitation Learning (fast, stable)
    The network learns to mimic the smart heuristic policy.
    This gives a strong warm start — far better than random init.

  Phase 2 — REINFORCE (policy gradient)
    The network fine-tunes directly on environment rewards.
    Improves beyond the heuristic teacher.

Usage:
    python train.py                    # full training (both phases)
    python train.py --phase imitation  # imitation only (faster)
    python train.py --phase rl         # RL fine-tune only
    python train.py --episodes 200     # custom episode count
"""

import argparse
import json
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

from policy import (
    AgriPolicy, obs_to_tensor, save_policy, load_policy,
    ACTION_TO_IDX, IDX_TO_ACTION, NUM_ACTIONS, _resolve_crop_id
)

try:
    from server.agrirl_env_environment import AgriCoreEnv
except ModuleNotFoundError:
    from agrirl_env.server.agrirl_env_environment import AgriCoreEnv

try:
    from models import AgrirlAction as Action
except ModuleNotFoundError:
    from agrirl_env.models import AgrirlAction as Action

# ── Heuristic teacher (same logic as inference.py) ────────────────────────────

def heuristic_act(obs) -> Action:
    if not obs.crops:
        return Action(crop_id=0, action="wait")
    critical = [c for c in obs.crops if c.moisture < 20]
    if critical and obs.water >= 10 and obs.energy >= 5:
        return Action(crop_id=min(critical, key=lambda c: c.moisture).id, action="irrigate")
    if obs.pesticide >= 5 and obs.energy >= 2:
        infested = [c for c in obs.crops if c.pest_level > 2]
        if infested:
            return Action(crop_id=max(infested, key=lambda c: c.pest_level).id, action="pesticide")
    mature = [c for c in obs.crops if c.stage == "mature"]
    if mature and obs.market_price >= 1.0:
        return Action(crop_id=mature[0].id, action="harvest")
    if obs.forecast == "rainy" and not critical:
        if all(c.moisture >= 35 for c in obs.crops):
            return Action(crop_id=0, action="wait")
    dry = [c for c in obs.crops if c.moisture < 40]
    if dry and obs.water >= 10 and obs.energy >= 5:
        return Action(crop_id=min(dry, key=lambda c: c.moisture).id, action="irrigate")
    if obs.fertilizer >= 5 and obs.energy >= 3 and obs.day <= 20:
        growing = [c for c in obs.crops if c.stage in ("vegetative", "flowering") and c.fertilized_times < 2]
        if growing:
            return Action(crop_id=growing[0].id, action="fertilize")
    if obs.day > 20 and mature:
        return Action(crop_id=mature[0].id, action="harvest")
    return Action(crop_id=0, action="wait")


# ── Phase 1: Imitation Learning ───────────────────────────────────────────────

def train_imitation(
    policy: AgriPolicy,
    episodes: int = 300,
    tasks: list | None = None,
    lr: float    = 1e-3,
    save_path: str = "agri_policy.pt",
) -> AgriPolicy:
    """
    Supervised imitation learning:
    Run heuristic teacher, collect (obs, action) pairs, train with cross-entropy.
    """
    tasks     = tasks or ["easy", "medium", "hard"]
    optimiser = Adam(policy.parameters(), lr=lr)
    scheduler = StepLR(optimiser, step_size=50, gamma=0.5)
    policy.train()

    print("\n[Train] Phase 1 — Imitation Learning")
    print(f"        Episodes: {episodes} | Tasks: {tasks}")
    print("        " + "─" * 50)

    best_loss = float("inf")

    for ep in range(1, episodes + 1):
        task = random.choice(tasks)
        env  = AgriCoreEnv()
        obs  = env.reset(task=task)

        obs_list, act_list = [], []

        while not obs.done:
            teacher_action = heuristic_act(obs)
            obs_t  = obs_to_tensor(obs)
            act_idx = ACTION_TO_IDX[teacher_action.action]

            obs_list.append(obs_t)
            act_list.append(act_idx)

            obs = env.step(teacher_action)

        if not obs_list:
            continue

        # Batch update
        obs_batch = torch.stack(obs_list)                          # (T, OBS_DIM)
        act_batch = torch.tensor(act_list, dtype=torch.long)       # (T,)

        optimiser.zero_grad()
        logits = policy(obs_batch)                                  # (T, NUM_ACTIONS)
        loss   = F.cross_entropy(logits, act_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimiser.step()

        if ep % 50 == 0:
            scheduler.step()
            print(f"        Ep {ep:>4} | Task: {task:<6} | Loss: {loss.item():.4f}")
            if loss.item() < best_loss:
                best_loss = loss.item()
                save_policy(policy, save_path)

    print(f"        Best loss: {best_loss:.4f} | Saved → {save_path}\n")
    return policy


# ── Phase 2: REINFORCE ─────────────────────────────────────────────────────────

def train_reinforce(
    policy: AgriPolicy,
    episodes: int  = 200,
    tasks: "list | None" = None,
    lr: float      = 3e-4,
    gamma: float   = 0.99,
    save_path: str = "agri_policy.pt",
) -> AgriPolicy:
    """
    REINFORCE (Monte Carlo policy gradient):
    Collect full episode trajectories, compute returns, update policy.
    """
    tasks     = tasks or ["easy", "medium", "hard"]
    optimiser = Adam(policy.parameters(), lr=lr)
    policy.train()

    print("[Train] Phase 2 — REINFORCE (Policy Gradient)")
    print(f"        Episodes: {episodes} | Tasks: {tasks} | γ={gamma}")
    print("        " + "─" * 50)

    best_avg_reward = -float("inf")
    reward_history  = []

    for ep in range(1, episodes + 1):
        task = random.choice(tasks)
        env  = AgriCoreEnv()
        obs  = env.reset(task=task)

        log_probs    = []
        rewards_ep   = []

        while not obs.done:
            obs_t  = obs_to_tensor(obs).unsqueeze(0)       # (1, OBS_DIM)
            logits = policy(obs_t)                          # (1, NUM_ACTIONS)
            probs  = F.softmax(logits, dim=-1)
            dist   = torch.distributions.Categorical(probs)
            idx    = dist.sample()

            log_probs.append(dist.log_prob(idx))

            action_str = IDX_TO_ACTION[int(idx.item())]
            crop_id    = _resolve_crop_id(obs, action_str)
            action     = Action(crop_id=crop_id, action=action_str)    #type: ignore

            obs = env.step(action)
            rewards_ep.append(obs.reward)

        if not log_probs:
            continue

        # Compute discounted returns
        returns = []
        G = 0.0
        for r in reversed(rewards_ep):
            G = r + gamma * G
            returns.insert(0, G)

        returns_t = torch.tensor(returns, dtype=torch.float32)

        # Normalise returns for stability
        if returns_t.std() > 1e-8:
            returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

        # Policy gradient loss
        log_prob_t = torch.stack(log_probs)
        loss       = -(log_prob_t * returns_t).mean()

        optimiser.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimiser.step()

        total_r = sum(rewards_ep)
        reward_history.append(total_r)

        if ep % 25 == 0:
            avg = sum(reward_history[-25:]) / min(len(reward_history), 25)
            print(f"        Ep {ep:>4} | Task: {task:<6} | "
                  f"Reward: {total_r:>7.2f} | Avg(25): {avg:>7.2f}")
            if avg > best_avg_reward:
                best_avg_reward = avg
                save_policy(policy, save_path)

    print(f"        Best avg reward: {best_avg_reward:.2f} | Saved → {save_path}\n")
    return policy


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_policy(policy: AgriPolicy, task: str = "easy", episodes: int = 5) -> dict:
    """Run N episodes with trained policy and report average score."""
    policy.eval()
    scores  = []
    rewards = []

    for _ in range(episodes):
        env   = AgriCoreEnv()
        obs   = env.reset(task=task)
        total = 0.0

        while not obs.done:
            action_str, crop_id, _ = policy.predict(obs)
            obs = env.step(Action(crop_id=crop_id, action=action_str))
            total += obs.reward

        score = obs.score if obs.score is not None else 0.0
        scores.append(score)
        rewards.append(total)

    result = {
        "task":          task,
        "episodes":      episodes,
        "avg_score":     round(sum(scores)  / len(scores),  3),
        "avg_reward":    round(sum(rewards) / len(rewards), 3),
        "best_score":    round(max(scores),  3),
    }
    print(json.dumps({"type": "EVAL", **result}))
    return result


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="AgriCore Policy Training")
    parser.add_argument("--phase",    choices=["imitation", "rl", "both"], default="both")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--save",     default="agri_policy.pt")
    parser.add_argument("--eval",     action="store_true")
    args = parser.parse_args()

    torch.manual_seed(42)
    random.seed(42)

    # Load existing or create fresh
    policy = load_policy(args.save) or AgriPolicy()

    if args.phase in ("imitation", "both"):
        eps = args.episodes or 300
        policy = train_imitation(policy, episodes=eps, save_path=args.save)

    if args.phase in ("rl", "both"):
        eps = args.episodes or 200
        # Load best imitation checkpoint before RL fine-tune
        policy = load_policy(args.save) or policy
        policy = train_reinforce(policy, episodes=eps, save_path=args.save)

    if args.eval or True:
        print("\n[Train] Final Evaluation")
        print("        " + "─" * 50)
        for task in ["easy", "medium", "hard"]:
            evaluate_policy(policy, task=task, episodes=5)

    print(f"\n[Train] Done. Model saved → {args.save}")
    print(f"        Load in inference.py: USE_TORCH=true python inference.py")


if __name__ == "__main__":
    main()