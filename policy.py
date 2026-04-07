# policy.py
"""
AgriCore Neural Policy — PyTorch
=================================
A trained neural network policy that maps observations to actions.
Used as the low-level executor in hybrid mode:

  Gemini Strategist  →  high-level directives (every 5 days)
  AgriPolicy (torch) →  high-frequency action selection (every step)

Architecture: 3-layer MLP
  Input  : flattened observation vector (obs_dim)
  Hidden : 128 → 64 with ReLU + Dropout
  Output : action logits over 5 actions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional

# ── Action mapping ─────────────────────────────────────────────────────────────
ACTION_LIST = ["wait", "irrigate", "fertilize", "harvest", "pesticide"]
ACTION_TO_IDX = {a: i for i, a in enumerate(ACTION_LIST)}
IDX_TO_ACTION = {i: a for i, a in enumerate(ACTION_LIST)}

# ── Observation dimensions ─────────────────────────────────────────────────────
# Per crop (4 crops × 6 features): moisture, growth, stage_enc, pest, fertilized, wait_days
CROP_FEATURES   = 6
NUM_CROPS       = 4
GLOBAL_FEATURES = 9   # water, fertilizer, pesticide, energy, day, weather_enc,
                       # forecast_enc, market_price, soil_health
OBS_DIM = NUM_CROPS * CROP_FEATURES + GLOBAL_FEATURES   # = 33
NUM_ACTIONS = 5

STAGE_MAP   = {"seed": 0, "vegetative": 1, "flowering": 2, "mature": 3}
WEATHER_MAP = {"sunny": 0, "cloudy": 1, "rainy": 2, "heatwave": 3, "frost": 4}


# ── Observation encoder ────────────────────────────────────────────────────────

def obs_to_tensor(obs, strategy=None) -> torch.Tensor:
    """
    Convert AgrirlObservation → flat float tensor of shape (OBS_DIM,).

    If a Strategy is provided, strategy thresholds are included as
    additional context (increases OBS_DIM by 4 — see OBS_DIM_WITH_STRATEGY).
    """
    features = []

    # Global features (normalised to ~[0,1])
    features.append(obs.water        / 120.0)
    features.append(obs.fertilizer   / 60.0)
    features.append(obs.pesticide    / 40.0)
    features.append(obs.energy       / 200.0)
    features.append(obs.day          / 30.0)
    features.append(WEATHER_MAP.get(obs.weather,  0) / 4.0)
    features.append(WEATHER_MAP.get(obs.forecast, 0) / 4.0)
    features.append(obs.market_price / 2.0)
    features.append(obs.soil_health  / 100.0)

    # Per-crop features (pad to NUM_CROPS if fewer crops present)
    for i in range(NUM_CROPS):
        if i < len(obs.crops):
            c = obs.crops[i]
            features.append(c.moisture        / 100.0)
            features.append(c.growth          / 100.0)
            features.append(STAGE_MAP.get(c.stage, 0) / 3.0)
            features.append(min(c.pest_level,  10) / 10.0)
            features.append(min(c.fertilized_times, 5) / 5.0)
            features.append(min(c.wait_days,   10) / 10.0)
        else:
            features.extend([0.0] * CROP_FEATURES)

    return torch.tensor(features, dtype=torch.float32)


# ── Policy Network ─────────────────────────────────────────────────────────────

class AgriPolicy(nn.Module):
    """
    3-layer MLP policy network.

    Input  → (batch, OBS_DIM)
    Output → (batch, NUM_ACTIONS)  action logits

    Usage:
        policy = AgriPolicy()
        obs_t  = obs_to_tensor(obs)
        logits = policy(obs_t.unsqueeze(0))
        action_idx = logits.argmax(dim=-1).item()
        action_str = IDX_TO_ACTION[action_idx]
    """

    def __init__(self, obs_dim: int = OBS_DIM, hidden1: int = 128, hidden2: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden1),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden2, NUM_ACTIONS),
        )
        # Initialise with small weights — stable at episode start
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def predict(self, obs, crop_id_hint: int = 0):
        """
        Given an observation, return (action_str, crop_id, confidence).
        crop_id is picked by the heuristic (driest crop for irrigate, etc.)
        """
        self.eval()
        with torch.no_grad():
            obs_t  = obs_to_tensor(obs).unsqueeze(0)   # (1, OBS_DIM)
            logits = self.forward(obs_t)                # (1, NUM_ACTIONS)
            probs  = F.softmax(logits, dim=-1)
            idx    = probs.argmax(dim=-1).item()
            conf = probs[0, int(idx)].item()

        action_str = IDX_TO_ACTION[int(idx)]

        # Resolve crop_id from action type
        crop_id = _resolve_crop_id(obs, action_str)

        return action_str, crop_id, round(conf, 3)


def _resolve_crop_id(obs, action: str) -> int:
    """Pick the most relevant crop for the chosen action."""
    if not obs.crops:
        return 0
    if action == "irrigate":
        return min(obs.crops, key=lambda c: c.moisture).id
    if action == "pesticide":
        infested = [c for c in obs.crops if c.pest_level > 0]
        return max(infested, key=lambda c: c.pest_level).id if infested else 0
    if action == "harvest":
        mature = [c for c in obs.crops if c.stage == "mature"]
        return mature[0].id if mature else 0
    if action == "fertilize":
        growing = [c for c in obs.crops if c.stage in ("vegetative", "flowering")]
        return growing[0].id if growing else 0
    return 0


# ── Model save / load ──────────────────────────────────────────────────────────

def save_policy(policy: AgriPolicy, path: str = "agri_policy.pt"):
    torch.save({"state_dict": policy.state_dict(), "obs_dim": OBS_DIM}, path)
    print(f"[Policy] Saved → {path}")


def load_policy(path: str = "agri_policy.pt") -> Optional[AgriPolicy]:
    try:
        ckpt   = torch.load(path, map_location="cpu", weights_only=True)
        policy = AgriPolicy(obs_dim=ckpt.get("obs_dim", OBS_DIM))
        policy.load_state_dict(ckpt["state_dict"])
        policy.eval()
        print(f"[Policy] Loaded ← {path}")
        return policy
    except FileNotFoundError:
        return None
    except Exception as e:
        print(f"[Policy] Load failed ({e}) — using untrained network")
        return AgriPolicy()