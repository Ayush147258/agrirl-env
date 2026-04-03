---
title: Agricore Environment
emoji: 🌾
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
app_port: 8000
---


# 🌾 AgriCore: High-Fidelity Agricultural Resource Allocation Environment

> **AgriCore is a High-Fidelity Resource Allocation Environment that treats agricultural management as a complex system optimization problem. It challenges agents to balance immediate yields against long-term soil sustainability, energy costs, and volatile market pricing. By incorporating predictive forecasting and diminishing returns, AgriCore provides a rigorous benchmark for evaluating an agent's ability to reason under multi-dimensional constraints.**

---

## 🏆 Meta-PyTorch OpenEnv Hackathon Submission

AgriCore is built on the **OpenEnv** framework and submitted to the Meta-PyTorch Hackathon as a novel RL benchmark environment. It goes beyond toy environments by simulating real-world agricultural complexity — a domain where every decision has cascading, delayed consequences.

---

## 🌍 Why AgriCore?

Modern agriculture is one of the world's most complex optimization problems. A farmer must simultaneously:

- Manage **limited water and fertilizer** across multiple crops
- Adapt to **unpredictable weather** (heatwaves, frost, rain)
- **Time harvests** against volatile market prices
- Prevent **soil degradation** from over-fertilization
- Combat **pest infestations** triggered by poor moisture management
- Make **deadline-aware** decisions under resource scarcity

This makes agriculture a perfect testbed for next-generation AI agents that must reason under **multi-dimensional, partially observable constraints** — far beyond what simple grid-world or bandit environments can offer.

---

## ⚙️ Environment Design

### Multi-Crop System
AgriCore manages **4 independent crops simultaneously**, each with its own state:

| Attribute | Description |
|---|---|
| `moisture` | Water level (0–100). Growth only occurs in range 30–80 |
| `growth` | Accumulated biomass. Drives stage progression and harvest value |
| `stage` | `seed → vegetative → flowering → mature` |
| `pest_level` | Increases when moisture > 80. Causes growth loss if > 2 |
| `fertilized_times` | Tracks fertilizer applications. Enables diminishing returns |
| `wait_days` | Days since last action. Penalizes neglected crops |

### Resource Management
Agents operate under **three constrained resources**:

| Resource | Starting Amount | Cost Per Action |
|---|---|---|
| 💧 Water | 120 units | 10 per irrigation |
| 🌿 Fertilizer | 60 units | 5 per fertilization |
| ⚡ Energy | 200 units | 3–5 per action |

### Weather System (Task-Aware)
Weather is **task-difficulty dependent** and affects all crops:

| Weather | Effect |
|---|---|
| ☀️ Sunny | −5 moisture per crop per day |
| 🌧️ Rainy | +10 moisture per crop per day |
| ☁️ Cloudy | Neutral |
| 🌡️ Heatwave | −20 moisture + −3 reward (hard only) |
| 🌨️ Frost | −2 growth (hard only) |

A **forecast** of tomorrow's weather is provided in every observation — rewarding agents that plan ahead.

---

## 🎯 Reward Function

AgriCore uses a **multi-objective reward function** designed to prevent simple exploitation:

```
reward += growth_gained × soil_health        # 🌱 Growth reward
reward += crop.growth × 2 × market_price     # 🌾 Harvest reward
reward += 5 / fertilized_times               # 📉 Diminishing fertilizer returns
reward -= 3  (if heatwave)                   # 🌡️ Weather penalty
reward -= 2  (if frost)                      # 🌨️ Frost penalty  
reward -= 1  (if pest_level > 2)             # 🐜 Pest penalty
reward -= 2  (if wait_days > 3)              # ⏳ Neglect penalty
reward -= 4  (if moisture > 85 on hard)      # 💦 Overwatering penalty
reward -= 5  (if harvesting non-mature crop) # ❌ Premature harvest penalty
```

**Key design principle:** No single greedy strategy dominates. Over-fertilizing degrades soil health. Over-irrigating causes pests. Waiting too long causes neglect penalties. Agents must learn **balanced, long-horizon strategies**.

---

## 📊 Difficulty Levels

AgriCore supports three task levels, enabling curriculum learning:

| Level | Weather | Scoring Formula | Challenge |
|---|---|---|---|
| 🟢 Easy | Mostly sunny + occasional rain | `min(1.0, total_growth / 300)` | Basic resource management |
| 🟡 Medium | Random sunny/rainy/cloudy | `max(0.0, (total_growth - day) / 300)` | Time-efficiency matters |
| 🔴 Hard | Full weather + heatwave/frost | `max(0.0, min(1.0, efficiency / 10))` | Extreme adaptability required |

---

## 🧠 Grading & Evaluation

Agents are evaluated against a **greedy baseline** (always irrigates the driest crop):

```
intelligence = smart_reward / greedy_reward   (clamped 0.0 – 1.0)
final_score  = _compute_score()               (task-normalized)
```

This dual evaluation ensures agents are judged on both **absolute performance** and **relative intelligence** against a known baseline.

---

## 🚀 Quick Start

### Local Execution
```bash
# Install dependencies
uv sync

# Run inference
python inference.py
```

### Docker
```bash
docker build -t agricore -f server/Dockerfile .
docker run -d -p 8000:8000 agricore
```

### API Access
```
http://localhost:8000/docs
```
[link](https://huggingface.co/spaces/ayush712145/agrirl-env)
---

## 📈 Sample Results

```
🏆 FINAL RESULTS
Easy Score  : 0.55
Medium Score: 0.52
Hard Score  : 0.82
Average     : 0.63
```

---

## 🏗️ Architecture

```
agrirl_env/
├── models.py                    # AgrirlAction, AgrirlObservation, Crop, WeatherType
├── inference.py                 # Smart baseline policy + episode runner
├── grader.py                    # Greedy vs Smart evaluation
├── client.py                    # OpenEnv WebSocket client
└── server/
    ├── agrirl_env_environment.py  # AgriCoreEnv — core logic
    ├── app.py                     # FastAPI server
    └── __init__.py
```

---

## 🔬 Research Directions

AgriCore opens several interesting research questions:

- **Multi-agent coordination** — Can multiple agents manage separate crops more efficiently?
- **Curriculum learning** — Does training on easy→medium→hard improve generalization?
- **Model-based planning** — Can agents exploit the weather forecast for lookahead planning?
- **Resource allocation under scarcity** — What strategies emerge when water is severely limited?
- **Market timing** — Can agents learn to time harvests against dynamic pricing?

---

## 📋 Observation Space

```python
class AgrirlObservation(Observation):
    crops: List[Crop]          # 4 crops with full state
    water: float               # Remaining water (0–120)
    fertilizer: float          # Remaining fertilizer (0–60)
    energy: float              # Remaining energy (0–200)
    day: int                   # Current day (1–30)
    weather: WeatherType       # Current weather
    forecast: WeatherType      # Tomorrow's weather (partial observability aid)
    market_price: float        # Dynamic harvest price (0.5–1.5)
    soil_health: float         # Degrades with over-fertilization (0.7 or 1.0)
    reward: float              # Step reward
    done: bool                 # Episode complete
    score: Optional[float]     # Final normalized score (only on done)
```

## 🎮 Action Space

```python
class AgrirlAction(Action):
    crop_id: int               # Which crop to act on (0–3)
    action: Literal[
        "irrigate",            # +15 moisture, costs 10 water + 5 energy
        "fertilize",           # +growth (diminishing), costs 5 fertilizer + 3 energy
        "harvest",             # Convert growth to reward × market_price
        "wait"                 # Skip turn, small penalty
    ]
```

---

## 👥 Team

Built for the **Meta-PyTorch OpenEnv Hackathon** using the OpenEnv framework.

---

*AgriCore — where every drop of water is a decision.*