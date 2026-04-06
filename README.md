---
title: Agricore Environment
emoji: 🌾
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
app_port: 8000
---

# 🌾 AgriCore: Intelligent Autonomous Farming Agent

> **Powered by Gemini 2.0 Flash · Digital Twin Weather Mapping for Punjab, Maharashtra, California & Midwest · Hierarchical Multi-Agent RL System**

## 🏆 Meta-PyTorch OpenEnv Hackathon Submission

AgriCore is a **High-Fidelity Agricultural Resource Allocation Environment** built on the OpenEnv framework — and extended with a production-grade Hierarchical Multi-Agent AI system.

It challenges agents to balance immediate crop yields against long-term soil sustainability, energy costs, and volatile market pricing. By incorporating real-world weather grounding, predictive forecasting, and diminishing returns, AgriCore provides a rigorous benchmark for evaluating an agent's ability to **reason under multi-dimensional, partially observable constraints**.

**What separates this from a standard RL environment:**

| Layer | Technology | Purpose |
|---|---|---|
| 🌍 Digital Twin | Open-Meteo API | Live weather for Punjab, Maharashtra, California, Midwest |
| 🧠 LLM Strategist | Gemini 2.0 Flash | Sets high-level goals every 5 days — no API call every step |
| ⚙️ RL Executor | Rule-based control system | Logic-heavy 8-priority decision tree driven by Strategist |
| 🔍 Post-Mortem Analyst | Gemini 2.0 Flash | Causal analysis + threshold patching after each episode |
| 📖 Agricultural Ledger | JSON persistence | Cross-task transfer: Easy lessons applied to Hard from Day 1 |
| 🛡️ Resilience Engine | Heuristic + Statistical | 100% uptime — works with zero API key |

---

## 🌍 Why Agriculture?

Modern farming is one of the world's most complex optimization problems. A farmer must simultaneously:

- Manage **limited water, fertilizer, and energy** across 4 independent crops
- Adapt to **unpredictable weather** — heatwaves, frost, rainy spells
- **Time harvests** against volatile, dynamic market prices
- Prevent **soil degradation** from over-fertilization
- Combat **pest infestations** triggered by poor moisture control
- Make **deadline-aware decisions** as resources deplete over 30 days

This makes agriculture a perfect testbed for AI agents that must reason under **real-world multi-dimensional constraints** — far beyond what grid-world or bandit environments can offer.

---

## 🏗️ System Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                     EPISODE LOOP (30 days)                   │
│                                                              │
│  🌍 Digital Twin          Real weather → physics scaling     │
│   Open-Meteo API    Punjab / Maharashtra / California        │
│         ↓                                                    │
│  🧠 Strategist Agent      High-level directives (every 5d)  │
│   Gemini 2.0 Flash  moisture · harvest · pest · water goals  │
│         ↓                                                    │
│  ⚙️  Executor (RL)        Low-level action + reasoning      │
│   8-priority tree   Emergency→Pest→Harvest→Irrigate→Wait     │
│         ↓                                                    │
│  🔍 Post-Mortem Analyst   Causal analysis + strategy patch   │
│   Gemini 2.0 Flash  root cause · key mistake · directive     │
│         ↓                                                    │
│  📖 Agricultural Ledger   Cross-task transfer learning       │
│   knowledge_base.json   Easy→Medium→Hard, zero warm-up       │
└──────────────────────────────────────────────────────────────┘
```

### Project Structure

```
agrirl_env/
├── app.py                       # Gradio dashboard (Hugging Face Spaces)
├── inference.py                 # Episode runner — all 5 layers wired
├── strategist.py                # Gemini LLM Strategist agent
├── post_mortem.py               # Post-episode reflection + patching
├── digital_twin.py              # Open-Meteo real-world weather grounding
├── knowledge_base.py            # Agricultural Ledger (JSON persistence)
├── visualizer.py                # Strategy timeline + matplotlib charts
├── grader.py                    # Greedy vs Smart policy evaluation
├── models.py                    # Pydantic data models
├── mock_responses.json          # Cached AI responses for demo mode
├── requirements.txt
└── server/
    ├── agrirl_env_environment.py  # AgriCoreEnv — core simulation logic
    ├── app.py                     # FastAPI server
    └── Dockerfile
```

---

## ⚙️ Environment Design

### Multi-Crop System
AgriCore manages **4 independent crops simultaneously**, each with its own state:

| Attribute | Description |
|---|---|
| `moisture` | Water level (0–100). Growth only occurs in range 30–80 |
| `growth` | Accumulated biomass. Drives stage progression and harvest value |
| `stage` | `seed → vegetative → flowering → mature` |
| `pest_level` | Rises when moisture > 80. Causes growth loss if > 2 |
| `fertilized_times` | Tracks fertilizer applications — enables diminishing returns |
| `wait_days` | Days since last action. Penalizes crop neglect |

### Resource Budget

| Resource | Starting Amount | Cost Per Action |
|---|---|---|
| 💧 Water | 120 units | 10 per irrigation |
| 🌿 Fertilizer | 60 units | 5 per fertilization |
| ⚡ Energy | 200 units | 3–5 per action |
| 🧪 Pesticide | 40 units | 8 per spray + 0.5% soil health penalty |

### Weather System (Task-Aware)

| Weather | Effect |
|---|---|
| ☀️ Sunny | −5 moisture per crop per day |
| 🌧️ Rainy | +10 moisture per crop per day |
| ☁️ Cloudy | Neutral |
| 🌡️ Heatwave | −20 moisture + −3 reward *(hard only)* |
| 🌨️ Frost | −2 growth *(hard only)* |

A **forecast of tomorrow's weather** is provided in every observation — rewarding agents that plan ahead.

---

## 🎯 Reward Function

AgriCore uses a **multi-objective reward function** designed to prevent single-strategy exploitation:

```
reward += growth_gained × soil_health        # 🌱 Healthy growth bonus
reward += crop.growth × 2 × market_price     # 🌾 Harvest value
reward += 5 / fertilized_times               # 📉 Diminishing fertilizer returns
reward -= 3   (if heatwave)                  # 🌡️ Weather penalty
reward -= 2   (if frost)                     # 🌨️ Frost penalty
reward -= 1   (if pest_level > 2)            # 🐜 Pest damage
reward -= 2   (if wait_days > 3)             # ⏳ Neglect penalty
reward -= 4   (if moisture > 85 on hard)     # 💦 Overwatering penalty
reward -= 5   (if harvesting non-mature)     # ❌ Premature harvest penalty
```

**Key design principle:** No single greedy strategy dominates. Over-fertilizing degrades soil health. Over-irrigating triggers pests. Waiting too long causes neglect penalties. Agents must learn **balanced, long-horizon strategies**.

---

## 📊 Difficulty Levels

| Level | Weather | Scoring Formula | Challenge |
|---|---|---|---|
| 🟢 Easy | Mostly sunny + occasional rain | `min(1.0, total_growth / 300)` | Basic resource management |
| 🟡 Medium | Random sunny/rainy/cloudy | `max(0.0, (total_growth - day) / 300)` | Time-efficiency matters |
| 🔴 Hard | Full weather + heatwave/frost | `max(0.0, min(1.0, efficiency / 10))` | Extreme adaptability required |

---

## 📋 Observation & Action Space

```python
class AgrirlObservation(Observation):
    crops: List[Crop]       # 4 crops with full state
    water: float            # Remaining water (0–120)
    fertilizer: float       # Remaining fertilizer (0–60)
    pesticide: float        # Remaining pesticide
    energy: float           # Remaining energy (0–200)
    day: int                # Current day (1–30)
    weather: WeatherType    # Current weather
    forecast: WeatherType   # Tomorrow's forecast (partial observability aid)
    market_price: float     # Dynamic harvest price (0.5–1.5)
    soil_health: float      # Degrades with over-fertilization
    reward: float           # Step reward
    done: bool              # Episode complete
    score: Optional[float]  # Final normalized score (only when done)

class AgrirlAction(Action):
    crop_id: int            # Which crop (0–3)
    action: Literal[
        "irrigate",         # +15 moisture | cost: 10 water + 5 energy
        "fertilize",        # +growth (diminishing) | cost: 5 fertilizer + 3 energy
        "pesticide",        # Reduce pests | cost: 8 chemical + 2 energy + 0.5% soil
        "harvest",          # Convert growth → reward × market_price
        "wait",             # Skip turn, small neglect penalty
    ]
```

---

## 🤖 Multi-Agent AI System

### Strategist Agent (Gemini 2.0 Flash)
- Called **once every 5 days** — not every step — conserving 83% of API quota
- Outputs structured JSON directives: `moisture_threshold`, `harvest_price_floor`, `pest_threshold`, `water_reserve_pct`, `fertilize_max_day`
- **Resilience:** falls back to deterministic heuristic expert system if API unavailable

### Executor (RL Control System)
- **Logic-heavy** — every threshold comes from `strategy.*`, nothing hardcoded
- 8-priority decision tree: `Emergency → Pesticide → Harvest → Conservation → Irrigate → Fertilize → Late Harvest → Wait`
- Every action produces a one-line **human-readable justification** for full explainability

### Post-Mortem Analyst (Gemini 2.0 Flash)
- Runs **once per episode end** (3 LLM calls total across all difficulty levels)
- Identifies: failure day, root cause, key mistake
- Patches strategy via **70/30 weighted blend** — new recommendation weighted at 30% to prevent overcorrection
- **Resilience:** falls back to pure-Python statistical analysis (reward curve, water usage, soil health) if API fails

---

## 🌍 Digital Twin Weather Grounding

Real-world meteorological data via **Open-Meteo API** (free, no key needed, 10,000 calls/day):

| Region | Location | Farming Context |
|---|---|---|
| `punjab` | Ludhiana, India | Wheat/rice belt — **default** |
| `maharashtra` | Nashik, India | Seasonal crop planning |
| `california` | Central Valley, USA | Drought and heat simulation |
| `midwest` | Iowa, USA | Corn/soybean belt |

**Physics scaling applied each step:**
- `evaporation_multiplier` = scales with real temperature (1.0 at 25°C → 2.0 at 45°C)
- `rain_bonus` = real precipitation directly added to crop moisture
- `sim_weather` = mapped from real conditions (38°C+ → heatwave, 4°C- → frost)

---

## 📊 Observability

**Strategy Evolution Timeline** — printed after each episode:
```
Day  1  [balanced   ]  moisture=35.0 | pest=2.0 | episode start
Day  6  [survival   ]  moisture=50.0 | pest=2.0 | heatwave day 6
Day 11  [profit     ]  moisture=32.0 | pest=2.5 | day 11 review
Day 16  [balanced   ]  moisture=37.5 | pest=2.1 | post-mortem adjusted
```
*▲ = threshold increased vs previous · ▼ = decreased*

**Live Dashboard** — 4-panel matplotlib chart saved as `agri_dashboard_{task}.png`:
- Step reward + cumulative reward curve
- Avg crop moisture vs strategy threshold (deficit zone highlighted in red)
- Water remaining over time
- Soil health with critical zone (<40) highlighted

---

## 📖 Cross-Task Transfer Learning

The **Agricultural Ledger** (`knowledge_base.json`) persists lessons across difficulty levels:

```
Easy episode ends   → ReflectionReport saved to ledger
Medium episode starts → loads Easy lessons, pre-tunes thresholds on Day 1
Medium episode ends  → lessons saved
Hard episode starts  → loads Medium lessons, zero warm-up needed
```

> *"Our agent features Cross-Task Transfer Learning. It remembered it over-irrigated during the Easy simulation and corrected that behaviour in the Hard simulation immediately — zero warm-up required."*

---

## 🛡️ Resilience Architecture

| Layer | Primary | Fallback | Guarantee |
|---|---|---|---|
| Strategist | Gemini 2.0 Flash | Deterministic heuristic | Always produces a strategy |
| Post-Mortem | Gemini 2.0 Flash | Statistical reward analysis | Always produces a report |
| Weather | Open-Meteo API | Neutral defaults (25°C) | Never crashes on network fail |
| Full Demo | Live AI | `USE_MOCK_AI=true` → mock_responses.json | Works with zero API key |

> *"Engineered with API Resilience Logic: fallback deterministic heuristic + cached mock-reasoning ensures 100% demo uptime regardless of API quota status."*

---

## 📈 Performance Results

```
🏆 FINAL RESULTS
Easy Score  : 0.55
Medium Score: 0.52
Hard Score  : 0.82
Average     : 0.63
```

| Task | Greedy Baseline | AgriRL Agent | Improvement |
|---|---|---|---|
| Easy | ~0.34 | ~0.55 | **+62%** |
| Medium | ~0.31 | ~0.52 | **+68%** |
| Hard | ~0.42 | ~0.82 | **+95%** |

---

## 🧠 Grading & Evaluation

Agents are evaluated against a greedy baseline (always irrigates the driest crop):

```
intelligence = smart_reward / greedy_reward   (clamped 0.0 – 1.0)
final_score  = _compute_score()               (task-normalized)
```

This dual evaluation ensures agents are judged on both **absolute performance** and **relative intelligence** against a known baseline.

---

## ⚡ Quick Start

### Local
```bash
git clone https://github.com/YOUR_USERNAME/agrirl.git
cd agrirl
pip install -r requirements.txt

# Get free API key → aistudio.google.com → API Keys
export GEMINI_API_KEY="your_key_here"

# Run full evaluation
python inference.py

# Launch Gradio dashboard
python app.py

# Demo mode — no API key needed
USE_MOCK_AI=true python inference.py
```

### Docker (Environment Server)
```bash
docker build -t agricore -f server/Dockerfile .
docker run -d -p 8000:8000 agricore
# API docs: http://localhost:8000/docs
```

---

## 🚀 Hugging Face Deployment

1. Create Space at **huggingface.co → Spaces → New Space** (Gradio SDK)
2. Link your GitHub repository
3. **Settings → Secrets → New Secret:** `GEMINI_API_KEY` = your key
4. Every `git push` auto-deploys ✅

**Live Demo:** [huggingface.co/spaces/ayush712145/agrirl-env](https://huggingface.co/spaces/ayush712145/agrirl-env)

---

## 🔬 Research Directions

- **Multi-agent coordination** — Can multiple agents manage separate crops more efficiently?
- **Curriculum learning** — Does training Easy→Medium→Hard improve generalization?
- **Model-based planning** — Can agents exploit the weather forecast for lookahead planning?
- **Resource scarcity** — What strategies emerge when water is cut to 50% of normal?
- **Market timing** — Can agents learn to hold crops for peak pricing windows?

---

## 📝 Technical Descriptions for Judges

| Feature | Description |
|---|---|
| Multi-Agent | *"Implemented a Hierarchical Multi-Agent System using a Gemini 2.0 LLM-Strategist to oversee RL-Executor sub-routines with structured JSON directive passing."* |
| Digital Twin | *"Integrated Real-Time Digital Twin grounding via Open-Meteo Meteorological API for dynamic environment physics scaling across Punjab, Maharashtra, California, and Midwest regions."* |
| Self-Refining | *"Engineered an Automated Reflection Loop performing post-hoc causal analysis to optimize agent thresholds via weighted blending across episodes."* |
| Transfer Learning | *"Implemented Cross-Task Transfer Learning via a persistent Agricultural Ledger enabling zero-warmup threshold adaptation from Easy to Hard difficulty."* |
| Observability | *"Built a Real-Time Observability Dashboard using Gradio Blocks to visualize multi-agent reasoning, Digital Twin metrics, and strategy evolution timelines."* |
| Resilience | *"Designed full API resilience with deterministic heuristic fallback and cached mock-reasoning engine ensuring 100% demo uptime regardless of quota status."* |

---

## 👥 Team

Built for the **Meta-PyTorch OpenEnv Hackathon** using the OpenEnv framework.

**Deployment:** [huggingface.co/spaces/ayush712145/agrirl-env](https://huggingface.co/spaces/ayush712145/agrirl-env)

---

*AgriCore — where every drop of water is a decision.*