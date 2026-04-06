# visualizer.py
"""
Visualizer — Strategy Timeline + Episode Dashboard Charts
----------------------------------------------------------
Provides two public utilities:

  StrategyTimeline
      Records how the Strategist's directives evolved across an episode.
      Call .record() at each strategy update, then .print_table() at the end
      for a formatted console summary. The .snapshots list is also iterable
      by app.py for live Gradio log output.

  save_episode_charts(episode_log, task, final_score, output_path)
      Renders a 2×3 matplotlib dashboard PNG covering:
        - Step reward (bar) + cumulative reward (line)
        - Average crop moisture with danger zone shading
        - Water remaining over time
        - Soil health over time
        - Action distribution (pie)
        - Weather breakdown (pie)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List

import matplotlib
matplotlib.use("Agg")           # non-interactive backend — safe for servers
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches


# ── Strategy Snapshot ──────────────────────────────────────────────────────────

@dataclass
class StrategySnapshot:
    """
    A point-in-time record of the Strategist's active directives.
    Stored by StrategyTimeline; fields are accessed directly by app.py.
    """
    day:                 int
    priority:            str
    moisture_threshold:  float
    harvest_price_floor: float
    fertilize_max_day:   int
    pest_threshold:      float
    water_reserve_pct:   float
    reasoning:           str
    trigger:             str        # human-readable reason this snapshot was taken


# ── Strategy Timeline ──────────────────────────────────────────────────────────

class StrategyTimeline:
    """
    Records strategy snapshots across an episode so you can see how
    the Strategist adapted over time.

    Usage:
        timeline = StrategyTimeline()
        timeline.record(day=1, strategy=strategy, trigger="episode start")
        ...
        timeline.record(day=obs.day, strategy=strategy, trigger="priority shift")
        timeline.print_table()   # prints formatted table to console
        timeline.snapshots       # list[StrategySnapshot] for programmatic access
    """

    def __init__(self):
        self.snapshots: List[StrategySnapshot] = []

    def record(self, day: int, strategy, trigger: str = "") -> None:
        """
        Capture the current strategy state as a snapshot.

        Args:
            day:      Simulation day number.
            strategy: A strategist.Strategy instance.
            trigger:  Short description of why this snapshot was taken.
        """
        snap = StrategySnapshot(
            day                 = day,
            priority            = strategy.priority,
            moisture_threshold  = strategy.moisture_threshold,
            harvest_price_floor = strategy.harvest_price_floor,
            fertilize_max_day   = strategy.fertilize_max_day,
            pest_threshold      = strategy.pest_threshold,
            water_reserve_pct   = strategy.water_reserve_pct,
            reasoning           = getattr(strategy, "reasoning", ""),
            trigger             = trigger,
        )
        self.snapshots.append(snap)

    def print_table(self) -> None:
        """Print a formatted strategy-evolution table to stdout."""
        if not self.snapshots:
            print("  [Timeline] No snapshots recorded.")
            return

        print("\n  ╔══════════════════════════════════════════════════════════════════════════╗")
        print("  ║                        📊  STRATEGY TIMELINE                           ║")
        print("  ╠═════╦═════════════╦══════════╦═══════════╦══════════╦══════════════════╣")
        print("  ║ Day ║ Priority    ║ Moisture ║ Pest Thr. ║ WaterRes ║ Trigger          ║")
        print("  ╠═════╬═════════════╬══════════╬═══════════╬══════════╬══════════════════╣")

        for s in self.snapshots:
            day      = str(s.day)
            priority = s.priority[:11]
            moisture = f"{s.moisture_threshold:.1f}"
            pest     = f"{s.pest_threshold:.1f}"
            water    = f"{s.water_reserve_pct*100:.0f}%"
            trigger  = s.trigger[:18]
            print(
                f"  ║ {day:<3} ║ {priority:<11} ║ {moisture:<8} ║ "
                f"{pest:<9} ║ {water:<8} ║ {trigger:<16} ║"
            )

        print("  ╚═════╩═════════════╩══════════╩═══════════╩══════════╩══════════════════╝")

    def summary_lines(self) -> List[str]:
        """
        Return snapshot data as a list of plain strings — useful for
        embedding in Gradio log boxes without console formatting.
        """
        lines = []
        for s in self.snapshots:
            lines.append(
                f"Day {s.day:>2} [{s.priority:<11}] "
                f"moisture={s.moisture_threshold:.1f} | "
                f"pest={s.pest_threshold:.1f} | "
                f"water_reserve={s.water_reserve_pct*100:.0f}% | "
                f"{s.trigger}"
            )
        return lines


# ── Episode Dashboard ──────────────────────────────────────────────────────────

# Dark dashboard palette — matches the Gradio app's dark card theme
_BG_FIGURE  = "#0f1117"
_BG_AXES    = "#1a1d2e"
_GRID       = dict(color="#2a2d3e", linewidth=0.5, linestyle="--")
_TEXT_COLOR = "white"
_MUTED      = "#9ca3af"
_SPINE      = "#2a2d3e"

_ACTION_COLORS = {
    "irrigate":  "#38bdf8",
    "fertilize": "#a78bfa",
    "harvest":   "#4ade80",
    "pesticide": "#f87171",
    "wait":      "#6b7280",
}

_WEATHER_COLORS = {
    "sunny":    "#f59e0b",
    "rainy":    "#38bdf8",
    "cloudy":   "#94a3b8",
    "heatwave": "#ef4444",
    "frost":    "#818cf8",
}


def _style_ax(ax: plt.Axes, title: str) -> None:
    """Apply consistent dark styling to a single axes."""
    ax.set_facecolor(_BG_AXES)
    ax.set_title(title, color=_TEXT_COLOR, fontsize=9, pad=5, fontweight="bold")
    ax.tick_params(colors=_MUTED, labelsize=7)
    for sp in ax.spines.values():
        sp.set_edgecolor(_SPINE)
    ax.grid(color="#2a2d3e", linewidth=0.5, linestyle="--")


def save_episode_charts(
    episode_log:  list,
    task:         str,
    final_score:  float,
    output_path:  str = "agri_dashboard.png",
) -> str:
    """
    Render and save a 2×3 matplotlib episode dashboard.

    Args:
        episode_log:  List of per-step dicts from the episode runner.
        task:         Task name ("easy" | "medium" | "hard").
        final_score:  Normalised final score (0.0 – 1.0).
        output_path:  Destination PNG path.

    Returns:
        The resolved output_path string.
    """
    if not episode_log:
        print(f"  [Visualizer] No episode data — skipping chart for {output_path}.")
        return output_path

    # ── Extract series ─────────────────────────────────────────────────────────
    days        = [e["day"]          for e in episode_log]
    rewards     = [e["reward"]       for e in episode_log]
    water       = [e["water"]        for e in episode_log]
    soil        = [e["soil_health"]  for e in episode_log]
    moisture    = [e["avg_moisture"] for e in episode_log]
    actions     = [e["action"]       for e in episode_log]
    weathers    = [e["weather"]      for e in episode_log]

    # Cumulative reward
    cum_rewards = []
    running = 0.0
    for r in rewards:
        running += r
        cum_rewards.append(round(running, 2))

    # Action + weather distributions
    action_counts  = {a: actions.count(a)  for a in set(actions)}
    weather_counts = {w: weathers.count(w) for w in set(weathers)}

    # ── Figure setup ──────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 8), facecolor=_BG_FIGURE)
    fig.suptitle(
        f"AgriRL Episode Dashboard  ·  Task: {task.upper()}  ·  Score: {final_score:.3f}",
        color=_TEXT_COLOR, fontsize=12, fontweight="bold", y=0.98,
    )

    gs   = gridspec.GridSpec(2, 3, hspace=0.50, wspace=0.35,
                             left=0.07, right=0.97, top=0.92, bottom=0.08)
    axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(3)]

    # ── Panel 0: Step reward + cumulative ─────────────────────────────────────
    ax = axes[0]
    bar_colors = [("#4ade80" if r >= 0 else "#f87171") for r in rewards]
    ax.bar(days, rewards, color=bar_colors, alpha=0.55, width=0.8, zorder=2)
    ax2 = ax.twinx()
    ax2.plot(days, cum_rewards, color="#f59e0b", linewidth=1.8, zorder=3)
    ax2.tick_params(colors="#f59e0b", labelsize=7)
    ax2.set_ylabel("Cumulative", color="#f59e0b", fontsize=7)
    for sp in ax2.spines.values():
        sp.set_edgecolor(_SPINE)
    _style_ax(ax, "Reward per Step")
    ax.set_xlabel("Day", color=_MUTED, fontsize=7)
    ax.axhline(0, color=_MUTED, linewidth=0.6, linestyle=":")

    # ── Panel 1: Average crop moisture ────────────────────────────────────────
    ax = axes[1]
    ax.plot(days, moisture, color="#38bdf8", linewidth=1.8, zorder=3)
    ax.axhline(35, color="#f87171", linewidth=1.0, linestyle=":", zorder=2)
    ax.axhline(80, color="#f59e0b", linewidth=1.0, linestyle=":", zorder=2)
    ax.fill_between(
        days, moisture, 35,
        where=[m < 35 for m in moisture],
        alpha=0.22, color="#f87171", zorder=1,
   )
    ax.fill_between(
        days, moisture, 80,
        where=[m > 80 for m in moisture],
        alpha=0.22, color="#f59e0b", zorder=1,
    )
    _style_ax(ax, "Avg Crop Moisture")
    ax.set_ylim(0, 105)
    ax.set_xlabel("Day", color=_MUTED, fontsize=7)
    # Danger zone legend
    ax.legend(
        handles=[
            mpatches.Patch(color="#f87171", alpha=0.5, label="< 35 (dry)"),
            mpatches.Patch(color="#f59e0b", alpha=0.5, label="> 80 (pest risk)"),
        ],
        fontsize=6, framealpha=0.3, facecolor=_BG_AXES,
        labelcolor=_MUTED, loc="upper right",
    )

    # ── Panel 2: Water remaining ──────────────────────────────────────────────
    ax = axes[2]
    ax.plot(days, water, color="#818cf8", linewidth=1.8, zorder=3)
    ax.fill_between(days, water, 0, alpha=0.15, color="#818cf8", zorder=1)
    ax.axhline(30, color="#f87171", linewidth=0.8, linestyle=":", zorder=2)
    _style_ax(ax, "Water Remaining")
    ax.set_ylim(0, 130)
    ax.set_xlabel("Day", color=_MUTED, fontsize=7)

    # ── Panel 3: Soil health ──────────────────────────────────────────────────
    ax = axes[3]
    ax.plot(days, soil, color="#a78bfa", linewidth=1.8, zorder=3)
    ax.fill_between(days, soil, 0, alpha=0.15, color="#a78bfa", zorder=1)
    ax.axhline(0.7, color="#f87171", linewidth=0.8, linestyle=":", zorder=2)
    _style_ax(ax, "Soil Health")
    ax.set_ylim(0, 1.1)
    ax.set_xlabel("Day", color=_MUTED, fontsize=7)

    # ── Panel 4: Action distribution (pie) ────────────────────────────────────
    ax = axes[4]
    ac_labels = list(action_counts.keys())
    ac_sizes  = list(action_counts.values())
    ac_colors = [_ACTION_COLORS.get(a, "#6b7280") for a in ac_labels]
    wedges, texts, autotexts = ax.pie(
        ac_sizes,
        labels=ac_labels,
        colors=ac_colors,
        autopct="%1.0f%%",
        startangle=90,
        wedgeprops=dict(linewidth=0.5, edgecolor=_BG_FIGURE),
        textprops=dict(color=_MUTED, fontsize=7),
    )
    for at in autotexts:
        at.set_fontsize(6.5)
        at.set_color(_TEXT_COLOR)
    ax.set_facecolor(_BG_AXES)
    ax.set_title("Action Distribution", color=_TEXT_COLOR, fontsize=9, pad=5, fontweight="bold")

    # ── Panel 5: Weather breakdown (pie) ──────────────────────────────────────
    ax = axes[5]
    wc_labels = list(weather_counts.keys())
    wc_sizes  = list(weather_counts.values())
    wc_colors = [_WEATHER_COLORS.get(w, "#6b7280") for w in wc_labels]
    wedges, texts, autotexts = ax.pie(
        wc_sizes,
        labels=wc_labels,
        colors=wc_colors,
        autopct="%1.0f%%",
        startangle=90,
        wedgeprops=dict(linewidth=0.5, edgecolor=_BG_FIGURE),
        textprops=dict(color=_MUTED, fontsize=7),
    )
    for at in autotexts:
        at.set_fontsize(6.5)
        at.set_color(_TEXT_COLOR)
    ax.set_facecolor(_BG_AXES)
    ax.set_title("Weather Breakdown", color=_TEXT_COLOR, fontsize=9, pad=5, fontweight="bold")

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True) \
        if os.path.dirname(output_path) else None

    fig.savefig(output_path, dpi=120, bbox_inches="tight", facecolor=_BG_FIGURE)
    plt.close(fig)
    print(f"  [Visualizer] Chart saved → {output_path}")
    return output_path