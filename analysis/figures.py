"""Generate IEEE-Access-ready figures from training TB events + frozen-eval CSVs.

Outputs (300 DPI PNGs):
    analysis/figures/fig_learning_curves.png   per-algo mean curve + per-seed thin lines
    analysis/figures/fig_success_bar.png       success rate bar with 95% bootstrap CI
    analysis/figures/fig_steps_box.png         steps-to-goal box plot per algo

Usage:
    conda activate tianshou       # tianshou env has numpy/pandas/matplotlib + tensorboard
    python code/analysis/figures.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


PROJECT_ROOT = Path(__file__).resolve().parents[2]

ALGO_COLORS = {"sac": "#1f77b4", "ppo": "#d62728", "rainbow": "#2ca02c"}
ALGO_LABELS = {"sac": "SAC", "ppo": "PPO", "rainbow": "Rainbow DQN"}


def load_reward_curve(run_dir: Path) -> tuple[np.ndarray, np.ndarray] | None:
    # ml-agents nests events under <run_dir>/<behavior_name>/
    for sub in run_dir.iterdir():
        if sub.is_dir() and any(p.name.startswith("events.out.tfevents") for p in sub.iterdir() if p.is_file()):
            ea = EventAccumulator(str(sub), size_guidance={"scalars": 0})
            ea.Reload()
            if "Environment/Cumulative Reward" not in ea.Tags()["scalars"]:
                return None
            events = ea.Scalars("Environment/Cumulative Reward")
            return np.array([e.step for e in events]), np.array([e.value for e in events])
    return None


def interpolate_to_common_grid(curves: list[tuple[np.ndarray, np.ndarray]], n_points: int = 200) -> tuple[np.ndarray, np.ndarray]:
    if not curves:
        return np.array([]), np.array([])
    max_step = min(c[0][-1] for c in curves)
    grid = np.linspace(c[0][0] if False else min(c[0][0] for c in curves), max_step, n_points)
    arr = np.empty((len(curves), n_points))
    for i, (s, v) in enumerate(curves):
        arr[i] = np.interp(grid, s, v)
    return grid, arr


def make_learning_curves(results_dir: Path, out_dir: Path, seeds: list[int]) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for algo in ["sac", "ppo", "rainbow"]:
        curves = []
        for s in seeds:
            run_dir = results_dir / f"{algo}_seed_{s}"
            if not run_dir.is_dir():
                continue
            curve = load_reward_curve(run_dir)
            if curve is not None:
                curves.append(curve)
        if not curves:
            continue
        # Per-seed thin lines
        color = ALGO_COLORS[algo]
        for s_idx, (steps, vals) in enumerate(curves):
            ax.plot(steps, vals, color=color, alpha=0.25, linewidth=0.7,
                    label=None)
        # Per-algo mean + bootstrap band on common grid
        grid, arr = interpolate_to_common_grid(curves)
        if arr.size == 0:
            continue
        mean = arr.mean(axis=0)
        # 95% bootstrap CI per step
        rng = np.random.default_rng(0)
        n_boot = 1000
        boots = np.empty((n_boot, len(grid)))
        for b in range(n_boot):
            sample = rng.choice(arr.shape[0], size=arr.shape[0], replace=True)
            boots[b] = arr[sample].mean(axis=0)
        lo = np.quantile(boots, 0.025, axis=0)
        hi = np.quantile(boots, 0.975, axis=0)
        ax.plot(grid, mean, color=color, linewidth=2.2, label=ALGO_LABELS[algo])
        ax.fill_between(grid, lo, hi, color=color, alpha=0.15)
    ax.set_xlabel("Environment steps")
    ax.set_ylabel("Cumulative reward (episode mean)")
    ax.set_title("Learning curves under stochastic wind + changing destination (3 seeds)")
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path = out_dir / "fig_learning_curves.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[done] {out_path}")


def make_success_bar(agg_csv: Path, out_dir: Path) -> None:
    df = pd.read_csv(agg_csv)
    order = ["sac", "ppo", "rainbow"]
    df = df.set_index("algo").reindex(order).reset_index()
    iqm = df["iqm_success"].values * 100
    lo = df["ci_lo"].values * 100
    hi = df["ci_hi"].values * 100
    yerr = np.stack([iqm - lo, hi - iqm], axis=0)
    colors = [ALGO_COLORS[a] for a in df["algo"]]

    fig, ax = plt.subplots(figsize=(5.5, 4))
    x = np.arange(len(order))
    bars = ax.bar(x, iqm, yerr=yerr, capsize=6, color=colors, alpha=0.85, edgecolor="black", linewidth=0.7)
    for b, v in zip(bars, iqm):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.05, f"{v:.2f}%",
                ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([ALGO_LABELS[a] for a in df["algo"]])
    ax.set_ylabel("Success rate (%)")
    ax.set_ylim(95, 101)
    ax.set_title("Frozen-policy success rate (N=200 per seed, 3 seeds)\nIQM with 95% stratified bootstrap CI (Agarwal et al., 2021)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out_path = out_dir / "fig_success_bar.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[done] {out_path}")


def make_steps_box(per_ep_csv: Path, out_dir: Path) -> None:
    df = pd.read_csv(per_ep_csv)
    goal_df = df[df["outcome"] == "goal"].copy()
    order = ["sac", "ppo", "rainbow"]
    data = [goal_df[goal_df["algo"] == a]["steps"].values for a in order]

    fig, ax = plt.subplots(figsize=(5.5, 4))
    bp = ax.boxplot(
        data,
        tick_labels=[ALGO_LABELS[a] for a in order],
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 1.5},
    )
    for patch, a in zip(bp["boxes"], order):
        patch.set_facecolor(ALGO_COLORS[a])
        patch.set_alpha(0.7)
    means = [d.mean() for d in data]
    ax.scatter(range(1, len(order) + 1), means, marker="D", color="black", s=30, zorder=10, label="mean")
    for i, m in enumerate(means):
        ax.text(i + 1, m + 1, f"{m:.1f}", ha="center", fontsize=9)
    ax.set_ylabel("Steps to goal")
    ax.set_title("Steps-to-goal on successful episodes (600 episodes per algo)")
    ax.legend(loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out_path = out_dir / "fig_steps_box.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[done] {out_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", type=Path, default=PROJECT_ROOT / "code" / "results")
    p.add_argument("--analysis-dir", type=Path, default=PROJECT_ROOT / "analysis")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = p.parse_args()

    out_dir = args.analysis_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    make_learning_curves(args.results_dir, out_dir, args.seeds)
    make_success_bar(args.analysis_dir / "eval_algo_aggregate.csv", out_dir)
    make_steps_box(args.analysis_dir / "eval_per_episode.csv", out_dir)


if __name__ == "__main__":
    main()
