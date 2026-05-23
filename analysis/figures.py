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

# Okabe-Ito colorblind-safe palette (safe for deuteranopia/protanopia/tritanopia).
ALGO_COLORS = {"sac": "#0072B2", "ppo": "#E69F00", "rainbow": "#009E73"}
ALGO_LABELS = {"sac": "SAC", "ppo": "PPO", "rainbow": "Rainbow DQN"}
# Redundant marker encoding so interval/scatter figures stay legible in grayscale / CVD.
ALGO_MARKERS = {"sac": "o", "ppo": "s", "rainbow": "D"}

# Shared typography so all figures render at a consistent size side-by-side in the paper.
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "savefig.bbox": "tight",
})


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
        # Per-seed thin lines (de-emphasized so the mean + band carries the message)
        color = ALGO_COLORS[algo]
        for s_idx, (steps, vals) in enumerate(curves):
            ax.plot(steps, vals, color=color, alpha=0.12, linewidth=0.5,
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
        ax.fill_between(grid, lo, hi, color=color, alpha=0.18)
    ax.set_xlabel("Environment steps")
    ax.set_ylabel("Cumulative reward (episode mean)")
    # Reward is sparse-terminal, hard-bounded to [-1, 1]. No data exists beyond +/-1.0;
    # the small +/-0.05 margin only lifts the spine off the ceiling/floor data so it
    # is not visually amputated by the frame (not a claim of headroom past the bound).
    ax.set_ylim(-1.05, 1.05)
    ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path = out_dir / "fig_learning_curves.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[done] {out_path}")


def make_success_bar(agg_csv: Path, out_dir: Path) -> None:
    # Dot + 95% CI (forest plot), NOT a bar chart. PPO and SAC sit at the ceiling with
    # overlapping CIs (a tie); Rainbow DQN trails with a clearly separated, lower CI.
    # Bars from zero would distort the small absolute spread; the zoomed forest plot
    # shows both the PPO/SAC overlap and the Rainbow gap honestly.
    df = pd.read_csv(agg_csv)
    order = ["sac", "ppo", "rainbow"]
    df = df.set_index("algo").reindex(order).reset_index()
    iqm = df["iqm_success"].values * 100
    lo = df["ci_lo"].values * 100
    hi = df["ci_hi"].values * 100

    fig, ax = plt.subplots(figsize=(5.5, 4))
    x = np.arange(len(order))
    for i, a in enumerate(order):
        # clip tiny negatives from degenerate CIs (zero-variance seeds → CI is a point)
        yerr = np.clip(np.array([[iqm[i] - lo[i]], [hi[i] - iqm[i]]]), 0.0, None)
        ax.errorbar(x[i], iqm[i], yerr=yerr, fmt=ALGO_MARKERS[a], color=ALGO_COLORS[a],
                    markersize=9, capsize=6, elinewidth=1.6, capthick=1.6,
                    markeredgecolor="black", markeredgewidth=0.6)
        # degenerate (zero-width) CI: draw a short horizontal cap so the point-interval
        # reads as intentional rather than a missing whisker (PPO: 3 identical seeds).
        if (hi[i] - lo[i]) < 0.01:
            ax.plot([x[i] - 0.11, x[i] + 0.11], [iqm[i], iqm[i]],
                    color=ALGO_COLORS[a], lw=1.6, solid_capstyle="butt", zorder=2)
        ax.annotate(f"{iqm[i]:.2f}%", xy=(x[i], iqm[i]), xytext=(12, 0),
                    textcoords="offset points", ha="left", va="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([ALGO_LABELS[a] for a in order])
    ax.set_xlim(-0.5, len(order) - 0.5 + 0.4)
    ax.set_ylabel("Success rate (%)")
    # span all CIs with margin (Rainbow's lower bound now well below the PPO/SAC ceiling)
    ax.set_ylim(float(np.floor(lo.min() - 1.5)), 100.6)
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

    box_w = 0.5
    fig, ax = plt.subplots(figsize=(5.5, 4))
    bp = ax.boxplot(
        data,
        widths=box_w,
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
    # Label sits in the inter-box gap (clear of the colored fill) for readability.
    for i, m in enumerate(means):
        ax.text(i + 1 + box_w / 2 + 0.04, m, f"{m:.1f}", ha="left", va="center", fontsize=9)
    ax.set_xlim(0.5, len(order) + 0.6)
    ax.set_ylabel("Steps to goal")
    ax.legend(loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out_path = out_dir / "fig_steps_box.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[done] {out_path}")


def make_trajectory(traj_json: Path, out_dir: Path, n_success: int = 6) -> None:
    """Fig 10: one panel per algo, several representative successful (x,z) paths +
    any wall-failure path overlaid. No wind overlay (sim wind is spatially uniform)."""
    import json
    if not traj_json.exists():
        print(f"[skip] {traj_json} missing — run evaluate.py --log-trajectories")
        return
    with open(traj_json) as f:
        trajs = json.load(f)
    order = ["sac", "ppo", "rainbow"]

    # Boat path (transform.localPosition) and target (targetTransform.localPosition) are
    # in different parent frames. On goal episodes the boat physically reaches the goal,
    # so (path_end - target) recovers the constant frame offset; apply it to express the
    # destination in the boat frame (residual ~goal-collider radius). See project memory.
    goal_off = np.array([
        np.array(t["path"][-1]) - np.array(t["target"])
        for t in trajs if t["outcome"] == "goal" and t["path"]
    ])
    offset = np.median(goal_off, axis=0) if len(goal_off) else np.zeros(2)

    def dest_of(t):
        return np.array(t["target"]) + offset

    # Shared axis extent from all paths + (frame-corrected) destinations.
    xs, zs = [], []
    for t in trajs:
        for x, z in t["path"]:
            xs.append(x); zs.append(z)
        d = dest_of(t); xs.append(d[0]); zs.append(d[1])
    xpad = 0.05 * (max(xs) - min(xs)); zpad = 0.05 * (max(zs) - min(zs))
    xlim = (min(xs) - xpad, max(xs) + xpad); zlim = (min(zs) - zpad, max(zs) + zpad)

    fig, axes = plt.subplots(1, len(order), figsize=(13, 6.0), sharex=True, sharey=True)
    for ax, algo in zip(axes, order):
        goals = [t for t in trajs if t["algo"] == algo and t["outcome"] == "goal"]
        walls = [t for t in trajs if t["algo"] == algo and t["outcome"] == "wall"]
        # Pick a spread of successful episodes by target location for visual variety.
        goals = sorted(goals, key=lambda t: (t["target"][0], t["target"][1]))
        step = max(1, len(goals) // n_success)
        picks = goals[::step][:n_success]
        color = ALGO_COLORS[algo]
        for t in picks:
            p = np.array(t["path"]); d = dest_of(t)
            ax.plot(p[:, 0], p[:, 1], color=color, alpha=0.7, linewidth=1.3)
            ax.plot(p[0, 0], p[0, 1], marker="o", color=color, markersize=6,
                    markeredgecolor="black", markeredgewidth=0.5, zorder=5)
            ax.plot(d[0], d[1], marker="*", color="black", markersize=12, zorder=6)
        for t in walls[:1]:  # show one real failure if it exists for this algo
            p = np.array(t["path"]); d = dest_of(t)
            ax.plot(p[:, 0], p[:, 1], color="#444444", alpha=0.9, linewidth=1.4, linestyle="--")
            ax.plot(p[0, 0], p[0, 1], marker="o", color="#444444", markersize=6,
                    markeredgecolor="black", markeredgewidth=0.5, zorder=5)
            ax.plot(d[0], d[1], marker="*", color="black", markersize=12, zorder=6)
            ax.plot(p[-1, 0], p[-1, 1], marker="x", color="#d62728", markersize=10,
                    markeredgewidth=2.0, zorder=7)
        ax.set_title(ALGO_LABELS[algo])
        ax.set_xlabel("x position")
        ax.set_xlim(xlim); ax.set_ylim(zlim)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("z position")
    # Single shared legend.
    handles = [
        plt.Line2D([], [], marker="o", color="gray", markeredgecolor="black",
                   linestyle="-", label="start → path"),
        plt.Line2D([], [], marker="*", color="black", linestyle="None", markersize=12, label="destination"),
        plt.Line2D([], [], color="#444444", linestyle="--",
                   marker="x", markerfacecolor="#d62728", markeredgecolor="#d62728",
                   label="wall failure"),
    ]
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=True,
               bbox_to_anchor=(0.5, 0.015))
    out_path = out_dir / "fig_trajectories.png"
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
    make_trajectory(args.analysis_dir / "eval_trajectories.json", out_dir)


if __name__ == "__main__":
    main()
