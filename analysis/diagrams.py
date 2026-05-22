"""Regenerate the v1 network-architecture diagrams at 300 DPI (no Unity needed).

Replaces the low-res (~96 DPI) v1 figures:
    fig_policy_net.png  <- v1 Figure 7 (policy network)  -- FAITHFUL 12-d input / 2-branch out
    fig_value_net.png   <- v1 Figure 8 (value network)   -- FAITHFUL 12-d input / scalar out

Out of scope here (user supplies own high-res from PPTX): the RL agent-environment loop
(v1 Fig 1) and the PPO training pipeline (v1 Fig 6). Unity-screenshot figures (env, boat
close-ups, arena grid, inspector) are re-captured in the Editor at high resolution.

Usage:
    python code/analysis/diagrams.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "analysis" / "figures"

plt.rcParams.update({
    "font.size": 12,
    "savefig.bbox": "tight",
    "savefig.dpi": 300,
})

NODE_FC = "#e8643c"   # warm accent for network nodes (CVD-safe vs black text)


# ──────────────────── Fig 7 / 8: network architecture (faithful) ────────────────────

# True observation = 12 floats (see MoveToGoalAgent.cs + Environment.prefab):
#   boat localPosition (3) + orientation quaternion (4) + target localPosition (3)
#   + weather-vane x, z (2) = full 2-D wind direction.
# (VectorObservationSize=12 since 2026-05-22; earlier builds set 11 and truncated vane.z.)
INPUT_LABELS = [
    r"$x^{b}$", r"$y^{b}$", r"$z^{b}$",
    r"$q_x$", r"$q_y$", r"$q_z$", r"$q_w$",
    r"$x^{d}$", r"$y^{d}$", r"$z^{d}$",
    r"$v_x$", r"$v_z$",
]
INPUT_GROUPS = [  # (start_idx, end_idx_inclusive, label)
    (0, 2, "boat\nposition (3)"),
    (3, 6, "orientation\n(quaternion, 4)"),
    (7, 9, "destination (3)"),
    (10, 11, "wind dir.\n($v_x, v_z$, 2)"),
]


def _node(ax, x, y, r=0.13, fc=NODE_FC):
    ax.add_patch(Circle((x, y), r, facecolor=fc, edgecolor="black", linewidth=0.8, zorder=3))


def _draw_net(ax, output_nodes_y, output_labels, output_group_labels):
    """Shared MLP skeleton: 11 inputs -> 128 -> 128 -> output."""
    xs = [0.0, 1.7, 3.4, 5.1]  # input, h1, h2, output
    n_in = len(INPUT_LABELS)
    in_y = np.linspace(0.5, n_in - 0.5, n_in)[::-1]
    # hidden: show 6 nodes + ellipsis
    hid_y = np.linspace(1.0, n_in - 1.0, 6)[::-1]

    # connections (representative, light)
    for y0 in in_y:
        for y1 in hid_y:
            ax.plot([xs[0], xs[1]], [y0, y1], color="#cccccc", linewidth=0.4, zorder=1)
    for y0 in hid_y:
        for y1 in hid_y:
            ax.plot([xs[1], xs[2]], [y0, y1], color="#cccccc", linewidth=0.4, zorder=1)
    for y0 in hid_y:
        for y1 in output_nodes_y:
            ax.plot([xs[2], xs[3]], [y0, y1], color="#cccccc", linewidth=0.4, zorder=1)

    # input nodes + labels + group brackets
    for y, lab in zip(in_y, INPUT_LABELS):
        _node(ax, xs[0], y, fc="#5ba0d0")
        ax.text(xs[0] - 0.35, y, lab, ha="right", va="center", fontsize=11)
    for s, e, glab in INPUT_GROUPS:
        ytop, ybot = in_y[s] + 0.28, in_y[e] - 0.28
        ax.plot([xs[0] - 1.05, xs[0] - 1.05], [ybot, ytop], color="black", linewidth=1.2)
        ax.plot([xs[0] - 1.05, xs[0] - 0.95], [ytop, ytop], color="black", linewidth=1.2)
        ax.plot([xs[0] - 1.05, xs[0] - 0.95], [ybot, ybot], color="black", linewidth=1.2)
        ax.text(xs[0] - 1.2, (ytop + ybot) / 2, glab, ha="right", va="center", fontsize=9.5)

    # hidden nodes + ellipsis + size label
    for xi in (xs[1], xs[2]):
        for y in hid_y:
            _node(ax, xi, y)
        ax.text(xi, 0.1, r"$\vdots$", ha="center", va="center", fontsize=16)
        ax.text(xi, n_in + 0.2, "128", ha="center", va="center", fontsize=10)
    ax.text((xs[1] + xs[2]) / 2, n_in + 0.9, "hidden layers (2)", ha="center", fontsize=10)

    # output nodes + labels
    for y, lab in zip(output_nodes_y, output_labels):
        _node(ax, xs[3], y, fc="#e8643c")
        ax.text(xs[3] + 0.32, y, lab, ha="left", va="center", fontsize=11)
    for (gy0, gy1, glab) in output_group_labels:
        ax.text(xs[3] + 1.35, (gy0 + gy1) / 2, glab, ha="left", va="center", fontsize=9.5)

    ax.text(xs[0], n_in + 0.9, f"input ({n_in})", ha="center", fontsize=10)
    ax.set_xlim(-3.0, 7.2); ax.set_ylim(-0.5, n_in + 1.4)
    ax.axis("off")


def make_policy_net() -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.6))
    # MultiDiscrete(3,3): two branches — steer {L,0,R}, motor {F,0,B}
    steer_y = [9.2, 8.2, 7.2]
    motor_y = [4.0, 3.0, 2.0]
    out_y = steer_y + motor_y
    out_lab = [r"$a^{s}_{L}$", r"$a^{s}_{0}$", r"$a^{s}_{R}$",
               r"$a^{m}_{F}$", r"$a^{m}_{0}$", r"$a^{m}_{B}$"]
    grp = [(steer_y[0], steer_y[-1], "steer\nbranch (3)"),
           (motor_y[0], motor_y[-1], "motor\nbranch (3)")]
    _draw_net(ax, out_y, out_lab, grp)
    ax.text(5.1, 11.0, "output:\nMultiDiscrete(3,3)", ha="center", fontsize=10)
    fig.tight_layout()
    out = OUT_DIR / "fig_policy_net.png"
    fig.savefig(out); plt.close(fig); print(f"[done] {out}")


def make_value_net() -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.6))
    out_y = [5.5]
    _draw_net(ax, out_y, [r"$v(s_t)$"], [(5.5, 5.5, "state value")])
    ax.text(5.1, 11.0, "output:\nscalar value", ha="center", fontsize=10)
    fig.tight_layout()
    out = OUT_DIR / "fig_value_net.png"
    fig.savefig(out); plt.close(fig); print(f"[done] {out}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    make_policy_net()
    make_value_net()


if __name__ == "__main__":
    main()
