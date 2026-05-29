"""Regenerate the v1 network-architecture diagrams at 300 DPI (no Unity needed).

Layer-box style (IEEE Access / mainstream ML literature convention):
each layer = one rectangle with size + activation, arrows between, grouped
input/output items shown as sub-items inside the relevant boxes. Compact
(~25% column height) and column-width-safe.

Replaces the low-res (~96 DPI) v1 figures:
    fig_policy_net.png  <- v1 Figure 7 (policy network)  -- FAITHFUL 12-d input / 2-branch out
    fig_value_net.png   <- v1 Figure 8 (value network)   -- FAITHFUL 12-d input / scalar out

Usage:
    python code/analysis/diagrams.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "analysis" / "figures"

plt.rcParams.update({
    "font.size": 9,
    "savefig.bbox": "tight",
    "savefig.dpi": 300,
})

INPUT_FILL = "#dbeafe"   # light blue (input)
HIDDEN_FILL = "#f4f4f4"  # neutral gray (hidden)
OUTPUT_FILL = "#fde2c4"  # light orange (output)
EDGE = "black"


def _box(ax, x, y_center, w, h, text, fill):
    """Rounded rectangle with multi-line left-aligned text inside."""
    patch = FancyBboxPatch(
        (x, y_center - h / 2), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        facecolor=fill, edgecolor=EDGE, linewidth=1.0,
    )
    ax.add_patch(patch)
    ax.text(x + 0.10, y_center, text, ha="left", va="center", fontsize=9)


def _arrow(ax, x, y0, y1):
    ax.annotate(
        "", xy=(x, y1), xytext=(x, y0),
        arrowprops=dict(arrowstyle="-|>", lw=1.0, color=EDGE,
                        shrinkA=0, shrinkB=0, mutation_scale=10),
    )


def _draw_layer_stack(ax, output_lines, output_box_h):
    """Stack: Input box → FC1 → FC2 → Output box.

    output_lines: list of text lines for the Output box (header + optional sub-items).
    output_box_h: height for the Output box (depends on number of lines).
    """
    box_x = 0.12
    box_w = 3.30

    input_lines = (
        "Input (12)\n"
        "  ├ boat position ($b_x$, $b_y$, $b_z$)\n"
        "  ├ orientation quaternion ($q_x$, $q_y$, $q_z$, $q_w$)\n"
        "  ├ destination position ($d_x$, $d_y$, $d_z$)\n"
        "  └ wind direction ($v_x$, $v_z$)"
    )

    # Layout from top down. Y positions are box centers.
    in_h = 1.30
    fc_h = 0.45
    gap = 0.22

    y_in = 4.30
    y_fc1 = y_in - in_h / 2 - gap - fc_h / 2
    y_fc2 = y_fc1 - fc_h - gap
    y_out = y_fc2 - fc_h / 2 - gap - output_box_h / 2

    _box(ax, box_x, y_in, box_w, in_h, input_lines, INPUT_FILL)
    _box(ax, box_x, y_fc1, box_w, fc_h, "Fully Connected 128 + ReLU", HIDDEN_FILL)
    _box(ax, box_x, y_fc2, box_w, fc_h, "Fully Connected 128 + ReLU", HIDDEN_FILL)
    _box(ax, box_x, y_out, box_w, output_box_h, "\n".join(output_lines), OUTPUT_FILL)

    # Arrows: each connects the bottom of the upper box to the top of the lower box.
    arrow_x = box_x + box_w / 2
    _arrow(ax, arrow_x, y_in - in_h / 2, y_fc1 + fc_h / 2)
    _arrow(ax, arrow_x, y_fc1 - fc_h / 2, y_fc2 + fc_h / 2)
    _arrow(ax, arrow_x, y_fc2 - fc_h / 2, y_out + output_box_h / 2)

    y_bot = y_out - output_box_h / 2
    ax.set_xlim(0, 3.55)
    ax.set_ylim(y_bot - 0.15, y_in + in_h / 2 + 0.15)
    ax.set_aspect("auto")
    ax.axis("off")


def make_policy_net() -> None:
    fig, ax = plt.subplots(figsize=(3.5, 3.4))
    output_lines = [
        "Output: MultiDiscrete(3, 3)",
        "  ├ steer: {turn left, neutral, turn right}",
        "  └ motor: {forward, neutral, backward}",
    ]
    _draw_layer_stack(ax, output_lines, output_box_h=0.85)
    fig.tight_layout()
    out = OUT_DIR / "fig_policy_net.png"
    fig.savefig(out); plt.close(fig); print(f"[done] {out}")


def make_value_net() -> None:
    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    output_lines = ["Output: scalar  $v(s_t)$"]
    _draw_layer_stack(ax, output_lines, output_box_h=0.45)
    fig.tight_layout()
    out = OUT_DIR / "fig_value_net.png"
    fig.savefig(out); plt.close(fig); print(f"[done] {out}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    make_policy_net()
    make_value_net()


if __name__ == "__main__":
    main()
