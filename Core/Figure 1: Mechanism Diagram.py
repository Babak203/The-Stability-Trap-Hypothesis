
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib.lines import Line2D


OUTPUT_DIR = r"/Users/MaJun/PycharmProjects/Python(Ajou University)/The Stability Trap Hypothesis/终稿代码/Figure/Figure1:机制图"


def add_box(ax, cx, cy, w, h, text, fontsize=8.5):
    """Draw a rectangle centered at (cx, cy) with text; return geometry info."""
    x = cx - w / 2
    y = cy - h / 2
    rect = Rectangle(
        (x, y),
        w,
        h,
        linewidth=0.9,
        edgecolor="black",
        facecolor="none",
    )
    ax.add_patch(rect)
    ax.text(
        cx,
        cy,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
    )
    return {
        "cx": cx,
        "cy": cy,
        "w": w,
        "h": h,
        "left": x,
        "right": x + w,
        "bottom": y,
        "top": y + h,
    }


def add_vertical_arrow(ax, parent_box, child_box):
    """Straight vertical arrow from parent bottom to child top, aligned to child centre."""
    x = child_box["cx"]
    y_from = parent_box["bottom"]
    y_to = child_box["top"]
    arr = FancyArrowPatch(
        (x, y_from),
        (x, y_to),
        arrowstyle="-|>",
        mutation_scale=7,
        linewidth=0.8,
        color="black",
    )
    ax.add_patch(arr)


def draw_framework(ax):
    ax.set_xlim(0.2, 10)
    ax.set_ylim(1.3, 7.5)
    ax.axis("off")

    # Top: environment
    top = add_box(
        ax,
        cx=5.0,
        cy=6.9,
        w=6.0,
        h=0.9,
        text="Institutional environment and policy shocks",
        fontsize=8.5,
    )

    # S, F, V row – narrower boxes with clear gaps
    y_sfv = 5.3
    w_sfv, h_sfv = 1.8, 0.8
    box_S = add_box(ax, cx=3.0, cy=y_sfv, w=w_sfv, h=h_sfv, text="Stability S")
    box_F = add_box(ax, cx=5.0, cy=y_sfv, w=w_sfv, h=h_sfv, text="Feedback F")
    box_V = add_box(ax, cx=7.0, cy=y_sfv, w=w_sfv, h=h_sfv, text="Volatility V")

    for child in (box_S, box_F, box_V):
        add_vertical_arrow(ax, top, child)

    # Reversible stability
    rev = add_box(
        ax,
        cx=5.0,
        cy=3.7,
        w=4.4,
        h=0.8,
        text="Reversible stability",
        fontsize=8.5,
    )

    # Arrows from S/F/V down to reversible (still straight)
    for parent in (box_S, box_F, box_V):
        add_vertical_arrow(ax, parent, rev)

    # Bottom regimes: four equal boxes, symmetric around x = 5
    y_reg = 2.0
    w_reg, h_reg = 2.1, 0.9
    gap = 0.4
    total_w = 4 * w_reg + 3 * gap
    left = 5.0 - total_w / 2

    centers = [left + w_reg / 2 + i * (w_reg + gap) for i in range(4)]

    box_frag = add_box(
        ax,
        cx=centers[0],
        cy=y_reg,
        w=w_reg,
        h=h_reg,
        text="Fragile regimes\nS < –1",
    )
    box_rigid = add_box(
        ax,
        cx=centers[1],
        cy=y_reg,
        w=w_reg,
        h=h_reg,
        text="Rigid trap\n–1 < S < 0, F < 0",
    )
    box_near = add_box(
        ax,
        cx=centers[2],
        cy=y_reg,
        w=w_reg,
        h=h_reg,
        text="Near trap\nS·F < 0",
    )
    box_adapt = add_box(
        ax,
        cx=centers[3],
        cy=y_reg,
        w=w_reg,
        h=h_reg,
        text="Stable adaptive\nS > 0, F > 0",
    )

    # Horizontal connector under Reversible stability
    y_mid = (rev["bottom"] + box_frag["top"]) / 2.0
    x_left = box_frag["cx"]
    x_right = box_adapt["cx"]
    line = Line2D([x_left, x_right], [y_mid, y_mid], linewidth=0.8, color="black")
    ax.add_line(line)

    # Arrow from Reversible stability down to the horizontal connector (center)
    trunk_arrow = FancyArrowPatch(
        (rev["cx"], rev["bottom"]),
        (rev["cx"], y_mid),
        arrowstyle="-|>",
        mutation_scale=7,
        linewidth=0.8,
        color="black",
    )
    ax.add_patch(trunk_arrow)

    # Vertical arrows from horizontal connector to each regime box
    for child in (box_frag, box_rigid, box_near, box_adapt):
        arr = FancyArrowPatch(
            (child["cx"], y_mid),
            (child["cx"], child["top"]),
            arrowstyle="-|>",
            mutation_scale=7,
            linewidth=0.8,
            color="black",
        )
        ax.add_patch(arr)


def main():
    # 183 mm width figure
    width_mm = 183.0
    width_in = width_mm / 25.4
    height_in = width_in * 0.45
    fig, ax = plt.subplots(figsize=(width_in, height_in))
    draw_framework(ax)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    png_path = os.path.join(OUTPUT_DIR, "Figure 1：机制图.png")
    svg_path = os.path.join(OUTPUT_DIR, "Figure1_stability_framework_styleA_branch_183mm.svg")

    fig.savefig(png_path, dpi=400, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    print("Saved:", png_path)
    print("Saved:", svg_path)


if __name__ == "__main__":
    main()
