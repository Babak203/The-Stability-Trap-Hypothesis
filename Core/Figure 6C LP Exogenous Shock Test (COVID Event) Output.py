#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Plot script for Task 21c: COVID event-study shock-response path (Rigid vs non-Rigid)

This script reads the event-study coefficients:
    covid_event_study_coeffs.csv

and generates a figure showing β_k (Rigid − non-Rigid growth differential)
by relative year (k = -4, -3, -2, -1, 1, 2, 3), with 95% confidence intervals.

Input path:
    /Users/MaJun/PycharmProjects/Python(Ajou University)/
        The Stability Trap Hypothesis/终稿代码/Figure/Figure6:COVID 事件型冲击/
        covid_event_study_coeffs.csv

Output:
    - Figure6:COVID 事件型冲击.png
    - Figure6:COVID 事件型冲击.svg
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ========== 0. Paths ==========

DATA_PATH = Path(
    "/Users/MaJun/PycharmProjects/Python(Ajou University)/The Stability Trap Hypothesis/终稿代码/Figure/Figure6:COVID 事件型冲击/covid_event_study_coeffs.csv"
)

OUTPUT_DIR = Path(
    "/Users/MaJun/PycharmProjects/Python(Ajou University)/The Stability Trap Hypothesis/终稿代码/Figure/Figure6:COVID 事件型冲击"
)
OUTPUT_FIG = OUTPUT_DIR / "Figure6:COVID 事件型冲击.png"


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Input coeff file not found: {DATA_PATH}")

    print(f"[INFO] Loading event-study coefficients from: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)

    # Expect columns: rel_year, beta, ci_low, ci_high
    required_cols = ["rel_year", "beta", "ci_low", "ci_high"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in coeff file: {missing}")

    # Sort by rel_year
    df = df.sort_values("rel_year")

    # X-axis: relative years and actual calendar years
    x = df["rel_year"].values
    beta = df["beta"].values
    ci_low = df["ci_low"].values
    ci_high = df["ci_high"].values

    years = 2019 + x  # map rel_year to actual year labels

    # ========== 1. Global style (journal-like) ==========
    plt.rcParams.update({
        "font.size": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
    })

    # 183mm ≈ 7.2 inches
    fig = plt.figure(figsize=(7.2, 3.8))
    ax = fig.add_subplot(111)

    # ========== 2. Plot: point estimates + 95% CI band ==========

    # 95% CI band
    ax.fill_between(
        x,
        ci_low,
        ci_high,
        alpha=0.15,
        label="95% confidence interval",)

    # Point estimates with line
    ax.plot(x, beta, marker="o", linewidth=1.5, label="Rigid − non-Rigid")

    # Zero line (solid, thin)
    ax.axhline(0.0, linestyle="--", linewidth=0.8)

    # Vertical line at shock year boundary (k = 0, 2019 baseline)
    ax.axvline(0.0, linestyle="--", linewidth=0.8)

    # Axis labels
    ax.set_xlabel("Relative year (k, 2019 = 0)")
    ax.set_ylabel("Growth differential: Rigid − non-Rigid (p.p.)")

    # X-ticks and labels: e.g. "2015 (-4)"
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{int(y)} ({int(k)})" for k, y in zip(x, years)],
        fontsize=7
    )

    # Optional: tighten x-limits a bit for nicer framing
    ax.set_xlim(x.min() - 0.5, x.max() + 0.5)
    ax.set_ylim(-6.5, 3.0)

    # Legend: moved to lower left, no frame
    ax.legend(loc="lower left", frameon=False)

    # Layout: slightly缩小左边留白
    fig.subplots_adjust(left=0.08, right=0.98, top=0.98, bottom=0.12)

    # ========== 3. Save figure (PNG + SVG, 300 dpi) ==========
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig.savefig(OUTPUT_FIG, dpi=400)

    base = OUTPUT_FIG.with_suffix("")  # remove .png suffix
    fig.savefig(str(base) + ".svg", dpi=400)

    print(f"[INFO] Figure saved to: {OUTPUT_FIG}")
    print(f"[INFO] SVG version saved to: {str(base) + '.svg'}")


if __name__ == "__main__":
    main()

