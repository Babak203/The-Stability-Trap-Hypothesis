
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# =========================================================
# 1) PATHS (按你要求：不猜，写死在你当前体系里)
# =========================================================
BASE_DIR = Path(r"/The Stability Trap Hypothesis/提交代码/附件/Figure_S5:5 年累计增长收益 Δ5")
IO_DIR = BASE_DIR / r"Figure 7: Scenario Path (China-US-Vietnam Baseline vs. Reform)"
IN_CSV = IO_DIR / "Figure_S5_Delta5y_30countries_sorted.csv"

# output files (same IO_DIR)
OUT_PNG = IO_DIR / "Figure_S5_Delta5y_30countries_sorted_BAR_FINAL.png"
OUT_SVG = IO_DIR / "Figure_S5_Delta5y_30countries_sorted_BAR_FINAL.svg"
OUT_CLEAN_CSV = IO_DIR / "Figure_S5_Delta5y_30countries_sorted_CLEAN.csv"


# =========================================================
# 2) STYLE (终稿美学：低饱和、紧凑、无报告味)
# =========================================================
REG_COLORS = {
    "Rigid Trap": "#c44e52",
    "Near Trap": "#dd8452",
    "Stable Adaptive": "#55a868",
    "Fragile regimes": "#8c613c",
}
REG_ORDER = ["Rigid Trap", "Near Trap", "Stable Adaptive", "Fragile regimes"]
REG_MARKERS = {"Rigid Trap": "o", "Near Trap": "D", "Stable Adaptive": "^", "Fragile regimes": "s"}

FIGSIZE = (8.6, 2.4)
DPI = 1000
BAR_WIDTH = 0.75


# =========================================================
# 3) HELPERS
# =========================================================
def pick_col(df: pd.DataFrame, candidates):
    # exact match
    for c in candidates:
        if c in df.columns:
            return c
    # case-insensitive match
    lower_map = {str(c).lower(): c for c in df.columns}
    for c in candidates:
        k = str(c).lower()
        if k in lower_map:
            return lower_map[k]
    return None


def norm_regime(x: str) -> str:
    s = str(x).strip()
    sl = s.lower()
    if "rigid" in sl:
        return "Rigid Trap"
    if "near" in sl:
        return "Near Trap"
    if "stable" in sl and "adaptive" in sl:
        return "Stable Adaptive"
    if "fragile" in sl:
        return "Fragile regimes"
    return s


def ensure_pp(series: pd.Series) -> pd.Series:
    """If values look like bp (e.g., max>30), convert to p.p. by /100."""
    x = pd.to_numeric(series, errors="coerce")
    mx = x.max(skipna=True)
    if mx is not None and mx > 30:
        return x / 100.0
    return x


def die_with_dir_listing(msg: str, folder: Path):
    listing = "\n".join([f"  - {p.name}" for p in sorted(folder.glob("*"))])
    raise FileNotFoundError(msg + "\n\nFiles under target folder:\n" + listing)


# =========================================================
# 4) MAIN
# =========================================================
def main():
    if not IO_DIR.exists():
        die_with_dir_listing(f"[ERROR] IO_DIR not found: {IO_DIR}", BASE_DIR)

    if not IN_CSV.exists():
        die_with_dir_listing(f"[ERROR] Input CSV not found: {IN_CSV}", IO_DIR)

    df = pd.read_csv(IN_CSV)

    # detect columns
    col_iso3 = pick_col(df, ["ISO3", "iso3", "iso3c", "country_code", "countryiso3code", "CountryIso3Code"])
    col_reg  = pick_col(df, ["Regime", "regime", "regime_norm", "REGIME"])
    col_d5y  = pick_col(df, ["Delta_5y", "Delta5y", "Δ5y", "Delta_5y_pp", "Delta5y_pp",
                             "Delta_5y_bp", "Delta5y_bp", "Δ5y_bp"])

    if col_iso3 is None:
        raise KeyError(f"Cannot find ISO3 column. Available columns: {list(df.columns)}")
    if col_reg is None:
        raise KeyError(f"Cannot find Regime column. Available columns: {list(df.columns)}")
    if col_d5y is None:
        raise KeyError(f"Cannot find Δ5y column. Available columns: {list(df.columns)}")

    out = pd.DataFrame({
        "ISO3": df[col_iso3].astype(str).str.strip().str.upper(),
        "Regime": df[col_reg].astype(str).str.strip().map(norm_regime),
        "Delta_5y": ensure_pp(df[col_d5y]),
    }).dropna(subset=["ISO3", "Regime", "Delta_5y"])

    # sort descending by gain
    out = out.sort_values("Delta_5y", ascending=False).reset_index(drop=True)

    # write clean sorted CSV (for downstream scripts)
    out.to_csv(OUT_CLEAN_CSV, index=False)

    # plotting style
    plt.rcParams.update({
        "font.size": 7,
        "axes.labelsize": 8,
        "xtick.labelsize": 6,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.grid": False,
        "axes.linewidth": 0.8,
    })

    fig, ax = plt.subplots(figsize=FIGSIZE)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    x = np.arange(len(out))
    y = out["Delta_5y"].to_numpy(float)

    colors = out["Regime"].map(REG_COLORS).fillna("gray").to_numpy()
    ax.bar(x, y, color=colors, width=BAR_WIDTH, align="center")
    ax.margins(x=0.01)
    ymax = float(np.nanmax(y)) if len(y) else 1.0
    ax.set_ylim(0, ymax * 1.12)

    ax.set_ylabel("Five-year cumulative growth (p.p.)")
    ax.set_xlabel("Country")

    ax.set_xticks(x)
    ax.set_xticklabels(out["ISO3"].tolist(), rotation=45, ha="center", rotation_mode="anchor")

    # clean spines (more journal-like)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # value labels (compact)
    for i, yi in enumerate(y):
        ax.text(i, yi + ymax * 0.02, f"{yi:.1f}", ha="center", va="bottom", fontsize=6)

    # legend (counts), no frame
    counts = out["Regime"].value_counts().to_dict()
    handles = []
    for k in REG_ORDER:
        if k in counts:
            handles.append(
                Line2D([0], [0],
                       marker=REG_MARKERS.get(k, "o"),
                       linestyle="None",
                       color="none",
                       markerfacecolor=REG_COLORS.get(k, "gray"),
                       markeredgecolor=REG_COLORS.get(k, "gray"),
                       markersize=5,
                       label=f"{k} ({counts[k]})")
            )
    ax.legend(
        handles=handles,
        loc="upper right",
        frameon=False,
        borderaxespad=0.3,
        handletextpad=0.4,
        labelspacing=0.25,
        markerscale=0.9
    )
    fig.tight_layout(pad=0.2)
    fig.savefig(OUT_PNG, dpi=DPI, bbox_inches="tight")
    fig.savefig(OUT_SVG, dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    print("[OK] Input :", IN_CSV)
    print("[OK] Clean :", OUT_CLEAN_CSV)
    print("[OK] PNG   :", OUT_PNG)
    print("[OK] SVG   :", OUT_SVG)


if __name__ == "__main__":
    main()