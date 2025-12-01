#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scheme A v2.9 — 主文版 S–F 散点 + 全球分型地图（全部国家标注，标签紧凑版，最新 Fragile–Trap–Adaptive 分型，红色虚线边界，无 F 阈值线）

调整点：
1. S–F 散点图：图例移入图内空白处，无 “Class” 标题；
2. 全球地图：图例一行显示，位于下方空白一行，无 “Class” 标题。

【技术规格修改】
- 图宽统一为 183 mm（顶刊双栏宽度），用英寸表示；
- 默认 dpi=300（可通过参数调整，但始终 ≥300 即可）；
- 所有主图同时输出 PNG + SVG；
- 地图内不再使用大标题，标题交给正文 caption。
"""

import argparse
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", message=".*tight_layout.*", category=UserWarning)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger("schemeA_v2_6_all_labels_tight")

# ========= 图形技术规格常量（统一宽度 183mm） =========
FIG_WIDTH_MM = 183.0
FIG_WIDTH_INCH = FIG_WIDTH_MM / 25.4  # ≈ 7.2 inch
SCATTER_ASPECT = 0.62  # 高 = 宽 * 0.62，略扁一些
MAP_ASPECT = 0.50      # 地图略矮一点，适合双栏排版

# ========= 默认路径 =========
DEFAULT_PANEL_CSV = "/Users/MaJun/PycharmProjects/Python(Ajou University)/.venv/enriched_panel.csv"
DEFAULT_OUT_DIR   = "/Users/MaJun/PycharmProjects/Python(Ajou University)/The Stability Trap Hypothesis/终稿代码/Figure/Figure2和3全球分型地图+30国 S–F 散点"
DEFAULT_COUNTRY   = "countryiso3code"
DEFAULT_MAPS_ZIP  = "/Users/MaJun/PycharmProjects/Python(Ajou University)/The Stability Trap Hypothesis/maps.zip"

# 类别顺序、配色、marker 形状
CLASS_ORDER = ["Rigid Trap","Near Trap","Stable Adaptive","Fragile regimes","Escaping"]
CLASS_COLORS = {
    "Rigid Trap":      "#a50026",
    "Near Trap":       "#fdae61",
    "Fragile regimes": "#8c510a",
    "Stable Adaptive": "#1a9850",
    "Escaping":        "#74add1",
    "No Data":         "#d3d3d3",
}
CLASS_MARKERS = {
    "Rigid Trap":      "o",  # 圆点
    "Fragile regimes": "s",  # 方块
    "Near Trap":       "D",  # 菱形
    "Stable Adaptive": "^",  # 上三角
    "Escaping":        "v",  # 下三角
}

# 标签偏移模式（points）
OFFSET_PATTERNS = [
    (0, 6),
    (0, -7),
    (6, 4),
    (-6, 4),
    (6, -4),
    (-6, -4),
    (8, 0),
    (-8, 0),
]

# ========= 参数解析 =========

def build_args():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--year", type=int, default=2023)
    ap.add_argument("--panel-path", type=str, default=DEFAULT_PANEL_CSV)
    ap.add_argument("--out-dir", type=str, default=DEFAULT_OUT_DIR)
    ap.add_argument("--country-col", type=str, default=DEFAULT_COUNTRY)
    ap.add_argument("--s-col", type=str, default=None)
    ap.add_argument("--f-col", type=str, default=None)
    ap.add_argument("--maps-zip", type=str, default=DEFAULT_MAPS_ZIP)
    ap.add_argument("--dpi", type=int, default=300)
    # 阈值
    ap.add_argument("--cutS", type=str, default="-1.0,-0.5,0.0", help="S 的三个阈值（逗号分隔）")
    ap.add_argument("--F0", type=float, default=-0.5, help="斜线阈值截距，F >= F0 + b*S")
    ap.add_argument("--b", type=float, default=0.25, help="斜率 b")
    ap.add_argument("--autoF", type=str, default="", help="自动 F0，如 'p50'（覆盖 --F0）")
    ap.add_argument("--escaping", type=int, default=0, help="1 启用 Escaping=(S<0 & F>=0) 覆盖")
    # 可视化
    ap.add_argument("--band", type=float, default=0.10, help="敏感带半宽")
    ap.add_argument("--size-mode", type=str, default="uniform",
                    choices=["uniform","pos","abs"],
                    help="点大小模式：uniform=统一；pos/abs 与旧版兼容（不推荐）")
    return ap.parse_args()

def parse_cutS(s: str):
    vals = [float(x.strip()) for x in s.split(",") if x.strip()]
    if len(vals) != 3:
        raise ValueError("--cutS 需要 3 个数，例如 '-1.0,-0.5,0.0'")
    vals.sort()
    return vals

def ensure_outdir(p: str) -> Path:
    out = Path(p)
    out.mkdir(parents=True, exist_ok=True)
    return out

# ========= S / F 列推断 =========

def first_present(cands, mapping):
    for c in cands:
        if c.lower() in mapping:
            return mapping[c.lower()]
    return None

def infer_SF(df, s_col, f_col):
    m = {c.lower(): c for c in df.columns}

    # S
    if s_col is None:
        s_col = first_present(["S_zscore","z_S","S","stability_z","stability","S_std","S_score"], m)
    if s_col is None:
        wgi_cols = [c for c in df.columns if c.startswith("WGI_") and c.endswith(".EST")]
        if len(wgi_cols) >= 3:
            tmp = df[wgi_cols].astype(float)
            s_raw = tmp.mean(axis=1, skipna=True)
            s_col = "_S_from_WGI_"
            zden = s_raw.std(ddof=0) or 1.0
            df[s_col] = (s_raw - s_raw.mean()) / zden
            log.info("[S] built from WGI_* -> %s", s_col)
        else:
            raise ValueError("没有 S 列，且 WGI_* 不足以构造 S；请用 --s-col 指定。")

    # F
    if f_col is None:
        f_col = first_present(["FQ_z_equal_weight","F_index","F_zscore","F","FQ","feedback_z","feedback"], m)
    if f_col is None:
        raise ValueError("没有 F 列（如 FQ_z_equal_weight）；请用 --f-col 指定。")

    return s_col, f_col

def pick_name_column(df):
    for cand in ["Country Name","country_name","country","name","admin","ADMIN","NAME"]:
        if cand in df.columns:
            return cand
    return None

# ========= 分类与自动参数 =========

def classify_sloped(s, f, s1, s2, s3, F0, b, band, escaping=False):
    """基于最新规则的 S–F 分型（矩形区域）：
    - S ≤ s1: Fragile regimes（极低稳定，脆弱区，非陷阱）
    - -1 < S < 0:
        * F > 0   → Near Trap（左上：稳定性偏低但反馈较强，可望“爬出”）
        * F ≤ 0   → Rigid Trap（左下：中等稳定 + 弱反馈，刚性陷阱）
    - S ≥ 0:
        * F ≥ 0  → Stable Adaptive（右上：高稳定 + 强反馈，稳健可调）
        * F < 0  → Near Trap（右下：高稳定 + 中等反馈，临近陷阱区，可上可下）

    escaping=1 时，对 S < 0 且 F ≥ 0 的国家额外标记为 Escaping。
    F0, b, band 仅用于记录参考斜线阈值，不再参与分类。
    """
    # 仍然计算一次参考阈值，便于导出表格，但不用于分类
    thresh = F0 + b * s

    if s <= s1:
        cls = "Fragile regimes"
    elif s < 0.0:
        # -1 < S < 0
        if f > 0.0:
            cls = "Near Trap"
        else:
            cls = "Rigid Trap"
    else:
        # S ≥ 0
        if f >= 0.0:
            cls = "Stable Adaptive"
        else:
            cls = "Near Trap"

    if escaping and (s < 0.0) and (f >= 0.0):
        cls = "Escaping"
    return cls, thresh


def auto_pick_F0(df, spec: str):
    if not spec:
        return None
    spec = spec.lower().strip()
    if spec.startswith("p") and spec[1:].isdigit():
        q = int(spec[1:]) / 100.0
        return float(df["F"].quantile(q))
    return None

# ========= 绘制 S–F 散点 =========

def plot_scatter(df, out_dir: Path, year: int, dpi: int,
                 s1, s2, s3, F0, b, band,
                 name_col=None, size_mode="uniform"):

    # 顶刊：宽度固定 183mm，高度按比例
    fig, ax = plt.subplots(figsize=(FIG_WIDTH_INCH, FIG_WIDTH_INCH * SCATTER_ASPECT), dpi=dpi)

    # 阴影：极简浅绿色窄带，只在 F=0 上下 band 范围内，S ∈ [s1, s3]
    ax.fill_between(
        [s1, s3],
        -band,
        band,
        color="#e7f3df",  # 更淡的浅绿色
        alpha=0.55,
        zorder=0,
    )

    # 边界线：S = s1, S = s3 使用中性灰色虚线；F = 0 使用略深的红色虚线（从 S=s1 到右侧边界）
    boundary_v = dict(color="#7f7f7f", lw=0.9, ls="--", zorder=1.5)
    boundary_h = dict(color="#d73027", lw=1.0, ls="--", zorder=1.5)
    ax.axvline(s1, **boundary_v)
    ax.axvline(s3, **boundary_v)

    if size_mode == "abs":
        sizes = 40 + 25 * (df["S"].abs() / (df["S"].abs().max() or 1.0))
    elif size_mode == "pos":
        Spos = np.maximum(df["S"], 0.0)
        sizes = 40 + 25 * (Spos / (Spos.max() or 1.0))
    else:
        sizes = np.full(len(df), 48.0)

    for c in CLASS_ORDER:
        mask = df["regimeA"] == c
        if mask.any():
            ax.scatter(
                df.loc[mask, "S"], df.loc[mask, "F"],
                s=sizes[mask] * 0.85,
                alpha=0.95,
                c=CLASS_COLORS.get(c, "#cccccc"),
                marker=CLASS_MARKERS.get(c, "o"),
                edgecolor="white",
                linewidth=0.55,
                label=f"{c} ({int(mask.sum())})",
                zorder=3,
            )

    lab = "iso3c"
    df_sorted = df.sort_values(["S", "F"]).reset_index(drop=True)

    for idx, (_, r) in enumerate(df_sorted.iterrows()):
        if idx % 2 == 0:
            dy = 6
            va = "bottom"
        else:
            dy = -7
            va = "top"
        if str(r[lab]) == "USA":
            dy = -6
            va = "top"

        ax.annotate(
            str(r[lab]),
            (r["S"], r["F"]),
            textcoords="offset points",
            xytext=(0, dy),
            fontsize=7.0,
            color="#666666",
            ha="center",
            va=va,
        )

    # 轴标签 & 刻度字体
    ax.tick_params(axis="both", which="major", labelsize=9)
    ax.set_xlabel("Stability (S)", fontsize=10)
    ax.set_ylabel("Feedback (F)", fontsize=10)

    # 只画水平浅灰网格线
    ax.grid(axis="y", color="#e0e0e0", linewidth=0.6, linestyle="-", zorder=0)

    # 图例放在图内右下角
    ax.legend(
        frameon=True,
        ncol=1,
        loc="lower right",
        bbox_to_anchor=(0.97, 0.04),
        framealpha=0.95,
        edgecolor="#cccccc",
        fontsize=8,
    )

    x_margin = (df["S"].max() - df["S"].min()) * 0.05
    y_margin = (df["F"].max() - df["F"].min()) * 0.08
    ax.set_xlim(df["S"].min() - x_margin, df["S"].max() + x_margin)
    ax.set_ylim(df["F"].min() - y_margin, df["F"].max() + y_margin)

    # 横向边界线：从 S=s1 到当前 x 轴最大值
    x_right = ax.get_xlim()[1]
    ax.hlines(0.0, s1, x_right, **boundary_h)

    plt.subplots_adjust(left=0.09, right=0.80, bottom=0.12, top=0.97)

    # 输出 PNG + SVG
    png_path = out_dir / f"Fig_SF_scatter_schemeA_v2_{year}.png"
    svg_path = out_dir / f"Fig_SF_scatter_schemeA_v2_{year}.svg"
    plt.savefig(png_path, dpi=dpi, bbox_inches="tight")
    plt.savefig(svg_path, dpi=dpi, bbox_inches="tight", format="svg")
    plt.close()
    return str(png_path), str(svg_path)

# ========= 地图 =========

def _load_world_offline(maps_zip: str):
    import geopandas as gpd

    def _read(p):
        return gpd.read_file(p)

    if maps_zip and Path(maps_zip).exists():
        try:
            world = _read(maps_zip)
        except Exception:
            world = _read(gpd.datasets.get_path("naturalearth_lowres"))
    else:
        world = _read(gpd.datasets.get_path("naturalearth_lowres"))

    up = {c.upper(): c for c in world.columns}
    if "ADM0_A3" in up:
        world["iso3"] = world[up["ADM0_A3"]]
    elif "ISO_A3" in up:
        world["iso3"] = world[up["ISO_A3"]]
    elif "ISO_A3_EH" in up:
        world["iso3"] = world[up["ISO_A3_EH"]]
    elif "ISO3" in up:
        world["iso3"] = world[up["ISO3"]]
    else:
        world["iso3"] = world.get("iso_a3", None)

    name_col = None
    for cand in ["ADMIN", "NAME", "name"]:
        if cand in world.columns:
            name_col = cand
            break
    if name_col is not None:
        world = world[world[name_col].astype(str).str.lower() != "antarctica"].copy()

    if world.crs is None:
        world.set_crs(4326, inplace=True)
    else:
        world = world.to_crs(4326)

    try:
        world["geometry"] = world.buffer(0)
    except Exception:
        pass

    if "iso3" in world.columns:
        world = world.drop_duplicates(subset=["iso3"]).copy()

    return world

def plot_world_map(world, df, out_dir: Path, year: int, dpi: int):
    """全球分型地图。"""
    import matplotlib.patches as mpatches

    merged = world.merge(
        df[["iso3c", "regimeA"]].rename(columns={"iso3c": "iso3"}),
        on="iso3",
        how="left",
    )
    merged["__color__"] = merged["regimeA"].map(CLASS_COLORS).fillna(CLASS_COLORS["No Data"])

    # 图宽 183mm，高按比例
    fig, ax = plt.subplots(figsize=(FIG_WIDTH_INCH, FIG_WIDTH_INCH * MAP_ASPECT), dpi=dpi)
    merged.plot(
        color=merged["__color__"],
        edgecolor="#FFFFFF",
        linewidth=0.35,
        ax=ax,
    )
    ax.set_axis_off()

    # 不在图内使用大标题，交给 caption 处理
    # ax.set_title(...)

    present = [c for c in CLASS_ORDER if c in merged["regimeA"].unique()]
    handles = [
        mpatches.Patch(
            color=CLASS_COLORS[c],
            label=f"{c} ({(merged['regimeA'] == c).sum()})",
        )
        for c in present
    ]
    if merged["regimeA"].isna().any():
        handles.append(
            mpatches.Patch(
                color=CLASS_COLORS["No Data"],
                label="No Data",
            )
        )

    # 图例：一行，在下方空白一行
    ax.legend(
        handles=handles,
        ncol=len(handles),
        loc="lower center",
        bbox_to_anchor=(0.5, -0.1),
        frameon=True,
        framealpha=1.0,
        edgecolor="#cccccc",
        fontsize=7,
    )
    # 下边距再加大一点，给 legend 腾空间
    plt.subplots_adjust(bottom=0.1, top=0.99, left=0.01, right=0.99)

    # 输出 PNG + SVG
    png_path = out_dir / f"Fig_boundary_world_schemeA_v2_{year}.png"
    svg_path = out_dir / f"Fig_boundary_world_schemeA_v2_{year}.svg"
    plt.savefig(png_path, dpi=dpi, bbox_inches="tight")
    plt.savefig(svg_path, dpi=dpi, bbox_inches="tight", format="svg")
    plt.close()
    return str(png_path), str(svg_path)

# ========= 主函数 =========

def main():
    args = build_args()
    out_dir = ensure_outdir(args.out_dir)

    df = pd.read_csv(args.panel_path)
    df = df[df["year"] == args.year].copy()
    if df.empty:
        raise SystemExit(f"No data for year {args.year} in {args.panel_path}")

    country_col = args.country_col if args.country_col in df.columns else "iso3c"
    s_col, f_col = infer_SF(df, args.s_col, args.f_col)
    name_col = pick_name_column(df)

    df = df.rename(columns={country_col: "iso3c", s_col: "S", f_col: "F"})

    s1, s2, s3 = parse_cutS(args.cutS)
    F0_auto = auto_pick_F0(df, args.autoF) if args.autoF else None
    F0 = F0_auto if F0_auto is not None else args.F0

    regimes, thresholds = [], []
    for s, f in zip(df["S"], df["F"]):
        cls, thr = classify_sloped(s, f, s1, s2, s3, F0, args.b, args.band, escaping=bool(args.escaping))
        regimes.append(cls)
        thresholds.append(thr)
    df["regimeA"] = regimes
    df["F_threshold"] = thresholds

    year0 = int(df["year"].iloc[0])
    table_path = out_dir / f"classification_schemeA_v2_{year0}.csv"
    keep_cols = ["iso3c", "S", "F", "F_threshold", "regimeA"]
    if name_col:
        df.rename(columns={name_col: "country_name"}, inplace=True)
        keep_cols = ["iso3c", "country_name", "S", "F", "F_threshold", "regimeA"]
    df[keep_cols].to_csv(table_path, index=False)
    log.info("[OUT] %s", table_path)

    scatter_png, scatter_svg = plot_scatter(
        df,
        out_dir,
        year0,
        args.dpi,
        s1,
        s2,
        s3,
        F0,
        args.b,
        args.band,
        name_col=("country_name" if "country_name" in df.columns else None),
        size_mode=args.size_mode,
    )

    world = _load_world_offline(args.maps_zip)
    map_png, map_svg = plot_world_map(world, df, out_dir, year0, args.dpi)

    log.info("[DONE] {'table': '%s', 'scatter_png': '%s', 'scatter_svg': '%s', 'map_png': '%s', 'map_svg': '%s'}",
             table_path, scatter_png, scatter_svg, map_png, map_svg)

if __name__ == "__main__":
    main()
