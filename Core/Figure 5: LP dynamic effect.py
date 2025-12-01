# -*- coding: utf-8 -*-
"""
M3 · Jordà Local Projections — paper-ready
绝对路径版（数据/γ* 固定，统一输出目录）
"""

import warnings, re
warnings.filterwarnings("ignore")

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from linearmodels import PanelOLS

# ========= 绝对路径（按需改动） =========
ENRICHED = "/Users/MaJun/PycharmProjects/Python(Ajou University)/.venv/enriched_panel.csv"
GAMMA_FILE = "/Users/MaJun/PycharmProjects/Python(Ajou University)/The Stability Trap Hypothesis/二稿代码/M1-M7/scripts/core/V3.3_M2_outputs/threshold_point_ci.txt"
# 统一输出目录：图和表都写在这里
OUT_DIR = "/Users/MaJun/PycharmProjects/Python(Ajou University)/The Stability Trap Hypothesis/终稿代码/Figure/Figure5:LP 动态效应"

# ========= 运行参数 =========
H_NEG: List[int] = [-3, -2, -1]
H_POS: List[int] = [0, 1, 2, 3, 4, 5]
MIN_SIDE_N: int = 30
WINSOR_P: float = 0.01
COMMON_SAMPLE: bool = True     # True = 公共样本（更严），False = 按每个 h 取样
AUTO_FALLBACK: bool = True     # 公共样本若空则自动回退
GAMMA_FALLBACK: float = -0.7604  # 找不到文件时的兜底
THRESH_BY = "S"   # "S" 表示用 S1 切分；若以后要按 F 切分，就改为 "F"

# ========= 输出准备 =========
OUT = Path(OUT_DIR)
OUT.mkdir(parents=True, exist_ok=True)

# ========= 工具函数 =========
def _first_float(s: str) -> float:
    m = re.search(r"[-+]?\d*\.\d+|[-+]?\d+", s)
    if not m:
        raise ValueError(f"无法从文本解析数值：{s}")
    return float(m.group(0))

def read_gamma_star() -> Tuple[float, str]:
    """优先读取给定路径；若不存在，尝试 CI 大写；都无则回退到固定值。"""
    p = Path(GAMMA_FILE)
    if p.exists():
        g = _first_float(p.read_text(encoding="utf-8").splitlines()[0])
        print(f"[LP] gamma* 来源：{p} -> {g:.6f}")
        return g, ""
    alt = p.with_name("threshold_point_CI.txt")
    if alt.exists():
        g = _first_float(alt.read_text(encoding="utf-8").splitlines()[0])
        print(f"[LP] gamma* 来源：{alt} -> {g:.6f}")
        return g, ""
    print(f"[LP] WARN: 未找到 {p} 或 {alt}，回退到固定 gamma*={GAMMA_FALLBACK}")
    return GAMMA_FALLBACK, "_forced"

def winsorize_inplace(df: pd.DataFrame, cols: List[str], p: float = 0.01) -> None:
    for c in cols:
        x = df[c].astype(float)
        lo, hi = np.nanpercentile(x, [100*p, 100*(1-p)])
        df[c] = x.clip(lo, hi)

def load_enriched() -> pd.DataFrame:
    p = Path(ENRICHED)
    if not p.exists():
        raise FileNotFoundError(f"找不到 enriched_panel.csv：{p}")
    df = pd.read_csv(p)
    if "year" in df.columns:
        df["year"] = df["year"].astype(int)

    # y: GDP 增长
    if "g" not in df.columns:
        for c in ["g", "WDI_GDP_Growth", "GDP_Growth", "growth", "g_it"]:
            if c in df.columns:
                df["g"] = df[c]
                break
    assert "g" in df.columns, "未找到 GDP 增长列 g"

    # 控制项
    if "WDI_GDPpc" in df.columns and "log_gdppc_lag" not in df.columns:
        df["log_gdppc"] = np.log(df["WDI_GDPpc"].replace({0: np.nan}))
        df["log_gdppc_lag"] = df.groupby("country")["log_gdppc"].shift(1)

    if "pi" not in df.columns:
        for c in ["WDI_Inflation", "inflation", "pi"]:
            if c in df.columns:
                df["pi"] = df[c]
                break
    assert "pi" in df.columns, "缺少通胀列 pi"
    df["pi2"] = df["pi"] ** 2

    # 指标口径
    assert "S_PCA" in df.columns, "缺少 S_PCA"
    assert "F_PCA_main" in df.columns, "缺少 F_PCA_main"
    df["S1"] = df["S_PCA"].shift(1)
    df["F1"] = df["F_PCA_main"].shift(1)

    # 轻微稳健处理
    winsorize_inplace(df, ["g", "pi"], p=WINSOR_P)
    df["pi2"] = df["pi"] ** 2
    return df

def y_for_h(df: pd.DataFrame, h: int) -> pd.DataFrame:
    if h >= 0:
        y = df.groupby("country")["g"].shift(-h)
    else:
        y = df.groupby("country")["g"].shift(abs(h))
    return df[["country", "year"]].assign(y=y.values)

def build_X(tmp: pd.DataFrame, gamma: float) -> pd.DataFrame:
    split_var = tmp["S1"] if THRESH_BY == "S" else tmp["F1"]
    return pd.DataFrame({
        "const": 1.0,
        "_S_low":  tmp["S1"] * (split_var <  gamma),
        "_S_high": tmp["S1"] * (split_var >= gamma),
        "log_gdppc_lag": tmp["log_gdppc_lag"].values,
        "pi":            tmp["pi"].values,
        "pi2":           tmp["pi2"].values,
    }, index=tmp.index)

def fit_panel(y: pd.Series, X: pd.DataFrame) -> Tuple[float, float, float]:
    mod = PanelOLS(y, X, entity_effects=True, time_effects=True)
    clusters = pd.Series(X.index.get_level_values(0), index=X.index, name="country")
    res = mod.fit(cov_type="clustered", clusters=clusters)
    bL = float(res.params.get("_S_low",  np.nan))
    bH = float(res.params.get("_S_high", np.nan))
    cov = res.cov
    var_diff = cov.loc["_S_high", "_S_high"] + cov.loc["_S_low", "_S_low"] - 2*cov.loc["_S_high", "_S_low"]
    se_diff = float(np.sqrt(max(var_diff, 0.0)))
    return bL, bH, se_diff

def nested_masks(wide: pd.DataFrame, y_bank: dict, hs: List[int]) -> dict:
    masks, mask = {}, None
    for h in sorted(hs):
        tmp = wide.merge(y_bank[h], on=["country", "year"], how="left")
        valid = tmp[["y","S1","F1","log_gdppc_lag","pi","pi2"]].notna().all(axis=1)
        mask = valid if mask is None else (mask & valid)
        masks[h] = mask.copy()
    return masks

def run_lp(df: pd.DataFrame, gamma: float, suffix: str = "") -> None:
    print(f"[LP] gamma* = {gamma:.6f}")
    base = ["country","year","S1","F1","log_gdppc_lag","pi","pi2","g"]
    wide = df[base].copy()
    H = H_NEG + H_POS

    y_bank = {h: y_for_h(df, h) for h in H}

    def fit_one(tmp: pd.DataFrame, h: int, diag: list):
        F1_eff = tmp["F1"]
        nL, nR = int((F1_eff < gamma).sum()), int((F1_eff >= gamma).sum())
        N = len(tmp)
        if min(nL, nR) < MIN_SIDE_N:
            diag.append((h, N, nL, nR, 0, "minN"))
            return (h, np.nan, np.nan, np.nan, np.nan, np.nan, N)
        try:
            bL, bH, se = fit_panel(tmp["y"], build_X(tmp, gamma))
            diag.append((h, N, nL, nR, 1, "ok"))
            return (h, bL, bH, bH-bL, np.nan, se, N)
        except Exception as e:
            diag.append((h, N, nL, nR, 0, f"err:{type(e).__name__}"))
            return (h, np.nan, np.nan, np.nan, np.nan, np.nan, N)

    def _estimate(common=True):
        rows, diag = [], []
        if common:
            # 预期窗口公共样本
            if H_NEG:
                mask_pre = np.ones(len(wide), dtype=bool)
                for h in H_NEG:
                    tmp = wide.merge(y_bank[h], on=["country","year"], how="left")
                    valid = tmp[["y","S1","F1","log_gdppc_lag","pi","pi2"]].notna().all(axis=1)
                    mask_pre &= valid
                for h in H_NEG:
                    tmp = wide.loc[mask_pre].merge(y_bank[h], on=["country","year"], how="left")
                    tmp = tmp.dropna(subset=["y","S1","F1","log_gdppc_lag","pi","pi2"]).set_index(["country","year"]).sort_index()
                    rows.append(fit_one(tmp, h, diag))
            # 事后窗口嵌套样本
            if H_POS:
                masks_post = nested_masks(wide, y_bank, H_POS)
                for h in H_POS:
                    tmp = wide.loc[masks_post[h]].merge(y_bank[h], on=["country","year"], how="left")
                    tmp = tmp.dropna(subset=["y","S1","F1","log_gdppc_lag","pi","pi2"]).set_index(["country","year"]).sort_index()
                    rows.append(fit_one(tmp, h, diag))
        else:
            # 每个 h 单独取样
            for h in H:
                tmp = wide.merge(y_bank[h], on=["country","year"], how="left")
                tmp = tmp.dropna(subset=["y","S1","F1","log_gdppc_lag","pi","pi2"]).set_index(["country","year"]).sort_index()
                if tmp.empty:
                    diag.append((h, 0, 0, 0, 0, "empty"))
                    rows.append((h, np.nan, np.nan, np.nan, np.nan, np.nan, 0))
                else:
                    rows.append(fit_one(tmp, h, diag))

        out = pd.DataFrame(rows, columns=["h","beta_low","beta_high","diff","se_low","se_diff","N"]).sort_values("h")
        diag = pd.DataFrame(diag, columns=["h","N","N_left","N_right","estimated","note"]).sort_values("h")
        return out, diag

    out, diag = _estimate(common=COMMON_SAMPLE)
    if AUTO_FALLBACK and COMMON_SAMPLE and (
        out[out["h"]>=0]["N"].fillna(0).sum()==0 or
        diag.query("h==0 and estimated==0").shape[0]>0
    ):
        print("[LP] post window empty under common-sample; fallback to per-h sample...")
        out, diag = _estimate(common=False)

    # ===== 导出（CSV） =====
    out.to_csv(OUT / f"lp_irf{suffix}.csv", index=False)
    diag.to_csv(OUT / f"lp_diag{suffix}.csv", index=False)
    print("DONE CSV ->", OUT / f"lp_irf{suffix}.csv", "|", OUT / f"lp_diag{suffix}.csv")

    # ===== 文本摘要 =====
    out["t_diff"] = out["diff"] / out["se_diff"]
    pre, post = out[out["h"]<0], out[out["h"]>=0]
    pre_sig_cnt = int((pre["t_diff"].abs()>=1.96).sum())
    pre_max_abs = float(pre["diff"].abs().max()) if not pre.empty else np.nan

    pos = post[(post["diff"]>0) & (post["t_diff"].abs()>=1.96)]
    runs, run, last = [], [], None
    pos_set = set(pos["h"].tolist())
    for h in post["h"].tolist():
        if h in pos_set:
            if last is None or h==last+1:
                run.append(int(h)); last=int(h)
            else:
                runs.append(run); run=[int(h)]; last=int(h)
        else:
            if run:
                runs.append(run); run=[]; last=None
    if run:
        runs.append(run)
    if not post["diff"].isna().all():
        idx = int(post["diff"].idxmax())
        peak_h  = int(post.loc[idx,"h"])
        peak_val= float(post.loc[idx,"diff"])
        peak_t  = float(post.loc[idx,"t_diff"])
    else:
        peak_h = peak_val = peak_t = np.nan

    (OUT / f"lp_summary{suffix}.txt").write_text(
        "LP summary\n"
        f"gamma* = {gamma:.6f}\nCOMMON_SAMPLE = {COMMON_SAMPLE}\nMIN_SIDE_N = {MIN_SIDE_N}\n"
        f"H_NEG = {H_NEG} | H_POS = {H_POS}\n"
        f"Pre-trend: sig_count(h<0)={pre_sig_cnt}, max|Δ_h|={pre_max_abs:.4f}\n"
        f"Significant positive windows: {runs}\n"
        f"Peak Δ at h={peak_h}: {peak_val:.4f} (t={peak_t:.2f})\n",
        encoding="utf-8"
    )

    # ===== 作图（183mm 宽 · dpi=400 · PNG+SVG） =====
    def _band(y, se, mult=1.96):
        return (y - mult*se, y + mult*se)

    dd = out.set_index("h").sort_index()

    # 统一基础字体
    plt.rcParams.update({
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8
    })

    c_low  = "#1f77b4"
    c_high = "#ff7f0e"
    c_band = "#999999"
    c_diff = "#000000"

    # === 主图：β^L_h, β^H_h 和 Δ_h 带（主文图） ===
    fig, ax = plt.subplots(figsize=(7.2, 4.3))  # 约 183mm × 110mm

    ax.plot(
        dd.index, dd["beta_low"],
        marker="o", linestyle="-", color=c_low,
        linewidth=0.9, markersize=3.5,
        label=r"Low feedback, $\beta^L_h$"
    )
    ax.plot(
        dd.index, dd["beta_high"],
        marker="s", linestyle="-", color=c_high,
        linewidth=0.9, markersize=3.5,
        label=r"High feedback, $\beta^H_h$"
    )

    if dd["se_diff"].notna().any():
        lo, hi = _band(dd["diff"], dd["se_diff"])
        ax.fill_between(
            dd.index, lo, hi,
            color=c_band, alpha=0.18, linewidth=0.0,
            label="95% CI"
        )
        ax.plot(
            dd.index, dd["diff"],
            marker="o", linestyle="-", color=c_diff,
            linewidth=0.9, markersize=3.5,
            label=r"$\Delta_h$")

    ax.axvline(0, linestyle="--", linewidth=0.8, color="black", alpha=0.6)
    ax.axhline(0, linestyle="--", linewidth=0.8, color="black", alpha=0.6)

    ax.set_xlabel("Horizon h")
    ax.set_ylabel("Marginal effect of S on growth", labelpad=5)
    ax.yaxis.set_label_coords(-0.055, 0.5)
    ax.tick_params(axis="both", which="both", length=3)

    ax.legend(loc="lower left", frameon=False)
    # 底部留白略收（约“半行”）
    fig.subplots_adjust(left=0.09, right=0.985, top=0.975, bottom=0.10)

    png_path = OUT / f"lp_irf_S_effect{suffix}.png"
    fig.savefig(png_path, dpi=400)
    fig.savefig(png_path.with_suffix(".svg"))
    plt.close(fig)

    # === 预趋势图（附录） ===
    predd = dd.loc[[h for h in dd.index if h in H_NEG]].copy()
    fig2, ax2 = plt.subplots(figsize=(7.2, 3.6))

    if not predd.empty:
        ax2.plot(
            predd.index, predd["beta_low"],
            marker="o", linestyle="-", color=c_low,
            linewidth=0.9, markersize=3.5,
            label=r"$\beta^L_h$ (h<0)"
        )
        ax2.plot(
            predd.index, predd["beta_high"],
            marker="s", linestyle="-", color=c_high,
            linewidth=0.9, markersize=3.5,
            label=r"$\beta^H_h$ (h<0)"
        )
        if predd["se_diff"].notna().any():
            lo2, hi2 = _band(predd["diff"], predd["se_diff"])
            ax2.fill_between(
                predd.index, lo2, hi2,
                color=c_band, alpha=0.18, linewidth=0.0,
                label="95% CI"
            )
            ax2.plot(
                predd.index, predd["diff"],
                marker="o", linestyle="-", color=c_diff,
                linewidth=0.85, markersize=3.5,
                label=r"$\Delta_h$ (h<0)"
            )

    ax2.axhline(0, linestyle="--", linewidth=0.8, color="black", alpha=0.6)
    ax2.set_xlabel("Horizon h (pre)")
    ax2.set_ylabel("Effect")
    ax2.tick_params(axis="both", which="both", length=3)
    ax2.legend(loc="upper right", frameon=False)
    fig2.subplots_adjust(left=0.09, right=0.985, top=0.975, bottom=0.18)

    pre_png = OUT / f"lp_pretrend_zoom{suffix}.png"
    fig2.savefig(pre_png, dpi=400)
    fig2.savefig(pre_png.with_suffix(".svg"))
    plt.close(fig2)

    # === 仅 Δ_h 图（附录差值图） ===
    fig3, ax3 = plt.subplots(figsize=(7.2, 4.0))

    if dd["se_diff"].notna().any():
        lo3, hi3 = _band(dd["diff"], dd["se_diff"])
        ax3.fill_between(
            dd.index, lo3, hi3,
            color=c_band, alpha=0.18, linewidth=0.0
        )

    ax3.plot(
        dd.index, dd["diff"],
        marker="o", linestyle="-", color=c_diff,
        linewidth=0.9, markersize=3.5
    )
    ax3.axhline(0, linestyle="--", linewidth=0.8, color="black", alpha=0.6)
    ax3.axvline(0, linestyle="--", linewidth=0.8, color="black", alpha=0.6)

    ax3.set_xlabel("Horizon h")
    ax3.set_ylabel(r"$\Delta_h$")
    ax3.tick_params(axis="both", which="both", length=3)
    fig3.subplots_adjust(left=0.09, right=0.985, top=0.975, bottom=0.14)

    diff_png = OUT / f"lp_irf_diff_clean{suffix}.png"
    fig3.savefig(diff_png, dpi=400)
    fig3.savefig(diff_png.with_suffix(".svg"))
    plt.close(fig3)

    print("DONE FIG ->", png_path, "|", pre_png, "|", diff_png)

def main():
    gamma, suffix = read_gamma_star()
    df = load_enriched()
    run_lp(df, gamma, suffix)

if __name__ == "__main__":
    main()
