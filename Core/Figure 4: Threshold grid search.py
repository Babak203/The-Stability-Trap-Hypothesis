# -*- coding: utf-8 -*-
"""
补丁V3.3  M2：两段“斜率断点” + 控制 + 国家聚类稳健SE + 投稿表格/稳健性打包
"""

import os, sys, warnings, bisect
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=RuntimeWarning)
plt.switch_backend("Agg")


# ============ 配置 ============
@dataclass
class Config:
    data_path: str = "/Users/MaJun/PycharmProjects/Python(Ajou University)/.venv/enriched_panel.csv"

    y_col: Optional[str] = None
    q_col: Optional[str] = None
    group_col: Optional[str] = None          # 推荐 "iso3c" 或 "country"
    controls: List[str] = None

    # 阈值搜索参数
    min_per_side: int = 25
    coarse_q_low: float = 0.10
    coarse_q_high: float = 0.90
    coarse_grid_n: int = 60
    refine_window_frac: float = 0.10
    refine_grid_n: int = 60

    jitter_std: float = 1e-10
    add_const: bool = True

    # 协方差
    use_cluster: bool = True
    hc_type: str = "HC1"

    # 稳健性
    use_jackknife: bool = True
    use_bootstrap: bool = False
    bootstrap_B: int = 200
    bootstrap_seed: int = 2025

    # 输出目录
    out_dir: str = os.path.join(os.path.dirname(__file__), "Figure4:阈值 grid search")
    random_seed: int = 42


CFG = Config()
if CFG.controls is None:
    CFG.controls = []


# ============ 列名识别 ============
Y_ALIASES = ["g", "growth", "gdp_growth", "gdppc_growth", "gdp_pc_growth", "NY.GDP.PCAP.KD.ZG"]
Q_ALIASES = ["Q", "S", "stability_score", "Stability_Score", "Stability_Score_0_1",
             "WGI_Composite", "WGI_stability_index", "S_index", "S_level"]
G_ALIASES = ["country", "iso3c", "economy", "code", "ccode", "country_id", "nation", "iso"]
Y_HINT = ["growth", "gdp", "pcap", "per_capita", "zg"]
Q_HINT = ["stab", "wgi", "composite", "index", "stability", "gov", "quality"]
G_HINT = ["country", "iso", "nation", "state", "economy", "code"]
CTRL_HINT = [
    ("investment", ["inv", "invest", "capital", "formation", "gcf"]),
    ("inflation",  ["infl", "cpi", "ppi", "prices"]),
    ("trade",      ["trade", "open", "export", "import"]),
    ("popg",       ["popg", "population", "pop_growth"]),
]


def _first_existing(df, cand):
    for c in cand:
        if c in df.columns:
            return c
    return None


def _hint_pick(df, tokens):
    scored = []
    for c in df.columns:
        name = str(c).lower()
        score = sum(t in name for t in tokens)
        if score > 0 and pd.api.types.is_numeric_dtype(df[c]):
            scored.append((score, c))
    scored.sort(reverse=True)
    return scored[0][1] if scored else None


def resolve_columns(df) -> Tuple[str, str, Optional[str], List[str]]:
    y_col = CFG.y_col if CFG.y_col in df.columns else (_first_existing(df, Y_ALIASES) or _hint_pick(df, Y_HINT))
    q_col = CFG.q_col if CFG.q_col in df.columns else (_first_existing(df, Q_ALIASES) or _hint_pick(df, Q_HINT))
    gcol = CFG.group_col if CFG.group_col in df.columns else (_first_existing(df, G_ALIASES) or _hint_pick(df, G_HINT))
    controls = []
    if CFG.controls:
        for c in CFG.controls:
            if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
                controls.append(c)
    else:
        for _, toks in CTRL_HINT:
            cand = _hint_pick(df, toks)
            if cand and cand not in (y_col, q_col) and cand not in controls:
                controls.append(cand)
        controls = controls[:6]
    if y_col is None or q_col is None:
        print("[columns] preview:", df.columns[:40].tolist())
        miss = []
        if y_col is None:
            miss.append("y_col(增长)")
        if q_col is None:
            miss.append("q_col(阈值)")
        raise KeyError(f"缺少必须列: {', '.join(miss)}。")
    print(f"[columns] y={y_col} | q={q_col} | group={gcol} | controls={controls}")
    return y_col, q_col, gcol, controls


# ============ 读数/清洗 ============
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def read_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"数据文件不存在: {path}")
    return pd.read_csv(path)


def clean_df(df, y_col, q_col, controls):
    df = df.replace([np.inf, -np.inf], np.nan).copy()
    for c in [y_col, q_col] + controls:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    before = len(df)
    df = df.dropna(subset=[y_col, q_col]).copy()
    if CFG.jitter_std > 0:
        rng = np.random.default_rng(CFG.random_seed)
        df[q_col] = df[q_col].to_numpy(dtype=float) + rng.normal(0, CFG.jitter_std, size=len(df))
    print(f"[clean] drop NA/inf: {before - len(df)} | remain {len(df)}")
    return df


# ============ 阈值/设计矩阵 ============
def _feasible_index(q_sorted, gamma, min_side):
    n = q_sorted.size
    if n < 2 * min_side:
        return None
    lo, hi = min_side, n - min_side - 1
    if lo > hi:
        return None
    j = bisect.bisect_left(q_sorted, gamma)
    return min(max(j, lo), hi)


def feasible_gamma(Q, gamma, min_side):
    q = np.sort(Q.dropna().to_numpy())
    j = _feasible_index(q, gamma, min_side)
    return None if j is None else float(q[j])


def build_exog_slope(df, gamma, y_col, q_col, controls):
    Q = df[q_col]
    g_star = feasible_gamma(Q, gamma, CFG.min_per_side)
    if g_star is None:
        raise ValueError("No feasible gamma")
    S = df[q_col].to_numpy()
    S_low = (S * (S < g_star)).reshape(-1, 1)
    S_high = (S * (S >= g_star)).reshape(-1, 1)
    X_parts = [S_low, S_high]
    names = ["S_low", "S_high"]
    for c in controls:
        X_parts.append(df[c].to_numpy().reshape(-1, 1))
        names.append(c)
    X = np.column_stack(X_parts)
    if CFG.add_const:
        X = sm.add_constant(X, has_constant="add")
        names = ["const"] + names
    left = int((Q < g_star).sum())
    right = int((Q >= g_star).sum())
    print(f"[exog] gamma={g_star:.6f} | left={left} right={right} | X={X.shape}")
    return X, names, g_star


def ols_fit(y, X, group=None):
    model = sm.OLS(y, X, missing="drop")
    if CFG.use_cluster and group is not None:
        try:
            return model.fit(cov_type="cluster", cov_kwds={"groups": group})
        except Exception as e:
            print(f"[warn] cluster-robust failed -> fallback to {CFG.hc_type}: {e}")
            return model.fit(cov_type=CFG.hc_type)
    else:
        return model.fit(cov_type=CFG.hc_type)


def slope_diff_tstat(res, names) -> Tuple[float, float, float]:
    i1 = names.index("S_low")
    i2 = names.index("S_high")
    b1 = res.params[i1]
    b2 = res.params[i2]
    V = res.cov_params()
    v11 = V[i1, i1]
    v22 = V[i2, i2]
    v12 = V[i1, i2]
    diff = b2 - b1
    se = np.sqrt(v11 + v22 - 2 * v12) if (v11 > 0 and v22 > 0) else np.nan
    t = diff / se if (se is not None and se > 0) else np.nan

    from math import erf

    def norm_cdf(z):
        return 0.5 * (1 + erf(z / np.sqrt(2)))

    p = 2 * (1 - norm_cdf(abs(t))) if (t == t) else np.nan
    return float(diff), float(t), float(p)


def make_gamma_grid(Q, low, high, n):
    qs = np.linspace(low, high, n)
    cand = np.quantile(Q.dropna().to_numpy(), qs)
    grid = []
    for g in cand:
        gg = feasible_gamma(Q, float(g), CFG.min_per_side)
        if gg is not None:
            grid.append(gg)
    uniq = []
    last = None
    for v in sorted(grid):
        if last is None or abs(v - last) > 1e-15:
            uniq.append(float(v))
            last = float(v)
    if not uniq:
        raise ValueError("No feasible gamma in grid.")
    return uniq


# ============ 搜索/拟合 ============
def grid_search(df, grid, y_col, q_col, controls, group_col):
    rows = []
    y = df[y_col].to_numpy()
    groups = df[group_col] if (group_col and group_col in df.columns) else None
    for g in grid:
        try:
            X, names, g_used = build_exog_slope(df, g, y_col, q_col, controls)
            res = ols_fit(y, X, groups)
            diff, t, p = slope_diff_tstat(res, names)
            rows.append({
                "gamma": float(g_used), "slope_diff": float(diff), "t_diff": float(t), "p_diff": float(p),
                "aic": float(res.aic), "bic": float(res.bic), "rss": float(res.ssr),
                "nobs": int(res.nobs), "r2": float(res.rsquared), "r2_adj": float(res.rsquared_adj)
            })
        except Exception as e:
            rows.append({"gamma": float(g), "slope_diff": np.nan, "t_diff": np.nan,
                         "p_diff": np.nan, "error": str(e)})
    return pd.DataFrame(rows).sort_values("gamma").reset_index(drop=True)


def refine_grid_around_best(df, res_coarse, q_col):
    if "t_diff" not in res_coarse.columns:
        res_coarse["t_diff"] = np.nan
    ok = res_coarse[res_coarse["t_diff"].notna()]
    if ok.empty:
        samp = []
        if "error" in res_coarse.columns:
            samp = res_coarse["error"].dropna().unique().tolist()[:3]
        raise ValueError("Coarse grid all NaN. Samples: %s" % samp)
    best = ok.iloc[ok["t_diff"].abs().argmax()]
    g_best = float(best["gamma"])
    print(f"[refine] coarse best gamma={g_best:.6f}, |t|={abs(best['t_diff']):.3f}")
    Q = df[q_col]
    vals = Q.dropna().to_numpy()
    g_q = (vals <= g_best).mean()
    lo_q = max(g_q - CFG.refine_window_frac, 0.0)
    hi_q = min(g_q + CFG.refine_window_frac, 1.0)
    return make_gamma_grid(Q, lo_q, hi_q, CFG.refine_grid_n)


def fit_at_gamma(df, gamma, y_col, q_col, controls, group_col):
    y = df[y_col].to_numpy()
    groups = df[group_col] if (group_col and group_col in df.columns) else None
    X, names, g_used = build_exog_slope(df, gamma, y_col, q_col, controls)
    res = ols_fit(y, X, groups)
    diff, t, p = slope_diff_tstat(res, names)
    return {"gamma": g_used, "res": res, "names": names,
            "slope_diff": diff, "t_diff": t, "p_diff": p}


# ============ 稳健性 ============
def jackknife_gamma(df, gamma_point, y_col, q_col, controls, group_col):
    rows = []
    for i in range(len(df)):
        dfi = df.drop(df.index[i]).reset_index(drop=True)
        try:
            ft = fit_at_gamma(dfi, gamma_point, y_col, q_col, controls, group_col)
            rows.append({"i_drop": i, "gamma_used": ft["gamma"],
                         "slope_diff": float(ft["slope_diff"]),
                         "t_diff": float(ft["t_diff"]), "p_diff": float(ft["p_diff"])})
        except Exception as e:
            rows.append({"i_drop": i, "error": str(e)})
    return pd.DataFrame(rows)


def bootstrap_gamma(df, gamma_point, y_col, q_col, controls, group_col):
    rng = np.random.default_rng(CFG.bootstrap_seed)
    n = len(df)
    rows = []
    for b in range(CFG.bootstrap_B):
        idx = rng.integers(0, n, size=n)
        dfb = df.iloc[idx].reset_index(drop=True)
        try:
            ft = fit_at_gamma(dfb, gamma_point, y_col, q_col, controls, group_col)
            rows.append({"b": b, "gamma_used": ft["gamma"],
                         "slope_diff": float(ft["slope_diff"]),
                         "t_diff": float(ft["t_diff"]), "p_diff": float(ft["p_diff"])})
        except Exception as e:
            rows.append({"b": b, "error": str(e)})
    return pd.DataFrame(rows)


# ============ 可视化（顶刊版） ============
def plot_grid(res_coarse, res_refine, gamma_point, out_png):
    """Figure 4A：阈值 grid search 图"""
    fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=400)

    # 粗网格 |t|
    if "t_diff" in res_coarse.columns:
        ax.plot(
            res_coarse["gamma"],
            res_coarse["t_diff"].abs(),
            lw=1.0,
            ls="-",
            label="Coarse grid |t|",
        )

    # 细网格 |t|
    if "t_diff" in res_refine.columns:
        ax.plot(
            res_refine["gamma"],
            res_refine["t_diff"].abs(),
            lw=1.0,
            ls="-",
            label="Refined grid |t|",
        )

    # 阈值竖线（实线）
    ax.axvline(
        gamma_point,
        ls="-",
        lw=0.9,
        label=rf"$\gamma^* ({gamma_point:.2f})$",
    )

    ax.set_xlabel(r"Candidate threshold $\gamma$", fontsize=9)
    ax.set_ylabel(r"$|t(\Delta \beta)|$", fontsize=9)
    ax.tick_params(axis="both", labelsize=8)

    ax.legend(frameon=False, fontsize=8, loc="upper right")

    for spine in ax.spines.values():
        spine.set_linewidth(0.6)

    # 你指定的留白
    fig.subplots_adjust(left=0.10, right=0.98, top=0.98, bottom=0.14)

    fig.savefig(out_png)
    base, ext = os.path.splitext(out_png)
    fig.savefig(base + ".svg")
    plt.close(fig)


def plot_marginal_effect(res, names, gamma_point, Smin, Smax, out_png):
    """Figure 4B：边际效应图"""
    i1 = names.index("S_low")
    i2 = names.index("S_high")

    b1 = res.params[i1]
    b2 = res.params[i2]
    V = res.cov_params()
    se1 = np.sqrt(V[i1, i1]) if V[i1, i1] > 0 else np.nan
    se2 = np.sqrt(V[i2, i2]) if V[i2, i2] > 0 else np.nan

    xs = np.linspace(Smin, Smax, 200)
    ys = np.where(xs < gamma_point, b1, b2)
    ses = np.where(xs < gamma_point, se1, se2)

    fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=400)

    ax.plot(xs, ys, lw=1.0, label=r"$\partial g / \partial S$")

    if np.isfinite(ses).all():
        ax.fill_between(
            xs,
            ys - 1.96 * ses,
            ys + 1.96 * ses,
            alpha=0.25,
            label="95% CI",
        )

    ax.axvline(
        gamma_point,
        ls="-",                 # 改成实线
        lw=0.9,
        label=rf"$\gamma^* ({gamma_point:.2f})$",
    )

    ax.set_xlabel(r"Stability $S$", fontsize=9)
    ax.set_ylabel("Marginal effect on growth", fontsize=9)
    ax.tick_params(axis="both", labelsize=8)
    ax.legend(frameon=False, fontsize=8, loc="upper right")

    for spine in ax.spines.values():
        spine.set_linewidth(0.6)

    fig.subplots_adjust(left=0.10, right=0.98, top=0.98, bottom=0.14)

    fig.savefig(out_png)
    base, ext = os.path.splitext(out_png)
    fig.savefig(base + ".svg")
    plt.close(fig)


# ============ 投稿表格导出 ============
def table_descriptives(df, cols, out_csv):
    desc = df[cols].describe(percentiles=[.1, .25, .5, .75, .9]).T
    desc.rename(columns={"mean": "Mean", "std": "Std", "min": "Min",
                         "10%": "P10", "25%": "P25", "50%": "P50",
                         "75%": "P75", "90%": "P90", "max": "Max"}, inplace=True)
    desc.to_csv(out_csv)


def table_m2_baseline(fit, names, gamma_point, out_csv):
    res = fit["res"]
    i1 = names.index("S_low")
    i2 = names.index("S_high")
    rows = []
    for idx, label in [(i1, "S_low (slope below γ)"),
                       (i2, "S_high (slope above γ)")]:
        rows.append({"param": label,
                     "coef": res.params[idx],
                     "se": res.bse[idx],
                     "t": res.tvalues[idx],
                     "p": res.pvalues[idx]})
    rows.append({"param": "Slope diff (high - low)",
                 "coef": fit["slope_diff"], "se": np.nan,
                 "t": fit["t_diff"], "p": fit["p_diff"]})
    out = pd.DataFrame(rows)
    extra = pd.DataFrame([
        {"param": "N", "coef": res.nobs},
        {"param": "R2", "coef": res.rsquared},
        {"param": "gamma*", "coef": gamma_point},
    ])
    out = pd.concat([out, extra], ignore_index=True)
    out.to_csv(out_csv, index=False)


# ============ 主流程 ============
def run_all(df, y_col, q_col, group_col, controls, suffix=""):
    ensure_dir(CFG.out_dir)
    tag = f"{suffix}".strip()

    if CFG.use_cluster and group_col and group_col in df.columns:
        miss = int(df[group_col].isna().sum())
        if miss > 0:
            print(f"[warn] group_col '{group_col}' has {miss} NA -> fill 'UNK'")
            df[group_col] = df[group_col].fillna("UNK")

    desc_csv = os.path.join(CFG.out_dir, f"table_desc{tag}.csv")
    desc_cols = [y_col, q_col] + controls
    table_descriptives(df, desc_cols, desc_csv)
    print("[out]", desc_csv)

    print("[stage] coarse grid …")
    coarse_grid = make_gamma_grid(df[q_col], CFG.coarse_q_low,
                                  CFG.coarse_q_high, CFG.coarse_grid_n)
    res_coarse = grid_search(df, coarse_grid, y_col, q_col, controls, group_col)
    csv1 = os.path.join(CFG.out_dir, f"threshold_grid_search{tag}.csv")
    res_coarse.to_csv(csv1, index=False)
    print("[out]", csv1)

    print("[stage] refine grid …")
    refine_grid = refine_grid_around_best(df, res_coarse, q_col)
    res_refine = grid_search(df, refine_grid, y_col, q_col, controls, group_col)
    csv2 = os.path.join(CFG.out_dir, f"threshold_grid_refined{tag}.csv")
    res_refine.to_csv(csv2, index=False)
    print("[out]", csv2)

    ok = res_refine.dropna(subset=["t_diff"])
    best = ok.iloc[ok["t_diff"].abs().argmax()]
    gamma_point = float(best["gamma"])

    fit = fit_at_gamma(df, gamma_point, y_col, q_col, controls, group_col)
    res = fit["res"]
    names = fit["names"]

    txt = os.path.join(CFG.out_dir, f"threshold_point_ci{tag}.txt")
    with open(txt, "w", encoding="utf-8") as f:
        f.write(f"gamma* = {fit['gamma']:.6f}\n")
        f.write(
            f"(beta_high - beta_low) = {fit['slope_diff']:.6f}, "
            f"t = {fit['t_diff']:.3f}, p = {fit['p_diff']:.4g}\n\n"
        )
        f.write(res.summary().as_text())
    print("[out]", txt)

    det = os.path.join(CFG.out_dir, f"threshold_detail{tag}.csv")
    pd.DataFrame([{
        "gamma_used": fit["gamma"], "nobs": int(res.nobs),
        "r2": float(res.rsquared), "r2_adj": float(res.rsquared_adj),
        "β_low": res.params[names.index("S_low")],
        "β_high": res.params[names.index("S_high")],
        "Δβ": fit["slope_diff"], "t(Δβ)": fit["t_diff"],
        "p(Δβ)": fit["p_diff"],
    }]).to_csv(det, index=False)
    print("[out]", det)

    tbl2 = os.path.join(CFG.out_dir, f"table_M2_baseline{tag}.csv")
    table_m2_baseline(fit, names, gamma_point, tbl2)
    print("[out]", tbl2)

    tbl3 = os.path.join(CFG.out_dir, f"table_M2_robust{tag}.csv")
    table_m2_robust_pack(df, y_col, q_col, group_col,
                         controls, gamma_point, tbl3)
    print("[out]", tbl3)

    png1 = os.path.join(CFG.out_dir, f"threshold_grid_gsr{tag}.png")
    plot_grid(res_coarse, res_refine, gamma_point, png1)
    print("[out]", png1)
    Smin = float(df[q_col].min())
    Smax = float(df[q_col].max())
    png2 = os.path.join(CFG.out_dir, f"marginal_effect{tag}.png")
    plot_marginal_effect(res, names, gamma_point, Smin, Smax, png2)
    print("[out]", png2)

    if CFG.use_jackknife:
        jk = jackknife_gamma(df, gamma_point, y_col, q_col, controls, group_col)
        jk_csv = os.path.join(CFG.out_dir, f"jackknife_gamma{tag}.csv")
        jk.to_csv(jk_csv, index=False)
        print("[out]", jk_csv)

    if CFG.use_bootstrap:
        bs = bootstrap_gamma(df, gamma_point, y_col, q_col, controls, group_col)
        bs_csv = os.path.join(CFG.out_dir, f"bootstrap_gamma{tag}.csv")
        bs.to_csv(bs_csv, index=False)
        print("[out]", bs_csv)

    sum_txt = os.path.join(CFG.out_dir, f"fit_summary{tag}.txt")
    with open(sum_txt, "w", encoding="utf-8") as f:
        f.write(res.summary().as_text())
    print("[out]", sum_txt)

    print("DONE ->",
          os.path.basename(csv1), "|", os.path.basename(csv2), "|",
          os.path.basename(tbl2), "|", os.path.basename(tbl3), "|",
          os.path.basename(png1), "|", os.path.basename(png2))


def table_m2_robust_pack(df, y_col, q_col, group_col,
                          controls, gamma_point, out_csv):
    def winsorize(s, p=0.01):
        lo, hi = s.quantile(p), s.quantile(1 - p)
        return s.clip(lo, hi)

    specs = []
    specs.append(("A: Baseline", dict(
        min_per_side=CFG.min_per_side,
        coarse=(CFG.coarse_q_low, CFG.coarse_q_high),
        w=None,
    )))
    specs.append(("B: min_per_side=20", dict(
        min_per_side=20,
        coarse=(CFG.coarse_q_low, CFG.coarse_q_high),
        w=None,
    )))
    specs.append(("C: Winsorize 1%", dict(
        min_per_side=CFG.min_per_side,
        coarse=(CFG.coarse_q_low, CFG.coarse_q_high),
        w=0.01,
    )))
    specs.append(("D: Coarse 0.05–0.95", dict(
        min_per_side=CFG.min_per_side,
        coarse=(0.05, 0.95),
        w=None,
    )))

    rows = []
    for label, opt in specs:
        df_ = df.copy()
        if opt["w"] is not None:
            df_[y_col] = winsorize(df_[y_col], opt["w"])
            df_[q_col] = winsorize(df_[q_col], opt["w"])

        orig_min = CFG.min_per_side
        orig_low, orig_high = CFG.coarse_q_low, CFG.coarse_q_high
        CFG.min_per_side = opt["min_per_side"]
        CFG.coarse_q_low, CFG.coarse_q_high = opt["coarse"]

        try:
            coarse = make_gamma_grid(
                df_[q_col],
                CFG.coarse_q_low,
                CFG.coarse_q_high,
                max(40, CFG.coarse_grid_n // 2),
            )
            res_c = grid_search(df_, coarse, y_col, q_col, controls, group_col)
            ok = res_c.dropna(subset=["t_diff"])
            if ok.empty:
                raise RuntimeError("no valid grid in spec %s" % label)
            g_best = float(ok.iloc[ok["t_diff"].abs().argmax()]["gamma"])
            fit = fit_at_gamma(df_, g_best, y_col, q_col, controls, group_col)

            rows.append({
                "Spec": label,
                "gamma*": fit["gamma"],
                "β_low": fit["res"].params[fit["names"].index("S_low")],
                "β_high": fit["res"].params[fit["names"].index("S_high")],
                "Δβ": fit["slope_diff"],
                "t(Δβ)": fit["t_diff"],
                "p(Δβ)": fit["p_diff"],
                "R2": fit["res"].rsquared,
                "N": int(fit["res"].nobs),
            })
        except Exception as e:
            rows.append({"Spec": label, "error": str(e)})
        finally:
            CFG.min_per_side = orig_min
            CFG.coarse_q_low, CFG.coarse_q_high = orig_low, orig_high

    pd.DataFrame(rows).to_csv(out_csv, index=False)


def main():
    data_path = CFG.data_path
    suffix = ""
    if len(sys.argv) >= 2 and sys.argv[1].strip():
        data_path = sys.argv[1].strip()
    if len(sys.argv) >= 3 and sys.argv[2].strip():
        suffix = str(sys.argv[2].strip())

    print(f"[data] using {os.path.abspath(data_path)} -> enriched_panel.csv")
    df_raw = read_data(data_path)

    y_col, q_col, group_col, controls = resolve_columns(df_raw)
    df = clean_df(df_raw, y_col, q_col, controls)

    if len(df) < CFG.min_per_side * 2 + 5:
        raise ValueError(f"样本过小：n={len(df)}，降低 min_per_side 或放宽过滤")
    run_all(df, y_col, q_col, group_col, controls, suffix=suffix)


if __name__ == "__main__":
    main()
