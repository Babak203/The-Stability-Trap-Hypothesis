#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 7 · 情景路径（中美越基线 vs 改革） + Table 4（Policy calculator）

- 主面板：enriched_panel.csv
- 若主面板缺少 F，则从 panel_with_FQ.csv 合并 FQ
- 使用 lp_irf.csv 的 LP 系数，构造 5 年预测路径
- 输出：
    Figure7a_China_...png / .svg
    Figure7b_UnitedStates_...png / .svg
    Figure7c_VietNam_...png / .svg
    Table4_policy_calculator_...csv
"""

from __future__ import annotations
import argparse
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ======= 默认路径（按你项目目录来的） =======

DEFAULT_IRF_CSV_CORE = (
    r"/Users/MaJun/PycharmProjects/Python(Ajou University)/"
    r"The Stability Trap Hypothesis/终稿代码/Figure/Figure5:LP 动态效应/lp_irf.csv"
)

DEFAULT_IRF_CSV_DIAG = (
    r"/Users/MaJun/PycharmProjects/Python(Ajou University)/二稿代码/M1-M7/诊断-稳健性与校验/tab/lp_irf.csv"
)

DEFAULT_PANEL_MAIN = (
    r"/Users/MaJun/PycharmProjects/Python(Ajou University)/.venv/enriched_panel.csv"
)

DEFAULT_PANEL_FQ = (
    r"/Users/MaJun/PycharmProjects/Python(Ajou University)/.venv/panel_with_FQ.csv"
)

DEFAULT_DIR = (
    r"/Users/MaJun/PycharmProjects/Python(Ajou University)/"
    r"The Stability Trap Hypothesis/终稿代码/Figure/Figure7:情景路径（中美越基线 vs 改革）"
)

H_LIST = [1, 2, 3, 4, 5]

# 一堆候选列名（和你原脚本保持一致）
COUNTRY_CAND_EXACT = [
    "country", "Country", "COUNTRY",
    "name", "Name", "NAME",
    "entity", "Entity", "ENTITY",
    "economy", "Economy", "ECONOMY",
    "LOCATION",
    "country_name", "Country Name", "COUNTRY_NAME",
    "countryName", "countryname",
    "countryIso3code", "countryiso3code", "CountryIso3Code",
    "iso3c", "ISO3C", "iso3", "ISO3", "ISO_A3", "iso_a3",
    "countrycode", "COUNTRYCODE", "Country Code",
    "country_x", "country_y",
    "国家", "国家名称", "国家或地区", "地区", "经济体", "名称",
]
COUNTRY_FUZZY_KEYS = [
    "country", "entity", "econom", "nation", "location",
    "iso3", "iso_a3", "countrycode", "alpha-3", "alpha3",
    "code3", "iso3code", "国家", "地区", "名称", "经济体",
]

YEAR_CAND_EXACT = [
    "year", "Year", "YEAR",
    "yr", "YR",
    "date", "Date", "DATE",
    "time", "Time", "TIME",
    "年份", "年度",
    "year_x", "year_y",
]
YEAR_FUZZY_KEYS = ["year", "yr", "date", "time", "年份", "年度"]

F_CAND = [
    "FQ", "F_Q", "FQ_x", "FQ_y",
    "F", "F_index", "Fscore",
    "Feedback", "feedback",
    "F_oriented", "F0", "F_current", "F_auto",
    "F_rigid", "F_adaptive",
]

S_CAND = [
    "Delta_Stability_Score",
    "S", "dS", "Delta_S",
    "shock_S", "S_shock",
    "Stability_Delta", "S_value", "S_amp", "Shock",
]

TERM_CAND = ["term", "variable", "var", "name", "Term", "VARIABLE"]
H_CAND = ["horizon", "h", "H", "Horizon"]
BETA_CAND = ["coef", "beta", "b", "estimate", "coefficient", "Beta", "Estimate", "value"]
SE_CAND = ["se", "std_err", "stderr", "std_error", "SE", "StdErr"]
S_TERM_NAMES = ["S", "dS", "shock_S", "S_shock", "Delta_S", "Delta_Stability_Score"]
SF_TERM_NAMES = ["SxF", "S*F", "S:F", "dSxF", "S_inter_F", "S*F_oriented"]

ISO3_LIKE = re.compile(
    r"^(.*iso.*3.*|.*iso_a3.*|.*countryiso3code.*|.*countrycode.*|.*alpha[-_ ]?3.*)$",
    re.IGNORECASE,
)

ISO3_TO_FULL = {
    "CHN": "China",
    "USA": "United States",
    "VNM": "Viet Nam",
    "GBR": "United Kingdom",
    "FRA": "France",
    "DEU": "Germany",
    "JPN": "Japan",
    "KOR": "Korea, Rep.",
    "IND": "India",
    "BRA": "Brazil",
    "RUS": "Russian Federation",
    "CAN": "Canada",
    "AUS": "Australia",
    "ITA": "Italy",
    "ESP": "Spain",
    "MEX": "Mexico",
    "IDN": "Indonesia",
}


# ========= 小工具函数 =========

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _strip_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() if isinstance(c, str) else c for c in df.columns]
    return df


def _pick_col_case_insensitive(df: pd.DataFrame, names) -> str | None:
    cols = list(df.columns)
    low = {c.lower(): c for c in cols if isinstance(c, str)}
    for n in names:
        if isinstance(n, str) and n.lower() in low:
            return low[n.lower()]
    return None


def _pick_col_fuzzy(df: pd.DataFrame, keys) -> str | None:
    cols = [c for c in df.columns if isinstance(c, str)]
    lows = [(c, c.lower()) for c in cols]
    for key in keys:
        k = key.lower()
        for col, low in lows:
            if k in low:
                return col
    return None


def smart_pick_col(df, override, exact_list, fuzzy_list, desc):
    if override:
        direct = override if override in df.columns else _pick_col_case_insensitive(df, [override])
        if direct:
            return direct
        raise KeyError(f"{desc}='{override}' 不在数据列：{list(df.columns)}")

    col = _pick_col_case_insensitive(df, exact_list)
    if col:
        return col
    col = _pick_col_fuzzy(df, fuzzy_list)
    if col:
        return col
    raise KeyError(f"无法识别 {desc}；候选(精准)={exact_list}；现有={list(df.columns)}")


def guess_col(df, cands):
    for c in cands:
        if c in df.columns:
            return c
    col = _pick_col_case_insensitive(df, cands)
    if col:
        return col
    col = _pick_col_fuzzy(df, cands)
    return col


def file_or_first_exist(cands):
    for p in cands:
        if p and Path(p).expanduser().exists():
            return p
    return None


def fetch_irf_arrays(
    irf: pd.DataFrame,
    irf_term_col: str | None,
    irf_h_col: str,
    irf_beta_col: str | None,
    irf_se_col: str | None,
    irf_bS_col: str | None,
    irf_bSF_col: str | None,
    irf_seS_col: str | None,
    irf_seSF_col: str | None,
):
    sub = irf.copy()
    sub[irf_h_col] = pd.to_numeric(sub[irf_h_col], errors="coerce").astype("Int64")

    # long 形式：一列 term
    if (irf_term_col is not None) and (irf_term_col in sub.columns):
        if irf_beta_col is None:
            irf_beta_col = guess_col(sub, BETA_CAND) or BETA_CAND[0]

        bS = (
            sub[sub[irf_term_col].isin(S_TERM_NAMES)]
            .set_index(irf_h_col)[irf_beta_col]
            .reindex(H_LIST)
            .astype(float)
            .values
        )
        bSF = (
            sub[sub[irf_term_col].isin(SF_TERM_NAMES)]
            .set_index(irf_h_col)[irf_beta_col]
            .reindex(H_LIST)
            .astype(float)
            .values
        )

        se_col = irf_se_col or guess_col(sub, SE_CAND)
        if se_col is not None:
            seS = (
                sub[sub[irf_term_col].isin(S_TERM_NAMES)]
                .set_index(irf_h_col)[se_col]
                .reindex(H_LIST)
                .astype(float)
                .values
            )
            seSF = (
                sub[sub[irf_term_col].isin(SF_TERM_NAMES)]
                .set_index(irf_h_col)[se_col]
                .reindex(H_LIST)
                .astype(float)
                .values
            )
        else:
            seS = np.zeros(len(H_LIST))
            seSF = np.zeros(len(H_LIST))

        return np.nan_to_num(bS), np.nan_to_num(bSF), np.nan_to_num(seS), np.nan_to_num(seSF)

    # wide 形式：每个 horizon 一行，bS/bSF 各一列
    irf_bS_col = irf_bS_col or guess_col(sub, ["beta_low", "bS", "beta_S", "S_coef", "coef_S", "low_beta"])
    irf_bSF_col = irf_bSF_col or guess_col(sub, ["beta_high", "bSF", "beta_SF", "SF_coef", "coef_SF", "high_beta"])
    irf_seS_col = irf_seS_col or guess_col(sub, ["se_low", "S_se", "se_bS", "se_S", "low_se"])
    irf_seSF_col = irf_seSF_col or guess_col(sub, ["se_high", "SF_se", "se_bSF", "se_SF", "high_se"])

    if (irf_bS_col is None) or (irf_bSF_col is None):
        raise KeyError("IRF 既无 term，也找不到 bS/bSF 列；需要手动指定。")

    bS = (
        sub.set_index(irf_h_col)[irf_bS_col]
        .reindex(H_LIST)
        .astype(float)
        .values
    )
    bSF = (
        sub.set_index(irf_h_col)[irf_bSF_col]
        .reindex(H_LIST)
        .astype(float)
        .values
    )

    if irf_seS_col:
        seS = (
            sub.set_index(irf_h_col)[irf_seS_col]
            .reindex(H_LIST)
            .astype(float)
            .values
        )
    else:
        seS = np.zeros_like(bS)

    if irf_seSF_col:
        seSF = (
            sub.set_index(irf_h_col)[irf_seSF_col]
            .reindex(H_LIST)
            .astype(float)
            .values
        )
    else:
        seSF = np.zeros_like(bSF)

    return np.nan_to_num(bS), np.nan_to_num(bSF), np.nan_to_num(seS), np.nan_to_num(seSF)


def build_S_path(mode: str, rho: float, pulses: int) -> np.ndarray:
    if mode == "persistent":
        return np.array([rho ** k for k in range(5)], float)
    if mode == "pulses":
        return np.array([1.0 if k < max(1, int(pulses)) else 0.0 for k in range(5)], float)
    # once
    return np.array([1.0, 0.0, 0.0, 0.0, 0.0], float)


def convolve(y_core, se_core, S_path, S_amp):
    y = np.zeros(5, float)
    se = np.zeros(5, float)
    for h in range(1, 6):
        s = 0.0
        v = 0.0
        for k in range(h):
            j = h - 1 - k
            w = S_path[k]
            s += w * y_core[j]
            v += (w * se_core[j]) ** 2
        y[h - 1] = S_amp * s
        se[h - 1] = abs(S_amp) * np.sqrt(v)
    return y, se


def nearest_year_row(panel: pd.DataFrame, country_col: str, year_col: str,
                     country, target_year: int) -> pd.DataFrame:
    sub = panel[panel[country_col] == country].copy()
    if sub.empty:
        return pd.DataFrame()
    le = sub[sub[year_col] <= target_year]
    if not le.empty:
        y = le[year_col].max()
        return sub[sub[year_col] == y].head(1)
    sub["__absdiff"] = (sub[year_col] - target_year).abs()
    sub = sub.sort_values(["__absdiff", year_col])
    return sub.head(1).drop(columns=["__absdiff"], errors="ignore")


def parse_args():
    p = argparse.ArgumentParser(description="Figure7: 情景路径（中美越基线 vs 改革） + Table 4")
    p.add_argument("--countries", default="China;United States;Viet Nam")
    p.add_argument("--year", type=int, default=2023)
    p.add_argument("--F-orient", choices=["rigid-high", "adaptive-high"], default="adaptive-high")
    p.add_argument("--shock-mode", choices=["once", "persistent", "pulses"], default="persistent")
    p.add_argument("--rho", type=float, default=0.6)
    p.add_argument("--pulses", type=int, default=3)
    p.add_argument("--rho-grid", type=str, default="0.5,0.7,0.9")
    p.add_argument("--robust", choices=["worst", "mean"], default="worst")
    p.add_argument("--plot-cum", action="store_true", default=True)
    p.add_argument("--lam", type=float, default=0.93)
    p.add_argument("--gamma", type=float, default=1e-4)
    p.add_argument("--tail-guard", type=int, default=3)
    p.add_argument("--F-grid-step", type=float, default=0.002)
    p.add_argument("--s-ref", choices=["const", "sd5", "sd_all"], default="sd5")
    p.add_argument("--S-amp", type=float, default=1.0)
    p.add_argument("--bp-scale", type=float, default=100.0)

    p.add_argument("--irf-csv", type=str, default=None)
    p.add_argument("--panel-csv", type=str, default=DEFAULT_PANEL_MAIN)
    p.add_argument("--fq-csv", type=str, default=DEFAULT_PANEL_FQ)

    p.add_argument("--country-col", type=str, default=None)
    p.add_argument("--year-col", type=str, default=None)
    p.add_argument("--F-col", type=str, default=None)
    p.add_argument("--S-col", type=str, default=None)

    p.add_argument("--irf-term-col", type=str, default=None)
    p.add_argument("--irf-h-col", type=str, default=None)
    p.add_argument("--irf-beta-col", type=str, default=None)
    p.add_argument("--irf-se-col", type=str, default=None)
    p.add_argument("--irf-bS-col", type=str, default=None)
    p.add_argument("--irf-bSF-col", type=str, default=None)
    p.add_argument("--irf-seS-col", type=str, default=None)
    p.add_argument("--irf-seSF-col", type=str, default=None)

    p.add_argument("--fig-dir", type=str, default=DEFAULT_DIR)
    p.add_argument("--out-dir", type=str, default=DEFAULT_DIR)
    return p.parse_args()


def main():
    warnings.filterwarnings("ignore", category=UserWarning)

    args = parse_args()

    # ====== 找 IRF / Panel 文件 ======
    irf_path = file_or_first_exist(
        [args.irf_csv, DEFAULT_IRF_CSV_CORE, DEFAULT_IRF_CSV_DIAG, "./lp_irf.csv"]
    )
    if irf_path is None:
        raise FileNotFoundError("找不到 IRF CSV (lp_irf.csv)")

    panel_path = file_or_first_exist(
        [args.panel_csv, DEFAULT_PANEL_MAIN, "./enriched_panel.csv"]
    )
    if panel_path is None:
        raise FileNotFoundError("找不到面板 CSV (enriched_panel)")

    irf = _strip_cols(pd.read_csv(irf_path))
    panel = _strip_cols(pd.read_csv(panel_path))

    # IRF 结构
    irf_h_col = smart_pick_col(irf, args.irf_h_col, H_CAND, ["h"], "IRF horizon")
    try:
        tmp = smart_pick_col(irf, args.irf_term_col, TERM_CAND, ["term", "var", "name"], "IRF term")
        irf_term_col = tmp if tmp in irf.columns else None
    except Exception:
        irf_term_col = None

    # 主面板：国家 & 年份列
    pan_country_col = smart_pick_col(
        panel, args.country_col, COUNTRY_CAND_EXACT, COUNTRY_FUZZY_KEYS, "面板 国家"
    )
    pan_year_col = smart_pick_col(
        panel, args.year_col, YEAR_CAND_EXACT, YEAR_FUZZY_KEYS, "面板 年份"
    )

    # 如果 pan_country_col 是 "country" 且看起来像 ISO3，而且有 countryiso3code，则换成后者
    if pan_country_col == "country" and "countryiso3code" in panel.columns:
        vals = panel[pan_country_col].dropna().astype(str)
        if (not vals.empty) and (vals.str.len().eq(3).mean() > 0.9) and (vals.str.isupper().mean() > 0.9):
            pan_country_col = "countryiso3code"

    # F 列
    def try_get_F_col(df: pd.DataFrame) -> str | None:
        if args.F_col and (args.F_col in df.columns):
            return args.F_col
        return guess_col(df, F_CAND)

    F_col_raw = try_get_F_col(panel)

    # 如果主面板没 F，就从 FQ 面板合并
    if F_col_raw is None:
        fq_path = file_or_first_exist([args.fq_csv, DEFAULT_PANEL_FQ])
        if fq_path is None:
            raise KeyError("主面板缺少 F 列，且未找到 FQ 来源。")

        fq = _strip_cols(pd.read_csv(fq_path))
        fq_country_col = smart_pick_col(
            fq, None, COUNTRY_CAND_EXACT, COUNTRY_FUZZY_KEYS, "FQ面板 国家"
        )
        fq_year_col = smart_pick_col(
            fq, None, YEAR_CAND_EXACT, YEAR_FUZZY_KEYS, "FQ面板 年份"
        )
        fq_F_col = guess_col(fq, F_CAND)
        if fq_F_col is None:
            raise KeyError(f"FQ 面板中未找到 F 列；现有列: {list(fq.columns)}")

        def norm_iso(series: pd.Series) -> pd.Series:
            return series.astype(str).str.strip().str.upper()

        def norm_name(series: pd.Series) -> pd.Series:
            return series.astype(str).str.strip()

        main_key = panel[pan_country_col]
        fq_key = fq[fq_country_col]

        if ISO3_LIKE.match(pan_country_col or "") and ISO3_LIKE.match(fq_country_col or ""):
            left_key, right_key = norm_iso(main_key), norm_iso(fq_key)
        else:
            left_key, right_key = norm_name(main_key), norm_name(fq_key)

        left = panel.assign(
            __key=left_key,
            __year=pd.to_numeric(panel[pan_year_col], errors="coerce"),
        )
        right = fq.assign(
            __key=right_key,
            __year=pd.to_numeric(fq[fq_year_col], errors="coerce"),
        )[
            ["__key", "__year", fq_F_col]
        ]

        merged = left.merge(right, on=["__key", "__year"], how="left")
        panel = merged.rename(columns={fq_F_col: "FQ_merge"})
        F_col_raw = "FQ_merge"

    # S 列
    S_col_guess = guess_col(panel, S_CAND)
    if args.S_col and (args.S_col in panel.columns):
        S_col = args.S_col
    elif S_col_guess:
        S_col = S_col_guess
    elif S_CAND[0] in panel.columns:
        S_col = S_CAND[0]
    else:
        S_col = S_CAND[0]
        panel[S_col] = 1.0  # 实在没有就给常数 1

    # F 方向：adaptive-high -> Rigid 高
    if args.F_orient == "adaptive-high":
        s = pd.to_numeric(panel[F_col_raw], errors="coerce")
        lo, hi = np.nanmin(s.values), np.nanmax(s.values)
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            panel["_F_rigid"] = 1.0 - ((s - lo) / (hi - lo))
        else:
            panel["_F_rigid"] = 1.0 - s
        F_col_use = "_F_rigid"
    else:
        F_col_use = F_col_raw

    # 国家列表
    countries_raw = [c.strip() for c in args.countries.split(";") if c.strip()]
    countries = countries_raw[:]
    show_names = countries_raw[:]

    # 如果面板国家列是 ISO3，就把输入的英文名映射成 ISO3，再反向拿全名
    if ISO3_LIKE.match(pan_country_col or ""):
        name_cols = [
            c for c in panel.columns
            if re.search(r"(country|name|econom|国家|地区)", c, re.IGNORECASE)
            and c != pan_country_col
        ]
        mapped = []
        display = []
        if name_cols:
            name_col = sorted(name_cols, key=lambda c: -panel[c].notna().sum())[0]
            look = panel[[name_col, pan_country_col]].dropna().drop_duplicates()
            to_iso = {
                str(a).strip().lower(): str(b).strip().upper()
                for a, b in zip(look[name_col], look[pan_country_col])
            }
            to_name = {
                str(b).strip().upper(): str(a).strip()
                for a, b in zip(look[name_col], look[pan_country_col])
            }
            for nm in countries_raw:
                key = nm.strip().lower()
                iso = to_iso.get(
                    key,
                    {"china": "CHN", "united states": "USA", "viet nam": "VNM", "vietnam": "VNM"}.get(
                        key, nm
                    ),
                )
                mapped.append(iso)
                display.append(ISO3_TO_FULL.get(iso, to_name.get(iso, nm)))
        else:
            for nm in countries_raw:
                iso = {"China": "CHN", "United States": "USA", "Viet Nam": "VNM", "Vietnam": "VNM"}.get(
                    nm, nm
                )
                mapped.append(iso)
                display.append(ISO3_TO_FULL.get(iso, nm))
        countries = mapped
        show_names = display

    # IRF 系数
    bS, bSF, seS, seSF = fetch_irf_arrays(
        irf,
        irf_term_col,
        irf_h_col,
        None,
        None,
        None,
        None,
        None,
        None,
    )

    # ====== 图形与美学设置 ======
    plt.rcParams.update({
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.grid": False,
        "axes.linewidth": 0.8,
    })

    rows = []            # Table 4
    country_results = [] # 存每个国家的路径，用于第二轮画图

    global_y_min, global_y_max = np.inf, -np.inf

    # 振幅
    def s_amplitude(country):
        if args.s_ref == "const":
            return float(args.S_amp)
        sub = panel[panel[pan_country_col] == country].sort_values(by=pan_year_col)
        ser = sub[S_col]
        if args.s_ref == "sd5":
            ser = sub[sub[pan_year_col] >= args.year - 4][S_col]
        val = float(pd.to_numeric(ser, errors="coerce").std(skipna=True))
        return val if np.isfinite(val) and val > 0 else 1.0

    # 给定 F, ρ, S_amp，生成 5 年路径
    def y_for(F, bS, bSF, seS, seSF, rho, S_amp):
        F = float(F)
        y_core = bS + bSF * F
        se_core = np.sqrt(seS**2 + (F**2) * seSF**2)
        S_path = build_S_path(args.shock_mode, rho, args.pulses)
        return convolve(y_core, se_core, S_path, S_amp)

    # ρ-grid
    if args.shock_mode == "persistent" and (args.rho_grid or "").strip():
        rho_list = [float(x) for x in args.rho_grid.split(",")]
    else:
        rho_list = [args.rho]

    lam = float(args.lam)
    gamma = float(args.gamma)
    tail = int(args.tail_guard)
    Hidx = [idx for idx, h in enumerate(H_LIST) if h >= tail]
    H = H_LIST

    # ========= 第 1 轮：算 F* / 路径 / Δ5y，并记下 y 范围 =========
    for country, show_name in zip(countries, show_names):
        pan_now = nearest_year_row(panel, pan_country_col, pan_year_col, country, args.year)
        if pan_now.empty:
            continue

        F0 = float(pd.to_numeric(pan_now[F_col_use], errors="coerce").astype(float).iloc[0])
        S_amp = s_amplitude(country)

        def loss(F):
            F = float(np.clip(F, 0.0, 1.0))
            vals = []
            for rho in rho_list:
                y, _ = y_for(F, bS, bSF, seS, seSF, rho, S_amp)
                neg_tail = float(np.sum(np.maximum(0.0, -y[Hidx])))
                neg_sum = float(-np.sum(y))
                vals.append((1 - lam) * neg_sum + lam * neg_tail)
            agg = max(vals) if args.robust == "worst" else float(np.mean(vals))
            return agg + gamma * (F - F0) ** 2

        step = max(1e-4, float(args.F_grid_step))
        grid = np.linspace(0.0, 1.0, int(1.0 / step) + 1)
        losses = np.array([loss(F) for F in grid])
        F_star = float(grid[np.argmin(losses)])

        # 用 ρ-list 的第一个 ρ 来画主文路径
        y0, se0 = y_for(F0, bS, bSF, seS, seSF, rho_list[0], S_amp)
        y1, se1 = y_for(F_star, bS, bSF, seS, seSF, rho_list[0], S_amp)
        cum0 = np.cumsum(y0)
        cum1 = np.cumsum(y1)

        gap5 = float(np.sum(y1 - y0))          # Δ5y (pp)
        gap5_bp = gap5 * float(args.bp_scale)  # bp，如需

        cur_min = np.min([y0.min(), y1.min(), cum0.min(), cum1.min()])
        cur_max = np.max([y0.max(), y1.max(), cum0.max(), cum1.max()])
        global_y_min = min(global_y_min, cur_min)
        global_y_max = max(global_y_max, cur_max)

        country_results.append({
            "country_code": country,
            "show_name": show_name,
            "F0": F0,
            "F_star": F_star,
            "S_amp": S_amp,
            "y0": y0,
            "y1": y1,
            "se0": se0,
            "se1": se1,
            "cum0": cum0,
            "cum1": cum1,
            "gap5": gap5,
            "gap5_bp": gap5_bp,
        })

        rows.append({
            "country_code": country,
            "country_name": show_name,
            "F0": round(F0, 4),
            "F_star": round(F_star, 4),
            "Delta_F": round(F_star - F0, 4),
            "S_amp": round(S_amp, 4),
            "Delta_5y": round(gap5, 4),
            "Delta_5y_bp": round(gap5_bp, 2),
        })

    if not country_results:
        print("[WARN] 没有可用国家行，未生成图和 Table 4。")
        return

    # 统一 y 轴范围
    y_pad = 0.5
    y_min = global_y_min - y_pad
    y_max = global_y_max + y_pad

    # ========= 第 2 轮：一国一图输出 =========
    fig_dir = Path(args.fig_dir).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    ensure_dir(fig_dir)
    ensure_dir(out_dir)

    tag = (
        f"impact_dual_{args.F_orient}_{args.year}_{args.shock_mode}"
        + ("_rhogrid" if (args.shock_mode == "persistent" and (args.rho_grid or '').strip()) else "")
    )

    letters = ["a", "b", "c", "d", "e", "f"]

    for idx, res in enumerate(country_results):
        fig, ax = plt.subplots(figsize=(7.2, 2.4))  # 183mm 宽，单面板

        ax.axvspan(
            tail - 0.5,
            max(H) + 0.5,
            color="tab:red",
            alpha=0.04,
            label=f"h≥{tail} guard",
        )
        ax.axhline(0, lw=0.8, color="k")

        ax.plot(H, res["y0"], marker="o", lw=1.3, ms=4, label="Baseline")
        ax.plot(H, res["y1"], marker="o", lw=1.3, ms=4, label="Reform")
        ax.plot(H, res["cum0"], "--", lw=1.1, alpha=0.9, label="Baseline (cum.)")
        ax.plot(H, res["cum1"], "--", lw=1.1, alpha=0.9, label="Reform (cum.)")

        ax.set_ylim(y_min, y_max)
        ax.set_xlabel("Horizon (years)")
        ax.set_ylabel("Predicted growth contribution (pp)")

        ax.set_title(f"{res['show_name']}  (Δ5y = {res['gap5']:+.2f} pp)")

        ax.legend(loc="lower left", frameon=False)

        letter = letters[idx]
        slug = re.sub(r"[^A-Za-z0-9]+", "", res["show_name"])
        png_path = fig_dir / f"Figure7{letter}_{slug}_{tag}.png"
        svg_path = png_path.with_suffix(".svg")

        fig.savefig(png_path, dpi=400, bbox_inches="tight")
        fig.savefig(svg_path, dpi=400, bbox_inches="tight")
        plt.close(fig)

        print(f"[OUT] {res['show_name']} -> {png_path}")

    # ========= Table 4 =========
    table_df = pd.DataFrame(rows)
    csv_out = out_dir / f"Table4_policy_calculator_{tag}.csv"
    table_df.to_csv(csv_out, index=False)
    print("[OUT] Table 4 ->", csv_out)

    print("IRF CSV:", irf_path)
    print("PANEL CSV (main):", panel_path)
    print("FQ CSV (if used):", file_or_first_exist([args.fq_csv, DEFAULT_PANEL_FQ]))


if __name__ == "__main__":
    main()
