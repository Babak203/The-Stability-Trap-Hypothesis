#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
download_panel_v2.py

功能：
  - 按“稳定=规则可预期性(S4: GE/RL/RQ/CC)、V=ΔS滚动波动、F=FreedomHouse+RSF+V-Dem（占位）”
    的口径，从 World Bank API 抓取 WDI+WGI（2004–2023）；
  - 计算 S4_raw / S4_trend(EWM) / V_5y（ΔS4_raw 的 5年滚动std），并识别 PSAV_shock（可选）；
  - 输出 5 个 CSV：
      countries_30.csv
      wdi_panel.csv
      wgi_panel.csv（含 S4_* 与 V_5y、PSAV_shock）
      feedback_panel_placeholder.csv（等你抓到 F 三源再填）
      panel_2004_2023.csv（WDI+WGI合并，可直接做首轮M1–M3）
    以及 completeness_report.csv（各国关键字段缺失率报告）。

用法：
  python download_panel_v2.py
  python download_panel_v2.py --years 2004 2023 --countries_csv ./国家样本_30国_v2.csv --outdir ./data

依赖：
  pip install pandas requests tqdm numpy
"""

from __future__ import annotations
import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional
from pathlib import Path
# —— 项目默认路径（写成你的实际目录）
PROJECT_DIR = Path("/Users/MaJun/Desktop/The Stability Trap Hypothesis/第二阶段/DATA")
DEFAULT_OUTDIR = PROJECT_DIR / "data"
DEFAULT_COUNTRIES_CSV = PROJECT_DIR / "数据/国家样本_30国_v2.csv"

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

# -----------------------
# 基本配置
# -----------------------
DEFAULT_YEARS = list(range(2004, 2024))  # 2004..2023
FALLBACK_ISO3 = [
    "CHN","USA","JPN","DEU","GBR","FRA","ITA","CAN","AUS","KOR",
    "IND","IDN","VNM","THA","PHL","MYS","RUS","TUR","SAU","IRN",
    "ZAF","NGA","EGY","KEN","BRA","MEX","ARG","CHL","PER","COL"
]

# WDI（点号写法）
WDI_INDICATORS: Dict[str, str] = {
    "GDP_Growth": "NY.GDP.MKTP.KD.ZG",
    "GDPpc":      "NY.GDP.PCAP.KD",
    "Investment": "NE.GDI.TOTL.ZS",
    "Inflation":  "FP.CPI.TOTL.ZG",
    "PopGrowth":  "SP.POP.GROW",
    "Trade":      "NE.TRD.GNFS.ZS",
}

# WGI：四项规则层 + 其他两项
WGI_INDICATORS: Dict[str, str] = {
    "CC.EST": "CC.EST",
    "GE.EST": "GE.EST",
    "RQ.EST": "RQ.EST",
    "RL.EST": "RL.EST",
    "VA.EST": "VA.EST",
    "PV.EST": "PV.EST",  # = PSAV
}

# -----------------------
# 工具函数
# -----------------------
def fetch_indicator(indicator_code: str, per_page: int = 20000, max_retries: int = 6, pause: float = 1.2) -> pd.DataFrame:
    """World Bank v2 JSON API 抓取单一指标（所有国家、所有年份）。"""
    url = f"https://api.worldbank.org/v2/country/all/indicator/{indicator_code}"
    params = {"format": "json", "per_page": per_page}
    headers = {"User-Agent": "stability-trap-fetcher/1.0"}
    for attempt in range(1, max_retries + 1):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=60)
            r.raise_for_status()
            payload = r.json()
            rows = payload[1]
            recs = []
            for it in rows:
                ct = it.get("country")
                ct_name = ct.get("value") if isinstance(ct, dict) else ct
                year_raw = it.get("date")
                year = int(year_raw) if str(year_raw).isdigit() else None
                recs.append({
                    "countryiso3code": it.get("countryiso3code"),
                    "country": ct_name,
                    "year": year,
                    "value": it.get("value"),
                    "indicator": indicator_code
                })
            return pd.DataFrame(recs)
        except Exception:
            if attempt == max_retries:
                raise
            time.sleep(pause * attempt)

def pull_many(ind_map: Dict[str, str], years: List[int], iso3_list: List[str], label_prefix: str) -> pd.DataFrame:
    """批量抓取多个指标；过滤年份与国家；列名统一为 {prefix}_{alias}。"""
    frames: List[pd.DataFrame] = []
    for alias, code in tqdm(ind_map.items(), desc=f"Fetching {label_prefix}"):
        df = fetch_indicator(code)
        df = df[df["year"].isin(years) & df["countryiso3code"].isin(iso3_list)].copy()
        df.rename(columns={"value": f"{label_prefix}_{alias}"}, inplace=True)
        frames.append(df[["countryiso3code","country","year", f"{label_prefix}_{alias}"]])
    out = frames[0]
    for i in range(1, len(frames)):
        out = out.merge(frames[i], on=["countryiso3code","country","year"], how="outer")
    out = out.sort_values(["countryiso3code","year"]).reset_index(drop=True)
    # 明确类型
    out["countryiso3code"] = out["countryiso3code"].astype(str)
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("Int64")
    return out

def to_01_scale_from_wgi(x: pd.Series) -> pd.Series:
    """WGI [-2.5, 2.5] -> [0,1]（带截断）。"""
    return ((pd.to_numeric(x, errors="coerce") + 2.5) / 5.0).clip(0, 1)

def winsorize_series(s: pd.Series, p: float = 0.01) -> pd.Series:
    """两端 p 分位温和截尾。"""
    if s.notna().sum() == 0:
        return s
    lo, hi = s.quantile(p), s.quantile(1 - p)
    return s.clip(lower=lo, upper=hi)

def build_S4_and_V(wgi: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """
    S4_raw = mean( GE/RL/RQ/CC（0-1缩放） )
    S4_trend = EWM(halflife=3)
    V_5y = rolling std of ΔS4_raw (window=5, min_periods=3)
    PSAV_shock：基于 PV.EST 年差的异常识别
    """
    df = wgi.copy().reset_index(drop=True)
    df["countryiso3code"] = df["countryiso3code"].astype(str)
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")

    # 0-1 缩放四项规则层
    for k in ["GE.EST","RL.EST","RQ.EST","CC.EST"]:
        col = f"WGI_{k}"
        if col in df.columns:
            df[col + "_01"] = df.groupby("countryiso3code")[col].transform(to_01_scale_from_wgi)

    s4_cols_01 = [c for c in df.columns if c.endswith("_01") and any(x in c for x in ["GE.EST","RL.EST","RQ.EST","CC.EST"])]
    df["S4_raw"] = df[s4_cols_01].mean(axis=1, skipna=True)

    df = df.sort_values(["countryiso3code","year"]).reset_index(drop=True)
    # transform 确保与原索引对齐（避免老问题）
    df["S4_trend"] = df.groupby("countryiso3code")["S4_raw"].transform(
        lambda s: s.ewm(halflife=3, adjust=False, min_periods=1).mean()
    )
    df["S4_diff"] = df.groupby("countryiso3code")["S4_raw"].diff()
    df["V_5y"] = df.groupby("countryiso3code")["S4_diff"].transform(
        lambda s: s.rolling(window=window, min_periods=3).std()
    )

    # PSAV shock（基于 PV.EST 年差）
    if "WGI_PV.EST" in df.columns:
        df["PV_diff"] = df.groupby("countryiso3code")["WGI_PV.EST"].diff()
        df["PV_rollsd"] = df.groupby("countryiso3code")["PV_diff"].transform(
            lambda s: s.rolling(7, min_periods=4).std()
        )
        df["PSAV_shock"] = ((df["PV_diff"].abs() > 1.5 * df["PV_rollsd"]).astype("Int64")).where(df["PV_rollsd"].notna())

    keep = [
        "countryiso3code","country","year",
        "WGI_CC.EST","WGI_GE.EST","WGI_RQ.EST","WGI_RL.EST","WGI_VA.EST","WGI_PV.EST",
        "WGI_GE.EST_01","WGI_RL.EST_01","WGI_RQ.EST_01","WGI_CC.EST_01",
        "S4_raw","S4_trend","S4_diff","V_5y","PSAV_shock"
    ]
    return df[keep]

def load_country_list(countries_csv: Optional[Path]) -> List[str]:
    """从CSV读取 ISO3 列（支持多种列名），否则用内置30国。"""
    if countries_csv and countries_csv.exists():
        dfc = pd.read_csv(countries_csv)
        for k in ["ISO3","iso3","countryiso3code","CountryISO3","country_code"]:
            if k in dfc.columns:
                col = k
                break
        else:
            col = dfc.columns[0]
        iso3 = (dfc[col].dropna().astype(str).str.upper().str.strip().unique().tolist())
        return iso3
    return FALLBACK_ISO3

# -----------------------
# 主流程
# -----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", nargs=2, type=int, default=[2004, 2023], help="起止年份（含）")
    ap.add_argument("--countries_csv", type=str, default="", help="30国清单（含 ISO3 列）")
    ap.add_argument("--outdir", type=str, default="data", help="输出目录")
    args = ap.parse_args()

    years = list(range(args.years[0], args.years[1] + 1))
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    iso3_list = load_country_list(Path(args.countries_csv) if args.countries_csv else None)

    # 0) 国家表
    pd.DataFrame({"ISO3": iso3_list}).to_csv(outdir / "countries_30.csv", index=False, encoding="utf-8-sig")

    # 1) 抓 WDI
    print("Downloading WDI …")
    wdi = pull_many(WDI_INDICATORS, years, iso3_list, "WDI")

    # 2) 抓 WGI
    print("Downloading WGI …")
    wgi_wide = pull_many(WGI_INDICATORS, years, iso3_list, "WGI")

    # 3) 计算 S4 与 V
    print("Building S⁴ and V …")
    wgi_proc = build_S4_and_V(wgi_wide)

    # 4) （占位）F 三源表
    f_cols = ["countryiso3code","country","year",
              "FreedomHouse_FIW_Total_Score_0_100",
              "RSF_World_Press_Freedom_Index_0_100",
              "VDem_v2x_freexp_altinf_0_1",
              "FQ_z_equal_weight"]
    feedback_panel = pd.DataFrame(columns=f_cols)
    feedback_panel.to_csv(outdir / "feedback_panel_placeholder.csv", index=False, encoding="utf-8-sig")

    # 5) 输出分表
    wdi.to_csv(outdir / "wdi_panel.csv", index=False, encoding="utf-8-sig")
    wgi_proc.to_csv(outdir / "wgi_panel.csv", index=False, encoding="utf-8-sig")

    # 6) 合并面板（可直接用于首轮回归）
    panel = (wdi.merge(wgi_proc, on=["countryiso3code","country","year"], how="outer")
                 .sort_values(["countryiso3code","year"])
                 .reset_index(drop=True))

    # 7) 温和截尾（可依据需要保留/删除）
    for col in ["WDI_Inflation","WDI_GDP_Growth"]:
        if col in panel.columns:
            panel[col] = winsorize_series(panel[col].astype(float), p=0.01)

    panel.to_csv(outdir / "panel_2004_2023.csv", index=False, encoding="utf-8-sig")

    # 8) 简要完备性报告
    econ_core = ["WDI_GDP_Growth","WDI_GDPpc","WDI_Investment","WDI_Inflation","WDI_PopGrowth","WDI_Trade"]
    rep = panel.groupby("countryiso3code")[econ_core + ["WGI_GE.EST","WGI_RL.EST","WGI_RQ.EST","WGI_CC.EST"]].apply(
        lambda g: g.notna().mean()
    ).round(3)
    rep.to_csv(outdir / "completeness_report.csv", encoding="utf-8-sig")

    print(f"Done. Outputs in: {outdir.resolve()}")
    print(" - countries_30.csv")
    print(" - wdi_panel.csv")
    print(" - wgi_panel.csv (含 S4_raw / S4_trend / V_5y / PSAV_shock)")
    print(" - feedback_panel_placeholder.csv  # 等你抓入 FH/RSF/V-Dem 再合成 FQ")
    print(" - panel_2004_2023.csv            # 可直接做 M1–M3")
    print(" - completeness_report.csv")

if __name__ == "__main__":
    main()

