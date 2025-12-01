#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Task 21a: 使用 panel_with_FQ.csv 构造 COVID 外生冲击样本（LP / DID 预备数据）

功能概述
--------
本脚本基于 panel_with_FQ.csv：
1. 从 S4_trend 构造标准化的稳定性指标 S_z（均值 0，标准差 1）。
2. 使用 FQ_z_equal_weight 作为反馈指标 F。
3. 选取 2015–2022 年：
   - pre：2015–2019（基线期）
   - post：2020–2022（疫情冲击期）
4. 按国家计算：
   - g_pre  ：基线期 5 年平均 GDP 增速
   - g_post ：冲击期 3 年平均 GDP 增速
   - delta_g_covid = g_post - g_pre
   - S_pre  ：2015–2019 的 S_z 平均值
   - F_pre  ：2015–2019 的 F 平均值
5. 定义 Rigid dummy：
   - Rigid = 1  当且仅当  S_pre >= GAMMA_STAR 且 F_pre < median(F_pre)
   - 其中 GAMMA_STAR = -0.76（来自阈值回归结果）
6. 在 2015–2022 的面板上：
   - 合并上述国家层指标
   - 构造 Post = 1( year >= 2020 )
7. 输出 covid_lp_sample_from_panelFQ.csv，用于后续 LP / DID 分析。

使用说明
--------
- 确认 INPUT_PATH / OUTPUT_DIR 是否需要修改为你本地的实际路径。
- 在终端中执行：
    python Figure 6A Sample and variable preparation for the ALP exogenous shock test (COVID event).py
"""

import pandas as pd
import numpy as np
from pathlib import Path


# ========== 0. 路径 & 参数配置 ==========

# 原始 30 国面板（含 WGI / WDI / FQ）
INPUT_PATH = Path(
    "/Users/MaJun/PycharmProjects/Python(Ajou University)/.venv/enriched_panel.csv"
)

# 输出目录（建议与你现有 M1–M7 数据保持一致）
OUTPUT_DIR = Path(
    "/Users/MaJun/PycharmProjects/Python(Ajou University)/The Stability Trap Hypothesis/终稿代码/Figure/Figure6:COVID 事件型冲击"
)
OUTPUT_PATH = OUTPUT_DIR / "covid_lp_sample_from_panelFQ.csv"

# 稳定性阈值 γ*
GAMMA_STAR = -0.76  # 与主文阈值回归保持一致


# ========== 1. 主函数 ==========

def main() -> None:
    # 1.1 读入数据
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"[ERROR] Input file not found: {INPUT_PATH}")

    print(f"[INFO] Loading data from: {INPUT_PATH}")
    df = pd.read_csv(INPUT_PATH)
    print(f"[INFO] Data loaded. Shape: {df.shape}")

    # 1.2 基本列名（本文件为固定结构，不做自动识别）
    country_col = "countryiso3code"
    year_col = "year"
    growth_col = "WDI_GDP_Growth"      # GDP growth
    s4_trend_col = "S4_trend"          # 稳定性基础指标
    f_comp_col = "FQ_z_equal_weight"   # 反馈综合指标

    required_cols = [country_col, year_col, growth_col, s4_trend_col, f_comp_col]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"[ERROR] Missing required columns: {missing}")

    # 1.3 从 S4_trend 构造 z-score 稳定性指标 S_z
    #     注意：在整份 2004–2023、30 国样本上标准化
    s_all = df[s4_trend_col].astype(float)
    s_mean = s_all.mean()
    s_std = s_all.std()
    df["S_z"] = (s_all - s_mean) / s_std

    print(f"[INFO] Constructed S_z from '{s4_trend_col}'. "
          f"Mean ≈ {df['S_z'].mean():.4f}, SD ≈ {df['S_z'].std():.4f}")

    # 1.4 F 指标直接使用 FQ_z_equal_weight（已经是 z-score 合成）
    df["F"] = df[f_comp_col].astype(float)

    # 1.5 限定年份：2015–2022
    df[year_col] = df[year_col].astype(int)
    mask_2015_2022 = (df[year_col] >= 2015) & (df[year_col] <= 2022)
    df_sub = df.loc[mask_2015_2022].copy()
    print(f"[INFO] Restricted to 2015–2022: {df_sub.shape[0]} rows.")

    # 1.6 pre / post 掩码
    pre_mask = df_sub[year_col].between(2015, 2019)
    post_mask = df_sub[year_col].between(2020, 2022)

    # 1.7 确保每个国家在 pre 与 post 都有观测
    pre_countries = set(df_sub.loc[pre_mask, country_col].unique())
    post_countries = set(df_sub.loc[post_mask, country_col].unique())
    common_countries = pre_countries & post_countries
    print(f"[INFO] Countries with both pre and post observations: {len(common_countries)}")

    # 1.8 计算国家层 summary：pre（g_pre, S_pre, F_pre）
    pre_summary = (
        df_sub.loc[pre_mask & df_sub[country_col].isin(common_countries)]
        .groupby(country_col, as_index=True)
        .agg(
            g_pre=(growth_col, "mean"),
            S_pre=("S_z", "mean"),
            F_pre=("F", "mean"),
        )
    )

    # 1.9 计算国家层 summary：post（g_post）
    post_summary = (
        df_sub.loc[post_mask & df_sub[country_col].isin(common_countries)]
        .groupby(country_col, as_index=True)
        .agg(
            g_post=(growth_col, "mean"),
        )
    )

    # 1.10 合并 pre & post
    summary = pre_summary.join(post_summary, how="inner")
    print(f"[INFO] Summary with pre/post averages: {summary.shape[0]} countries.")

    # 1.11 计算 delta_g_covid
    summary["delta_g_covid"] = summary["g_post"] - summary["g_pre"]

    # 1.12 计算 Rigid dummy（高 S + 低 F）
    F_pre_median = summary["F_pre"].median()
    print(f"[INFO] Median(F_pre) = {F_pre_median:.3f}")

    summary["Rigid"] = np.where(
        (summary["S_pre"] >= GAMMA_STAR) & (summary["F_pre"] < F_pre_median),
        1,
        0,
    )

    print("[INFO] Rigid group counts:")
    print(summary["Rigid"].value_counts())
    print("[INFO] Rigid group shares:")
    print(summary["Rigid"].value_counts(normalize=True).rename("share"))

    # 1.13 将 summary 合并回 2015–2022 面板
    df_sub = df_sub.merge(
        summary[["g_pre", "g_post", "delta_g_covid", "S_pre", "F_pre", "Rigid"]],
        on=country_col,
        how="left",
    )

    # 1.14 构造 Post_t dummy：2020–2022 = 1
    df_sub["Post"] = np.where(df_sub[year_col] >= 2020, 1, 0)

    # 1.15 删除没有完整 summary 的观测（理论上不会发生，但以防万一）
    before_drop = df_sub.shape[0]
    df_sub = df_sub.dropna(subset=["g_pre", "g_post", "delta_g_covid", "S_pre", "F_pre"])
    after_drop = df_sub.shape[0]
    if after_drop < before_drop:
        print(f"[INFO] Dropped rows with missing pre/post info: {before_drop} -> {after_drop}")

    # 1.16 输出结果
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df_sub.to_csv(OUTPUT_PATH, index=False)

    print(f"[INFO] Saved COVID LP sample to: {OUTPUT_PATH}")
    print("[INFO] Output columns:")
    print(df_sub.columns.tolist())


if __name__ == "__main__":
    main()
