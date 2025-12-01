#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Task 21b: COVID 外生冲击的 DID / 事件研究（LP 风格）回归脚本（修正版）

数据来源：
    - 使用 Task 21a 生成的：
      /Users/MaJun/PycharmProjects/Python(Ajou University)/The Stability Trap Hypothesis/二稿代码/M1-M7/scripts/core/LP外生冲击检验/LP外生冲击检验（COVID）/covid_lp_sample_from_panelFQ.csv

主要内容：
1. 读取 30 国 × 2015–2022 面板，检查关键变量。
2. 基准 DID 回归：
   g_it = α_i + λ_t + β (Rigid_i × Post_t) + ε_it
   - g_it: WDI_GDP_Growth
   - Rigid_i: 2015–2019 高稳定 + 低反馈 国家
   - Post_t: 2020–2022 = 1
   - 固定效应：国家 FE、年份 FE
   - 标准误：按国家聚类
3. 事件研究 / LP 风格回归：
   - 定义相对年份 rel_year = year - 2019（2019 为最后一个“正常年”）
       2015→-4, 2016→-3, 2017→-2, 2018→-1, 2019→0, 2020→1, 2021→2, 2022→3
   - 以 rel_year = 0 (2019) 作为基准，构造哑变量 D_k (k ≠ 0)
   - 回归：
       g_it = α_i
              + Σ_{k≠0} γ_k D_k(t)
              + Σ_{k≠0} β_k (Rigid_i × D_k(t))
              + ε_it
     其中 β_k 就是 “Rigid 国家在相对年份 k 相比非 Rigid 的额外增速差”，
     相对于 2019 年的差异。

输出：
    - 文本：
        covid_did_results.txt
        covid_event_study_results.txt
    - 事件研究系数表：
        covid_event_study_coeffs.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
import statsmodels.api as sm


# ========== 0. 路径配置 ==========

DATA_PATH = Path(
    "/Users/MaJun/PycharmProjects/Python(Ajou University)/The Stability Trap Hypothesis/终稿代码/Figure/Figure6:COVID 事件型冲击/covid_lp_sample_from_panelFQ.csv"
)

OUTPUT_DIR = Path(
    "/Users/MaJun/PycharmProjects/Python(Ajou University)/The Stability Trap Hypothesis/终稿代码/Figure/Figure6:COVID 事件型冲击"
)


# ========== 1. 读取数据 & 基本检查 ==========

def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"[ERROR] Input data not found: {DATA_PATH}")

    print(f"[INFO] Loading COVID LP sample from: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    print(f"[INFO] Data shape: {df.shape}")

    required_cols = [
        "countryiso3code",
        "year",
        "WDI_GDP_Growth",
        "Rigid",
        "Post",
        "g_pre",
        "g_post",
        "delta_g_covid",
        "S_pre",
        "F_pre",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"[ERROR] Missing required columns: {missing}")

    # 类型处理
    df["year"] = df["year"].astype(int)
    df["Rigid"] = df["Rigid"].astype(int)
    df["Post"] = df["Post"].astype(int)

    # 只保留 2015–2022（理论上已经是）
    df = df[(df["year"] >= 2015) & (df["year"] <= 2022)].copy()

    print("[INFO] Basic overview: obs by year")
    print(df[["year", "countryiso3code"]].groupby("year").size())

    # 简单检查：Rigid vs 非 Rigid 的 COVID 增长损失
    grp = df.drop_duplicates("countryiso3code").groupby("Rigid")["delta_g_covid"].mean()
    print("\n[INFO] Average COVID growth loss (delta_g_covid) by Rigid group:")
    print(grp)

    # 确保因变量和关键解释变量为 float
    df["WDI_GDP_Growth"] = pd.to_numeric(df["WDI_GDP_Growth"], errors="coerce")
    df["g_pre"] = pd.to_numeric(df["g_pre"], errors="coerce")
    df["g_post"] = pd.to_numeric(df["g_post"], errors="coerce")
    df["delta_g_covid"] = pd.to_numeric(df["delta_g_covid"], errors="coerce")
    df["S_pre"] = pd.to_numeric(df["S_pre"], errors="coerce")
    df["F_pre"] = pd.to_numeric(df["F_pre"], errors="coerce")

    # 丢掉有缺失的行，避免 statsmodels 报错
    df = df.dropna(subset=["WDI_GDP_Growth", "Rigid", "Post"])

    return df


# ========== 2. 工具函数：构造虚拟变量矩阵 ==========

def build_dummies(series: pd.Series, prefix: str) -> pd.DataFrame:
    """
    给定一个分类变量（如国家、年份），构造虚拟变量，丢弃第一个水平作为基准。
    """
    dummies = pd.get_dummies(series, prefix=prefix, drop_first=True)
    return dummies


# ========== 3. 基准 DID 回归 ==========

def run_did_regression(df: pd.DataFrame) -> sm.regression.linear_model.RegressionResultsWrapper:
    """
    g_it = α_i + λ_t + β (Rigid_i × Post_t) + ε_it

    - 国家 FE: via country dummies
    - 年份 FE: via year dummies
    - 标准误: 按国家聚类
    """
    print("\n[INFO] Running baseline DID regression...")

    # 因变量
    y = df["WDI_GDP_Growth"].astype(float)

    # 国家 FE
    country_dummies = build_dummies(df["countryiso3code"], prefix="cty")

    # 年份 FE
    year_dummies = build_dummies(df["year"], prefix="year")

    # DID 变量
    rigid = df["Rigid"].astype(float).rename("Rigid")
    post = df["Post"].astype(float).rename("Post")
    rigid_post = (df["Rigid"] * df["Post"]).astype(float).rename("Rigid_x_Post")

    # 设计矩阵
    X_did = pd.concat(
        [country_dummies, year_dummies, rigid, post, rigid_post],
        axis=1
    )

    # 加上常数项
    X_did = sm.add_constant(X_did)

    # 强制转成 float，避免 "object" dtype
    X_did = X_did.astype(float)

    print("[INFO] X_did dtypes summary:")
    print(X_did.dtypes.value_counts())

    # 回归：OLS + 按国家聚类的标准误
    model = sm.OLS(y, X_did)
    results = model.fit(
        cov_type="cluster",
        cov_kwds={"groups": df["countryiso3code"]}
    )

    print("[INFO] DID regression finished.")
    print("[INFO] Coefficient on Rigid_x_Post (main DID effect):")
    if "Rigid_x_Post" in results.params.index:
        beta = results.params["Rigid_x_Post"]
        se = results.bse["Rigid_x_Post"]
        tval = results.tvalues["Rigid_x_Post"]
        pval = results.pvalues["Rigid_x_Post"]
        print(f"  beta = {beta:.4f}, se = {se:.4f}, t = {tval:.2f}, p = {pval:.4f}")
    else:
        print("  [WARNING] Rigid_x_Post not found in parameter list.")

    return results


# ========== 4. 事件研究 / LP 风格回归 ==========

def run_event_study_regression(df: pd.DataFrame) -> sm.regression.linear_model.RegressionResultsWrapper:
    """
    事件研究回归：

    1. 定义 rel_year = year - 2019
       2015→-4, 2016→-3, 2017→-2, 2018→-1, 2019→0, 2020→1, 2021→2, 2022→3
    2. 基准年份：rel_year = 0 (2019)，不建哑变量
    3. 对 k ∈ {-4, -3, -2, -1, 1, 2, 3}：
         D_rel_k(t) = 1{ rel_year = k }
         Rigid_x_D_rel_k(i,t) = Rigid_i × D_rel_k(t)
    4. 回归：
         g_it = α_i
                + Σ_{k≠0} γ_k D_rel_k(t)
                + Σ_{k≠0} β_k Rigid_x_D_rel_k(i,t)
                + ε_it

       β_k = Rigid vs 非 Rigid 在相对年份 k 的额外增速差，
             相对于 2019 年（k=0）的基准。
    """
    print("\n[INFO] Running event-study / LP-style regression...")

    df = df.copy()
    df["rel_year"] = df["year"] - 2019

    # 可用的相对年份（数据范围内）
    print("[INFO] Unique relative years (rel_year):", sorted(df["rel_year"].unique()))

    # 只保留 2015–2022 的 rel_year ∈ [-4, 3]
    df = df[(df["rel_year"] >= -4) & (df["rel_year"] <= 3)].copy()

    # 设定基准年 rel_year = 0 (2019) —— 不设哑变量
    rel_year_values = [-4, -3, -2, -1, 1, 2, 3]

    # 构造 D_rel_k 和交互项
    D_cols = []
    Rigid_D_cols = []
    for k in rel_year_values:
        d_col = f"D_rel_{k}"
        rd_col = f"Rigid_x_D_rel_{k}"

        df[d_col] = (df["rel_year"] == k).astype(float)
        df[rd_col] = df["Rigid"].astype(float) * df[d_col]

        D_cols.append(d_col)
        Rigid_D_cols.append(rd_col)

    # 因变量
    y = df["WDI_GDP_Growth"].astype(float)

    # 国家固定效应（不含年份 FE，时间维度由 D_rel_k 吸收）
    country_dummies = build_dummies(df["countryiso3code"], prefix="cty")

    # 设计矩阵
    X_es = pd.concat([country_dummies, df[D_cols], df[Rigid_D_cols]], axis=1)
    X_es = sm.add_constant(X_es)

    # 强制转成 float
    X_es = X_es.astype(float)

    print("[INFO] X_es dtypes summary:")
    print(X_es.dtypes.value_counts())

    # 回归：OLS + 按国家聚类的标准误
    model_es = sm.OLS(y, X_es)
    results_es = model_es.fit(
        cov_type="cluster",
        cov_kwds={"groups": df["countryiso3code"]}
    )

    print("[INFO] Event-study regression finished.")

    # 提取 β_k（Rigid × D_rel_k）
    coef_rows = []
    for k, rd_col in zip(rel_year_values, Rigid_D_cols):
        if rd_col not in results_es.params.index:
            continue
        beta = results_es.params[rd_col]
        se = results_es.bse[rd_col]
        tval = results_es.tvalues[rd_col]
        pval = results_es.pvalues[rd_col]
        ci_low, ci_high = results_es.conf_int().loc[rd_col]

        coef_rows.append(
            {
                "rel_year": k,
                "param_name": rd_col,
                "beta": beta,
                "se": se,
                "t": tval,
                "p": pval,
                "ci_low": ci_low,
                "ci_high": ci_high,
            }
        )

    coef_df = pd.DataFrame(coef_rows).sort_values("rel_year")
    print("\n[INFO] Event-study Rigid effect by relative year (β_k):")
    print(coef_df)

    # 保存结果到 CSV
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    coef_path = OUTPUT_DIR / "covid_event_study_coeffs.csv"
    coef_df.to_csv(coef_path, index=False)
    print(f"[INFO] Saved event-study coefficients to: {coef_path}")

    return results_es


# ========== 5. 主流程 ==========

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()

    # 5.1 基准 DID 回归
    did_results = run_did_regression(df)

    did_txt_path = OUTPUT_DIR / "covid_did_results.txt"
    with open(did_txt_path, "w", encoding="utf-8") as f:
        f.write(did_results.summary().as_text())
    print(f"[INFO] Saved DID regression summary to: {did_txt_path}")

    # 5.2 事件研究 / LP 风格回归
    es_results = run_event_study_regression(df)

    es_txt_path = OUTPUT_DIR / "covid_event_study_results.txt"
    with open(es_txt_path, "w", encoding="utf-8") as f:
        f.write(es_results.summary().as_text())
    print(f"[INFO] Saved event-study regression summary to: {es_txt_path}")

    print("\n[INFO] Task 21b finished. You now have:")
    print("  - covid_did_results.txt (DID 回归全文)")
    print("  - covid_event_study_results.txt (事件研究回归全文)")
    print("  - covid_event_study_coeffs.csv (按相对年份的 β_k 系数表)")


if __name__ == "__main__":
    main()
