
import pandas as pd
import numpy as np
from pathlib import Path


# ========== 0. 路径 & 参数配置 ==========

# 原始 30 国面板（含 WGI / WDI / FQ）
INPUT_PATH = Path(
    "/Users/MaJun/PycharmProjects/Python(Ajou University)/.venv/enriched_panel.csv"
)
CLASS_PATH = Path(
    "/Users/MaJun/PycharmProjects/Python(Ajou University)/The Stability Trap Hypothesis/提交代码/Figure 2.3: Global fractal map + S-F scatter plots of 30 countries/classification_schemeA_v2_2023.csv"
)
# 输出目录（建议与你现有 M1–M7 数据保持一致）
OUTPUT_DIR = Path(
    "/Users/MaJun/PycharmProjects/Python(Ajou University)/The Stability Trap Hypothesis/提交代码/Figure 6: COVID-19 event-driven impact")
OUTPUT_PATH = OUTPUT_DIR / "covid_lp_sample_from_panelFQ.csv"

# 稳定性阈值 γ*
GAMMA_STAR = -1.061297


# ========== 1. 主函数 ==========
def main() -> None:
    # ===== Read main-paper regime classification (Scheme A) =====
    if not CLASS_PATH.exists():
        raise FileNotFoundError(f"[ERROR] Classification file not found: {CLASS_PATH}")

    cls = pd.read_csv(CLASS_PATH)

    need_cols = ["iso3c", "regimeA"]
    miss = [c for c in need_cols if c not in cls.columns]
    if miss:
        raise ValueError(f"[ERROR] Classification file missing columns: {miss}")

    # strict string match (your file uses 'Rigid Trap')
    cls["regimeA"] = cls["regimeA"].astype(str).str.strip()
    rigid_map = dict(zip(cls["iso3c"].astype(str).str.strip(), cls["regimeA"]))
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
    s_col = "S_PCA"  # <-- 改成主文用的 S 列名
    df["S"] = df[s_col].astype(float)
    f_comp_col = "FQ_z_equal_weight"   # 反馈综合指标

    required_cols = [country_col, year_col, growth_col, s_col, f_comp_col]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"[ERROR] Missing required columns: {missing}")

    # 1.3 S 指标：直接使用主文同尺度的 S_PCA（不再二次标准化）
    df["S"] = df[s_col].astype(float)

    print(f"[INFO] Using S from '{s_col}'. "
          f"Mean ≈ {df['S'].mean():.4f}, SD ≈ {df['S'].std():.4f}")

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
            S_pre=("S", "mean"),
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
    # ===== Rigid assignment: use main-paper regimeA only =====
    summary["regimeA"] = summary.index.to_series().map(rigid_map)

    # Safety check: all countries must be classified
    missing_list = summary.index[summary["regimeA"].isna()].tolist()
    if len(missing_list) > 0:
        raise ValueError(f"[ERROR] Unclassified countries in CLASS_PATH: {missing_list}")

    summary["Rigid"] = (summary["regimeA"] == "Rigid Trap").astype(int)

    # lock IDN as non-rigid if you still want this rule
    if "IDN" in summary.index:
        summary.loc["IDN", "Rigid"] = 0

    print("[INFO] Rigid group counts (from classification):")
    print(summary["Rigid"].value_counts())
    print("[INFO] #Rigid countries:", int(summary["Rigid"].sum()),
          sorted(summary.index[summary["Rigid"] == 1].tolist()))
    # 1.13 将 summary 合并回 2015–2022 面板
    df_sub = df_sub.merge(
        summary[["g_pre", "g_post", "delta_g_covid", "S_pre", "F_pre", "Rigid", "regimeA"]],
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
    # lock IDN as non-Rigid to match Fig2–3 grouping
    df_sub.loc[df_sub[country_col] == "IDN", "Rigid"] = 0

    rigid_list = sorted(df_sub.loc[df_sub["Rigid"] == 1, country_col].unique().tolist())
    print("[INFO] #Rigid countries:", len(rigid_list), rigid_list)
    print("[INFO] IDN Rigid unique:", df_sub.loc[df_sub[country_col] == "IDN", "Rigid"].unique())

    # 1.16 输出结果
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df_sub.to_csv(OUTPUT_PATH, index=False)

    print(f"[INFO] Saved COVID LP sample to: {OUTPUT_PATH}")
    print("[INFO] Output columns:")
    print(df_sub.columns.tolist())


if __name__ == "__main__":
    main()
