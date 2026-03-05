# -*- coding: utf-8 -*-
# 把 /.../.venv/panel_with_FQ.csv 规范化成 enriched_panel.csv 供后续脚本直接使用
import numpy as np, pandas as pd
from pathlib import Path

CSV_PATH = "/Users/MaJun/PycharmProjects/Python(Ajou University)/.venv/panel_with_FQ.csv"
OUT_PATH = Path("/Users/MaJun/PycharmProjects/Python(Ajou University)/.venv/enriched_panel101.csv")

df = pd.read_csv(CSV_PATH)

# --- 标准列名 ---
df["country"] = df["countryiso3code"]
df["year"] = df["year"].astype(int)
df["g"] = df["WDI_GDP_Growth"]

# --- 控制项：log GDPpc 滞后、通胀及其平方 ---
gdppc = df["WDI_GDPpc"].replace({0: np.nan})
df["log_gdppc"] = np.log(gdppc)
df["log_gdppc_lag"] = df.groupby("country")["log_gdppc"].shift(1)
df["pi"] = df["WDI_Inflation"]
df["pi2"] = df["pi"]**2

# --- 年内 z-score 工具 ---
def z_by_year(series):
    return series.groupby(df["year"]).transform(lambda x: (x - x.mean())/x.std(ddof=0))

# --- S：WGI（去掉 VoA），逐年 z → PCA(PC1)，方向与 GE 同向=“越大=更稳定” ---
wgi_cols = [c for c in ["WGI_CC.EST","WGI_GE.EST","WGI_RQ.EST","WGI_RL.EST","WGI_PV.EST"] if c in df.columns]
Zs = pd.concat({c: z_by_year(df[c]) for c in wgi_cols}, axis=1)
mask_s = Zs.notna().all(axis=1)
X = Zs[mask_s].to_numpy(); X = X - X.mean(0)
U,s,Vt = np.linalg.svd(X, full_matrices=False)
pc1 = (X @ Vt.T[:,0]); pc1 = (pc1 - pc1.mean())/pc1.std()
ref = Zs.loc[mask_s, "WGI_GE.EST"] if "WGI_GE.EST" in Zs.columns else Zs.loc[mask_s, wgi_cols[0]]
corr = np.corrcoef(pc1, ref.values)[0,1]
if not np.isnan(corr) and corr < 0: pc1 = -pc1
S_full = np.full(len(Zs), np.nan); S_full[mask_s] = pc1
df["S_PCA"] = S_full

# --- F：FIW/RSF/V-DEM 三件套，逐年 z → 先对齐方向 → PCA(PC1)
#     目标口径：越大 = 反馈更强（与 FIW 同向）

fiw = z_by_year(df["z_FH"])    if "z_FH"   in df.columns else None
rsf_raw = z_by_year(df["z_RSF"])   if "z_RSF"  in df.columns else None
vdem = z_by_year(df["z_VDEM"]) if "z_VDEM" in df.columns else None

# 先拼出一个临时矩阵（未对齐）
Zf_raw = pd.concat({"FIW": fiw, "RSF": rsf_raw, "VDEM": vdem}, axis=1)

# 用 FIW 作为参考对齐 RSF 方向：让 corr(RSF, FIW) >= 0
mask_rf = Zf_raw[["FIW","RSF"]].notna().all(axis=1)
if mask_rf.any():
    corr_rf = np.corrcoef(Zf_raw.loc[mask_rf,"RSF"], Zf_raw.loc[mask_rf,"FIW"])[0,1]
else:
    corr_rf = np.nan

# 对齐后的 RSF（强度口径：越大=反馈越强）
if np.isnan(corr_rf) or corr_rf >= 0:
    rsf_aligned = Zf_raw["RSF"]
else:
    rsf_aligned = -Zf_raw["RSF"]

# 用“对齐后的 RSF”重新构造 PCA 输入矩阵
Zf = pd.concat({"FIW": Zf_raw["FIW"], "RSF": rsf_aligned, "VDEM": Zf_raw["VDEM"]}, axis=1)

# PCA：三者都不缺失的共同样本
mask_f = Zf.notna().all(axis=1)
X = Zf[mask_f].to_numpy()
X = X - X.mean(0)

U, s, Vt = np.linalg.svd(X, full_matrices=False)
pc1 = (X @ Vt.T[:, 0])
pc1 = (pc1 - pc1.mean()) / pc1.std()

# 再次确保 PC1 与 FIW 同向（越大=反馈更强）
corr_pc = np.corrcoef(pc1, Zf.loc[mask_f, "FIW"].values)[0,1]
if (not np.isnan(corr_pc)) and corr_pc < 0:
    pc1 = -pc1

F_full = np.full(len(Zf), np.nan)
F_full[mask_f] = pc1
df["F_PCA_main"] = F_full

# ===== 组件输出（删源稳健性 / 追溯用）=====
df["FIW_ExprBelief"] = Zf["FIW"]

# RSF：同时保留 weak 与 strong（strong=已对齐、应该用于论文 F）
df["RSF_PressIndex_weak"] = Zf_raw["RSF"]         # 未对齐版本（你原来的 z_RSF 年内标准化）
df["RSF_PressIndex_rev"]  = Zf["RSF"]             # 已对齐版本（推荐口径：越大=反馈更强）

df["VDEM_media_leg_audit"] = Zf["VDEM"]

# --- 冲击列：PSAV_shock → shock_tot（可被 LP 预趋势脚本识别）---
if "PSAV_shock" in df.columns:
    df["shock_tot"] = df["PSAV_shock"]

# --- 导出最小必需列（其余保留也无妨） ---
need = ["country","year","g","S_PCA","F_PCA_main",
        "FIW_ExprBelief","RSF_PressIndex_rev","VDEM_media_leg_audit",
        "log_gdppc_lag","pi","pi2","shock_tot"]
df.to_csv(OUT_PATH, index=False)
print("Saved:", OUT_PATH, "| rows:", len(df))
