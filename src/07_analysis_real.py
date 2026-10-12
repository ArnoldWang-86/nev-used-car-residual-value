# -*- coding: utf-8 -*-
r"""07_analysis_real.py —— 真实新能源车源专项分析（实证锚点）

数据：由 src/collect.py 从瓜子二手车官方 AI 数据出口采集的真实挂牌数据
      （合规依据：robots.txt 明示 Allow: /car-detail/*.md）

⚠️ 样本量说明（必须与结论一同出现）：
  受站点 IP 级风控限制，实际采集到的真实车源数量有限（十余条量级）。
  因此本脚本**不做模型训练**，只做两件在统计上仍然成立的事：
    1. 特征工程示范：把真实新能源车源的字段加工成可直接进模型的特征表
    2. 描述性分析：给出真实残值率的区间与结构，作为量级锚点
  任何"用 N 条数据训练出模型"的说法都是不成立的，本脚本不会那样做。

真实数据的独特价值（公开数据集无法替代）：
  - 电池容量、电池健康度(SOH)、换电/BaaS 标记 —— 新能源残值的核心变量
  - 中国市场口径的「挂牌价 / 新车指导价」—— 与公开数据集口径不同，可做对照

用法：
    python src/07_analysis_real.py
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np
import pandas as pd

# --- 中文 Windows 控制台默认 GBK，强制 UTF-8 输出，避免打印特殊字符时崩溃 ---
try:
    import sys as _sys
    _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    _sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_JSONL = os.path.join(ROOT, "data", "raw", "detail_raw.jsonl")
OUT_CSV = os.path.join(ROOT, "data", "clean", "cars_real.csv")
REPORT = os.path.join(ROOT, "output", "real_data_report.txt")
OUTDIR = os.path.join(ROOT, "results")
os.makedirs(OUTDIR, exist_ok=True)

# 这些列直接对应真实新能源车的业务含义，是公开数据集完全缺失的部分
EV_FEATURES = ["battery_kwh", "soh_pct", "is_battery_swap", "model_range_km",
               "cltc_km", "battery_chem", "drive_mode"]

NUMERIC = ["age_months", "mileage", "mileage_per_year", "battery_kwh",
           "soh_pct", "cltc_km", "model_range_km", "appearance_score",
           "transfer_times", "claim_times"]
BOOLS = ["is_battery_swap", "has_structural_repair", "is_first_owner"]
CATS = ["brand", "series", "battery_chem", "drive_mode", "city_name",
        "condition_grade", "energy_type"]


def load_real() -> pd.DataFrame:
    if not os.path.exists(RAW_JSONL):
        return pd.DataFrame()
    rows = []
    with open(RAW_JSONL, encoding="utf-8") as f:
        for ln in f:
            try:
                rows.append(json.loads(ln))
            except json.JSONDecodeError:
                continue
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).drop_duplicates(subset=["car_id"], keep="last")
    return df.reset_index(drop=True)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """按真实新能源车的业务逻辑构造特征。

    这里体现的是「结合业务场景构建有效特征」：
      - 车龄年 + 年均里程：使用强度的两个正交维度
      - 电池容量分档：75/100 kWh 对应不同购车成本与残值结构
      - SOH 与车龄的交互：同样是 5 年车，SOH 90% 与 80% 价值不同
      - 换电标记：BaaS 车型的挂牌价不含电池，口径与其他车不可直接比
    """
    d = df.copy()
    for c in NUMERIC:
        d[c] = pd.to_numeric(d.get(c), errors="coerce") if c in d.columns else np.nan
    for c in BOOLS:
        d[c] = (d[c].astype("boolean").fillna(False).astype(int)
                if c in d.columns else 0)
    for c in CATS:
        d[c] = d[c].astype("string") if c in d.columns else pd.NA

    d["age_years"] = (d["age_months"] / 12).round(2)

    # 电池容量分档（新能源特有）
    def kwh_band(v):
        if pd.isna(v):
            return "未知"
        if v <= 70:
            return "≤70kWh"
        if v <= 80:
            return "75kWh档"
        if v <= 90:
            return "80-90kWh"
        return "≥100kWh"
    d["battery_band"] = d["battery_kwh"].map(kwh_band)

    # SOH 与车龄交互：衡量「电池相对车龄的健康程度」
    d["soh_per_year"] = np.where(
        (d["age_years"] > 0) & d["soh_pct"].notna(),
        ((100 - d["soh_pct"]) / d["age_years"]).round(2), np.nan)

    # 使用强度分档
    def usage_band(v):
        if pd.isna(v):
            return "未知"
        if v < 10000:
            return "低强度(<1万/年)"
        if v < 20000:
            return "中强度(1-2万/年)"
        return "高强度(>2万/年)"
    d["usage_band"] = d["mileage_per_year"].map(usage_band)

    # 残值率复核：用原始价格字段重算，防止上游口径漂移
    d["residual_rate_recalc"] = (d["full_payment"] / d["guide_price"]).round(6)
    diff = (d["residual_rate_recalc"] - pd.to_numeric(d["residual_rate"],
                                                      errors="coerce")).abs()
    d["rate_mismatch"] = (diff > 1e-6).astype(int)

    d["data_source"] = "guazi-real-listing"
    return d


def report(d: pd.DataFrame) -> str:
    L = []
    P = L.append
    n = len(d)
    P("=" * 84)
    P("真实新能源车源数据报告（瓜子二手车公开 AI 数据出口）")
    P("=" * 84)
    P("")
    P("【合规来源】robots.txt 明示 Allow: /car-detail/*.md；仅用路径 URL，")
    P("           未使用被 Disallow 的查询参数，速率受限含风控熔断。")
    P("")
    P(f"【样本量】{n} 条")
    if n < 30:
        P("  ⚠️ 样本量不足以训练模型。本报告只做描述性分析与特征工程示范，")
        P("     任何基于该样本量的模型结论都不成立。")
    P("")
    if n == 0:
        return "\n".join(L)

    P("【口径自检】")
    bad = int(d["rate_mismatch"].sum())
    P(f"  残值率重算不一致的记录：{bad} 条（应为 0，非 0 说明上游解析漂移）")
    P("")
    P("【残值率分布（挂牌价 / 新车指导价）】")
    s = d["residual_rate"].astype(float)
    for k, v in s.describe(percentiles=[.25, .5, .75]).items():
        P(f"  {k:<8} {v:,.4f}")
    P("")
    P("【逐条明细】")
    cols = ["series", "model", "first_register", "age_months", "mileage",
            "guide_price", "full_payment", "residual_rate", "condition_grade",
            "battery_kwh", "soh_pct", "is_battery_swap"]
    cols = [c for c in cols if c in d.columns]
    P(d[cols].to_string(index=False))
    P("")
    P("【按车系汇总】")
    g = (d.groupby("series")
         .agg(样本数=("residual_rate", "size"),
              残值率中位=("residual_rate", "median"),
              车龄中位月=("age_months", "median"),
              里程中位=("mileage", "median"),
              指导价中位=("guide_price", "median"))
         .round(3))
    P(g.to_string())
    P("")
    P("【关键字段覆盖率】")
    for c in NUMERIC + BOOLS + ["condition_grade", "battery_band", "usage_band"]:
        if c in d.columns:
            cov = d[c].notna().mean() if d[c].dtype != object else d[c].notna().mean()
            P(f"  {c:<22} {cov:>6.1%}")
    P("")
    P("【新能源专属维度（公开数据集完全缺失）】")
    if "battery_band" in d.columns:
        P("  电池容量分档：")
        for k, v in d["battery_band"].value_counts().items():
            P(f"    {k:<16} {v} 条")
    if "soh_pct" in d.columns:
        soh = d[d["soh_pct"].notna()]
        P(f"  有 SOH 记录 {len(soh)} 条：{soh['soh_pct'].tolist()}")
    P(f"  换电/BaaS 车型：{int(d['is_battery_swap'].sum())} 条")
    P("")
    P("【与公开数据集的对照（口径不同，仅作量级参考）】")
    P("  真实中国市场：2021 年上牌车残值率约 22%，2023-2024 年上牌车约 38-40%")
    P("  公开数据集：车龄折旧比 0-1年 1.02 / 3-5年 0.72 / 8-12年 0.43")
    P("  两者不可直接比较——前者分母是官方指导价，后者分母是同车系准新车中位价。")
    return "\n".join(L)


def main() -> None:
    df = load_real()
    print(f"读取真实车源：{len(df)} 条")
    if df.empty:
        print("无数据。可能尚未采集或采集被风控中断。")
        txt = report(df)
        with open(REPORT, "w", encoding="utf-8") as f:
            f.write(txt)
        return

    d = build_features(df)
    txt = report(d)
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write(txt)
    print(txt)

    d.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n已输出 {OUT_CSV}")

    # 保存逐条明细供后续人工核对
    d.to_csv(os.path.join(OUTDIR, "n01_真实车源明细.csv"),
             index=False, encoding="utf-8-sig")
    print(f"已输出 results/n01_真实车源明细.csv")


if __name__ == "__main__":
    main()
