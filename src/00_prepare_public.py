# -*- coding: utf-8 -*-
r"""00_prepare_public.py —— 基于公开二手车市场数据的建模数据集构建

数据源：HuggingFace `imbalet/used_cars`
  432,227 条真实二手车挂牌记录（俄罗斯市场），关键字段无缺失。
  选它而不是 UCI imports-85 的原因见 README「数据源选择」章节：
  UCI 是 1985 年欧美燃油车、**没有车龄字段**，无法做时间维度的折旧建模，
  实测所有模型 R² 为负、跑不过中位数基线。

目标变量：deprec_ratio（相对折旧比）
    = 本车价格 / 同(品牌,车型,发动机,变速箱)组内「车龄<=2年」的中位价格

⚠️ 口径声明（必须与结论一同出现）：
  数据源**不含官方新车指导价**，因此该比值是「相对最新年款的折旧比」，
  **不等于**中国市场的「残值率 = 二手车价/新车指导价」。
  它度量的是同一车系内部、随时间推移的相对贬值程度，趋势可信、绝对水平不可跨市场套用。

产物：
  data/clean/cars_public.csv      建模用宽表
  output/public_data_report.txt   数据质量报告
"""
from __future__ import annotations

import os
import sys

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
SRC = os.path.join(ROOT, "data", "raw", "public", "used_cars_ru.csv")
OUT = os.path.join(ROOT, "data", "clean", "cars_public.csv")
REPORT = os.path.join(ROOT, "output", "public_data_report.txt")

NOW_YEAR = 2025          # 数据抓取年份，用于计算车龄
CACHE = os.path.join(ROOT, "data", "raw", "public", "used_cars_ru.parquet")

HF_REPO = "imbalet/used_cars"
HF_FILE = "dataset.csv"
USE_COLS = ["price", "year", "mileage", "mark", "model", "engine",
            "transmission", "power", "displacement", "body_type_name",
            "region", "owners", "steering_wheel", "gear_type"]

GROUP_KEY = ["mark", "model", "engine", "transmission"]


def ensure_raw() -> str:
    """确保原始数据就位（优先本地缓存，其次 HuggingFace 下载）。"""
    if os.path.exists(CACHE):
        print(f"使用本地 parquet 缓存：{CACHE}")
        return CACHE
    if os.path.exists(SRC):
        return SRC

    import requests
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    url = f"https://huggingface.co/datasets/{HF_REPO}/resolve/main/{HF_FILE}"
    print(f"下载 {url}")
    with requests.get(url, stream=True, timeout=(30, 600),
                      headers={"User-Agent": "Mozilla/5.0 (research)"}) as r:
        r.raise_for_status()
        tmp = SRC + ".part"
        got = 0
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1024 * 512):
                if chunk:
                    f.write(chunk)
                    got += len(chunk)
        os.replace(tmp, SRC)
    print(f"已下载 {got/1024/1024:.1f} MB -> {SRC}")
    return SRC


def load(path: str) -> pd.DataFrame:
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    df = pd.read_csv(path, low_memory=False, usecols=USE_COLS)
    # 首次读入后转 parquet，后续读取快很多（92MB CSV -> 约 10MB parquet）
    try:
        df.to_parquet(CACHE, index=False)
        print(f"已缓存 parquet：{CACHE}")
    except Exception as e:  # noqa: BLE001
        print(f"parquet 缓存失败（不影响流程）：{type(e).__name__}")
    return df


def build(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    notes = []
    n0 = len(df)
    notes.append(f"原始记录 {n0:,} 条")

    d = df.copy()
    # --- 类型与基础清洗 ---
    for c in ["price", "year", "mileage", "power", "displacement", "owners"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")

    d["age_years"] = NOW_YEAR - d["year"]
    bad_age = (d["age_years"] < 0) | (d["age_years"] > 40)
    notes.append(f"剔除车龄越界（<0 或 >40 年）{int(bad_age.sum()):,} 条")
    d = d[~bad_age]

    bad = (d["price"] <= 0) | (d["mileage"] <= 0)
    notes.append(f"剔除价格/里程非正 {int(bad.sum()):,} 条")
    d = d[~bad]

    lo, hi = d["price"].quantile([0.001, 0.999])
    trim = (d["price"] < lo) | (d["price"] > hi)
    notes.append(f"价格 0.1% 截尾剔除 {int(trim.sum()):,} 条（保留 [{lo:,.0f}, {hi:,.0f}] 卢布）")
    d = d[~trim]

    # --- 目标变量：相对折旧比 ---
    newest = (d[d["age_years"] <= 2]
              .groupby(GROUP_KEY)["price"].median()
              .rename("new_ref_price"))
    d = d.join(newest, on=GROUP_KEY)
    has_ref = d["new_ref_price"].notna()
    notes.append(f"可计算折旧比 {int(has_ref.sum()):,} 条"
                 f"（{has_ref.mean():.1%}，其余组内无 2 年内准新车，无法定基准）")
    d = d[has_ref].copy()
    d["deprec_ratio"] = d["price"] / d["new_ref_price"]
    # 折旧比理论上应 (0, 1.2]；>1.2 说明基准异常
    abn = d["deprec_ratio"] > 1.2
    notes.append(f"折旧比 >1.2 的异常记录 {int(abn.sum()):,} 条（保留但可识别）")
    d = d[d["deprec_ratio"] > 0]

    d["residual_rate"] = d["deprec_ratio"]     # 下游脚本统一用这个列名

    # --- 派生特征 ---
    d["price_wan"] = (d["price"] / 1e4).round(2)
    d["new_ref_wan"] = (d["new_ref_price"] / 1e4).round(2)
    age = d["age_years"].replace(0, np.nan)
    d["km_per_year"] = (d["mileage"] / age).round(0)
    d["power_per_l"] = (d["power"] / (d["displacement"] / 1000).replace(0, np.nan)).round(1)
    d["log_mileage"] = np.log1p(d["mileage"])
    d["log_new_ref"] = np.log1p(d["new_ref_price"])
    d["is_new_gen"] = (d["age_years"] <= 2).astype(int)
    d["engine_type"] = d["engine"].astype(str)
    d["drive_mode"] = d["gear_type"].astype(str)
    d["body_style"] = d["body_type_name"].astype(str)
    d["brand"] = d["mark"].astype(str)
    d["series"] = d["mark"].astype(str) + " " + d["model"].astype(str)
    d["city_name"] = d["region"].astype(str)

    # --- 与真实车源 schema 对齐的占位列（便于合并与交叉验证）---
    for c in ["battery_kwh", "soh_pct", "battery_chem", "cltc_km",
              "condition_grade", "condition_desc", "appearance_score"]:
        d[c] = np.nan
    d["is_battery_swap"] = 0
    d["has_structural_repair"] = 0
    d["is_first_owner"] = (d["owners"] == 1).astype(int)
    d["transfer_times"] = (d["owners"] - 1).clip(lower=0)
    d["claim_times"] = np.nan
    d["data_source"] = "HF-imbalet-used_cars-RU"
    d["car_id"] = ["ru%06d" % i for i in range(len(d))]

    cols = ["car_id", "data_source", "brand", "series", "model", "body_style",
            "guide_price", "new_ref_price", "new_ref_wan", "full_payment",
            "price", "price_wan", "residual_rate", "deprec_ratio",
            "age_years", "mileage", "log_mileage", "km_per_year",
            "power", "displacement", "power_per_l", "engine_type",
            "transmission", "drive_mode", "steering_wheel", "owners",
            "transfer_times", "is_first_owner", "city_name", "log_new_ref",
            "is_new_gen", "battery_kwh", "soh_pct", "battery_chem", "cltc_km",
            "condition_grade", "condition_desc", "appearance_score",
            "is_battery_swap", "has_structural_repair", "claim_times"]
    d["guide_price"] = d["new_ref_price"]
    d["full_payment"] = d["price"]
    return d[cols].reset_index(drop=True), notes


def report(d: pd.DataFrame, notes: list[str]) -> str:
    L = []
    P = L.append
    P("=" * 78)
    P("公开数据集数据质量报告")
    P("=" * 78)
    P("来源：HuggingFace imbalet/used_cars（二手车轮牌记录）")
    P("")
    P("【清洗步骤】")
    for n in notes:
        P(f"  - {n}")
    P("")
    P(f"【建模样本量】{len(d):,}")
    P("")
    P("【目标变量 deprec_ratio = 本车价 / 同车系最新年款中位价】")
    q = d["deprec_ratio"].describe(percentiles=[.05, .25, .5, .75, .95])
    for k, v in q.items():
        P(f"  {k:<8} {v:,.4f}")
    P("")
    P("【按车龄的折旧比（核心业务表）】")
    g = (d.groupby(pd.cut(d["age_years"], [-0.1, 1, 2, 3, 5, 8, 12, 20, 40]),
                    observed=True)
         .agg(样本数=("deprec_ratio", "size"),
              折旧比中位=("deprec_ratio", "median"),
              价格中位万卢布=("price_wan", "median"),
              里程中位=("mileage", "median"))
         .round(3))
    P(g.to_string())
    P("")
    P("【特征相关性（与 deprec_ratio）】")
    for c in ["age_years", "log_mileage", "km_per_year", "power",
              "displacement", "owners", "log_new_ref"]:
        P(f"  {c:<16} {d['deprec_ratio'].corr(d[c]):+.4f}")
    P("")
    P("【品牌分布 Top 12】")
    for k, v in d["brand"].value_counts().head(12).items():
        P(f"  {k:<24} {v:>7,}")
    P("")
    P("【类别基数】")
    for c in ["brand", "series", "body_style", "engine_type", "drive_mode", "city_name"]:
        P(f"  {c:<16} {d[c].nunique():>6,} 个取值")
    return "\n".join(L)


def main() -> None:
    raw = ensure_raw()
    print("读取原始数据 ...")
    df = load(raw)
    print(f"原始 {df.shape}")

    d, notes = build(df)
    print(f"\n建模样本 {len(d):,} 条")

    txt = report(d, notes)
    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write(txt)
    print(txt)

    d.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"\n已输出 {OUT}")


if __name__ == "__main__":
    main()
