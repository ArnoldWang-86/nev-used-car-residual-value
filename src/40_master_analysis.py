# -*- coding: utf-8 -*-
r"""40_master_analysis.py —— 权威分析脚本（修正全部已发现问题）

本脚本取代 src/10~15 的全部分析产出，修正：
  · EMM 置信区间口径（原用 14 城 SE 的算术平均，偏宽约 2 倍）
     正确：边际均值的方差 = R̄' V R̄，其中 R̄ 为对各城市取平均的对比向量
  · 覆盖率的双分母问题（统一用剔除燃油车后的 726）
  · 里程效应量的错误引用（1.5pp → 实测值）
  · 定价误差基数（挂牌价口径）
  · 稳健性检验（原因 patsy 参照水平 bug 全为 nan）
  · 输出结构化 JSON，供绘图与交付物使用
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.stats.anova as anova_lm

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "data", "raw", "detail_from_cache.jsonl")
OUT = os.path.join(ROOT, "output")
FOCUS = ["蔚来ES6", "蔚来EC6", "Model Y"]
BANDS = ["0-4年", "4年+"]

# ============================================================
# 数据准备
# ============================================================
raw = pd.DataFrame([json.loads(l) for l in open(SRC, encoding="utf-8") if l.strip()])
N_RAW = len(raw)
df = raw[raw["energy_type"] == "新能源"].copy()
N_NEV = len(df)
df = df[df["age_months"].notna()].copy()
N_AGE = len(df)
df["age_band"] = np.where(df["age_months"] < 48, "0-4年", "4年+")
df["mileage_wan"] = df["mileage"] / 10000
df["price_wan"] = df["guide_price"] / 10000
CC = df["city_name"].value_counts()
major = CC[CC >= 8].index.tolist()
df["city_grp"] = np.where(df["city_name"].isin(major), df["city_name"], "其他")

R = {}   # 结果汇总
R["_meta"] = {"n_raw": N_RAW, "n_nev": N_NEV, "n_age_valid": N_AGE,
              "excluded_fuel": N_RAW - N_NEV,
              "snapshot": "2026-10-12"}

# 覆盖率：以【全部新能源样本】为分母（含 age_months 缺失的那 1 条）
nev_all = raw[raw["energy_type"] == "新能源"]
cov = {}
for c in ["guide_price", "full_payment", "mileage", "first_register",
          "series", "city_name", "condition_grade", "appearance_score",
          "age_months", "soh_pct", "battery_kwh", "is_battery_swap",
          "claim_times", "transfer_times"]:
    v = nev_all[c]
    cov[c] = {"n": int(v.notna().sum()), "pct": float(v.notna().mean()),
              "denom": len(nev_all)}
R["coverage"] = cov

# ============================================================
# 2. 折旧曲线（6 段）
# ============================================================
df["age6"] = pd.cut(df["age_months"], [0, 12, 24, 36, 48, 60, 999],
                    labels=["0-12月", "12-24月", "24-36月", "36-48月",
                            "48-60月", "60月+"], right=False)
dep = []
for lb, g in df.groupby("age6", observed=True):
    v = g["residual_rate"]
    se = v.std(ddof=1) / np.sqrt(len(v))
    dep.append({"band": str(lb), "n": int(len(v)),
                "median": float(v.median()), "mean": float(v.mean()),
                "sd": float(v.std(ddof=1)),
                "ci_lo": float(v.mean() - 1.96 * se),
                "ci_hi": float(v.mean() + 1.96 * se)})
R["depreciation"] = dep

# σ_within（与 ANCOVA 同口径：3 车系 × 2 段）
sub = df[df["series"].isin(FOCUS)].copy()
gs = [sub.loc[(sub["series"] == s) & (sub["age_band"] == b),
              "residual_rate"].values for s in FOCUS for b in BANDS]
num = sum((len(g) - 1) * np.var(g, ddof=1) for g in gs if len(g) >= 2)
den = sum(len(g) - 1 for g in gs if len(g) >= 2)
SIGMA_W = float(np.sqrt(num / den))
R["sigma_within_3series"] = SIGMA_W
gs_all = [df.loc[df["age6"] == b, "residual_rate"].values
          for b in df["age6"].cat.categories]
n2 = sum((len(g) - 1) * np.var(g, ddof=1) for g in gs_all if len(g) >= 2)
d2 = sum(len(g) - 1 for g in gs_all if len(g) >= 2)
R["sigma_within_all_6seg"] = float(np.sqrt(n2 / d2))

# 车龄单变量 R²
b1 = np.polyfit(df["age_months"], df["residual_rate"], 1)
pred = np.polyval(b1, df["age_months"])
ss_res = ((df["residual_rate"] - pred) ** 2).sum()
ss_tot = ((df["residual_rate"] - df["residual_rate"].mean()) ** 2).sum()
R["age_r2_linear"] = float(1 - ss_res / ss_tot)

# ============================================================
# 3. 交叉表
# ============================================================
cross = []
for s in FOCUS:
    for b in BANDS:
        g = sub[(sub["series"] == s) & (sub["age_band"] == b)]["residual_rate"]
        se = g.std(ddof=1) / np.sqrt(len(g))
        cross.append({"series": s, "band": b, "n": int(len(g)),
                      "median": float(g.median()), "mean": float(g.mean()),
                      "sd": float(g.std(ddof=1)),
                      "ci_lo": float(g.mean() - 1.96 * se),
                      "ci_hi": float(g.mean() + 1.96 * se)})
R["crosstab"] = cross

# ============================================================
# 4. ANCOVA + 正确的 EMM 置信区间
# ============================================================
FML = ("residual_rate ~ C(series, Treatment('蔚来ES6')) * C(age_band) "
       "+ age_months + mileage_wan + price_wan + C(city_grp)")
model = smf.ols(FML, data=sub).fit()
tbl = anova_lm.anova_lm(model, typ=2)
ssr = tbl.loc["Residual", "sum_sq"]
LAB = {"C(series, Treatment('蔚来ES6'))": "车系",
       "C(age_band)": "车龄段", "C(city_grp)": "城市",
       "C(series, Treatment('蔚来ES6')):C(age_band)": "车系×车龄段",
       "age_months": "车龄(连续)", "mileage_wan": "里程", "price_wan": "指导价"}
eff = []
for k, lab in LAB.items():
    if k in tbl.index:
        ss = tbl.loc[k, "sum_sq"]
        eff.append({"effect": lab, "F": float(tbl.loc[k, "F"]),
                    "p": float(tbl.loc[k, "PR(>F)"]),
                    "eta2p": float(ss / (ss + ssr))})
R["ancova"] = {"n": int(model.nobs), "r2": float(model.rsquared),
               "r2_adj": float(model.rsquared_adj), "effects": eff}

# --- 正确的 EMM：对所有城市取边际 ---
cov_g = {"age_months": float(sub["age_months"].mean()),
         "mileage_wan": float(sub["mileage_wan"].mean()),
         "price_wan": float(sub["price_wan"].mean())}
cities = sorted(sub["city_grp"].unique())
nc = len(cities)

emm = {}
for s in FOCUS:
    for b in BANDS:
        # 构造 nc 行（每城一行），取预测均值的平均 → R̄
        rows = pd.DataFrame({"series": [s] * nc, "age_band": [b] * nc,
                             "age_months": [cov_g["age_months"]] * nc,
                             "mileage_wan": [cov_g["mileage_wan"]] * nc,
                             "price_wan": [cov_g["price_wan"]] * nc,
                             "city_grp": cities})
        pr = model.get_prediction(rows)
        m_ = float(pr.predicted_mean.mean())
        # R̄' V R̄：对各城市的预测做平均，等价于对比向量的线性组合
        # get_prediction 给出每行的 se；但均值不是单行预测，
        # 正确方差 = (1/nc²) Σ_i Σ_j Cov(ŷ_i, ŷ_j)
        # 用模型协方差矩阵直接算
        dm = pr.model.exog if hasattr(pr, "model") else None
        # 稳健做法：用设计矩阵
        X = model.model.data.orig_exog if False else None
        emm[(s, b)] = {"mean": m_, "rows": rows}

# 用设计矩阵精确计算 EMM 的 SE
from patsy import dmatrix  # noqa: E402
design_info = model.model.data.model_spec.design_info
V = model.cov_params().values


def emm_exact(series, band, cities):
    rows = pd.DataFrame({"series": [series] * len(cities),
                         "age_band": [band] * len(cities),
                         "age_months": [cov_g["age_months"]] * len(cities),
                         "mileage_wan": [cov_g["mileage_wan"]] * len(cities),
                         "price_wan": [cov_g["price_wan"]] * len(cities),
                         "city_grp": cities})
    X = np.asarray(dmatrix(design_info, rows))
    Rbar = X.mean(axis=0)            # 对各城市取平均的对比向量
    est = float(Rbar @ model.params.values)
    var = float(Rbar @ V @ Rbar)
    return est, var


emm_out = []
for s in FOCUS:
    for b in BANDS:
        est, var = emm_exact(s, b, cities)
        se = np.sqrt(var)
        emm_out.append({"series": s, "band": b, "emm": est, "se": se,
                        "ci_lo": est - 1.96 * se, "ci_hi": est + 1.96 * se,
                        "n": int(((sub["series"] == s) &
                                  (sub["age_band"] == b)).sum())})
R["emm"] = emm_out


def gap(s1, b1_, s2, b2_):
    e1, v1 = emm_exact(s1, b1_, cities)
    e2, v2 = emm_exact(s2, b2_, cities)
    rows1 = pd.DataFrame({"series": [s1] * len(cities), "age_band": [b1_] * len(cities),
                          "age_months": [cov_g["age_months"]] * len(cities),
                          "mileage_wan": [cov_g["mileage_wan"]] * len(cities),
                          "price_wan": [cov_g["price_wan"]] * len(cities),
                          "city_grp": cities})
    rows2 = rows1.copy(); rows2["series"] = s2; rows2["age_band"] = b2_
    X1 = np.asarray(dmatrix(design_info, rows1)).mean(axis=0)
    X2 = np.asarray(dmatrix(design_info, rows2)).mean(axis=0)
    Rb = X1 - X2
    d = float(Rb @ model.params.values)
    se = float(np.sqrt(Rb @ V @ Rb))
    return d, se


gaps = []
for b in BANDS:
    d, se = gap("Model Y", b, "蔚来ES6", b)
    gaps.append({"contrast": f"Model Y − 蔚来ES6（{b}）", "diff": d, "se": se,
                 "ci_lo": d - 1.96 * se, "ci_hi": d + 1.96 * se})
    d, se = gap("Model Y", b, "蔚来EC6", b)
    gaps.append({"contrast": f"Model Y − 蔚来EC6（{b}）", "diff": d, "se": se,
                 "ci_lo": d - 1.96 * se, "ci_hi": d + 1.96 * se})
    d, se = gap("蔚来ES6", b, "蔚来EC6", b)
    gaps.append({"contrast": f"蔚来ES6 − 蔚来EC6（{b}）", "diff": d, "se": se,
                 "ci_lo": d - 1.96 * se, "ci_hi": d + 1.96 * se})
R["contrasts"] = gaps

# 交互项 MDE
n_min = min(e["n"] for e in emm_out)
R["mde"] = {"n_min": n_min, "sigma": SIGMA_W,
            "mde_pp": float((1.959964 + 0.8416212) * SIGMA_W
                            * np.sqrt(4.0 / n_min) * 100),
            "note": "交互项可检出下限（两组差异之差），单位 pp"}

# ============================================================
# 5. 里程 / 指导价 效应（修正 1.5pp 错误）
# ============================================================
mile_coef = float(model.params.get("mileage_wan", np.nan))
mile_ci = model.conf_int().loc["mileage_wan"].tolist() \
    if "mileage_wan" in model.params.index else [np.nan, np.nan]
price_coef = float(model.params.get("price_wan", np.nan))
R["marginal_effects"] = {
    "mileage_per_wan_km_pp": mile_coef * 100,
    "mileage_ci_pp": [mile_ci[0] * 100, mile_ci[1] * 100],
    "price_per_wan_yuan_pp": price_coef * 100 * 1,   # 每 1 万元
    "note": "来自 3 车系 ANCOVA（n=343），协变量：age_months/city_grp",
}

# ============================================================
# 6. 稳健性检验（修正版，5 项）
# ============================================================
rob = []


def run_rob(name, data, use_city=True):
    d = data.copy()
    if use_city:
        fml = ("residual_rate ~ C(series, Treatment('蔚来ES6')) * C(age_band) "
               "+ age_months + mileage_wan + price_wan + C(city_grp)")
    else:
        fml = ("residual_rate ~ C(series, Treatment('蔚来ES6')) * C(age_band) "
               "+ age_months + mileage_wan + price_wan")
    m = smf.ols(fml, data=d).fit()
    di = m.model.data.model_spec.design_info
    vv = m.cov_params().values
    out = {"name": name, "n": int(m.nobs), "r2": float(m.rsquared)}
    for b in BANDS:
        r1 = pd.DataFrame({"series": ["Model Y"] * nc, "age_band": [b] * nc,
                           "age_months": [d["age_months"].mean()] * nc,
                           "mileage_wan": [d["mileage_wan"].mean()] * nc,
                           "price_wan": [d["price_wan"].mean()] * nc,
                           "city_grp": cities})
        r2 = r1.copy(); r2["series"] = "蔚来ES6"
        if use_city:
            X1 = np.asarray(dmatrix(di, r1)).mean(axis=0)
            X2 = np.asarray(dmatrix(di, r2)).mean(axis=0)
        else:
            X1 = np.asarray(dmatrix(di, r1.drop(columns=["city_grp"]))).mean(axis=0)
            X2 = np.asarray(dmatrix(di, r2.drop(columns=["city_grp"]))).mean(axis=0)
        Rb = X1 - X2
        out[f"gap_{b}"] = float(Rb @ m.params.values) * 100
    return out


rob.append(run_rob("主分析", sub))
lo, hi = sub["mileage"].quantile([0.01, 0.99])
rob.append(run_rob("剔除里程极端值", sub[(sub["mileage"] >= lo) & (sub["mileage"] <= hi)]))
lo, hi = sub["guide_price"].quantile([0.01, 0.99])
rob.append(run_rob("剔除指导价极端值",
                   sub[(sub["guide_price"] >= lo) & (sub["guide_price"] <= hi)]))
sub7 = df[df["series"].isin(FOCUS + ["小鹏G6", "极氪7X", "小米SU7", "乐道L60"])]
rob.append(run_rob("扩展7车系", sub7, use_city=False))
subs = sub[sub["soh_pct"].notna()]
if len(subs) > 60:
    rob.append(run_rob("仅SOH非空子集", subs))
R["robustness"] = rob

# ============================================================
# 7. 换电 / SOH（附带数据质量警示）
# ============================================================
nio = df[df["series"].isin(["蔚来ES6", "蔚来EC6"])]
m_swap = smf.ols("residual_rate ~ is_battery_swap + age_months + mileage_wan "
                 "+ price_wan + C(series)", data=nio).fit()
R["battery_swap"] = {
    "n": int(m_swap.nobs),
    "coef_pp": float(m_swap.params["is_battery_swap[T.True]"] * 100),
    "p": float(m_swap.pvalues["is_battery_swap[T.True]"]),
    "ci_pp": [float(x * 100) for x in m_swap.conf_int().loc["is_battery_swap[T.True]"]],
    "caveat": "is_battery_swap 由页面文案推断，非结构化字段；"
              "审计发现 287 台蔚来/乐道中 128 台被标 False，"
              "结论仅作方向性参考",
}
sohd = df[df["soh_pct"].notna()]
m_soh = smf.ols("residual_rate ~ soh_pct + age_months + mileage_wan "
                "+ price_wan + C(series)", data=sohd).fit()
R["soh"] = {"n": int(m_soh.nobs),
            "coef_pp_per_pct": float(m_soh.params["soh_pct"] * 100),
            "p": float(m_soh.pvalues["soh_pct"]),
            "coverage_pct": float(df["soh_pct"].notna().mean()),
            "caveat": "soh_pct 全部来自平台文案，无实测依据，仅方向性参考"}

# ============================================================
# 8. 城市分布（修正分母）
# ============================================================
R["cities"] = [{"city": k, "n": int(v), "pct": float(v / len(df))}
               for k, v in df["city_name"].value_counts().items()]
R["n_city_groups"] = len(major) + 1

# 挂牌价均值（用于把 MAE 换算成金额）
R["price_context"] = {
    "full_payment_mean": float(df["full_payment"].mean()),
    "guide_price_mean": float(df["guide_price"].mean()),
}

os.makedirs(OUT, exist_ok=True)
with open(os.path.join(OUT, "master_results.json"), "w", encoding="utf-8") as f:
    json.dump(R, f, ensure_ascii=False, indent=1)

# ---------------- 打印摘要 ----------------
print("=" * 88)
print("权威分析结果（修正版）")
print("=" * 88)
print(f"\n样本：原始 {N_RAW} → 新能源 {N_NEV}（剔除燃油车 {N_RAW-N_NEV}）"
      f" → 车龄有效 {N_AGE}")
print(f"σ_within（3车系×2段）= {SIGMA_W:.4f}")
print(f"车龄单变量 R² = {R['age_r2_linear']:.4f}")
print(f"挂牌价均值 = {df['full_payment'].mean():,.0f} 元，"
      f"指导价均值 = {df['guide_price'].mean():,.0f} 元")

print("\n【ANCOVA】")
print(f"  n={R['ancova']['n']}  R²={R['ancova']['r2']:.4f}")
for e in R["ancova"]["effects"]:
    print(f"    {e['effect']:<12} F={e['F']:>8.2f}  p={e['p']:>9.4f}  "
          f"偏η²={e['eta2p']:.4f}")

print("\n【调整后边际均值（正确 CI）】")
for e in emm_out:
    print(f"  {e['series']:<10}{e['band']:<8} {e['emm']:.4f} "
          f"[{e['ci_lo']:.4f}, {e['ci_hi']:.4f}]  n={e['n']}")

print("\n【关键对比（正确 CI）】")
for g in gaps:
    print(f"  {g['contrast']:<26} {g['diff']*100:+6.2f}pp "
          f"[{g['ci_lo']*100:+5.2f}, {g['ci_hi']*100:+5.2f}]")

print(f"\n【交互项可检出下限】n_min={n_min}  σ={SIGMA_W:.4f}  "
      f"MDE=±{R['mde']['mde_pp']:.1f}pp")

print("\n【边际效应】")
print(f"  里程：{R['marginal_effects']['mileage_per_wan_km_pp']:+.2f}pp / 万km  "
      f"CI[{R['marginal_effects']['mileage_ci_pp'][0]:+.2f}, "
      f"{R['marginal_effects']['mileage_ci_pp'][1]:+.2f}]")
print(f"  指导价：{R['marginal_effects']['price_per_wan_yuan_pp']:+.2f}pp / 万元")

print("\n【稳健性检验】")
for r in rob:
    print(f"  {r['name']:<16} n={r['n']:>4}  R²={r['r2']:.3f}  "
          f"0-4年 {r['gap_0-4年']:+5.1f}pp  4年+ {r['gap_4年+']:+5.1f}pp")

print("\n【换电】", f"系数 {R['battery_swap']['coef_pp']:+.2f}pp "
      f"p={R['battery_swap']['p']:.4f} n={R['battery_swap']['n']}")
print("【SOH】", f"系数 {R['soh']['coef_pp_per_pct']:+.3f}pp/百分点 "
      f"p={R['soh']['p']:.4f} n={R['soh']['n']}")
print(f"\n已写出 {OUT}\\master_results.json")
