# -*- coding: utf-8 -*-
r"""crosstab_feasibility.py —— 车型×车龄交叉分析需要多少条？

核心观点：交叉分析的约束**不是总量，而是每个格子的条数以及分布的偏斜程度**。
真实车源在不同车型、不同车龄上的分布高度不均（ES6 可能几十条，
EP9 可能 0 条；准新车多、老车少），因此"总量翻倍"不等于"每格翻倍"。

本脚本：
  1. 用实测数据估计真实的车型分布与车龄分布
  2. 按几种网格粒度模拟 1000 条的落格情况
  3. 算出每格的置信区间宽度与达标格子比例
  4. 给出「1000 条能做到哪一档粒度」的明确结论
"""
from __future__ import annotations

import io
import os
import re
import sys

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REAL = os.path.join(ROOT, "data", "clean", "cars_real.csv")
GUAZI = os.path.join(ROOT, "data", "clean", "cars_guazi.csv")
OUT = os.path.join(ROOT, "output", "交叉分析可行性.txt")

out = io.StringIO()
w = lambda *a: print(*a, file=out)
ZA = 1.959964
RNG = np.random.RandomState(20261012)

# ---------------------------------------------------------------- 实测参数
sigma = 0.0871
n_ev = 0
if os.path.exists(REAL):
    d = pd.read_csv(REAL, encoding="utf-8-sig")
    rr = pd.to_numeric(d["residual_rate"], errors="coerce").dropna()
    if len(rr) > 2:
        sigma = float(rr.std(ddof=1))
        n_ev = len(rr)

# 车型分布（用真实数据）
series_counts = {}
if os.path.exists(GUAZI):
    g = pd.read_csv(GUAZI, encoding="utf-8-sig")
    g["车系"] = g["title"].astype(str).str.extract(r"^([^\s]+)")
    series_counts = g["车系"].value_counts().to_dict()

# 车龄分布（用真实数据）
age_counts = {}
if os.path.exists(GUAZI):
    g = pd.read_csv(GUAZI, encoding="utf-8-sig")

    def yr(s):
        m = re.search(r"(20\d{2})", str(s))
        return int(m.group(1)) if m else np.nan
    g["y"] = g["license_date"].map(yr)
    g["age"] = 2026 - g["y"]
    age_counts = g["age"].dropna().astype(int).value_counts().to_dict()

w("=" * 94)
w("车型 × 车龄 交叉分析：1000 条到底能不能做？")
w("=" * 94)
w("")
w("【为什么不是「总量够就行」】")
w("  交叉分析把样本切成网格，每个格子都要单独估计。")
w("  格子数 = 车型数 × 车龄段数。总量不变时，格子越细，每格越少。")
w("  更麻烦的是**分布偏斜**：真实车源里 ES6 可能有几十条、EP9 一条都没有，")
w("  准新车多、7 年以上老车少。平均每格 20 条，可能一半格子只有 2~3 条。")
w("")

w("【实测的参数】")
w(f"  残值率标准差 σ = {sigma:.4f}（{n_ev} 条有指导价的记录）")
if series_counts:
    w(f"  已采车系的观测份额（注意：这是采样偏差，不是真实库存结构）：")
    tot = sum(series_counts.values())
    for k, v in list(series_counts.items())[:12]:
        w(f"     {k:<12} {v:>3} 条  ({v/tot:>5.1%})")
    w("")
    w("  ⚠️ 上面 ES6 占近 68%，是因为列表页第一屏 ES6 扎堆，")
    w("     不代表蔚来二手库存的真实车型结构。")
    w("     下面模拟改用【更接近真实结构的车型份额】，并给出敏感性对比。")

# 车型份额：两种情形对比
#   情形 A：按已采样本（高度偏斜，ES6 主导）
#   情形 B：按蔚来真实销量结构（更均衡，作为主模拟）
SHARES_SAMPLED = (np.array(list(series_counts.values()), dtype=float)
                  if series_counts else np.array([0.68, 0.10, 0.09, 0.07, 0.05, 0.02]))
SHARES_SAMPLED = SHARES_SAMPLED / SHARES_SAMPLED.sum()

# 蔚来二手库存车型结构（按销量与保有量估算，用于主模拟）
# ES6 保有量最大，EC6/ET5/ET5T/ES8 次之，ET7/EC7 中等，ES7/ES9/ET9/EP9 稀少
SHARES_REAL = np.array([0.26, 0.15, 0.13, 0.12, 0.11, 0.08, 0.06, 0.05, 0.03, 0.01])
SHARES_REAL = SHARES_REAL / SHARES_REAL.sum()
w("")
w("  主模拟采用的车型份额（近真实结构）：")
labels_real = ["ES6", "EC6", "ET5", "ET5T", "ES8", "ET7", "EC7", "ES7", "ES9/ET9", "其他"]
for lb, sh in zip(labels_real, SHARES_REAL):
    w(f"     {lb:<10} {sh:>6.1%}")

w("")
w("=" * 94)
w("一、不同网格粒度下，1000 条会怎么分布")
w("=" * 94)
w("")

# 网格方案：(名称, 车型维数, 车龄段数, 说明)
GRIDS = [
    ("粗：4 车型族 × 4 车龄段", 4, 4, "SUV/轿跑/轿车/新势力，车龄 0-2/2-4/4-6/6+"),
    ("中：6 车型 × 5 车龄段", 6, 5, "ES6/ES8/EC6/EC7/ET5/ET7，车龄 0-1/1-2/3-4/5-6/7+"),
    ("细：9 车型 × 7 车龄段", 9, 7, "蔚来全系，车龄按年分段"),
    ("极细：12 车型 × 8 车龄段", 12, 8, "含乐道，车龄细分"),
]

TOTAL = 1000

# 真实的车型偏斜：用观测到的份额，若不足则用典型长尾
if series_counts:
    shares = np.array(list(series_counts.values()), dtype=float)
    shares = shares / shares.sum()
else:
    shares = np.array([0.45, 0.20, 0.12, 0.08, 0.05, 0.04, 0.03, 0.02, 0.01])

# 车龄偏斜：新车多老车少
age_shares_default = np.array([0.10, 0.16, 0.22, 0.20, 0.14, 0.10, 0.05, 0.03])


def simulate(total: int, n_model: int, n_age: int, reps: int = 300,
             shares_use: np.ndarray | None = None):
    """模拟多次落格，返回每格条数的统计（分布偏斜下会很不均）。"""
    if shares_use is None:
        shares_use = SHARES_REAL
    all_cells = []
    for _ in range(reps):
        s = (shares_use[:n_model] if len(shares_use) >= n_model else
             np.concatenate([shares_use,
                             np.full(n_model - len(shares_use),
                                     shares_use.min() / 2)]))
        s = s / s.sum()
        base = (age_shares_default[:n_age] if n_age <= len(age_shares_default)
                else np.concatenate([age_shares_default,
                                     np.full(n_age - len(age_shares_default), 0.01)]))
        a = base / base.sum()
        cells = (np.outer(s, a) * total).astype(int)
        all_cells.append(cells.ravel())
    return np.array(all_cells)


w(f"  {'网格方案':<26}{'格子数':>7}{'平均每格':>10}{'中位每格':>10}"
  f"{'<10条的格':>11}{'≥20条的格':>11}")
for name, nm, na, note in GRIDS:
    C = simulate(TOTAL, nm, na)
    cells = nm * na
    mean_cell = C.mean()
    median_cell = np.median(C, axis=1).mean()
    frac_lt10 = (C < 10).mean()
    frac_ge20 = (C >= 20).mean()
    w(f"  {name:<26}{cells:>7}{mean_cell:>10.1f}{median_cell:>10.1f}"
      f"{frac_lt10:>10.0%}{frac_ge20:>10.0%}")

w("")
w("  说明：「<10条的格」占比越高，这些格子里的残值率估计越不可靠。")
w("")
w("  敏感性：若用「已采样本」那种高度偏斜的分布（ES6 占 68%），结果会更差：")
w(f"  {'网格方案':<26}{'≥20条的格(真实结构)':>20}{'≥20条的格(偏斜分布)':>22}")
for name, nm, na, note in GRIDS:
    C1 = simulate(TOTAL, nm, na, shares_use=SHARES_REAL)
    C2 = simulate(TOTAL, nm, na, shares_use=SHARES_SAMPLED)
    w(f"  {name:<26}{(C1>=20).mean():>19.0%}{(C2>=20).mean():>21.0%}")
w("")
w("  → 同样的 1000 条，分布越偏斜，可用格子越少。")
w("     这说明：**做交叉分析时，分布比总量更重要。**")

w("=" * 94)
w("二、每格条数 → 该格残值率的置信区间宽度")
w("=" * 94)
w("")
w(f"  {'每格条数':<10}{'95% 区间半宽':>16}{'能不能用':>34}")
for n in [3, 5, 10, 15, 20, 30, 50]:
    half = ZA * sigma / np.sqrt(n)
    if n < 10:
        judge = "不可用，只能当个案描述"
    elif n < 20:
        judge = "勉强，仅供趋势参考"
    elif n < 30:
        judge = "可用，误差约 ±3~4pp"
    else:
        judge = "较可靠，可写进结论"
    w(f"  {n:<10}±{half*100:>14.1f}pp{judge:>34}")
w("")
w("  换算：若每格只有 5 条，该格残值率的 95% 区间宽达 ±7.6pp，")
w("        而品牌间的真实差异往往也就 3~9pp —— 结论会被噪声淹没。")
w("")

w("=" * 94)
w("三、直接回答：1000 条能做到哪一档")
w("=" * 94)
w("")

results = []
for name, nm, na, note in GRIDS:
    C = simulate(TOTAL, nm, na)
    cells = nm * na
    mean_cell = C.mean()
    frac_ok = (C >= 20).mean()          # 每格 ≥20 视为可用
    half = ZA * sigma / np.sqrt(max(1, mean_cell))
    verdict = ("✅ 可以做" if frac_ok >= 0.8 else
               "🔸 可做但需合并格子" if frac_ok >= 0.4 else
               "❌ 太细，格子会空")
    results.append((name, cells, mean_cell, frac_ok, half, verdict))
    w(f"  {name}")
    w(f"     格子数 {cells} | 平均每格 {mean_cell:.1f} 条 | "
      f"≥20 条的格占 {frac_ok:.0%} | 该格精度 ±{half*100:.1f}pp")
    w(f"     判定：{verdict}")
    w("")

w("=" * 94)
w("四、结论与建议")
w("=" * 94)
w("")
w("  1. **1000 条可以支撑「中等粒度」的车型×车龄交叉**，")
w("     但不能支撑「蔚来全系 9 车型 × 7 车龄段」这种细网格。")
w("")
w("     推荐网格：**4~6 个车型 × 4~5 个车龄段 = 16~30 个格子**，")
w("     平均每格 33~62 条，多数格子能达到 ≥20 条的可用线。")
w("")
w("  2. **比总量更关键的是「按格子配额采集」。**")
w("     现在的采集方式是抓列表页，无法控制车型/车龄的落格分布，")
w("     结果会高度偏斜：ES6 一大堆、EP9 零条、准新车多、老车少。")
w("     若目标就是交叉分析，应在采集时【按车型+车龄设定目标配额】，")
w("     缺的格子定向补采。")
w("")
w("  3. **一个更实际的替代方案：不做交叉，改做交互项回归。**")
w("     与其把数据切碎成格子，不如建一个含交互项的模型：")
w("        残值率 ~ 车龄 + 里程 + 车型 + 车型×车龄")
w("     这样每个车型不需要独立成格，而是共享信息来估计斜率差异。")
w("     1000 条足以估计少量交互项（经验法则：每个交互项 50~100 条）。")
w("     这比硬切网格更省样本，也是统计上更推荐的做法。")
w("")
w("  4. 若坚持做细网格（9 车型 × 7 车龄段 = 63 格）且每格要 20 条，")
w(f"     理论总量需 63 × 20 = {63*20} 条，且**必须按配额采集**才能达到。")
w("")
w("  ⚠️ 所有计算基于 σ = %.4f（仅 %d 条估出），样本增加后应重算。" % (sigma, n_ev))

open(OUT, "w", encoding="utf-8").write(out.getvalue())
print(out.getvalue())
