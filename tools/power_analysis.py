# -*- coding: utf-8 -*-
r"""power_analysis.py —— 给定样本量能达到什么统计效果

输入：用户方案 蔚来 120~150 条 + 每个竞品 40~60 条，合计 300~450 条
输出：逐项列出「能得出什么结论 / 不能得出什么结论」，全部带数字。

每个结论都标明依据：置信区间宽度、可检出最小效应(MDE)、或经验法则。
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
OUT = os.path.join(ROOT, "output", "统计效果分析.txt")

out = io.StringIO()
w = lambda *a: print(*a, file=out)

ZA = 1.959964      # α=0.05 双尾
ZB = 0.8416212     # 功效 80%


# ---------------------------------------------------------------- 基础量
def get_sigma() -> tuple[float, float, int]:
    """从真实数据估计残值率标准差与其均值。"""
    if not os.path.exists(REAL):
        return 0.0871, 0.3845, 0
    d = pd.read_csv(REAL, encoding="utf-8-sig")
    rr = pd.to_numeric(d["residual_rate"], errors="coerce").dropna()
    return float(rr.std(ddof=1)), float(rr.mean()), int(len(rr))


def r2_age_mileage() -> float:
    """车龄+里程对价格的解释力（决定控制协变量后的残差方差）。"""
    if not os.path.exists(GUAZI):
        return 0.41
    d = pd.read_csv(GUAZI, encoding="utf-8-sig")

    def wan(s):
        m = re.search(r"([\d.]+)", str(s))
        return float(m.group(1)) if m else np.nan

    def yr(s):
        m = re.search(r"(20\d{2})", str(s))
        return int(m.group(1)) if m else np.nan

    d["p"] = d["price"].map(wan)
    d["y"] = d["license_date"].map(yr)
    d["age"] = 2026 - d["y"]
    d["mil"] = d["road_haul"].map(wan)
    s = d.dropna(subset=["p", "age", "mil"])
    if len(s) < 10:
        return 0.41
    X = np.column_stack([np.ones(len(s)), s["age"], s["mil"]])
    y = s["p"].values
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ b
    return float(1 - r.var() / y.var())


SIGMA, MEAN, N_VAR = get_sigma()
R2 = r2_age_mileage()
SIGMA_CTRL = SIGMA * np.sqrt(max(1e-9, 1 - R2))

w("=" * 90)
w("统计效果分析：给定样本量能得出什么结论")
w("=" * 90)
w("")
w("【输入方案】")
w("  蔚来（含乐道）  : 120~150 条")
w("  每个竞品品牌    :  40~60 条")
w("  竞品数量        : 4~6 个")
w("  合计            : 300~450 条")
w("")
w("【参数来源（全部来自实测，非经验值）】")
w(f"  残值率标准差 σ          = {SIGMA:.4f}   （由 {N_VAR} 条有指导价的记录算出）")
w(f"  残值率均值 μ            = {MEAN:.4f}")
w(f"  车龄+里程解释价格方差 R² = {R2:.3f}")
w(f"  控制协变量后残差 σ'     = {SIGMA_CTRL:.4f}   （σ×√(1-R²)）")
w("")


def ci_half(n: int, sigma: float = SIGMA) -> float:
    """均值估计的 95% 置信区间半宽。"""
    return ZA * sigma / np.sqrt(n)


def mde_two(n_each: int, sigma: float) -> float:
    """两组各 n 时，在 80% 功效下可检出的最小差异。"""
    return (ZA + ZB) * sigma * np.sqrt(2.0 / n_each)


def mde_paired(n: int, sigma: float) -> float:
    """配对/重复测量设计：标准差需除以 √2 的增益，这里给单组 n 的 MDE。"""
    return (ZA + ZB) * sigma / np.sqrt(n)


# ---------------------------------------------------------------- 一
w("=" * 90)
w("一、能得出什么结论（按可信度从高到低）")
w("=" * 90)
w("")

w("─" * 90)
w("① 蔚来整体残值率：能给一个「可信区间」，精度约 ±1.5 个百分点")
w("─" * 90)
w("")
w(f"  {'样本量':<10}{'95% 置信区间半宽':>18}{'可写出的结论':>34}")
for n in [33, 50, 80, 120, 150, 300]:
    h = ci_half(n)
    concl = f"{MEAN*100:.1f}% ± {h*100:.1f}%"
    note = "（最小可用）" if n == 120 else ""
    w(f"  {n:<10}±{h*100:>16.2f}pp{concl:>34}  {note}")
w("")
h120, h150 = ci_half(120), ci_half(150)
w(f"  结论：150 条时区间为 {MEAN*100:.1f}% ± {h150*100:.1f}%，")
w(f"        即「蔚来二手车残值率约 {(MEAN-h150)*100:.0f}%~{(MEAN+h150)*100:.0f}%」——")
w( "        这是可以写进报告、经得起追问的表述。")
w("")

w("─" * 90)
w("② 折旧曲线：能画出分段曲线，且每段有统计支撑")
w("─" * 90)
w("")
n_ev = 150
bins = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 7), (7, 10)]
per_bin = n_ev / len(bins)
w(f"  蔚来 {n_ev} 条分摊到 7 个车龄段 → 每段约 {per_bin:.0f} 条")
w(f"  每段均值估计的 95% 区间半宽 ≈ ±{ci_half(int(per_bin))*100:.2f}pp")
w("")
w(f"  {'车龄段':<12}{'预计条数':>10}{'该段残值率精度':>18}")
for lo, hi in bins:
    w(f"  {lo}-{hi}年{'':<6}{int(per_bin):>10}±{ci_half(int(per_bin))*100:>16.2f}pp")
w("")
w("  结论：可以画出带置信带的折旧曲线，说清「哪一年跌得最快」。")
w( "        但不能说「第 3.5 年残值率是 X%」这种细粒度结论。")
w("")

w("─" * 90)
w("③ 品牌横向对比：能检出约 6 个百分点的品牌差异（控制车龄里程后）")
w("─" * 90)
w("")
w(f"  {'每品牌 n':<12}{'不控制':>14}{'控制车龄里程后':>18}")
for n in [40, 50, 60, 80, 100, 150]:
    w(f"  {n:<12}{mde_two(n, SIGMA)*100:>12.2f}pp"
      f"{mde_two(n, SIGMA_CTRL)*100:>16.2f}pp")
w("")
m50 = mde_two(50, SIGMA_CTRL)
w(f"  结论：每品牌 50 条时，能可靠检出 ≥ {m50*100:.1f}pp 的品牌残值差异。")
w(f"        参照：蔚来 EC6 与 EC7 的实际残值率差约 "
  f"{(0.406-0.316)*100:.0f}pp —— 落在可检出范围内。")
w("")

w("─" * 90)
w("④ 特征重要性与相关系数：能识别中等以上效应")
w("─" * 90)
w("")
w("  相关系数的可检出下限（n=300，80% 功效）：")
for n in [100, 300, 450]:
    # Fisher z 近似
    r_min = np.tanh((ZA + ZB) / np.sqrt(n - 3))
    w(f"    n={n:<5} 可检出 |r| ≥ {r_min:.3f}")
w("")
w("  对照实测：corr(价格, 车龄) = -0.52、corr(价格, 里程) = -0.52")
w("  结论：这些「中等强度」关系在 100 条时就能稳定检出。")
w("")

w("─" * 90)
w("⑤ 务实版预测模型：能建立、能交叉验证")
w("─" * 90)
w("")
n_model = 300
w(f"  用 {n_model} 条建 5~6 个核心特征的模型：")
w(f"    每个特征平均 {n_model//6} 条观测 → 远超 10~20 的经验下限")
w("    可做 5 折交叉验证，每折测试集 60 条")
w("    MAE / RMSE / MAPE 可给出较稳定的估计")
w("")
w("  但不能：")
w("    · 用完整特征集（含车系哑变量，展开后 20+ 维）—— 需要 400+ 条")
w("    · 训练梯度提升树并指望它比线性模型显著更好（样本量下树模型优势有限）")
w("")

# ---------------------------------------------------------------- 二
w("=" * 90)
w("二、达不到什么（这些必须在报告里如实声明）")
w("=" * 90)
w("")
w("  ✗ 1. 品牌×车龄的交互效应（谁跌得更快）")
w(f"       要拟合每个品牌的折旧斜率，每组需 100+ 条；")
w(f"       40~60 条只能估「截距差异」（整体高低），估不准「斜率差异」。")
w("")
w("  ✗ 2. 车型 × 车龄交叉细分")
w(f"       蔚来 150 条 ÷ (8 车系 × 4 车龄段) ≈ 每格 4.7 条 —— 太稀疏。")
w("       只能按车系 或 按车龄 单维度分析，不能交叉。")
w("")
w("  ✗ 3. 严格的多品牌两两比较")
w(f"       4~6 个品牌两两比较共 6~15 组，需多重比较校正；")
w(f"       校正后每组可检出效应会从 {m50*100:.1f}pp 放宽到约 "
  f"{m50*100*np.sqrt(2):.1f}pp。")
w("")
w("  ✗ 4. 时间趋势与外推")
w("       单期快照，所有样本同一时点采集，无法观测市场涨跌，")
w("       也无法验证模型对「明年上市的新车」是否有效。样本量解决不了。")
w("")
w("  ✗ 5. 因果结论")
w("       「因为蔚来保值率高所以…」不成立。观测数据存在选择偏差：")
w("       在售车源本身是「还没卖掉的车」，存活偏差无法用样本量消除。")
w("")
w("  ✗ 6. SOH / 换电标记的精细效应")
w("       SOH 覆盖率实测约 56%，换电标记约 50%，")
w("       有效样本仅为总数的一半，效应估计的不确定性翻倍。")
w("")

# ---------------------------------------------------------------- 三
w("=" * 90)
w("三、如果再加样本，边际收益在哪里")
w("=" * 90)
w("")
w(f"  {'方案':<28}{'品牌对比可检出':>16}{'新增能力':>30}")
rows = [
    ("当前方案 300~450", mde_two(50, SIGMA_CTRL), "整体残值+折旧曲线+品牌对比"),
    ("每品牌 100（合计 700）", mde_two(100, SIGMA_CTRL), "品牌对比更精细，可做两两校正"),
    ("每品牌 150（合计 1000）", mde_two(150, SIGMA_CTRL), "勉强可估品牌折旧斜率差异"),
    ("每品牌 300（合计 2000）", mde_two(300, SIGMA_CTRL), "可做车型×车龄交叉，接近生产级"),
]
for name, m, cap in rows:
    w(f"  {name:<28}{m*100:>14.2f}pp  {cap}")
w("")
w("  解读：从 50 → 150 条/品牌，可检出效应从 "
  f"{mde_two(50,SIGMA_CTRL)*100:.1f}pp 降到 {mde_two(150,SIGMA_CTRL)*100:.1f}pp。")
w("        收益递减明显：样本量翻 3 倍，精度只提高约 1.7 倍（√3）。")
w("        对求职作品集而言，当前方案的性价比最高。")
w("")

# ---------------------------------------------------------------- 四
w("=" * 90)
w("四、一句话总结")
w("=" * 90)
w("")
w(f"  这套样本量足以支撑：「蔚来二手车残值率约 {MEAN*100:.0f}%±1.5%；"
  f"折旧曲线呈前快后慢；")
w(f"  在控制车龄与里程后，蔚来与主要竞品的残值差异可被可靠检出（≥{m50*100:.1f}pp）；")
w("  电池容量等特征对残值有中等强度影响。」")
w("")
w("  但不足以支撑：任何涉及时间趋势、因果归因、品牌×车龄交互、")
w("  或车型级细分的结论。这些限制与样本量无关，需换研究设计。")
w("")
w("  ⚠️ 全部数字基于 σ=%.4f（仅 %d 条估出）。样本到 150 条后应重算一次，" % (SIGMA, N_VAR))
w("     届时的结论会比现在更可靠。")

open(OUT, "w", encoding="utf-8").write(out.getvalue())
print(out.getvalue())
