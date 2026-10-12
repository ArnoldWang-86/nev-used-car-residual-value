# -*- coding: utf-8 -*-
r"""sample_size.py —— 反推最小样本量

思路：不做拍脑袋估计，而是**用手上已有的真实数据（瓜子 59 条）估计方差**，
再按不同研究问题的精度要求反推所需样本量。

四个问题各自对应一种样本量算法：
  Q1  蔚来整体残值率是多少？            → 单样本均值估计（置信区间宽度）
  Q2  蔚来 vs 竞品 谁更保值？            → 两样本均值比较（统计功效）
  Q2b 同上，但控制车龄里程                → 协方差校正后的功效（样本量大幅下降）
  Q3  残值预测模型要多少数据？            → 回归经验法则（每特征 10~20 条）
  Q4  折旧曲线要拟合到多细？              → 分年龄段的分层样本量

输出：output/样本量测算报告.txt
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
GUAZI = os.path.join(ROOT, "data", "clean", "cars_guazi.csv")
REAL = os.path.join(ROOT, "data", "clean", "cars_real.csv")
OUT = os.path.join(ROOT, "output", "样本量测算报告.txt")

out = io.StringIO()
w = lambda *a: print(*a, file=out)

# ---------------------------------------------------------------- 工具
def n_for_mean(sigma: float, margin: float, z: float = 1.96) -> int:
    """估计总体均值，要求在 95% 置信度下误差不超过 margin。"""
    return int(np.ceil((z * sigma / margin) ** 2))


def n_for_two_means(sigma: float, delta: float, alpha: float = 0.05,
                    power: float = 0.80) -> int:
    """两独立样本均值比较，每组所需样本量。"""
    from math import ceil
    z_a = 1.959964 if abs(alpha - 0.05) < 1e-6 else 1.96
    z_b = 0.8416212 if abs(power - 0.80) < 1e-6 else 0.84
    return int(ceil(2 * (z_a + z_b) ** 2 * sigma ** 2 / delta ** 2))


def n_per_group_effect(delta_over_sigma: float, alpha=0.05, power=0.80) -> int:
    from math import ceil
    za, zb = 1.959964, 0.8416212
    return int(ceil(2 * (za + zb) ** 2 / delta_over_sigma ** 2))


# ---------------------------------------------------------------- 读真实数据
w("=" * 88)
w("样本量测算报告")
w("=" * 88)
w("")
w("数据基础：用已采到的真实瓜子车源估方差，而非套用通用经验值。")
w("")

df = pd.read_csv(GUAZI, encoding="utf-8-sig") if os.path.exists(GUAZI) else pd.DataFrame()
if df.empty:
    w("缺少瓜子车源数据，无法估算。")
    open(OUT, "w", encoding="utf-8").write(out.getvalue())
    print(out.getvalue())
    raise SystemExit(0)


def to_wan(s):
    m = re.search(r"([\d.]+)", str(s))
    return float(m.group(1)) if m else np.nan


def to_year(s):
    m = re.search(r"(20\d{2})", str(s))
    return int(m.group(1)) if m else np.nan


df["价格万"] = df["price"].map(to_wan)
df["里程万"] = df["road_haul"].map(to_wan)
df["上牌年"] = df["license_date"].map(to_year)
df["车龄年"] = 2026 - df["上牌年"]
df["车系"] = df["title"].astype(str).str.extract(r"^([^\s]+)")

# 残值率：优先用详情数据里的真实「挂牌价/官方指导价」
real = pd.read_csv(REAL, encoding="utf-8-sig") if os.path.exists(REAL) else pd.DataFrame()

w("-" * 88)
w("一、我们现有数据的方差结构")
w("-" * 88)
w("")
w(f"  瓜子列表车源（含车龄、价格、里程）: {len(df)} 条")
if not real.empty and "residual_rate" in real.columns:
    rr = pd.to_numeric(real["residual_rate"], errors="coerce").dropna()
    w(f"  有官方指导价可算残值率的        : {len(rr)} 条")
    w(f"    残值率 均值={rr.mean():.4f} 标准差={rr.std(ddof=1):.4f} "
      f"区间=[{rr.min():.3f}, {rr.max():.3f}]")
    sigma_total = float(rr.std(ddof=1))
else:
    sigma_total = np.nan

# 用列表数据的「价格」在同车龄内估变异（作为残值率变异的代理）
w("")
w("  用车龄分层看价格离散度（残值率变异的代理指标）")
w("")
w(f"  {'车龄':<8}{'样本':>5}{'价格中位万':>11}{'价格标准差':>11}{'变异系数':>9}")
for age in sorted(df["车龄年"].dropna().unique()):
    sub = df[df["车龄年"] == age]["价格万"].dropna()
    if len(sub) < 3:
        continue
    cv = sub.std(ddof=1) / sub.mean()
    w(f"  {int(age):<8}{len(sub):>5}{sub.median():>11.2f}"
      f"{sub.std(ddof=1):>11.2f}{cv:>9.3f}")

# 残值率的变异系数（用已有 16 条）
if not np.isnan(sigma_total):
    cv_rate = sigma_total / rr.mean()
    w("")
    w(f"  残值率变异系数 CV = {cv_rate:.3f}"
      f"（该值决定后面所有样本量计算）")

w("")
w("-" * 88)
w("二、Q1 蔚来整体残值率估计 —— 需要多少条才能把误差控制住？")
w("-" * 88)
w("")
w("  公式：n = (1.96 × σ / 允许误差)²")
w("")
if not np.isnan(sigma_total):
    w(f"  取 σ = {sigma_total:.4f}（实测残值率标准差）")
    w("")
    w(f"  {'允许误差':<14}{'所需样本量':>12}   说明")
    for m, note in [(0.05, "粗略量级，只想知道「大概三四折」"),
                    (0.03, "能说清「约 40%±3%」，业务可用"),
                    (0.02, "较精细，可写进分析报告"),
                    (0.01, "很精细，通常没必要")]:
        w(f"  ±{m:<13}{n_for_mean(sigma_total, m):>12}   {note}")
    w("")
    w("  结论：只要 ±3% 的精度，约 "
      f"{n_for_mean(sigma_total, 0.03)} 条就够。")
    w("        这正是「最小样本量」的第一个答案。")

w("")
w("-" * 88)
w("三、Q2 蔚来 vs 竞品「谁更保值」 —— 这是最吃样本量的问题")
w("-" * 88)
w("")
w("  公式（两独立样本）：n_每组 = 2(z_α/2 + z_β)² σ² / Δ²")
w("  其中 Δ 是我们要检出的残值率差异")
w("")
if not np.isnan(sigma_total):
    w("  【不做任何控制时】σ 取残值率标准差，所需每组样本量：")
    w("")
    w(f"  {'要检出的差异 Δ':<18}{'每组样本量':>12}{'两组合计':>10}")
    for d in [0.05, 0.08, 0.10, 0.12, 0.15]:
        n = n_for_two_means(sigma_total, d)
        w(f"  {d:<18.2f}{n:>12}{2*n:>10}")
    w("")
    w("  解读：想检出「蔚来比竞品高 5 个百分点」，每组要几百条 —— 不现实。")
    w("        想检出 12 个百分点，每组约 "
      f"{n_for_two_means(sigma_total, 0.12)} 条。")

    # ---- 关键：控制车龄后的功效 ----
    w("")
    w("  【控制车龄后的关键改进】")
    # 用列表数据估计：车龄能解释多少价格方差
    sub = df.dropna(subset=["价格万", "车龄年"])
    r2_age = sub["价格万"].corr(sub["车龄年"]) ** 2 if len(sub) > 5 else np.nan
    sub2 = df.dropna(subset=["价格万", "车龄年", "里程万"])
    if len(sub2) > 5:
        X = np.column_stack([np.ones(len(sub2)), sub2["车龄年"], sub2["里程万"]])
        y = sub2["价格万"].values
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        r2_full = 1 - resid.var() / y.var()
    else:
        r2_full = r2_age
    w(f"    车龄单独解释价格方差 R² = {r2_age:.3f}")
    w(f"    车龄+里程共同解释   R² = {r2_full:.3f}")
    sigma_ctrl = sigma_total * np.sqrt(max(0.0, 1 - r2_full))
    w(f"    控制后残差标准差 σ' = σ×√(1-R²) = {sigma_ctrl:.4f}"
      f"（从 {sigma_total:.4f} 降到 {sigma_ctrl:.4f}）")
    w("")
    w("    因为比较「品牌溢价」时会把车龄里程放进模型做协变量，")
    w("    所以真正该用的是 σ'：")
    w("")
    w(f"  {'要检出的差异 Δ':<18}{'每组样本量':>12}{'两组合计':>10}")
    for d in [0.05, 0.08, 0.10, 0.12, 0.15]:
        n = n_for_two_means(sigma_ctrl, d)
        w(f"  {d:<18.2f}{n:>12}{2*n:>10}")
    w("")
    n12 = n_for_two_means(sigma_ctrl, 0.12)
    n10 = n_for_two_means(sigma_ctrl, 0.10)
    n08 = n_for_two_means(sigma_ctrl, 0.08)
    w(f"  结论：控制车龄里程后，检出 12 个百分点差异每组约 {n12} 条，"
      f"\n        检出 10 个百分点约 {n10} 条，检出 8 个百分点约 {n08} 条。")
    w("")
    w("  ⚠️ 这个「控制车龄后样本量骤降」的结论有一个**前提条件**：")
    w("     两个品牌的车龄分布必须【有重叠】。")
    w("     若蔚来的车普遍 5 年、竞品普遍 1 年，模型就是在做外推，")
    w("     协变量控制不成立，样本量再大也救不了。")
    w("     因此采集时应【按车龄分层】取数，而不是随机抓。")

w("")
w("-" * 88)
w("四、Q3 残值预测模型需要多少数据")
w("-" * 88)
w("")
w("  回归经验法则：每个自变量至少 10~20 个观测，否则系数不稳、易过拟合。")
w("")
n_feat_list = [5, 8, 12, 15, 20]
w(f"  {'自变量个数':<14}{'下限(10/变量)':>14}{'稳健(20/变量)':>14}")
for k in n_feat_list:
    w(f"  {k:<14}{k*10:>14}{k*20:>14}")
w("")
w("  我们计划用的特征（含类别展开前的原始维度）：")
w("    车龄、里程、年均里程、电池容量、SOH、换电标记、")
w("    车系、驱动形式、车况等级、城市 → 约 10 个原始特征")
w("")
w("  ⚠️ 重要：类别特征在模型里会展开成多个哑变量，")
w("     所以「10 个原始特征」的实际维度远不止 10。")
w("     例如车系若覆盖蔚来 8 个车系 + 竞品 20 个车型 = 28 个哑变量。")
w("     因此上表的 k 应当按【展开后的维度】理解，")
w("     而我们的实际维度大约在 12~20 之间。")
w("")
w("  结论：① 建「务实版」模型（车龄+里程+电池容量等 5~6 个数值特征，")
w("           类别只保留品牌）→ 50~120 条可用；")
w("        ② 想用完整特征集（含车系哑变量）建稳健模型 → 需要 200~400 条。")
w("        **这是「最小样本量」的第三个答案。**")

w("")
w("-" * 88)
w("五、Q4 折旧曲线要拟合到多细 —— 决定分年龄段的样本量")
w("-" * 88)
w("")
w("  若要把折旧曲线按车龄分段画出来，每个年龄段都需要足够样本。")
w("")
AGE_BINS = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 7), (7, 10)]
w(f"  {'车龄段':<12}{'现有条数':>10}{'建议最少':>10}")
need_total = 0
for lo, hi in AGE_BINS:
    have = int(((df["车龄年"] >= lo) & (df["车龄年"] < hi)).sum())
    need = 15          # 每段 15 条可画出可用的段均值
    need_total += need
    w(f"  {lo}-{hi}年{'':<6}{have:>10}{need:>10}")
w("")
w(f"  按每段 15 条计，画 7 段曲线需要约 {need_total} 条。")
w("  **这是「最小样本量」的第四个答案。**")

w("")
w("=" * 88)
w("六、汇总结论与采集建议")
w("=" * 88)
w("")
if not np.isnan(sigma_total):
    a1 = n_for_mean(sigma_total, 0.03)
    a2 = n_for_two_means(sigma_ctrl, 0.12)
    a3 = 100            # 务实版模型
    a4 = need_total
    w("  四个问题各自的最小样本量：")
    w("")
    w(f"    Q1 描述蔚来整体残值率（±3%）        : {a1:>4} 条")
    w(f"    Q2 品牌横向对比（检出 12pp，控制车龄）: {a2:>4} 条/每品牌")
    w(f"    Q3 建务实版预测模型（5~6 特征）      : {a3:>4} 条")
    w(f"    Q4 画出分段折旧曲线                  : {a4:>4} 条")
    w("")
    w("  " + "-" * 60)
    w("  推荐的最小可行方案（同时满足以上四项）：")
    w("  " + "-" * 60)
    w("")
    w("    蔚来（含乐道）   : 120~150 条   ← 自家是核心，要给足")
    w("    每个竞品品牌     :  40~60 条    ← 4~6 个品牌做对标")
    w("    ─────────────────────────────")
    w("    合计             : 300~450 条")
    w("")
    w("  这正好对应：")
    w("    · 蔚来 8 个城市页 + 8 个车系页，去重后约 250~320 条")
    w("    · 竞品 7 个品牌页，去重后约 60~100 条")
    w("    · 合计已超过上述下限")
    w("")
    w("  也就是说：**你现有清单里的「第一批 + 第四批」就能满足最小要求**，")
    w("  第二批（乐道）、第三批（车系页）是提升余量，第三/五批可不再采。")
    w("")
    w("  ⚠️ 需要说明的限制：")
    w("    · σ=0.0871 只由 16 条（有官方指导价的那批）估出，")
    w("      本身不确定性很大；样本增加后必须重算这套数字。")
    w("      这也是为什么报告的每一个结论都要写「基于当前 N 条」。")
    w("    · 单期快照无法做时间外推验证，这是样本量解决不了的问题。")
    w("    · 品牌对比要求各品牌车龄分布有重叠，否则协变量控制不成立。")
    w("    · 若某品牌实际只有 20 条在售，不要为凑样本反复翻页")
    w("      （robots 明确 Disallow: /*?* ，带 ?page= 的 URL 不能抓）。")

open(OUT, "w", encoding="utf-8").write(out.getvalue())
print(out.getvalue())
