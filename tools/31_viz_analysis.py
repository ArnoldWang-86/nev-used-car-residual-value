# -*- coding: utf-8 -*-
r"""31_viz_analysis.py —— 图2/3/4/5：交叉、箱线、效应量、调整均值

口径统一：ANCOVA 效应量与 EMM 全部取自 output/master_results.json
         （由 src/40_master_analysis.py 产出，已修正原 CI 偏宽 bug）
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from viz_style import C as PAL, NIO_COLORS, TESLA_COLOR, clean, title, footnote, save  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "data", "raw", "detail_from_cache.jsonl")
FIG = os.path.join(ROOT, "output", "figures")
FOCUS = ["蔚来ES6", "蔚来EC6", "Model Y"]
COL = {"蔚来ES6": NIO_COLORS[0], "蔚来EC6": NIO_COLORS[1], "Model Y": TESLA_COLOR}
BANDS = ["0-4年", "4年+"]
MR = json.load(open(os.path.join(ROOT, "output", "master_results.json"),
                    encoding="utf-8"))

df = pd.DataFrame([json.loads(l) for l in open(SRC, encoding="utf-8") if l.strip()])
df = df[df["energy_type"] == "新能源"].copy()
df = df[df["age_months"].notna()].copy()
df["age_band"] = np.where(df["age_months"] < 48, "0-4年", "4年+")
sub = df[df["series"].isin(FOCUS)].copy()

# ============================================================
# 图 2：3 车系 × 2 车龄段 交叉热力图
# ============================================================
M = np.zeros((3, 2)); N = np.zeros((3, 2), dtype=int)
for i, s in enumerate(FOCUS):
    for j, b in enumerate(BANDS):
        v = sub[(sub["series"] == s) & (sub["age_band"] == b)]["residual_rate"]
        M[i, j] = v.median(); N[i, j] = len(v)

fig, ax = plt.subplots(figsize=(8.6, 5.8))
im = ax.imshow(M, cmap="RdYlGn", vmin=0.20, vmax=0.56, aspect="auto")
for i in range(3):
    for j in range(2):
        tc = "white" if M[i, j] < 0.30 or M[i, j] > 0.50 else "#1A1D24"
        ax.text(j, i - 0.10, f"{M[i,j]:.3f}", ha="center", va="center",
                fontsize=16, fontweight="bold", color=tc)
        ax.text(j, i + 0.21, f"n = {N[i,j]}", ha="center", va="center",
                fontsize=9.8, color=tc, alpha=0.9)
ax.set_xticks(range(2)); ax.set_xticklabels(BANDS, fontsize=11.5)
ax.set_yticks(range(3)); ax.set_yticklabels(FOCUS, fontsize=11.5)
ax.set_xlabel("车龄段")
ax.grid(False)
for sp in ax.spines.values():
    sp.set_visible(False)
title(ax, "3 车系 × 2 车龄段：残值率中位数",
      "全部格子 n ≥ 45，满足交叉分析每格 ≥20 的最低样本要求")
cb = fig.colorbar(im, ax=ax, shrink=0.82, pad=0.03)
cb.set_label("残值率中位数", fontsize=9.8)
cb.outline.set_visible(False)
footnote(fig, f"数据来源：瓜子二手车 · 残值率 = 挂牌价 ÷ 官方指导价 · n={len(sub)}")
save(fig, os.path.join(FIG, "fig2_交叉热力图.png"))
print("fig2 done")

# ============================================================
# 图 3：车系箱线图（按车龄段分面）
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.6), sharey=True)
for ax, b in zip(axes, BANDS):
    data, cols = [], []
    for s in FOCUS:
        v = sub[(sub["series"] == s) & (sub["age_band"] == b)]["residual_rate"].values
        data.append(v); cols.append(COL[s])
    bp = ax.boxplot(data, patch_artist=True, widths=0.52, showfliers=False,
                    medianprops=dict(color="white", lw=2.0),
                    whiskerprops=dict(color=PAL["subtext"], lw=1.1),
                    capprops=dict(color=PAL["subtext"], lw=1.1))
    for patch, c in zip(bp["boxes"], cols):
        patch.set_facecolor(c); patch.set_alpha(0.80); patch.set_edgecolor("none")
    rng = np.random.default_rng(7)
    for k, (v, c) in enumerate(zip(data, cols)):
        jx = rng.normal(k + 1, 0.062, len(v))
        ax.scatter(jx, v, s=11, c=c, alpha=0.32, edgecolors="none", zorder=3)
        ax.text(k + 1, np.median(v) + 0.020, f"{np.median(v):.3f}",
                ha="center", va="bottom", fontsize=9.6, color=c,
                fontweight="bold", zorder=5)
        ax.text(k + 1, 0.095, f"n={len(v)}", ha="center", va="bottom",
                fontsize=8.8, color=PAL["subtext"])
    ax.set_xticks(range(1, 4)); ax.set_xticklabels(FOCUS, fontsize=10.5)
    ax.set_title(f"车龄 {b}", fontsize=12, loc="left", pad=10)
    clean(ax)
axes[0].set_ylabel("残值率")
axes[0].set_ylim(0.08, 0.72)
fig.suptitle("三车系残值率分布：Model Y 在两个车龄段均居首",
             fontsize=14.5, fontweight="bold", x=0.008, ha="left", y=1.03)
fig.text(0.008, 0.968, "箱体为四分位距，白线为中位数，散点为单车源",
         fontsize=9.8, color=PAL["subtext"], ha="left")
footnote(fig, "数据来源：瓜子二手车 · 已剔除燃油车 · 未控制协变量，"
              "控制后的对比见图5")
save(fig, os.path.join(FIG, "fig3_车系箱线图.png"))
print("fig3 done")

# ============================================================
# 图 4：ANCOVA 效应量（取自 master_results.json）
# ============================================================
rows = [{"效应": e["effect"], "F": e["F"], "p": e["p"], "eta2p": e["eta2p"]}
        for e in MR["ancova"]["effects"]]
rows.sort(key=lambda r: r["eta2p"])

fig, ax = plt.subplots(figsize=(10.4, 5.8))
for i, r in enumerate(rows):
    sig = r["p"] < 0.05
    c = PAL["primary"] if sig else PAL["muted"]
    ax.barh(i, r["eta2p"], height=0.52, color=c, alpha=0.92)
    ax.text(r["eta2p"] + 0.004, i, f"{r['eta2p']:.3f}", va="center",
            fontsize=10.5, color=c, fontweight="bold")
    star = "***" if r["p"] < 0.001 else "**" if r["p"] < 0.01 else \
           "*" if r["p"] < 0.05 else "n.s."
    ax.text(-0.006, i, star, va="center", ha="right", fontsize=9.6,
            color=PAL["good"] if sig else PAL["subtext"])
ax.set_yticks(range(len(rows)))
ax.set_yticklabels([r["效应"] for r in rows], fontsize=10.5)
ax.set_xlabel("偏 η²（效应量）")
xmax = max(r["eta2p"] for r in rows) * 1.24
ax.set_xlim(-0.030, xmax)
for xv, lb in [(0.01, "小"), (0.06, "中"), (0.14, "大")]:
    if xv < xmax:
        ax.axvline(xv, color=PAL["grid"], lw=1, ls=":")
        ax.text(xv, -0.78, lb, ha="center", fontsize=8.4, color=PAL["subtext"])
ax.set_ylim(len(rows) - 0.45, -0.95)
clean(ax)
title(ax, "各因素对残值率的效应量（ANCOVA，Type II）",
      f"偏 η² 越大影响越强 · n={MR['ancova']['n']} · "
      "*** p<0.001, ** p<0.01, * p<0.05, n.s. 不显著")
it = [r["eta2p"] for r in rows if "交互" in r["效应"]]
footnote(fig, "数据来源：瓜子二手车 · 模型已控制车龄/里程/指导价/城市 · "
              + (f"交互项偏 η²={it[0]:.3f}（n.s.）" if it else ""))
save(fig, os.path.join(FIG, "fig4_系数图.png"))
print("fig4 done")

# ============================================================
# 图 5：调整后边际均值（取自 master_results.json）
# ============================================================
EMM = {(e["series"], e["band"]): (e["emm"], e["ci_lo"], e["ci_hi"])
       for e in MR["emm"]}
GP = {g["contrast"]: g["diff"] for g in MR["contrasts"]}
d_0_4 = GP["Model Y − 蔚来ES6（0-4年）"] * 100
d_4p = GP["Model Y − 蔚来ES6（4年+）"] * 100
gap_ec = GP["蔚来ES6 − 蔚来EC6（0-4年）"] * 100

fig, ax = plt.subplots(figsize=(11.0, 6.0))
w = 0.34
xs = np.arange(3)
for j, (b, alpha) in enumerate([("0-4年", 0.95), ("4年+", 0.58)]):
    vals = [EMM[(s, b)][0] for s in FOCUS]
    err = [[EMM[(s, b)][0] - EMM[(s, b)][1] for s in FOCUS],
           [EMM[(s, b)][2] - EMM[(s, b)][0] for s in FOCUS]]
    ax.bar(xs + (j - 0.5) * w, vals, w, yerr=err, capsize=4,
           color=[COL[s] for s in FOCUS], alpha=alpha,
           edgecolor="white", linewidth=1.2,
           error_kw=dict(ecolor=PAL["text"], lw=1.2, capthick=1.2),
           label=b, zorder=3)
    for x_, v in zip(xs + (j - 0.5) * w, vals):
        ax.text(x_, v + 0.013, f"{v:.3f}", ha="center", va="bottom",
                fontsize=10.2, fontweight="bold", color=PAL["text"], zorder=4)

y0 = 0.555
ax.annotate("", xy=(2 - 0.17, y0), xytext=(0 + 0.17, y0),
            arrowprops=dict(arrowstyle="<->", color=PAL["accent"], lw=1.7))
ax.text(1, y0 + 0.011, f"0-4年：Model Y − 蔚来ES6 = {d_0_4:+.1f}pp",
        ha="center", fontsize=10.4, color=PAL["accent"], fontweight="bold")
ax.text(1, y0 - 0.036,
        f"4年+ 段该差距为 {d_4p:+.1f}pp　·　蔚来 ES6 ≈ EC6（差 {gap_ec:+.1f}pp，CI 跨 0）",
        ha="center", fontsize=9.6, color=PAL["subtext"])
ax.set_xticks(xs); ax.set_xticklabels(FOCUS, fontsize=12.5)
ax.set_ylabel("调整后残值率")
ax.set_ylim(0, 0.62)
clean(ax)
title(ax, "控制里程 / 指导价 / 车龄 / 城市后的残值率",
      "ANCOVA 调整后边际均值（全局协变量标准化，城市取边际）· "
      f"误差棒为 95% CI · n={MR['ancova']['n']}")
ax.legend(loc="upper left", ncol=2)
_pp = [e["p"] for e in MR["ancova"]["effects"] if "车系×" in e["effect"]]
_pe = f"{_pp[0]:.3f}" if _pp else "n.s."
footnote(fig, "数据来源：瓜子二手车 · 交互项 p=" + _pe
              + f" 不显著 · 交互项可检出下限 ±{MR['mde']['mde_pp']:.1f}pp")
save(fig, os.path.join(FIG, "fig5_调整均值.png"))
print("fig5 done")
