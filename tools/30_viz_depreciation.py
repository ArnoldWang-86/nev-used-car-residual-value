# -*- coding: utf-8 -*-
r"""30_viz_depreciation.py —— 图1：折旧曲线（高级版）

改进点（对比原版）：
  1. 增加【分车系】曲线（ES6 / EC6 / Model Y），这是项目核心
  2. 增加 LOWESS 置信带（bootstrap）
  3. 车龄段 n 标注改为顶部独立行，不再堆叠
  4. 加入关键业务标注（如"前 24 个月跌掉 33%"）
  5. 右侧加"残值率保留比例"对照轴
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
from statsmodels.nonparametric.smoothers_lowess import lowess

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from viz_style import C, NIO_COLORS, TESLA_COLOR, clean, title, footnote, save  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "data", "raw", "detail_from_cache.jsonl")
FIG = os.path.join(ROOT, "output", "figures")

df = pd.DataFrame([json.loads(l) for l in open(SRC, encoding="utf-8") if l.strip()])
df = df[df["energy_type"] == "新能源"].copy()
df = df[df["age_months"].notna()].copy()
FOCUS = ["蔚来ES6", "蔚来EC6", "Model Y"]
COL = {"蔚来ES6": NIO_COLORS[0], "蔚来EC6": NIO_COLORS[1], "Model Y": TESLA_COLOR}

# ============================================================
# 图 1a：全体折旧曲线（散点 + LOWESS + bootstrap 置信带）
# ============================================================
x = df["age_months"].values.astype(float)
y = df["residual_rate"].values.astype(float)
xs = np.linspace(x.min(), x.max(), 220)
sm = lowess(y, x, frac=0.45, return_sorted=True)

rng = np.random.default_rng(42)
boots = []
for _ in range(300):
    idx = rng.integers(0, len(x), len(x))
    s = lowess(y[idx], x[idx], frac=0.45, return_sorted=True)
    boots.append(np.interp(xs, s[:, 0], s[:, 1]))
boots = np.vstack(boots)
lo = np.percentile(boots, 2.5, axis=0)
hi = np.percentile(boots, 97.5, axis=0)

fig, ax = plt.subplots(figsize=(11.2, 6.2))
ax.scatter(x, y, s=14, c=C["band"], alpha=0.20, edgecolors="none",
           label=f"单车源（n={len(df)}）", zorder=1)
ax.fill_between(xs, lo, hi, color=C["accent"], alpha=0.16, zorder=2,
                label="LOWESS 95% 置信带（bootstrap ×300）")
ax.plot(sm[:, 0], sm[:, 1], color=C["accent"], lw=2.8, zorder=3,
        label="LOWESS 平滑")

# 车龄段 n：分段条带标注（画在图内顶部，避免与 x 轴标题冲突）
BINS = [(0, 12), (12, 24), (24, 36), (36, 48), (48, 60), (60, 200)]
ymax = y.max()
strip_y = ymax + 0.005
for a, b in BINS:
    m = (x >= a) & (x < b)
    if m.sum() == 0:
        continue
    bb = min(b, x.max())
    ax.plot([a + 0.5, bb - 0.5], [strip_y, strip_y],
            color=C["muted"], lw=3.2, solid_capstyle="butt", zorder=1)
    ax.text((a + bb) / 2, strip_y + 0.008, f"n={m.sum()}",
            ha="center", va="bottom", fontsize=8.4, color=C["subtext"])

# 关键业务标注
m24 = y[x < 24].mean()
m48 = y[(x >= 36) & (x < 48)].mean()
ax.annotate(
    f"0–24 个月均值 {m24:.1%}\n24–48 个月跌至 {m48:.1%}",
    xy=(24, np.interp(24, sm[:, 0], sm[:, 1])),
    xytext=(34, 0.86), fontsize=9.2, color=C["text"],
    arrowprops=dict(arrowstyle="->", color=C["subtext"], lw=1.1,
                    connectionstyle="arc3,rad=-0.2"),
    bbox=dict(boxstyle="round,pad=0.45", fc="#F7F9FB", ec="#DDE3EA", lw=0.8))

ax.set_xlabel("车龄（月）")
ax.set_ylabel("残值率（挂牌价 / 官方指导价）")
ax.set_ylim(0.08, ymax + 0.05)
ax.set_xlim(-2, x.max() + 3)
clean(ax)
title(ax, "新能源二手车折旧曲线：前 24 个月跌去约三成",
      "全部新能源车源（n=725）· 单期快照 2026-10-12 · 纵轴为残值率中位趋势")
ax.legend(loc="upper right", ncol=1)
footnote(fig, "数据来源：瓜子二手车公开详情页 · 残值率 = 挂牌价 ÷ 官方指导价 · "
              "LOWESS frac=0.45 · 阴影为 bootstrap 300 次 95% 区间")
save(fig, os.path.join(FIG, "fig1_折旧曲线.png"))
print("fig1 done")

# ============================================================
# 图 1b：分车系折旧曲线（核心对比）
# ============================================================
fig, ax = plt.subplots(figsize=(11.2, 6.4))
for s in FOCUS:
    d = df[df["series"] == s]
    xs_ = d["age_months"].values.astype(float)
    ys_ = d["residual_rate"].values.astype(float)
    ax.scatter(xs_, ys_, s=13, c=COL[s], alpha=0.16, edgecolors="none", zorder=1)
    gx = np.linspace(xs_.min(), xs_.max(), 160)
    gm = lowess(ys_, xs_, frac=0.75, return_sorted=True)
    ax.plot(gm[:, 0], gm[:, 1], color=COL[s], lw=2.9, zorder=3,
            label=f"{s}（n={len(d)}）")

# 标注 4 年处的差异
for s in FOCUS:
    d = df[(df["series"] == s) & (df["age_months"] >= 60)]
    if len(d):
        v = d["residual_rate"].median()
        ax.plot([60], [v], "o", ms=7, mfc="white", mec=COL[s], mew=2.2, zorder=4)
        ax.annotate(f"{v:.3f}", xy=(60, v), xytext=(8, 0),
                    textcoords="offset points", fontsize=9,
                    color=COL[s], fontweight="bold", va="center")

ax.axvline(48, color=C["muted"], ls="--", lw=1.1, zorder=0)
ax.text(48.6, 0.90, "48 月\n（4 年）", fontsize=8.6, color=C["subtext"], va="top")
ax.set_xlabel("车龄（月）")
ax.set_ylabel("残值率")
ax.set_ylim(0.10, 1.02)
clean(ax)
title(ax, "蔚来 ES6 / EC6 与 Model Y 折旧路径对比",
      "三条曲线取样区间不同（受上市时间限制）· 白点为 60 月以上中位数")
ax.legend(loc="upper right")
footnote(fig, "数据来源：瓜子二手车 · 各车系 LOWESS frac=0.75 · "
              "曲线外推部分不代表真实数据范围")
save(fig, os.path.join(FIG, "fig1b_分车系折旧.png"))
print("fig1b done")
