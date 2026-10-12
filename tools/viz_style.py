# -*- coding: utf-8 -*-
r"""viz_style.py —— 统一可视化风格（高级版）

设计要点：
  · 统一的品牌色板与语义色
  · 高 DPI、无边框、克制的网格线
  · 每个图自带数据来源与样本量标注
  · 支持中文（Microsoft YaHei / SimHei）
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# ---------------- 字体 ----------------
_CJK_CANDIDATES = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC",
                   "Source Han Sans SC", "PingFang SC", "WenQuanYi Zen Hei"]
_avail = {f.name for f in font_manager.fontManager.ttflist}
CJK = next((f for f in _CJK_CANDIDATES if f in _avail), "DejaVu Sans")

matplotlib.rcParams.update({
    "font.sans-serif": [CJK, "DejaVu Sans"],
    "font.family": "sans-serif",
    "axes.unicode_minus": False,
    "figure.dpi": 110,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#D8DCE3",
    "axes.linewidth": 0.9,
    "axes.labelcolor": "#2B2F36",
    "axes.titlecolor": "#141821",
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 10.5,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": "#EDF0F4",
    "grid.linewidth": 0.8,
    "xtick.color": "#5A616E",
    "ytick.color": "#5A616E",
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "legend.frameon": False,
    "legend.fontsize": 9.5,
    "lines.linewidth": 2.0,
    "figure.autolayout": False,
})

# ---------------- 语义色板 ----------------
C = {
    # 车系（项目主色）
    "蔚来ES6":   "#0F5C4A",     # NIO 深绿
    "蔚来EC6":   "#4E9C86",     # NIO 浅绿
    "Model Y":   "#D64541",     # Tesla 红
    "小鹏G6":    "#2E7BC4",
    "极氪7X":    "#E8912B",
    "小米SU7":   "#8B5CF6",
    "乐道L60":   "#0E7490",
    # 通用
    "primary":   "#1F4E5F",
    "accent":    "#E8503A",
    "neutral":   "#8892A4",
    "muted":     "#B8C0CC",
    "good":      "#2E9E6B",
    "warn":      "#E8A33D",
    "bad":       "#D64541",
    "grid":      "#EDF0F4",
    "text":      "#2B2F36",
    "subtext":   "#6B7382",
    "band":      "#1F4E5F",
}

NIO_COLORS = ["#0F5C4A", "#4E9C86", "#7FC0AC"]
TESLA_COLOR = "#D64541"


def figsize(w=10.0, h=6.0):
    return (w, h)


def clean(ax, spines=("top", "right")):
    """去掉指定边框，保留干净外观。"""
    for s in spines:
        ax.spines[s].set_visible(False)
    return ax


def title(ax, main, sub=None):
    """主标题 + 副标题（副标题小字灰色）。"""
    ax.set_title(main, pad=18 if sub else 10, loc="left")
    if sub:
        ax.text(0.0, 1.015, sub, transform=ax.transAxes, ha="left",
                va="bottom", fontsize=9.5, color=C["subtext"])
    return ax


def footnote(fig, text):
    """图底部来源/口径脚注。"""
    fig.text(0.008, -0.012, text, ha="left", va="top",
             fontsize=8.2, color=C["subtext"], wrap=True)
    return fig


def annotate_n(ax, x, y, s, **kw):
    kw.setdefault("fontsize", 8.8)
    kw.setdefault("color", C["subtext"])
    kw.setdefault("ha", "center")
    kw.setdefault("va", "bottom")
    ax.text(x, y, s, **kw)


def save(fig, path, tight=True):
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white",
                pad_inches=0.28)
    plt.close(fig)
    return path


def ci95(sd, n):
    import math
    return 1.959964 * sd / math.sqrt(n) if n > 0 else 0.0
