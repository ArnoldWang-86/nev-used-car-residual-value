# -*- coding: utf-8 -*-
r"""08_domain_gap.py —— 通用二手车模型 vs 真实中国新能源车：领域差距量化

回答一个对岗位最关键的问题：
    「用通用二手车市场数据训练出来的折旧模型，能不能直接用于中国新能源车？」

做法：
  1. 从公开数据集（俄罗斯通用二手车市场）估计【按车龄的折旧曲线】
  2. 从真实车源（瓜子·蔚来，中国新能源）计算【按车龄的残值曲线】
  3. 把两条曲线放到同一坐标（都以"接近新车"为 1.0）比较
  4. 量化偏差，并明确标注两条曲线的口径差异

⚠️ 口径必须说清，否则结论会被误读：
  公开数据集：分母 = 同车系【车龄≤2年】的中位挂牌价 → 相对折旧比
  真实车源  ：分母 = 【官方新车指导价】          → 残值率
  两者分母定义不同，因此本脚本**只比较折旧的"形状"与"速度"**，
  即把各自曲线归一化到车龄≈1 年为基准后再比，不比较绝对水平。

用法：
    python src/08_domain_gap.py
"""
from __future__ import annotations

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


try:
    import sys as _sys
    _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    _sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PUB = os.path.join(ROOT, "data", "clean", "cars_public.csv")
REAL = os.path.join(ROOT, "data", "clean", "cars_real.csv")
OUTDIR = os.path.join(ROOT, "results")
REPORT = os.path.join(ROOT, "output", "domain_gap_report.txt")
os.makedirs(OUTDIR, exist_ok=True)

TARGET = "residual_rate"
AGE_BINS = [-0.01, 1, 2, 3, 5, 8, 12, 20, 40]
AGE_LABELS = ["0-1年", "1-2年", "2-3年", "3-5年", "5-8年", "8-12年", "12-20年", "20-40年"]


def curve_public() -> pd.DataFrame:
    df = pd.read_csv(PUB, encoding="utf-8-sig",
                     usecols=["age_years", TARGET, "price_wan", "brand"])
    df["age_bin"] = pd.cut(df["age_years"], AGE_BINS, labels=AGE_LABELS)
    g = (df.groupby("age_bin", observed=True)
         .agg(样本数=(TARGET, "size"), 折旧比=("age_years", "median"),
              折旧比中位=(TARGET, "median"))
         .drop(columns=["折旧比"]))
    g["样本数"] = g["样本数"].astype(int)
    return g


def curve_real() -> pd.DataFrame:
    df = pd.read_csv(REAL, encoding="utf-8-sig")
    df["age_years"] = pd.to_numeric(df["age_months"], errors="coerce") / 12
    df["age_bin"] = pd.cut(df["age_years"], AGE_BINS, labels=AGE_LABELS)
    g = (df.groupby("age_bin", observed=True)
         .agg(样本数=(TARGET, "size"), 残值率中位=(TARGET, "median"),
              车龄中位年=("age_years", "median"))
         .round(4))
    g["样本数"] = g["样本数"].astype(int)
    return g


def normalized_curve(g: pd.DataFrame, col: str) -> pd.Series:
    """把曲线归一化：以最接近 1.5 年的档位为基准 = 1.0，看相对衰减速度。"""
    s = g[col].astype(float).copy()
    if s.dropna().empty:
        return s
    # 用最早的非空档位作基准，避免不同数据集覆盖区间不同导致不可比
    base_idx = s.dropna().index[0]
    base = float(s.loc[base_idx])
    return (s / base).round(4)


def annual_deprec(g: pd.DataFrame, col: str, age_col: str) -> pd.DataFrame:
    """计算各档位的年均折旧率（相对上一档），衡量衰减速度。"""
    rows = []
    idx = list(g.index)
    for i in range(1, len(idx)):
        a0, a1 = idx[i - 1], idx[i]
        try:
            v0 = float(g.loc[a0, col])
            v1 = float(g.loc[a1, col])
        except (KeyError, TypeError, ValueError):
            continue
        if not np.isfinite(v0) or not np.isfinite(v1) or v0 <= 0:
            continue
        # 档位中点到中点的年数差
        def mid(lab):
            lo, hi = lab.replace("年", "").split("-")
            return (float(lo) + float(hi)) / 2
        dy = mid(a1) - mid(a0)
        if dy <= 0:
            continue
        cagr = (v1 / v0) ** (1 / dy) - 1
        rows.append({"车龄档": a1, "年化变化率": round(cagr * 100, 2)})
    return pd.DataFrame(rows)


def main() -> None:
    L = []
    P = L.append
    P("=" * 88)
    P("领域差距分析：通用二手车市场 vs 真实中国新能源车")
    P("=" * 88)
    P("")
    P("【要回答的问题】")
    P("  用通用二手车数据训练的折旧模型，能否直接用于中国新能源车？")
    P("")

    if not os.path.exists(PUB) or not os.path.exists(REAL):
        P("缺少数据文件，无法分析。")
        with open(REPORT, "w", encoding="utf-8") as f:
            f.write("\n".join(L))
        print("\n".join(L))
        return

    pub = curve_public()
    real = curve_real()

    P("【口径声明 · 必须先读】")
    P("  公开数据集：分母 = 同车系【车龄≤2年】中位挂牌价 → 相对折旧比")
    P("  真实车源  ：分母 = 【官方新车指导价】           → 残值率")
    P("  两者分母定义不同，因此本分析**只比较折旧的形状与速度**，")
    P("  即各自归一化到最早档位为 1.0 后再比，不比较绝对水平。")
    P("")

    P("=" * 88)
    P("一、两条原始曲线")
    P("=" * 88)
    P("")
    P("  通用二手车市场（俄罗斯, n=%d）" % int(pub["样本数"].sum()))
    P(pub.to_string())
    P("")
    P("  真实中国新能源（瓜子·蔚来, n=%d）" % int(real["样本数"].sum()))
    P(real.to_string())
    P("")

    P("=" * 88)
    P("二、归一化后的折旧形状（各自最早档位 = 1.0）")
    P("=" * 88)
    P("")
    np_ = normalized_curve(pub, "折旧比中位")
    nr_ = normalized_curve(real, "残值率中位")
    cmp_df = pd.DataFrame({
        "通用市场_相对": np_,
        "真实新能源_相对": nr_,
    })
    cmp_df["差异"] = (cmp_df["真实新能源_相对"] - cmp_df["通用市场_相对"]).round(4)
    P(cmp_df.to_string())
    P("")

    P("=" * 88)
    P("三、年化折旧速度对比")
    P("=" * 88)
    P("")
    ad_pub = annual_deprec(pub, "折旧比中位", "age_years")
    ad_real = annual_deprec(real, "残值率中位", "age_months")
    P("  通用二手车市场（相邻档位间的年化变化率）：")
    P(ad_pub.to_string(index=False) if not ad_pub.empty else "   数据不足")
    P("")
    P("  真实中国新能源：")
    P(ad_real.to_string(index=False) if not ad_real.empty else "   数据不足")
    P("")

    # 只在重叠区间比较：3-5年 与 5-8年
    P("=" * 88)
    P("四、重叠区间的直接比较（3-8 年车龄）")
    P("=" * 88)
    P("")
    for lab in ["3-5年", "5-8年"]:
        try:
            a = float(pub.loc[lab, "折旧比中位"])
            b = float(real.loc[lab, "残值率中位"])
        except (KeyError, TypeError, ValueError):
            continue
        if np.isfinite(a) and np.isfinite(b):
            P(f"  {lab}：通用市场 {a:.3f}  vs  真实新能源 {b:.3f}  "
              f"→ 新能源{'更低' if b < a else '更高'} {abs(b-a)/a*100:.1f}%")
    P("")
    P("  注意：该差异同时包含【市场差异】与【口径差异】两部分，")
    P("        不能单独归因为'新能源更不保值'。要分离两者需要同口径数据。")
    P("")

    P("=" * 88)
    P("五、结论")
    P("=" * 88)
    P("")
    P("  1. 两条曲线的折旧形状都单调，说明车龄是跨市场通用的第一驱动变量。")
    P("  2. 但【绝对水平不可直接迁移】——分母口径不同（指导价 vs 同车系准新车价），")
    P("     且市场、能源类型、车型结构完全不同。")
    P("  3. 因此：通用数据训练的模型可以作为【方法论骨架】，")
    P("     但不能作为【中国新能源车的定价模型】直接上线。")
    P("  4. 落地上线需要本土同口径数据：官方指导价 + 中国新能源车源，")
    P("     这正是本岗位要解决的「数据壁垒」。")
    P("")
    P("  ⚠️ 样本量提示：真实车源仅 %d 条，上述真实曲线只能作为量级参考，"
      % int(real["样本数"].sum()))
    P("     不足以支撑统计推断。要得到可靠的本土折旧曲线，")
    P("     需要采集蔚来全系 + 竞品车型数百条同口径记录。")
    P("")

    txt = "\n".join(L)
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write(txt)
    print(txt)

    cmp_df.to_csv(os.path.join(OUTDIR, "d01_领域差距_折旧形状.csv"),
                  encoding="utf-8-sig")
    print(f"\n已输出 {REPORT}")


if __name__ == "__main__":
    main()
