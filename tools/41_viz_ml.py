# -*- coding: utf-8 -*-
r"""41_viz_ml.py —— 图6/7/8：模型对比、学习曲线、特征重要性

数据来源：master_results.json（权威结果）+ 现场复算 ML
口径统一：全部在 GroupKFold OOF 预测上计算 MAE / R²
"""
from __future__ import annotations

import copy
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from viz_style import C as PAL, clean, title, footnote, save  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
from sklearn.ensemble import RandomForestRegressor  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402
from sklearn.metrics import mean_absolute_error, r2_score  # noqa: E402
from sklearn.model_selection import GroupKFold, KFold, learning_curve  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

warnings.filterwarnings("ignore")

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "data", "raw", "detail_from_cache.jsonl")
FIG = os.path.join(ROOT, "output", "figures")
MSTR = json.load(open(os.path.join(ROOT, "output", "master_results.json"),
                      encoding="utf-8"))

# ---------------- 数据 ----------------
df = pd.DataFrame([json.loads(l) for l in open(SRC, encoding="utf-8") if l.strip()])
df = df[df["energy_type"] == "新能源"].copy()
df = df[df["age_months"].notna()].copy()
df["mileage_wan"] = df["mileage"] / 10000
df["price_wan"] = df["guide_price"] / 10000

TARGET = "residual_rate"
NUM = ["age_months", "mileage_wan", "price_wan", "appearance_score"]
CAT = ["series", "city_name"]
EXTRA = ["soh_pct", "battery_kwh", "is_battery_swap", "claim_times",
         "condition_grade"]

y = df[TARGET].values
groups = df["series"].values


def build(feats_num, feats_cat, feats_extra):
    X = df[feats_num].copy()
    for c in feats_cat:
        X = pd.concat([X, pd.get_dummies(df[c], prefix=c, dtype=float)], axis=1)
    for c in feats_extra:
        if c == "condition_grade":
            X = pd.concat([X, pd.get_dummies(df[c], prefix=c, dtype=float)], axis=1)
        else:
            X[c] = pd.to_numeric(df[c], errors="coerce")
    return X


def oof(model_factory, X, y, groups, n_splits=5):
    oofp = np.full(len(y), np.nan)
    folds = []
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups):
        m = model_factory()
        m.fit(X.iloc[tr], y[tr])
        p = m.predict(X.iloc[te])
        oofp[te] = p
        folds.append({"n": int(len(te)),
                      "mae": float(mean_absolute_error(y[te], p)),
                      "r2": float(r2_score(y[te], p))})
    return oofp, folds


def mk_ridge():
    return Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("sc", StandardScaler()), ("m", Ridge(alpha=1.0))])


def mk_rf():
    return Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("m", RandomForestRegressor(n_estimators=300,
                                                 min_samples_leaf=5,
                                                 random_state=42, n_jobs=-1))])


X_base = build(NUM, CAT, [])
X_batt = build(NUM, CAT, EXTRA)

# ---------------- 模型对比 ----------------
print("running models ...")
MODELS = [("常数基线", None), ("Ridge", mk_ridge), ("随机森林", mk_rf)]
try:
    import lightgbm as lgb
    MODELS.append(("LightGBM", lambda: Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("m", lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05,
                                min_child_samples=5, verbose=-1,
                                random_state=42))])))
except ImportError:
    pass
try:
    import xgboost as xgb
    MODELS.append(("XGBoost", lambda: Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("m", xgb.XGBRegressor(n_estimators=300, learning_rate=0.05,
                               min_child_weight=5, random_state=42,
                               verbosity=0))])))
except ImportError:
    pass

rows = []
oofs = {}
for name, mk in MODELS:
    if mk is None:
        oofp = np.full(len(y), np.nan)
        for tr, te in GroupKFold(n_splits=5).split(X_batt, y, groups):
            oofp[te] = np.median(y[tr])
        folds = []
        mae = mean_absolute_error(y, oofp)
        r2 = r2_score(y, oofp)
    else:
        oofp, folds = oof(mk, X_batt, y, groups)
        mae = mean_absolute_error(y, oofp)
        r2 = r2_score(y, oofp)
    oofs[name] = oofp
    rows.append({"model": name, "mae": mae, "r2": r2})

# 融合
if "LightGBM" in oofs and "XGBoost" in oofs:
    f = (oofs["LightGBM"] + oofs["XGBoost"]) / 2
    rows.append({"model": "LGBM+XGB 融合", "mae": mean_absolute_error(y, f),
                 "r2": r2_score(y, f)})

# 随机 KFold 反例（用随机森林）
oof_rand = np.full(len(y), np.nan)
for tr, te in KFold(n_splits=5, shuffle=True, random_state=42).split(X_batt):
    m = mk_rf(); m.fit(X_batt.iloc[tr], y[tr]); oof_rand[te] = m.predict(X_batt.iloc[te])
r2_rand = r2_score(y, oof_rand)

# ============================================================
# 图 6：模型对比
# ============================================================
fig, ax = plt.subplots(figsize=(10.6, 5.8))
names = [r["model"] for r in rows]
maes = [r["mae"] for r in rows]
cols = [PAL["muted"]] + [PAL["primary"]] * (len(rows) - 1)
bars = ax.barh(range(len(rows)), maes, height=0.58, color=cols, alpha=0.92)
best = min(range(len(maes)), key=lambda i: maes[i])
bars[best].set_color(PAL["accent"])
for i, (m_, r2) in enumerate(zip(maes, [r["r2"] for r in rows])):
    ax.text(m_ + 0.0022, i, f"MAE {m_:.4f}   R² {r2:+.3f}", va="center",
            fontsize=10, color=PAL["text"])
ax.set_yticks(range(len(rows)))
ax.set_yticklabels(names, fontsize=10.5)
ax.invert_yaxis()
ax.set_xlabel("样本外 MAE（GroupKFold OOF，越小越好）")
ax.set_xlim(0, max(maes) * 1.45)
ax.axvline(min(maes), color=PAL["accent"], ls=":", lw=1.2, alpha=0.7)
clean(ax)
title(ax, "六模型预测性能对比：随机森林略优，集成未带来增益",
      f"GroupKFold 按车系留出（5 折）· n={len(y)} · 全部在同一份 OOF 预测上计算")
footnote(fig, "数据来源：瓜子二手车 · 特征含车龄/里程/指导价/外观/车系/城市/车况/"
              "SOH/电池容量/换电 · 常数基线 = 训练集中位数")
save(fig, os.path.join(FIG, "fig6_模型对比.png"))
print("fig6 done")

# ============================================================
# 图 7：学习曲线 + 随机 KFold 对比
# ============================================================
ts, trs, vas = learning_curve(
    mk_ridge(), X_batt, y, groups=groups, cv=GroupKFold(n_splits=5),
    train_sizes=np.linspace(0.2, 1.0, 8), scoring="neg_mean_absolute_error")
trm = -trs.mean(axis=1); vam = -vas.mean(axis=1); vse = vas.std(axis=1) / np.sqrt(5)

fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.4),
                         gridspec_kw={"width_ratios": [1.25, 1]})
ax = axes[0]
ax.plot(ts, trm, "o-", color=PAL["primary"], lw=2.4, ms=5.5, label="训练误差",
        zorder=3)
ax.plot(ts, vam, "o-", color=PAL["accent"], lw=2.4, ms=5.5, label="验证误差",
        zorder=4)
ax.fill_between(ts, vam - 1.96 * vse, vam + 1.96 * vse,
                color=PAL["accent"], alpha=0.14, zorder=2)
# 训练/验证差距带
for i in range(len(ts)):
    ax.plot([ts[i], ts[i]], [trm[i], vam[i]], color=PAL["muted"], lw=1.0,
            ls=":", zorder=1)
ax.text(ts[-1], (trm[-1] + vam[-1]) / 2, f" 差距\n {(vam[-1]-trm[-1]):.4f}",
        fontsize=8.8, color=PAL["subtext"], va="center")
ax.annotate(f"验证误差 {vam[0]:.4f} → {vam[-1]:.4f}\n仍在下降，未收敛",
            xy=(ts[-1], vam[-1]), xytext=(ts[1], vam[1] + 0.0035),
            fontsize=9.6, color=PAL["text"],
            arrowprops=dict(arrowstyle="->", color=PAL["subtext"], lw=1.1,
                            connectionstyle="arc3,rad=0.18"),
            bbox=dict(boxstyle="round,pad=0.42", fc="#F7F9FB",
                      ec="#DDE3EA", lw=0.8), zorder=6)
ax.set_xlabel("训练样本量"); ax.set_ylabel("MAE")
ax.set_ylim(min(trm.min(), vam.min()) - 0.004,
            max(trm.max(), vam.max()) + 0.009)
clean(ax)
title(ax, "学习曲线：验证误差尚未趋平", "Ridge + GroupKFold · 阴影为 5 折 ±95%")
ax.legend(loc="upper right")

ax = axes[1]
vals = [r2_rand, max(r["r2"] for r in rows)]
labs = ["随机 KFold\n(预测已知车系的新车源)", "GroupKFold\n(预测全新车系)"]
b = ax.bar([0, 1], vals, width=0.5, color=[PAL["muted"], PAL["primary"]],
           alpha=0.92)
for i, v in enumerate(vals):
    ax.text(i, v + 0.018, f"{v:.3f}", ha="center", fontsize=12,
            fontweight="bold", color=PAL["text"])
ax.annotate("", xy=(1, vals[0] - 0.035), xytext=(0, vals[0] - 0.035),
            arrowprops=dict(arrowstyle="<->", color=PAL["accent"], lw=1.6))
ax.text(0.5, vals[0] - 0.012, f"差 {abs(vals[0]-vals[1])*100:.1f}pp",
        ha="center", fontsize=10.5, color=PAL["accent"], fontweight="bold")
ax.set_xticks([0, 1]); ax.set_xticklabels(labs, fontsize=10)
ax.set_ylabel("R²"); ax.set_ylim(0, 1.02)
clean(ax)
title(ax, "两种切分回答不同问题", "两者都真实，非「泄漏 vs 正确」")
footnote(fig, "数据来源：瓜子二手车 · GroupKFold 按 series 分组（56 个取值）· "
              "随机 KFold 会让同车系车源跨训练/验证集")
save(fig, os.path.join(FIG, "fig7_学习曲线.png"))
print("fig7 done")

# ============================================================
# 图 8：特征重要性（置换重要性）
# ============================================================
from sklearn.inspection import permutation_importance  # noqa: E402

rf = mk_rf()
tr_idx, te_idx = next(iter(GroupKFold(n_splits=5).split(X_batt, y, groups)))
rf.fit(X_batt.iloc[tr_idx], y[tr_idx])
pi = permutation_importance(rf, X_batt.iloc[te_idx], y[te_idx],
                            n_repeats=30, random_state=42,
                            scoring="neg_mean_absolute_error")
imp = pd.DataFrame({"feature": X_batt.columns,
                    "imp": pi.importances_mean,
                    "sd": pi.importances_std})
# 把哑变量按前缀合并
imp["group"] = imp["feature"].str.replace(r"^(series|city_name|condition_grade)_.*",
                                          r"\1（哑变量组）", regex=True)
g = imp.groupby("group")["imp"].sum().sort_values(ascending=False).head(10)

fig, ax = plt.subplots(figsize=(10.2, 5.6))
cols_ = [PAL["accent"] if i == 0 else PAL["primary"] for i in range(len(g))]
ax.barh(range(len(g)), g.values, height=0.58, color=cols_, alpha=0.9)
for i, v in enumerate(g.values):
    ax.text(v + max(g.values) * 0.012, i, f"{v:.4f}", va="center",
            fontsize=9.6, color=PAL["text"])
ax.set_yticks(range(len(g)))
ax.set_yticklabels([s.replace("（哑变量组）", "\n（哑变量组）") for s in g.index],
                   fontsize=9.8)
ax.invert_yaxis()
ax.set_xlabel("置换重要性（MAE 增量，越大越重要）")
ax.set_xlim(0, max(g.values) * 1.28)
clean(ax)
title(ax, "特征重要性：车龄、里程、指导价构成预测主力",
      f"随机森林 · 置换重要性 ×30 次 · 折外样本 n={len(te_idx)}")
footnote(fig, "数据来源：瓜子二手车 · 哑变量组已按前缀合并 · "
              "appearance_score 单变量 R²=0.499，与挂牌价存在同源风险")
save(fig, os.path.join(FIG, "fig8_特征重要性.png"))
print("fig8 done")

# 保存 ML 结果
with open(os.path.join(ROOT, "output", "ml_results.json"), "w",
          encoding="utf-8") as f:
    json.dump({"models": rows, "r2_random_kfold": float(r2_rand),
               "learning_curve": {"train_sizes": [int(t) for t in ts],
                                  "train_mae": [float(v) for v in trm],
                                  "val_mae": [float(v) for v in vam]},
               "importance": [{"feature": k, "imp": float(v)}
                              for k, v in g.items()]},
              f, ensure_ascii=False, indent=1)
print("ml_results.json written")
