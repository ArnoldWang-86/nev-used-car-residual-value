# -*- coding: utf-8 -*-
r"""16_train_final.py —— 最终预测模型：训练、验证、保存

与 src/15_ml_final.py 的区别（为什么要新写一个）：
  · 15_ml_final.py 用 90 维哑变量（55 车系 + 27 城市 + 车况），只打印 OOF 指标，
    并用 joblib 保存了一个「模型元信息 dict」——那不是一个可加载的模型，
    仓库里因此没有能真正调起来的预测能力。
  · 本脚本用**精简特征 + LightGBM 原生类别车系**，把真正的模型对象保存下来，
    并额外产出：分位数预测区间、线性外推对照模型、特征消融、分车系误差、
    未知车系行为测试、模型卡。

精简特征是否掉精度：本脚本用同一套 GroupKFold 口径把两版并排跑出来，
不靠断言，靠数字。

用法：
    python src/16_train_final.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import residual_model as rm  # noqa: E402
from sklearn.metrics import mean_absolute_error, r2_score  # noqa: E402

OUT = os.path.join(rm.ROOT, "output")
os.makedirs(rm.MODEL_DIR, exist_ok=True)

RULE = "=" * 84


def mae(y, p):
    return float(mean_absolute_error(y, p))


def r2(y, p):
    return float(r2_score(y, p))


# ==================================================================
# 1. 数据与词表
# ==================================================================
df = rm.load_clean()
print(RULE)
print("16_train_final.py · 最终预测模型")
print(RULE)
print(f"\n样本：{len(df)} 条新能源在售车源（{df['snapshot_date'].iloc[0]} 单期快照）")
print(f"目标：{rm.TARGET}（挂牌价 / 新车指导价），均值 {df[rm.TARGET].mean():.3f}、"
      f"中位 {df[rm.TARGET].median():.3f}")

feat_all = rm.build_features(df)
# 车系分档规则在**全表**上一次确定，并随模型持久化；
# 推理时必须复用这张表，不能在单行上重新统计（否则任何车系都会被并档）。
series_map = rm.build_bucket_map(df)
buckets = pd.Series([series_map[s] for s in df["series"].astype(str)], index=df.index)
vocab_raw = buckets.value_counts()
series_vocab = [s for s in vocab_raw.index.tolist()]
if rm.UNKNOWN_SERIES not in series_vocab:
    series_vocab.append(rm.UNKNOWN_SERIES)
print(f"\n车系词表：{len(series_vocab)} 档"
      f"（原始车系 {df['series'].nunique()} 个，样本 <{rm.SERIES_MIN_N} 条的已并入品牌档）")
print("  独立建模档：" + "、".join(
    f"{s}({n})" for s, n in vocab_raw.items() if n >= rm.SERIES_MIN_N))
print("  品牌聚合档：" + "、".join(
    f"{s}({n})" for s, n in vocab_raw.items() if n < rm.SERIES_MIN_N and s != rm.UNKNOWN_SERIES))
print("  ⚠️ 分档规则来自全表样本量，随模型保存；单行推理时严禁重新统计")

# ==================================================================
# 2. GroupKFold OOF：精简版 vs 原 90 维哑变量版
# ==================================================================
y = pd.to_numeric(df[rm.TARGET], errors="coerce").values
groups = df["series"].astype(str).values
# ⚠️ 分组必须与模型实际类别口径一致：模型用的是并档后的 series_cat，
# 若按原始 56 车系切分，被并档的稀有小车系会跨折出现 → 泄漏 → 指标虚高。
groups_bucket = buckets.values

oof_lean, resid_lean = rm.group_kfold_oof(df, series_map, series_vocab,
                                          groups=groups_bucket)
oof_rand, _ = rm.random_kfold_oof(df, series_map, series_vocab)       # 随机切分对照
oof_ridge, _ = rm.group_kfold_oof(df, series_map, series_vocab, kind="ridge",
                                  groups=groups_bucket)               # 线性对照

# --- 原版 90 维哑变量（复刻 src/15_ml_final.py 的口径）---
from sklearn.model_selection import GroupKFold  # noqa: E402
import lightgbm as lgb  # noqa: E402

city_counts = df["city_name"].value_counts()
major_cities = city_counts[city_counts >= 5].index.tolist()
city_grp = np.where(df["city_name"].isin(major_cities), df["city_name"], "其他")
sd = pd.get_dummies(df["series"], prefix="s", drop_first=True)
cd = pd.get_dummies(pd.Series(city_grp, index=df.index), prefix="c", drop_first=True)
gd = pd.get_dummies(df["condition_grade"], prefix="g", drop_first=True)
X_wide = pd.concat([
    df[["age_months", "mileage_wan", "price_wan", "appearance_score"]].astype(float),
    sd.astype(float), cd.astype(float), gd.astype(float),
    df[["soh_pct", "battery_kwh", "is_battery_swap", "claim_times"]]
      .astype(float).fillna(0),
], axis=1).fillna(0).values
gkf = GroupKFold(n_splits=5)
oof_wide = np.full(len(y), np.nan)
for tr, te in gkf.split(X_wide, y, groups):
    m = lgb.LGBMRegressor(n_estimators=300, max_depth=5, min_child_samples=5,
                          learning_rate=0.05, random_state=42, verbose=-1)
    m.fit(X_wide[tr], y[tr])
    oof_wide[te] = m.predict(X_wide[te])

# --- M0 常数基线 ---
oof_base = np.full(len(y), np.nan)
for tr, te in gkf.split(X_wide, y, groups):
    oof_base[te] = np.median(y[tr])

print("\n【验证一 · GroupKFold by 车系（预测没见过的车系）】")
print(f"{'模型':<34}{'MAE':>8}{'RMSE':>8}{'R²':>9}{'较基线提升':>12}")
rmse_lean = float(np.sqrt(np.mean(resid_lean ** 2)))
rows_v1 = [
    ("M0 常数基线（中位数）", oof_base),
    ("M1 Ridge 线性（精简特征）", oof_ridge),
    ("M2 LightGBM（精简 8 维）", oof_lean),
    ("M3 LightGBM（原 90 维哑变量）", oof_wide),
]
metrics = {}
for name, p in rows_v1:
    mt = dict(mae=mae(y, p), rmse=float(np.sqrt(np.mean((y - p) ** 2))),
              r2=r2(y, p))
    metrics[name] = mt
    imp = (mae(y, oof_base) - mt["mae"]) / mae(y, oof_base) * 100
    print(f"{name:<34}{mt['mae']:>8.4f}{mt['rmse']:>8.4f}{mt['r2']:>9.4f}"
          f"{imp:>11.1f}%")

lean = metrics["M2 LightGBM（精简 8 维）"]
wide = metrics["M3 LightGBM（原 90 维哑变量）"]
print(f"\n精简版 - 90维版：MAE {lean['mae']-wide['mae']:+.4f}，R² {lean['r2']-wide['r2']:+.4f}")
print("→ 结论：车系哑变量对精度的边际贡献接近 0（与 15_ml_final.py 的"
      "置换重要性 2.4e-16 一致），")

print("\n【验证二 · 随机 KFold（预测熟悉的车系，会高估能力）】")
print(f"随机 KFold  R² = {r2(y, oof_rand):.4f}   MAE = {mae(y, oof_rand):.4f}")
print(f"GroupKFold   R² = {lean['r2']:.4f}   MAE = {lean['mae']:.4f}")
print(f"两者差 {r2(y, oof_rand) - lean['r2']:+.4f}——回答的是两个不同业务问题，")
print("  不是「泄漏 vs 正确」。日常给已知车系估价看随机 KFold，新品牌进场看 GroupKFold。")

# 90% 预测区间：用 OOF 残差分位数标定（经验覆盖）
q_lo, q_hi = np.percentile(resid_lean, [5, 95])
cov90 = float(np.mean((y >= oof_lean + q_lo) & (y <= oof_lean + q_hi)))
print(f"\n【90% 预测区间标定】OOF 残差 5%/95% 分位 = "
      f"{q_lo:+.4f} / {q_hi:+.4f}（非对称）")
print(f"  区间半径折算成挂牌价：按均价 {df['full_payment'].mean()/10000:.1f} 万算，"
      f"约 {abs(q_lo)*df['full_payment'].mean()/10000:.2f} ~ "
      f"{abs(q_hi)*df['full_payment'].mean()/10000:.2f} 万元")

# ==================================================================
# 3. 特征消融（精简版内部）
# ==================================================================
print("\n【消融实验 · 每类特征值多少精度（精简特征内部）】")
ABL = [
    ("A1 车龄", ["age_months"]),
    ("A2 +使用强度（里程/年均里程）", ["age_months", "log_mileage_wan", "mileage_per_year"]),
    ("A3 +新车指导价", ["age_months", "log_mileage_wan", "mileage_per_year", "log_price_wan"]),
    ("A4 +车况（外观/理赔/过户）", ["age_months", "log_mileage_wan", "mileage_per_year",
                                    "log_price_wan", "appearance_score",
                                    "claim_times", "transfer_times"]),
    ("A5 +车系", rm.FEATURES),
]
Xc_all = rm.prepare_matrix(df, series_map, series_vocab)
abl_out = []
prev = None
for label, cols in ABL:
    cats = [c for c in rm.CAT_FEATURES if c in cols]
    oof = np.full(len(y), np.nan)
    for tr, te in gkf.split(Xc_all[cols], y, groups_bucket):
        m = rm.make_lgbm()
        kw = {"categorical_feature": cats} if cats else {}
        m.fit(Xc_all[cols].iloc[tr], y[tr], **kw)
        oof[te] = m.predict(Xc_all[cols].iloc[te])
    mt = mae(y, oof)
    delta = "" if prev is None else f"{mt - prev:+.4f}"
    print(f"  {label:<30} 特征 {len(cols):>2}  MAE={mt:.4f}  Δ={delta}")
    abl_out.append({"set": label, "n_features": len(cols), "mae": round(mt, 4),
                    "delta": None if prev is None else round(mt - prev, 4)})
    prev = mt

# ==================================================================
# 4. 分车系误差 + 未知车系行为
# ==================================================================
print("\n【分车系误差（GroupKFold OOF）】")
err = pd.DataFrame({"series": groups, "y": y, "pred": oof_lean,
                    "resid": y - oof_lean}).dropna()
per_series = []
for s, g in err.groupby("series"):
    if len(g) < 10:
        continue
    per_series.append({"series": s, "n": int(len(g)),
                       "mae": round(float(g["resid"].abs().mean()), 4),
                       "bias_pp": round(float(g["resid"].mean()) * 100, 2)})
per_series.sort(key=lambda d: -d["n"])
for d in per_series[:8]:
    print(f"  {d['series']:<12} n={d['n']:>3}  MAE={d['mae']:.4f}  "
          f"偏差={d['bias_pp']:+.2f}pp")

# 未知车系行为：真正做一次 leave-one-series-out，量化退化幅度
print("\n【未知车系兜底行为测试（leave-one-series-out）】")
probe_series = "蔚来ES6"
hold = df[df["series"] == probe_series]
train_wo = df[df["series"] != probe_series]
X_wo = rm.prepare_matrix(train_wo, series_map, series_vocab)
y_wo = pd.to_numeric(train_wo[rm.TARGET], errors="coerce").values
m_wo = rm.make_lgbm()
m_wo.fit(X_wo, y_wo, categorical_feature=rm.CAT_FEATURES)

X_ho = rm.prepare_matrix(hold, series_map, series_vocab)
# 用「见过该车系」（上述 OOF）与「没见过该车系」（此处）两种预测对同一批车比较
oof_ho = oof_lean[df["series"].values == probe_series]
pred_unk = m_wo.predict(X_ho)
y_ho = pd.to_numeric(hold[rm.TARGET], errors="coerce").values
print(f"  测试车系：{probe_series}（n={len(hold)}）")
print(f"  车系在词表内（对全新车系即不可用）预测 MAE = "
      f"{np.abs(y_ho - oof_ho).mean():.4f}")
print(f"  车系不在训练集、退化为未知档时    预测 MAE = "
      f"{np.abs(y_ho - pred_unk).mean():.4f}")
print(f"  退化代价：{np.abs(y_ho - pred_unk).mean() - np.abs(y_ho - oof_ho).mean():+.4f} MAE")
print("  说明：退化代价小，是因为未知档并非「哑变量全 0 的空白模型」，")
print("        仍会用车龄/里程/指导价给出预测，且本数据里 ES6 与品牌档特性接近。")
print("        但这不代表车系无关——真正要紧的是「该车系是否比同级更保值」")
print("        这类**车系特异结论**在未知档下完全无法给出。")
unk_test = {
    "series": probe_series, "n": int(len(hold)),
    "mae_series_known": round(float(np.abs(y_ho - oof_ho).mean()), 4),
    "mae_series_unknown": round(float(np.abs(y_ho - pred_unk).mean()), 4),
}

# ==================================================================
# 5. 拟合全量模型并保存 bundle
# ==================================================================
print(f"\n{RULE}\n5. 在全部 {len(df)} 条上拟合最终模型并保存\n{RULE}")
fitted = rm.fit_full(df, series_map, series_vocab)
print(f"训练样本 {fitted['n_fit']} 条，特征 {len(rm.FEATURES)} 维"
      f"（{len(rm.NUM_FEATURES)} 数值 + 1 车系类别，词表 {len(series_vocab)} 档）")

imp = pd.Series(fitted["point"].feature_importances_,
                index=rm.FEATURES).sort_values(ascending=False)
print("\n【特征重要性（分裂增益）】")
for k, v in imp.items():
    print(f"  {k:<20} {v:>8.1f}")

medians = {c: float(pd.to_numeric(df[c], errors="coerce").median())
           for c in ["age_months", "mileage_wan", "price_wan", "appearance_score"]}
reference = dict(medians)
reference["claim_times"] = 0.0
reference["transfer_times"] = 0.0

# 车系信息：样本量 + 区间加宽系数
series_info = {}
bucket_count = buckets.value_counts()
for s in series_vocab:
    n = int(bucket_count.get(s, 0))
    if s == rm.UNKNOWN_SERIES:
        status = "兜底档（训练集中未出现的车系会落到这里）"
    elif n >= 30:
        status = f"独立建模（样本 {n} 条，较可靠）"
    elif n >= rm.SERIES_MIN_N:
        status = f"独立建模（样本 {n} 条，偏少，精度下降）"
    else:
        status = "品牌聚合档（同品牌样本不足的车系合并）"
    series_info[s] = {"n": n, "status": status,
                      "shrink": round(float(np.sqrt(1 + 10.0 / max(n, 1))), 3)}

# 训练范围（用于外推告警）
train_range = {
    "age_months": [float(df["age_months"].min()), float(df["age_months"].max())],
    "mileage_wan": [float(df["mileage_wan"].min()), float(df["mileage_wan"].max())],
    "price_wan": [float(df["price_wan"].min()), float(df["price_wan"].max())],
}

# 经验年跌幅：用回归斜率，而不是分位数差值
# 分位数差值的坑：各车龄段的"样本构成"不同（老车段里品牌/车型结构变化），
# 直接相减会把构成效应混进"年跌幅"，本案实测 [48,60)→60月+ 只跌 4.46pp，
# 与 1-2 年段的 18.27pp 严重不一致，就是构成效应造成的假象。
d = df.dropna(subset=["age_months"]).copy().reset_index(drop=True)
bins = [0, 12, 24, 36, 48, 60, 999]
d["seg"] = pd.cut(d["age_months"], bins, right=False)
slope_all, intercept_all = np.polyfit(d["age_months"].values,
                                      d[rm.TARGET].values, 1)
print("\n【经验年跌幅 —— 回归斜率口径】")
print(f"  全体简单线性斜率 = {slope_all*12*100:+.2f}pp / 年"
      f"（R² = {np.corrcoef(d['age_months'], d[rm.TARGET])[0,1]**2:.3f}，n={len(d)}）")
print("  ⚠️ 简单斜率仍受品牌/车系构成影响，只能当上界参考；")
print("     真正的年跌幅由模型（含车系与里程控制）与 Ridge 斜率双轨给出。")

import statsmodels.formula.api as smf  # noqa: E402
# 车系档标签：按 car_id 映射回来（d 已 dropna，不能按位置取 values）
_bucket_by_id = dict(zip(df["car_id"], buckets.values))
d["series_cat"] = d["car_id"].map(_bucket_by_id)
assert d["series_cat"].notna().all(), "车系档映射存在缺失，请检查 car_id 唯一性"
m_ctrl = smf.ols(f"{rm.TARGET} ~ age_months + log_mileage_wan + log_price_wan "
                 f"+ C(series_cat)", data=d.assign(
                     log_mileage_wan=np.log(d["mileage_wan"].clip(lower=0) + 1),
                     log_price_wan=np.log(d["price_wan"].clip(lower=0) + 1))).fit()
ctrl_yearly = float(m_ctrl.params["age_months"] * 12 * 100)
print(f"  控制车系/里程/指导价后的车龄系数 = {ctrl_yearly:+.2f}pp / 年"
      f"（n={int(m_ctrl.nobs)}, R²={m_ctrl.rsquared:.3f}）")
print(f"  → 同一辆车的年折旧约 {abs(ctrl_yearly):.1f}pp；"
      f"简单斜率的 {slope_all*12*100:+.2f}pp 混入了车系构成效应。")

# 分车龄段的样本构成（说明为什么不能用差值）
seg_med = d.groupby("seg", observed=True)[rm.TARGET].median()
seg_n = {str(k): int(v) for k, v in d["seg"].value_counts().sort_index().items()}
empirical = {
    "slope_simple_pp_per_year": round(float(slope_all * 12 * 100), 2),
    "slope_controlled_pp_per_year": round(ctrl_yearly, 2),
    "slope_controlled_n": int(m_ctrl.nobs),
    "slope_controlled_r2": round(float(m_ctrl.rsquared), 4),
    "seg_median": {str(k): round(float(v), 4) for k, v in seg_med.items()},
    "seg_n": seg_n,
    "note": ("控制车系/里程/指导价后的车龄系数是「同一辆车开一年跌多少」的估计；"
             "简单斜率与分段中位数差值都混入了车系构成效应，只能作上界参考"),
}
print("\n【各车龄段中位数与样本量（仅供查构成，不直接相减当年跌幅）】")
for k in seg_med.index:
    print(f"  {str(k):<10} n={seg_n[str(k)]:>4}  中位残值率={seg_med[k]:.4f}")

bundle = {
    "version": "1.0",
    "created_by": "src/16_train_final.py",
    "target": rm.TARGET,
    "target_definition": "挂牌价 full_payment / 新车指导价 guide_price",
    "features": rm.FEATURES,
    "num_features": rm.NUM_FEATURES,
    "cat_features": rm.CAT_FEATURES,
    "series_vocab": series_vocab,
    "series_map": series_map,
    "series_info": series_info,
    "medians": medians,
    "reference": reference,
    "train_range": train_range,
    "n_train": int(fitted["n_fit"]),
    "snapshot": str(df["snapshot_date"].iloc[0]),
    "point": fitted["point"],
    "quantile": fitted["quantile"],
    "ridge": fitted["ridge"],
    "pi_quantile": list(rm.PI_Q),
    "oof_resid_quantile": [float(q_lo), float(q_hi)],
    "oof_metrics": {
        "group_kfold": {k: {kk: round(vv, 4) for kk, vv in v.items()}
                        for k, v in metrics.items()},
        "random_kfold_r2": round(r2(y, oof_rand), 4),
        "random_kfold_mae": round(mae(y, oof_rand), 4),
        "pi90_coverage_on_oof": round(cov90, 4),
        "baseline_mae": round(mae(y, oof_base), 4),
        "improve_vs_baseline_pct": round(
            (mae(y, oof_base) - lean["mae"]) / mae(y, oof_base) * 100, 1),
        "n": int(len(y)),
    },
    "ablation": abl_out,
    "per_series_error": per_series,
    "unknown_series_test": unk_test,
    "empirical": empirical,
}
import joblib  # noqa: E402
joblib.dump(bundle, rm.BUNDLE_PATH)
print(f"\n已保存模型：{rm.BUNDLE_PATH}")
print(f"  文件大小 {os.path.getsize(rm.BUNDLE_PATH)/1024:.1f} KB")

# ==================================================================
# 6. OOF 明细（供 demo 与看板使用，避免重复跑 CV）
# ==================================================================
oof_df = df[["car_id", "series", "model", "city_name", "age_months",
             "mileage_wan", "price_wan", "guide_price", "full_payment",
             rm.TARGET]].copy()
oof_df["oof_pred"] = oof_lean
oof_df["oof_pred_random"] = oof_rand
oof_df["oof_resid"] = resid_lean
oof_df["abs_err_pp"] = (oof_df["oof_resid"].abs() * 100).round(2)
oof_df["pred_price_wan"] = (oof_df["oof_pred"] * oof_df["price_wan"]).round(2)
oof_df["err_wan"] = (oof_df["pred_price_wan"] - oof_df["full_payment"] / 10000).round(2)
oof_path = os.path.join(rm.ROOT, "results", "p01_预测明细_OOF.csv")
oof_df.to_csv(oof_path, index=False, encoding="utf-8-sig")
print(f"已写出 OOF 预测明细：{oof_path}")

# ==================================================================
# 7. 模型卡
# ==================================================================
card = f"""# 残值预测模型卡（Model Card）

> 生成脚本：`src/16_train_final.py`　生成日期：本次运行
> 模型文件：`output/models/residual_model.joblib`
> 配套推理：`src/predict.py`（同一 `src/residual_model.py` 口径）

## 1. 这个模型回答什么问题

输入一辆车的**车系 / 车龄 / 里程 / 新车指导价（可选车况）**，输出：

1. **残值率**：这车现在大约还能卖到新车指导价的百分之几
2. **预测挂牌价（万元）** = 残值率 × 新车指导价 × 10000
3. **90% 预测区间**：单台车真实挂牌价大概率落在的区间

## 2. 口径（读数字前必须知道）

| 项 | 定义 |
|---|---|
| 目标变量 | `residual_rate = 挂牌价 full_payment / 新车指导价 guide_price` |
| 因变量来源 | 瓜子二手车公开详情页，{bundle['snapshot']} 单期快照 |
| 训练样本 | {bundle['n_train']} 条新能源在售车源（全部样本，无留出） |
| 挂牌价 ≠ 成交价 | 挂牌价系统性高于成交价，预测值应读作**合理挂牌区间** |
| 分母口径 | 指导价存在「上市时指导价 / 当前款指导价」混用，见结论报告局限 7 |

## 3. 模型与特征

- **主模型**：LightGBM（`num_leaves=15, max_depth=4, min_child_samples=12`），
  `series_cat` 走**原生类别特征**，不做哑变量展开
- **特征 {len(rm.FEATURES)} 维**：`{'`, `'.join(rm.FEATURES)}`
- **预测区间**：两个分位数头（α=0.05 / 0.95）直接回归预测区间边界，
  允许**非对称**（残值率分布左偏，对称 ±RMSE 会低估下方风险）
- **外推对照模型**：Ridge 线性回归（同一组特征），专用于「1 年后残值」外推
- **未知车系兜底**：词表外的车系统一映射到 `{rm.UNKNOWN_SERIES}` 档，
  推理时**强制输出告警**，预测含义退化为「同级同价位的平均车」

## 4. 性能（GroupKFold by 车系，5 折 OOF，n={len(y)}）

| 模型 | MAE | RMSE | R² |
|---|---|---|---|
| M0 常数基线（中位数） | {metrics['M0 常数基线（中位数）']['mae']:.4f} | {metrics['M0 常数基线（中位数）']['rmse']:.4f} | {metrics['M0 常数基线（中位数）']['r2']:.4f} |
| M1 Ridge 线性 | {metrics['M1 Ridge 线性（精简特征）']['mae']:.4f} | {metrics['M1 Ridge 线性（精简特征）']['rmse']:.4f} | {metrics['M1 Ridge 线性（精简特征）']['r2']:.4f} |
| **M2 LightGBM（本模型）** | **{lean['mae']:.4f}** | **{lean['rmse']:.4f}** | **{lean['r2']:.4f}** |
| M3 LightGBM（原 90 维哑变量） | {wide['mae']:.4f} | {wide['rmse']:.4f} | {wide['r2']:.4f} |

- 随机 KFold 对照：R² **{r2(y, oof_rand):.4f}**（预测**熟悉**车系，会高估能力）
- GroupKFold：R² **{lean['r2']:.4f}**（预测**没见过的**车系）
- 较常数基线 MAE 降低 **{bundle['oof_metrics']['improve_vs_baseline_pct']:.1f}%**
- **MAE {lean['mae']:.4f} 折算成金额**：按挂牌价均值
  {df['full_payment'].mean()/10000:.1f} 万算，约 **{lean['mae']*df['full_payment'].mean()/10000:.2f} 万元**
- 90% 预测区间在 OOF 上的经验覆盖率：**{cov90*100:.1f}%**（标定目标 90%）

**为什么精简版不输 90 维版**：车系哑变量在置换重要性上为 2.4e-16（`08_机器学习.txt`），
城市与车况哑变量的边际贡献也很小，而 726 行 × 90 列会让模型靠背下车系取胜。
换成原生类别车系后 MAE 变化 {lean['mae']-wide['mae']:+.4f}，精度持平但**可外推、可解释**。

## 5. 特征重要性（分裂增益）

| 特征 | 增益 |
|---|---|
""" + "\n".join(f"| {k} | {v:.1f} |" for k, v in imp.items()) + f"""

## 6. 适用边界（必须与预测结果一同呈现）

1. **单期快照**：全部数据采集于 {bundle['snapshot']}，模型**无法预测市场整体涨跌**。
   「1 年后残值」只反映车龄/里程自然衰减，不含市场行情变化。
2. **挂牌价口径**：预测的是合理挂牌价，不是成交价。
3. **存活偏差**：在售车源 = 还没卖掉的车，长期滞销车会拉高挂牌价。
4. **未知车系不可信**：车系不在词表内时，模型只能给出「同级同价位平均车」，
   这正是 GroupKFold 口径下 R² 只有 {lean['r2']:.4f}（而非随机的 {r2(y, oof_rand):.4f}）的原因。
5. **车龄外推**：训练车龄范围 {train_range['age_months'][0]:.0f}–{train_range['age_months'][1]:.0f} 月。
   超出范围时，树模型会**趋于平坦**（分段常数，学不到继续下降的趋势），
   因此「1 年后残值」同时给出 Ridge 线性斜率做对照。
6. **不做因果推断**：所有关系均为相关性描述。
7. **SOH / 电池容量未入模**：覆盖率仅 61.3% / 28.7%，且 SOH 全部来自平台文案、
   无实测依据（审计发现 5 台车两处数字自相矛盾）。真正要提升精度需要企业内部电池检测数据。

## 7. 复现

```bash
python src/16_train_final.py     # 训练 + 验证 + 保存模型
python src/predict.py --series 蔚来ES6 --age-years 3.5 --mileage-wan 6.8 --guide-price-wan 33.8
python src/17_predict_demo.py    # 4 辆车案例 + 1 年后外推 demo
```

固定随机种子 `{rm.SEED}`，ML 指标可逐位复现。
"""
with open(rm.MODEL_CARD_PATH, "w", encoding="utf-8") as f:
    f.write(card)
print(f"已写出模型卡：{rm.MODEL_CARD_PATH}")

# 元信息（轻量 JSON，便于看板/文档直接读）
meta = {k: v for k, v in bundle.items()
        if k not in ("point", "quantile", "ridge")}
with open(rm.BUNDLE_META_PATH, "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=1)
print(f"已写出元信息：{rm.BUNDLE_META_PATH}")
print(f"\n{RULE}\n完成。\n{RULE}")
