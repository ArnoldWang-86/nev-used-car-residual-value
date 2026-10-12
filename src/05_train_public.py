# -*- coding: utf-8 -*-
r"""05_train_public.py —— 公开数据集上的残值/折旧预测建模

目标：deprec_ratio = 本车价 / 同车系最新年款中位价（口径见 00_prepare_public.py）

模型谱系（覆盖 JD 要求的全部主流模型）：
    M0 常数基线（中位数）        —— 判断模型到底有没有价值
    M1 Ridge 线性回归            —— 可解释基线
    M2 随机森林 RandomForest     —— bagging 基线
    M3 LightGBM                  —— 主力
    M4 XGBoost                   —— 主力（与 LGBM 交叉验证）
    M5 LGBM+XGB 融合

验证设计（本脚本的核心方法学）：
  1. GroupKFold by series —— 同车系不能跨训练/验证集，否则模型"背车系"作弊
  2. 时间外推 —— 新车上牌年份切分，模拟「用老车预测新车」
  3. 两者都跑并对比 —— 差异本身就是结论

评估：MAE / RMSE / MAPE / SMAPE / R²，并统一报「相对常数基线的提升」

用法：
    python src/05_train_public.py
    python src/05_train_public.py --sample 80000      # 抽样加速
    python src/05_train_public.py --target raw
"""
from __future__ import annotations

import argparse
import json
import os
import warnings

import numpy as np
import pandas as pd

# --- 中文 Windows 控制台默认 GBK，强制 UTF-8 输出，避免打印特殊字符时崩溃 ---
try:
    import sys as _sys
    _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    _sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass


warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "clean", "cars_public.csv")
OUTDIR = os.path.join(ROOT, "results")
os.makedirs(OUTDIR, exist_ok=True)

TARGET = "residual_rate"          # 即 deprec_ratio
RANDOM_STATE = 42
N_SPLITS = 5

# 特征分层（用于消融）
NUM_BASE = ["age_years"]
NUM_USAGE = ["log_mileage", "km_per_year"]
NUM_SPEC = ["power", "displacement", "power_per_l", "owners"]
NUM_MARKET = ["log_new_ref"]
CAT_SPEC = ["engine_type", "transmission", "drive_mode", "steering_wheel"]
CAT_HIER = ["brand", "series", "body_style", "city_name"]

ALL_FEATS = NUM_BASE + NUM_USAGE + NUM_SPEC + NUM_MARKET + CAT_SPEC + CAT_HIER
CATS = CAT_SPEC + CAT_HIER


def fwd(y, mode):
    y = np.clip(np.asarray(y, dtype=float), 1e-4, 1 - 1e-4)
    if mode == "logit":
        return np.log(y / (1 - y))
    if mode == "log":
        return np.log(y)
    return y


def inv(z, mode):
    z = np.asarray(z, dtype=float)
    if mode == "logit":
        return 1.0 / (1.0 + np.exp(-z))
    if mode == "log":
        return np.exp(z)
    return z


def metrics(y, p) -> dict:
    y = np.asarray(y, float); p = np.asarray(p, float)
    ok = np.isfinite(y) & np.isfinite(p)
    y, p = y[ok], p[ok]
    mae = float(np.mean(np.abs(y - p)))
    rmse = float(np.sqrt(np.mean((y - p) ** 2)))
    nz = y != 0
    mape = float(np.mean(np.abs((y[nz] - p[nz]) / y[nz])) * 100) if nz.any() else np.nan
    den = (np.abs(y) + np.abs(p)) / 2
    nzd = den != 0
    smape = float(np.mean(np.abs(y[nzd] - p[nzd]) / den[nzd]) * 100) if nzd.any() else np.nan
    ss_res = float(np.sum((y - p) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot else np.nan
    return {"MAE": round(mae, 4), "RMSE": round(rmse, 4), "MAPE": round(mape, 2),
            "SMAPE": round(smape, 2), "R2": round(r2, 4), "n": int(len(y))}


def prepare_xy(df, feats):
    feats = [c for c in feats if c in df.columns]
    X = df[feats].copy()
    for c in X.columns:
        if c in CATS:
            X[c] = pd.Categorical(X[c]).codes.astype("float64")
        else:
            X[c] = pd.to_numeric(X[c], errors="coerce").astype("float64")
    return X


def make_models(cat_idx, n_est):
    from lightgbm import LGBMRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Ridge
    from xgboost import XGBRegressor

    kw = {"categorical_feature": cat_idx} if cat_idx else {}
    return {
        "M1_Ridge": Ridge(alpha=1.0, random_state=RANDOM_STATE),
        "M2_随机森林": RandomForestRegressor(
            n_estimators=max(100, n_est // 2), min_samples_leaf=3, n_jobs=-1,
            random_state=RANDOM_STATE),
        "M3_LightGBM": LGBMRegressor(
            n_estimators=n_est, learning_rate=0.05, num_leaves=63,
            min_child_samples=20, subsample=0.9, colsample_bytree=0.9,
            random_state=RANDOM_STATE, verbose=-1, **kw),
        "M4_XGBoost": XGBRegressor(
            n_estimators=n_est, learning_rate=0.05, max_depth=6,
            subsample=0.9, colsample_bytree=0.9, min_child_weight=3,
            random_state=RANDOM_STATE, n_jobs=-1, tree_method="hist",
            enable_categorical=False, verbosity=0),
    }


def fit_predict(name, model, Xtr, ytr, Xte):
    """Ridge 需要标准化与缺失填充，树模型直接吃 NaN。"""
    if name == "M1_Ridge":
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        med = Xtr.median(numeric_only=True)
        Xtr2 = Xtr.fillna(med).fillna(0)
        Xte2 = Xte.fillna(med).fillna(0)
        pipe = make_pipeline(StandardScaler(), model)
        pipe.fit(Xtr2, ytr)
        return pipe.predict(Xte2)
    model.fit(Xtr, ytr)
    return model.predict(Xte)


def group_cv(df, feats, mode, n_est):
    from sklearn.model_selection import GroupKFold

    X = prepare_xy(df, feats)
    cat_idx = [X.columns.get_loc(c) for c in feats if c in CATS]
    y = df[TARGET].astype(float).values
    groups = df["series"].astype(str).values
    k = min(N_SPLITS, max(2, len(np.unique(groups))))
    models = make_models(cat_idx, n_est)

    oof = {n: np.full(len(y), np.nan) for n in models}
    oof["M0_常数基线"] = np.full(len(y), np.nan)
    for tr, te in GroupKFold(n_splits=k).split(X, y, groups):
        ytr_t = fwd(y[tr], mode)
        oof["M0_常数基线"][te] = float(np.median(y[tr]))
        for n, m in models.items():
            try:
                oof[n][te] = inv(fit_predict(n, m, X.iloc[tr], ytr_t, X.iloc[te]), mode)
            except Exception as e:  # noqa: BLE001
                print(f"    [警告] {n} 失败: {type(e).__name__}: {str(e)[:60]}")
    oof["M5_融合_LGBM_XGB"] = np.nanmean(
        np.vstack([oof["M3_LightGBM"], oof["M4_XGBoost"]]), axis=0)

    rows = []
    for n, p in oof.items():
        ok = ~np.isnan(p)
        rows.append({"模型": n, **metrics(y[ok], p[ok])})
    res = pd.DataFrame(rows)
    b = float(res.loc[res.模型 == "M0_常数基线", "MAE"].iloc[0])
    res["MAE较基线提升%"] = ((b - res["MAE"]) / b * 100).round(1)
    bm = float(res.loc[res.模型 == "M0_常数基线", "MAPE"].iloc[0])
    res["MAPE较基线提升%"] = ((bm - res["MAPE"]) / bm * 100).round(1)
    return res, oof, y


def time_holdout(df, feats, mode, n_est):
    """时间外推：老车训练 → 新车测试（模拟上线后遇到更新的车）。"""
    X = prepare_xy(df, feats)
    cat_idx = [X.columns.get_loc(c) for c in feats if c in CATS]
    y = df[TARGET].astype(float).values
    age = df["age_years"].values
    cut = float(np.quantile(age, 0.7))          # 车龄小的 30% 作为"未来"
    tr = np.where(age > cut)[0]                 # 老车训练
    te = np.where(age <= cut)[0]                # 新车测试
    if len(tr) < 500 or len(te) < 200:
        return None, None, None
    models = make_models(cat_idx, n_est)
    ytr_t = fwd(y[tr], mode)
    preds = {"M0_常数基线": np.full(len(te), float(np.median(y[tr])))}
    for n, m in models.items():
        try:
            preds[n] = inv(fit_predict(n, m, X.iloc[tr], ytr_t, X.iloc[te]), mode)
        except Exception as e:  # noqa: BLE001
            print(f"    [警告] 时间外推 {n} 失败: {type(e).__name__}")
    preds["M5_融合_LGBM_XGB"] = np.nanmean(
        np.vstack([preds["M3_LightGBM"], preds["M4_XGBoost"]]), axis=0)
    rows = [{"模型": n, **metrics(y[te], p)} for n, p in preds.items()]
    res = pd.DataFrame(rows)
    b = float(res.loc[res.模型 == "M0_常数基线", "MAE"].iloc[0])
    res["MAE较基线提升%"] = ((b - res["MAE"]) / b * 100).round(1)
    return res, y[te], cut


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=["logit", "log", "raw"], default="logit")
    ap.add_argument("--n-estimators", type=int, default=400)
    ap.add_argument("--sample", type=int, default=0, help="随机抽样条数，0=全量")
    ap.add_argument("--data", default=DATA)
    a = ap.parse_args()

    if not os.path.exists(a.data):
        print(f"找不到 {a.data}，请先运行 src/00_prepare_public.py")
        return
    df = pd.read_csv(a.data, encoding="utf-8-sig")
    if a.sample and len(df) > a.sample:
        df = df.sample(a.sample, random_state=RANDOM_STATE).reset_index(drop=True)
        print(f"已抽样至 {len(df):,} 条")

    print("=" * 80)
    print("公开二手车数据集 · 折旧比建模")
    print("=" * 80)
    print(f"样本 {len(df):,} | 特征 {len(ALL_FEATS)} | 目标变换 {a.target}")
    print(f"目标 deprec_ratio：均值 {df[TARGET].mean():.4f} "
          f"中位 {df[TARGET].median():.4f} 标准差 {df[TARGET].std():.4f}")

    print("\n" + "=" * 80)
    print("验证一 · GroupKFold（按车系分组，防止车系泄漏）")
    print("=" * 80)
    res, oof, y = group_cv(df, ALL_FEATS, a.target, a.n_estimators)
    print(res.to_string(index=False))
    res.to_csv(os.path.join(OUTDIR, "r01_模型对比_GroupKFold.csv"),
               index=False, encoding="utf-8-sig")

    print("\n" + "=" * 80)
    print("验证二 · 时间外推（老车训练 → 新车测试）")
    print("=" * 80)
    t2, yte, cut = time_holdout(df, ALL_FEATS, a.target, a.n_estimators)
    if t2 is None:
        print("样本不足，跳过")
    else:
        print(f"切分点：车龄 > {cut:.1f} 年训练，<= {cut:.1f} 年测试 | 测试 {len(yte):,} 条")
        print(t2.to_string(index=False))
        t2.to_csv(os.path.join(OUTDIR, "r02_模型对比_时间外推.csv"),
                  index=False, encoding="utf-8-sig")

    # 保存预测明细（供误差分析）
    keep = [c for c in ["car_id", "brand", "series", "body_style", "age_years",
                        "mileage", "owners", "power", "displacement",
                        "transmission", "engine_type", "city_name",
                        TARGET, "price_wan", "new_ref_wan"] if c in df.columns]
    out = df[keep].copy()
    for n, p in oof.items():
        out[f"pred_{n}"] = p
    out.to_csv(os.path.join(OUTDIR, "r03_预测明细.csv"),
               index=False, encoding="utf-8-sig")
    print(f"\n已保存 results/r03_预测明细.csv")

    best = res[res.模型 != "M0_常数基线"].sort_values("MAE").iloc[0]
    summary = {
        "dataset": "HF imbalet/used_cars (RU used-car listings)",
        "n_samples": int(len(df)),
        "target": "deprec_ratio = 本车价 / 同(品牌,车型,发动机,变速箱)组内车龄<=2年中位价",
        "caveat": "数据源无官方指导价，该口径为相对折旧比，非中国口径残值率",
        "target_transform": a.target,
        "n_features": len(ALL_FEATS),
        "best_model": best["模型"], "best_MAE": float(best["MAE"]),
        "best_R2": float(best["R2"]),
        "group_cv": res.to_dict("records"),
    }
    if t2 is not None:
        summary["time_holdout"] = t2.to_dict("records")
    with open(os.path.join(OUTDIR, "r00_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    print("已保存 results/r00_summary.json")


if __name__ == "__main__":
    main()
