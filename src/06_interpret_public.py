# -*- coding: utf-8 -*-
r"""06_interpret_public.py —— 特征归因、消融实验与误差结构分析

回答三个业务问题：
  1. **各类特征各自值多少精度？** —— 分层消融实验
  2. **什么在驱动折旧？**        —— Permutation Importance + SHAP
  3. **模型在哪些车上错得最多？** —— 分段误差分析（这是最有业务价值的部分）

消融分层设计：
    L1 车龄                —— 只有"用了多久"
    L2 +使用强度            —— 里程、年均里程
    L3 +车况代理            —— 过户次数（一手车溢价）
    L4 +车辆规格            —— 功率、排量、升功率
    L5 +市场基准            —— 同车系准新车价（控制车型档次）
    L6 +类别分层            —— 品牌/车系/车身/地区/变速箱/驱动

误差分析设计：
  折旧预测真正难的是「同车龄同里程，为什么这辆更保值」。
  因此按车龄分段、按价格段分段、按品牌分段统计 MAE，
  找出模型失效的区间 —— 这比一个平均 MAE 有用得多。

用法：
    python src/06_interpret_public.py
    python src/06_interpret_public.py --no-shap
    python src/06_interpret_public.py --sample 80000
"""
from __future__ import annotations

import argparse
import os
import sys
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "clean", "cars_public.csv")
OUTDIR = os.path.join(ROOT, "results")
IMGDIR = os.path.join(ROOT, "images")
for d in (OUTDIR, IMGDIR):
    os.makedirs(d, exist_ok=True)

TARGET = "residual_rate"
RANDOM_STATE = 42

LAYERS = {
    "L1_车龄": ["age_years"],
    "L2_+使用强度": ["log_mileage", "km_per_year"],
    "L3_+过户次数": ["owners"],
    "L4_+车辆规格": ["power", "displacement", "power_per_l"],
    "L5_+市场基准": ["log_new_ref"],
    "L6_+类别分层": ["brand", "series", "body_style", "city_name",
                     "engine_type", "transmission", "drive_mode"],
}
ALL_FEATS = [c for v in LAYERS.values() for c in v]
CATS = {"brand", "series", "body_style", "city_name", "engine_type",
        "transmission", "drive_mode"}


def fwd(y):
    y = np.clip(np.asarray(y, dtype=float), 1e-4, 1 - 1e-4)
    return np.log(y / (1 - y))


def inv(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=float)))


def prepare_xy(df, feats):
    feats = [c for c in feats if c in df.columns]
    X = df[feats].copy()
    for c in X.columns:
        if c in CATS:
            X[c] = pd.Categorical(X[c]).codes.astype("float64")
        else:
            X[c] = pd.to_numeric(X[c], errors="coerce").astype("float64")
    return X


def make_lgbm(cat_idx, n_est=400):
    from lightgbm import LGBMRegressor
    kw = {"categorical_feature": cat_idx} if cat_idx else {}
    return LGBMRegressor(n_estimators=n_est, learning_rate=0.05, num_leaves=63,
                         min_child_samples=20, subsample=0.9, colsample_bytree=0.9,
                         random_state=RANDOM_STATE, verbose=-1, **kw)


def cv_pred(df, feats, n_est=400):
    """按车系分组交叉验证，返回 OOF 预测。"""
    from sklearn.model_selection import GroupKFold
    X = prepare_xy(df, feats)
    cat_idx = [X.columns.get_loc(c) for c in X.columns if c in CATS]
    y = df[TARGET].astype(float).values
    groups = df["series"].astype(str).values
    k = min(5, max(2, len(np.unique(groups))))
    oof = np.full(len(y), np.nan)
    for tr, te in GroupKFold(n_splits=k).split(X, y, groups):
        m = make_lgbm(cat_idx, n_est)
        m.fit(X.iloc[tr], fwd(y[tr]))
        oof[te] = inv(m.predict(X.iloc[te]))
    return oof, y


def ablation(df, n_est):
    print("\n" + "=" * 84)
    print("消融实验 · 每一类特征的边际价值（LightGBM，GroupKFold by 车系）")
    print("=" * 84)
    rows, cum, prev = [], [], None
    for name, feats in LAYERS.items():
        cum = cum + feats
        oof, y = cv_pred(df, cum, n_est)
        ok = ~np.isnan(oof)
        mae = float(np.mean(np.abs(y[ok] - oof[ok])))
        rmse = float(np.sqrt(np.mean((y[ok] - oof[ok]) ** 2)))
        delta = "" if prev is None else f"{mae - prev:+.4f}"
        rows.append({"特征集": name, "特征数": len(cum), "MAE": round(mae, 4),
                     "RMSE": round(rmse, 4), "较上一层": delta})
        print(f"  {name:<16} 特征 {len(cum):>2} | MAE {mae:.4f} | "
              f"RMSE {rmse:.4f} | 变化 {delta}")
        prev = mae
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUTDIR, "r04_消融实验.csv"),
               index=False, encoding="utf-8-sig")
    print("\n  解读：每一层的 MAE 降幅 = 该类信息的边际价值。")
    print("        若某层几乎没降，说明该维度在当前数据上区分度有限。")
    return res


def segment_error(df, n_est):
    """分段误差分析：模型在哪类车上失效。"""
    print("\n" + "=" * 84)
    print("分段误差分析 · 模型在哪些车上错得最多")
    print("=" * 84)
    oof, y = cv_pred(df, ALL_FEATS, n_est)
    d = df.copy()
    d["_pred"] = oof
    d["_err"] = np.abs(y - oof)
    d["_bias"] = oof - y          # 正 = 高估
    d = d[d["_pred"].notna()]

    print(f"\n  整体 MAE {d['_err'].mean():.4f} | "
          f"中位误差 {d['_err'].median():.4f} | "
          f"P90 误差 {d['_err'].quantile(0.9):.4f}")

    print("\n  【按车龄分段】")
    d["_age_bin"] = pd.cut(d["age_years"], [-0.1, 1, 2, 3, 5, 8, 12, 20, 40])
    g = d.groupby("_age_bin", observed=True).agg(
        n=("_err", "size"), MAE=("_err", "mean"),
        偏差=("_bias", "mean"), 实际折旧比中位=(TARGET, "median")).round(4)
    print(g.to_string())

    print("\n  【按价格段分段】")
    d["_price_bin"] = pd.qcut(d["price_wan"], 5,
                              labels=["最低20%", "20-40%", "40-60%", "60-80%", "最高20%"])
    g2 = d.groupby("_price_bin", observed=True).agg(
        n=("_err", "size"), MAE=("_err", "mean"),
        偏差=("_bias", "mean"), 价格中位万=("price_wan", "median")).round(4)
    print(g2.to_string())

    print("\n  【按品牌（样本量 Top 10）】")
    top = d["brand"].value_counts().head(10).index
    g3 = (d[d["brand"].isin(top)].groupby("brand")
          .agg(n=("_err", "size"), MAE=("_err", "mean"),
               偏差=("_bias", "mean"), 折旧比中位=(TARGET, "median"))
          .round(4).sort_values("MAE", ascending=False))
    print(g3.to_string())

    print("\n  【偏差方向分析】")
    over = (d["_bias"] > 0).mean()
    print(f"    高估占比 {over:.1%} | 低估占比 {1-over:.1%}")
    print("    偏差接近 50/50 说明模型没有系统性高估或低估。")

    for nm, gg in [("车龄", g), ("价格段", g2), ("品牌", g3)]:
        gg.to_csv(os.path.join(OUTDIR, f"r05_分段误差_{nm}.csv"),
                  encoding="utf-8-sig")
    return d


def importance_shap(df, n_est, do_shap=True):
    from sklearn.inspection import permutation_importance
    X = prepare_xy(df, ALL_FEATS)
    cat_idx = [X.columns.get_loc(c) for c in X.columns if c in CATS]
    y = df[TARGET].astype(float).values
    m = make_lgbm(cat_idx, n_est)
    m.fit(X, fwd(y))

    def score(est, Xs, ys):
        return -np.mean(np.abs(ys - inv(est.predict(Xs))))

    idx = np.random.RandomState(RANDOM_STATE).choice(
        len(X), min(20000, len(X)), replace=False)
    r = permutation_importance(m, X.iloc[idx], y[idx], scoring=score,
                              n_repeats=6, random_state=RANDOM_STATE)
    imp = pd.DataFrame({"特征": list(X.columns),
                        "重要性": r.importances_mean.round(5),
                        "标准差": r.importances_std.round(5)}) \
        .sort_values("重要性", ascending=False)
    print("\n" + "=" * 84)
    print("Permutation Importance（MAE 恶化程度，越大越重要）")
    print("=" * 84)
    for _, row in imp.iterrows():
        bar = "#" * max(1, int(row["重要性"] * 300))
        print(f"  {row['特征']:<18} {row['重要性']:>9.5f}  {bar}")
    imp.to_csv(os.path.join(OUTDIR, "r06_特征重要性.csv"),
               index=False, encoding="utf-8-sig")

    if not do_shap:
        return imp
    try:
        import shap
        expl = shap.TreeExplainer(m)
        sample = X.sample(min(3000, len(X)), random_state=RANDOM_STATE)
        sv = np.asarray(expl.shap_values(sample))
        if sv.ndim == 3:
            sv = sv[:, :, 0]
        sdf = pd.DataFrame({"特征": list(sample.columns),
                            "平均|SHAP|": np.abs(sv).mean(axis=0)}) \
            .sort_values("平均|SHAP|", ascending=False)
        print("\n" + "=" * 84)
        print("SHAP 平均绝对贡献")
        print("=" * 84)
        for _, row in sdf.iterrows():
            bar = "#" * max(1, int(row["平均|SHAP|"] * 60))
            print(f"  {row['特征']:<18} {row['平均|SHAP|']:>9.5f}  {bar}")
        sdf.to_csv(os.path.join(OUTDIR, "r07_SHAP归因.csv"),
                   index=False, encoding="utf-8-sig")

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
        plt.figure(figsize=(9, 6))
        shap.summary_plot(sv, sample, show=False, max_display=14)
        plt.tight_layout()
        plt.savefig(os.path.join(IMGDIR, "fig_public_shap.png"), dpi=150,
                    bbox_inches="tight")
        plt.close()
        print("\n  SHAP 图已保存：images/fig_public_shap.png")
    except Exception as e:  # noqa: BLE001
        print(f"\n  SHAP 失败（不影响其他结果）：{type(e).__name__}: {str(e)[:80]}")
    return imp


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-shap", action="store_true")
    ap.add_argument("--n-estimators", type=int, default=400)
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--data", default=DATA)
    a = ap.parse_args()

    if not os.path.exists(a.data):
        print(f"找不到 {a.data}，请先运行 src/00_prepare_public.py")
        return
    df = pd.read_csv(a.data, encoding="utf-8-sig")
    if a.sample and len(df) > a.sample:
        df = df.sample(a.sample, random_state=RANDOM_STATE).reset_index(drop=True)
        print(f"已抽样至 {len(df):,} 条")
    print(f"样本 {len(df):,} 条")

    ablation(df, a.n_estimators)
    segment_error(df, a.n_estimators)
    importance_shap(df, a.n_estimators, do_shap=not a.no_shap)


if __name__ == "__main__":
    main()
