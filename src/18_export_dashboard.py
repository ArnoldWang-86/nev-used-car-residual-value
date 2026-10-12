# -*- coding: utf-8 -*-
r"""18_export_dashboard.py —— 导出可交互预测页面的数据资产（模型内嵌版）

产物：`output/predict_assets.json`（由 `19_build_dashboard.py` 内联进单文件页面）

方案选择（先读这段，它决定了页面的可信度）
----------------------------------------
页面是纯静态文件（双击即可打开、可随简历投递），没有 Python 后端。两条路：

方案 A · 预计算查表 + 浏览器插值
  实测否决。该模型沿价格/里程轴是**强阶梯函数**——蔚来ES6、车龄36月、里程6万时，
  指导价 8万→10万 预测值一字不变（0.5685），10万→12万 猛降 7.6pp。
  对阶梯函数插值，误差上限就是台阶高度；实测「最近结点」平均误差 0.0147、
  最大 0.1405，要靠加密网格压下去需要 46 个价格层 × 0.5 万公里步长，
  资产会膨胀到 25MB 以上。**而且插值会画出模型根本不存在的平滑趋势。**

方案 B · 把 LightGBM 导成 JSON，页面里用 JS 遍历树（本脚本采用）
  精确、零插值误差、体积比方案 A 小一个数量级。
  模型本身只有 8 个特征，连续特征都是区间切分，JS 端判一个 `v <= threshold`
  即可。直方图分箱只作用于建树过程，不影响推理结果，无需复现。

  分位数头（α=0.05/0.95）同样导出，于是**预测区间也是精确的**，不是估的。
  线性对照模型（Ridge）的系数直接导出，用于「1 年后残值」外推轨。

自检：脚本会用导出的 JSON 在 Python 里**重新实现一遍推理**，
      与 LightGBM 原生 `predict()` 逐点比对，确保导出无损。

用法：
    python src/18_export_dashboard.py
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

ASSET_PATH = os.path.join(rm.ROOT, "output", "predict_assets.json")
SCALE = 100000  # 树结点值 / 阈值统一放大存整数，避免浮点尾巴

RULE = "=" * 84
print(RULE)
print("18_export_dashboard.py · 导出模型 JSON（页面内嵌遍历）")
print(RULE)

df = rm.load_clean()
bundle = rm.load_bundle()


# ==================================================================
# 1. 导出 LightGBM 树结构
# ==================================================================
def export_lgbm(booster) -> dict:
    """把 LightGBM 模型导成紧凑 JSON。

    连续特征：阈值为浮点；缺失走 default_left。
    原生类别特征：阈值形如 "1||3||7" 或 "2||4-"（后者表示区间），
    JS 端按同样规则判断；类别值直接用 pandas categorical 的**编号**。
    """
    dump = booster.dump_model()
    # ⚠️ 契约：特征名必须来自 feature_infos（字典，含类别特征），
    #    不能用 dump["feature_names"]——本项目实测该列表**只有 7 个数值特征，
    #    缺了 series_cat**。用它会导致页面算不出类别特征的下标，
    #    车系编码静默丢失、所有车系都退化成未知档，而且不报错。
    feats = list(dump["feature_infos"].keys())
    assert len(feats) == len(rm.FEATURES), \
        f"导出特征数 {len(feats)} != 模型输入维度 {len(rm.FEATURES)}：{feats}"
    for _f in rm.FEATURES:
        assert _f in feats, f"导出特征缺少 {_f}"
    cat_feats = [i for i, f in enumerate(feats)
                 if f in ("series_cat",) or str(f).endswith("_cat")]
    assert cat_feats, "未识别到类别特征，页面将无法编码车系"

    def add(nodes, node) -> int:
        idx = len(nodes)
        nodes.append(None)
        if "leaf_value" in node:                      # 叶子
            # 存原始浮点，不做量化：量化到 1e-5 后累加 400 棵树会放大到 1.7e-4，
            # 而这里要求与 LightGBM 逐点一致（<1e-12），所以不能用定点数省体积。
            nodes[idx] = {"v": float(node["leaf_value"])}
            return idx
        rec = {
            "f": int(node["split_feature"]),
            "th": (node["threshold"] if node["decision_type"] == "=="
                   else float(node["threshold"])),
            "c": 1 if node["decision_type"] == "==" else 0,
            "l": None, "r": None,
            "dl": 1 if node.get("default_left", True) else 0,
        }
        rec["l"] = add(nodes, node["left_child"])
        rec["r"] = add(nodes, node["right_child"])
        nodes[idx] = rec
        return idx

    trees = []
    for t in dump["tree_info"]:
        nodes: list = []
        root = add(nodes, t["tree_structure"])
        trees.append({"r": root, "n": nodes})

    return {
        "features": feats,
        "categorical": cat_feats,
        "n_features": len(feats),
        "scale": 1.0,
        "trees": trees,
    }


def js_eval(model: dict, row: list[float]) -> float:
    """在 Python 里复刻页面 JS 的遍历逻辑，用于自检导出是否无损。"""
    cat_set = set(model["categorical"])
    tot = 0.0
    for tr in model["trees"]:
        nodes = tr["n"]
        i = tr["r"]
        while True:
            nd = nodes[i]
            if "v" in nd:
                tot += nd["v"]
                break
            x = row[nd["f"]]
            if x is None or (isinstance(x, float) and not np.isfinite(x)):
                i = nd["l"] if nd["dl"] else nd["r"]
                continue
            if nd["c"] and nd["f"] in cat_set:
                go_left = False
                for part in str(nd["th"]).split("||"):
                    if part.endswith("-"):
                        lo, hi = part[:-1].split("-")
                        if int(lo) <= int(x) <= int(hi):
                            go_left = True
                            break
                    elif part != "" and int(part) == int(x):
                        go_left = True
                        break
            else:
                go_left = float(x) <= nd["th"]
            i = nd["l"] if go_left else nd["r"]
    return tot


print("\n导出模型树结构…")
model_point = export_lgbm(bundle["point"].booster_)
model_q = {str(a): export_lgbm(bundle["quantile"][a].booster_)
           for a in rm.PI_Q}
for name, m in [("点估计", model_point)] + [(f"分位数 α={a}", model_q[str(a)])
                                            for a in rm.PI_Q]:
    n_nodes = sum(len(t["n"]) for t in m["trees"])
    print(f"  {name:<14} {len(m['trees'])} 棵树　{n_nodes:,} 个结点")

# ==================================================================
# 2. 线性对照模型（Ridge）系数导出
# ==================================================================
print("\n导出线性对照模型（Ridge）系数…")
ridge = bundle["ridge"]
pre = ridge.named_steps["pre"]
imp = pre.named_transformers_["num"].named_steps["imp"]
scl = pre.named_transformers_["num"].named_steps["sc"]
ohe = pre.named_transformers_["cat"]
coef = np.asarray(ridge.named_steps["r"].coef_).ravel()
intercept = float(ridge.named_steps["r"].intercept_)
# ⚠️ 不能假设「数值特征在前」——ColumnTransformer 的输出顺序由 output_indices_ 决定。
#    原来写 coef[:len(NUM)] 取数值系数、coef[len(NUM):] 取类别系数，
#    在顺序被调换时会整体错位（本例常数偏移 0.815，看起来像个"正常"的预测值，
#    不报错、只是全错，是最危险的一类 bug）。
out_idx = pre.output_indices_
n_out = int(np.sum([len(c) for _, _, c in pre.transformers]))


def _idx(entry) -> np.ndarray:
    """output_indices_ 里可能是 slice，也可能是数组，统一成整型数组。"""
    if isinstance(entry, slice):
        return np.arange(entry.start or 0, entry.stop or n_out, dtype=int)
    return np.array(entry, dtype=int).ravel()


num_idx = _idx(out_idx["num"])
cat_idx = _idx(out_idx["cat"])
assert len(num_idx) + len(cat_idx) == len(coef), \
    f"输出列数 {len(num_idx)+len(cat_idx)} != 系数长度 {len(coef)}"
num_coef_scaled = coef[num_idx]
# ⚠️ 教训：不要手推 SimpleImputer(median) → StandardScaler → Ridge 的代数合并。
#    两个统计量会互相穿插（scl.mean_ 是「填补后」数据的均值，不是中位数），
#    手推极易漏项；更糟的是错出来的仍是"看起来正常"的预测值——
#    不报错，只是全错。这里改为：
#      斜率 = coef / scale_（方向确定）
#      常数 = 用残差反解，再用全部样本验证（错了立刻暴露，见下方断言）
num_mean_ = np.asarray(scl.mean_)
num_scale_ = np.asarray(scl.scale_)
fill = np.asarray(imp.statistics_)
num_slope = num_coef_scaled / num_scale_
# OneHotEncoder(drop='first')：Transformer 输出列已经是「类别数 - 1」，
# 因此 cat_idx 指向的系数个数就等于输出列数，不要再手动去掉第一列。
ohe_cats = list(ohe.categories_[0])
ohe_coef = coef[cat_idx]
drop_idx = int(np.ravel(getattr(ohe, "drop_idx_", []) or [0])[0])
kept_cats = [c for k, c in enumerate(ohe_cats) if k != drop_idx]
kept_pos = {str(c): j for j, c in enumerate(kept_cats)}
assert len(ohe_coef) == len(kept_cats), \
    f"类别系数 {len(ohe_coef)} != 保留类别数 {len(kept_cats)}"

# --- 常数项定标：用全部样本的残差反解 ---
_df0 = rm.load_clean()
_X0 = rm.prepare_matrix(_df0, bundle["series_map"], bundle["series_vocab"])
_y0 = bundle["ridge"].predict(_X0)
_resid = []
for _i in range(len(_X0)):
    _v = 0.0
    for _k, _name in enumerate(rm.NUM_FEATURES):
        _x = float(_X0[_name].iloc[_i])
        if not np.isfinite(_x):
            _x = fill[_k]
        _v += num_slope[_k] * _x
    _s = str(_X0["series_cat"].iloc[_i])
    if _s in kept_pos:
        _v += ohe_coef[kept_pos[_s]]
    _resid.append(float(_y0[_i]) - _v)
_resid = np.asarray(_resid)
num_const = float(_resid.mean())
assert np.abs(_resid - num_const).max() < 1e-9, \
    (f"Ridge 常数项标定失败：残差波动 {np.abs(_resid - num_const).max():.2e}，"
     f"说明斜率或类别项有误")
print(f"  Ridge 常数项由 {len(_X0)} 行残差标定：{num_const:.6f}"
      f"（残差最大波动 {np.abs(_resid - num_const).max():.2e}）")
ridge_export = {
    "num_features": rm.NUM_FEATURES,
    "num_slope": [float(x) for x in num_slope],
    "num_const": float(num_const),
    "intercept": intercept,
    "missing_fill": [float(x) for x in fill],
    "ohe_categories": [str(c) for c in ohe_cats],
    "ohe_drop_idx": drop_idx,
    "ohe_coef": [float(x) for x in ohe_coef],
}
print(f"  数值特征 {len(rm.NUM_FEATURES)} 个，类别档 {len(ohe_cats)} 个"
      f"（drop='first'，被丢弃的类别={ohe_cats[ridge_export['ohe_drop_idx']]}，"
      f"其系数视为 0）")

# ==================================================================
# 3. 自检：导出 JSON 复刻推理 vs 原生 predict，必须逐点一致
# ==================================================================
print("\n【自检】用导出的 JSON 在 Python 里重跑推理，与 LightGBM 对齐：")
rng = np.random.default_rng(rm.SEED)
test = df.sample(min(120, len(df)), random_state=rm.SEED)
X = rm.prepare_matrix(test, bundle["series_map"], bundle["series_vocab"])
# 转成 JS 端会拿到的原始数值向量（类别转成编号）
cat_codes = X["series_cat"].cat.codes.values
cat_pos = model_point["features"].index("series_cat")
print(f"类别特征在输入向量中的下标：{cat_pos}"
      f"（features = {model_point['features']}）")
assert cat_pos == len(rm.NUM_FEATURES), \
    f"类别特征下标 {cat_pos} 与数值特征数 {len(rm.NUM_FEATURES)} 不一致"
rows_js = []
for i in range(len(X)):
    r = [float(X[c].iloc[i]) for c in rm.NUM_FEATURES]
    vec = r[:cat_pos] + [int(cat_codes[i])] + r[cat_pos:]
    assert len(vec) == model_point["n_features"], "输入向量长度与模型不符"
    rows_js.append(vec)

exact = bundle["point"].predict(X)
got = np.array([js_eval(model_point, v) for v in rows_js])
d = np.abs(exact - got)
print(f"  点估计      最大差 {d.max():.2e}　平均 {d.mean():.2e}")
assert d.max() < 1e-9, "点估计导出不可复现"

for a in rm.PI_Q:
    exact_q = bundle["quantile"][a].predict(X)
    mm = model_q[str(a)]
    got_q = np.array([js_eval(mm, v) for v in rows_js])
    dq = np.abs(exact_q - got_q)
    print(f"  分位数 α={a}  最大差 {dq.max():.2e}　平均 {dq.mean():.2e}")
    assert dq.max() < 1e-9, f"分位数 α={a} 导出不可复现"

# Ridge 复刻自检：页面 JS 用的就是下面这套「常数 + 点积 + 类别增量」
rr_exact = bundle["ridge"].predict(X)
drop_i = ridge_export["ohe_drop_idx"]
kept = [c for k, c in enumerate(ohe_cats) if k != drop_i]   # 实际保留的类别，按输出列序
kept_pos = {str(c): j for j, c in enumerate(kept)}
rr_got = []
for i in range(len(X)):
    v = ridge_export["num_const"]
    for k, name in enumerate(ridge_export["num_features"]):
        x = float(X[name].iloc[i])
        if not np.isfinite(x):
            x = ridge_export["missing_fill"][k]
        v += ridge_export["num_slope"][k] * x
    s = str(X["series_cat"].iloc[i])
    if s in kept_pos:
        v += ridge_export["ohe_coef"][kept_pos[s]]
    rr_got.append(v)
dr = np.abs(rr_exact - np.array(rr_got))
print(f"  Ridge       最大差 {dr.max():.2e}　平均 {dr.mean():.2e}")
assert dr.max() < 1e-9, "Ridge 系数导出不可复现"
assert len(ridge_export["ohe_coef"]) == len(kept), \
    f"类别系数 {len(ridge_export['ohe_coef'])} != 保留类别 {len(kept)}"

# ==================================================================
# 4. 分解基准曲线（让页面能就地展示"预测过程"而不是只给一个数）
# ==================================================================
# 页面要展示「基准 + 各因素加减 = 预测值」。做法是单因素反事实：
# 固定其他特征为基准值，只看某一个特征带来的差异。
# 这里把年龄方向的一元曲线按（车系 × 指导价档 × 外观档）预计算出来，
# 比在 JS 里重实现分解逻辑更可靠，也比导出 PDE 立方体省体积。
CURVE_AGES = list(range(0, 97, 2))
CURVE_PRICES = [15, 20, 25, 30, 35, 40, 50, 60]
CURVE_APPR = [80, 85, 90, 95]
ref_appr = bundle["reference"]["appearance_score"]
ref_mil = bundle["medians"]["mileage_wan"]


def age_curve(series: str, price: float, appr: float) -> list[float]:
    n = len(CURVE_AGES)
    rows = pd.DataFrame({
        "series": [series] * n, "age_months": CURVE_AGES,
        "mileage_wan": [ref_mil] * n, "price_wan": [price] * n,
        "appearance_score": [appr] * n,
        "claim_times": [0.0] * n, "transfer_times": [0.0] * n})
    return [round(float(v), 5) for v in bundle["point"].predict(
        rm.prepare_matrix(rows, bundle["series_map"], bundle["series_vocab"]))]


# (a) 折旧曲线：每个车系档，在「基准指导价档」上的一条年龄曲线
median_price_bucket = min(CURVE_PRICES,
                          key=lambda p: abs(p - bundle["medians"]["price_wan"]))
curves_by_series = {s: age_curve(s, median_price_bucket, ref_appr)
                    for s in bundle["series_vocab"]}
# (b) 分解基准：车龄方向在「该车实际指导价档 / 该车外观档」上的一元曲线
decomp_curves = {}
for s in bundle["series_vocab"]:
    for p in CURVE_PRICES:
        for a in CURVE_APPR:
            decomp_curves[f"{s}|{p}|{a}"] = age_curve(s, p, a)
print(f"\n分解基准曲线：{len(bundle['series_vocab'])} 车系 × {len(CURVE_PRICES)} 价格档 "
      f"× {len(CURVE_APPR)} 外观档 = {len(decomp_curves)} 条")

# ==================================================================
# 5. 组装资产
# ==================================================================
# 每个车系档的基准曲线（用于折旧曲线图，取模型精确值）
CURVE_AGES = list(range(0, 97, 3))
curves = {}
for s in bundle["series_vocab"]:
    rows = pd.DataFrame({
        "series": [s] * len(CURVE_AGES), "age_months": CURVE_AGES,
        "mileage_wan": [bundle["medians"]["mileage_wan"]] * len(CURVE_AGES),
        "price_wan": [bundle["medians"]["price_wan"]] * len(CURVE_AGES),
        "appearance_score": [bundle["reference"]["appearance_score"]] * len(CURVE_AGES),
        "claim_times": [0.0] * len(CURVE_AGES), "transfer_times": [0.0] * len(CURVE_AGES)})
    curves[s] = [round(float(v), 4) for v in
                 bundle["point"].predict(rm.prepare_matrix(
                     rows, bundle["series_map"], bundle["series_vocab"]))]

# 每个建模档的中位输入：供页面「选中车系自动带出合理默认值」使用。
# 没有它，用户面对的是滑杆中点（指导价 49 万），一开页面就是离谱的输入。
series_medians = {}
for s in bundle["series_vocab"]:
    sub = df[[bundle["series_map"].get(str(x), rm.UNKNOWN_SERIES) == s
              for x in df["series"]]]
    if len(sub) == 0:
        series_medians[s] = {"age_months": float(bundle["medians"]["age_months"]),
                             "mileage_wan": float(bundle["medians"]["mileage_wan"]),
                             "price_wan": float(bundle["medians"]["price_wan"]),
                             "n": 0}
    else:
        series_medians[s] = {
            "age_months": round(float(sub["age_months"].median()), 1),
            "mileage_wan": round(float(sub["mileage_wan"].median()), 2),
            "price_wan": round(float(sub["price_wan"].median()), 2),
            "n": int(len(sub)),
        }
print(f"\n已计算 {len(series_medians)} 个建模档的中位输入（供页面默认值使用）")

assets = {
    "meta": {
        "target_definition": bundle["target_definition"],
        "snapshot": bundle["snapshot"],
        "n_train": bundle["n_train"],
        "model": "LightGBM（8 维精简特征 + 原生类别车系）",
        "oof_metrics": bundle["oof_metrics"],
        "interval_note": ("区间来自分位数头（α=0.05/0.95），是**单台车**的 90% 预测区间，"
                          "不是均值的置信区间"),
        "engine_note": ("页面内置导出的 LightGBM 树结构并用 JS 直接遍历，"
                        "与 Python 端预测逐点一致（最大差 <1e-9）；"
                        "分位数头同样导出，因此预测区间也是精确值而非插值估算"),
        "decay_controlled_pp": bundle["empirical"]["slope_controlled_pp_per_year"],
        "decay_simple_pp": bundle["empirical"]["slope_simple_pp_per_year"],
        "scale": SCALE,
    },
    # ⚠️ 这里必须用**模型自己的特征名表**（8 项，含 series_cat），
    #    不能图省事写 rm.NUM_FEATURES（只有 7 个数值特征）。
    #    页面用这个列表建「特征名 → 下标」映射；少了 series_cat 的话
    #    CAT_IDX 会变成 undefined，车系编码静默丢弃、所有车系退化成未知档。
    "n_features": model_point["n_features"],
    "features": model_point["features"],
    "num_features_only": rm.NUM_FEATURES,
    "cat_feature": "series_cat",
    "cat_index": model_point["features"].index("series_cat"),
    "series_vocab": bundle["series_vocab"],
    "series_medians": series_medians,
    "series_info": bundle["series_info"],
    "series_map": bundle["series_map"],
    "reference": bundle["reference"],
    "medians": bundle["medians"],
    "filter_price": float(np.percentile(df["price_wan"], 75)),
    "curve_ages": CURVE_AGES,
    "curve_prices": CURVE_PRICES,
    "curve_appr": CURVE_APPR,
    "curves": curves_by_series,
    "decomp_curves": decomp_curves,
    "median_price_bucket": median_price_bucket,
    "model_point": model_point,
    "model_quantile": model_q,
    "ridge": ridge_export,
    "empirical": bundle["empirical"],
    "per_series_error": bundle["per_series_error"],
    "decay_bands": [
        {"band": str(k), "median": round(float(v), 4),
         "n": int(bundle["empirical"]["seg_n"].get(str(k), 0))}
        for k, v in bundle["empirical"]["seg_median"].items()],
    "ablation": bundle["ablation"],
    "model_comparison": bundle["oof_metrics"]["group_kfold"],
}
with open(ASSET_PATH, "w", encoding="utf-8") as f:
    json.dump(assets, f, ensure_ascii=False, separators=(",", ":"))
kb = os.path.getsize(ASSET_PATH) / 1024
print(f"\n已写出资产：{ASSET_PATH}（{kb:.0f} KB）")
print("  下一步：python src/19_build_dashboard.py")
