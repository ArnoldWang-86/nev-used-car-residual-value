# -*- coding: utf-8 -*-
r"""residual_model.py —— 残值预测模型（训练 + 推理）共享模块

单一事实来源。`16_train_final.py`（训练）与 `predict.py`（推理）共用本模块，
保证特征工程、编码口径、时点口径在训练与推理两侧完全一致——
这是"评估指标能复现"的前提，也是审计最容易抓的漏洞。

口径定义（必须与结论报告第五章一致）：
    残值率 residual_rate = 挂牌价 full_payment / 新车指导价 guide_price
    预测挂牌价 predicted_price = 预测残值率 × 新车指导价

目标变量边界：本数据残值率为「挂牌价/指导价」，非成交价口径；
挂牌价系统性高于成交价，因此预测值应理解为「合理挂牌区间」而非成交价。

用法：
    from residual_model import load_bundle, predict_one, InputCase

    bundle = load_bundle()
    res = predict_one(bundle, InputCase(series="蔚来ES6", age_months=42,
                                        mileage_wan=6.8, price_wan=33.8))
    print(res["residual_rate"], res["price_wan"], res["pi90"])
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ------------------------------------------------------------------
# 常量与路径
# ------------------------------------------------------------------
TARGET = "residual_rate"
SEED = 42

MODEL_DIR = os.path.join(ROOT, "output", "models")
BUNDLE_PATH = os.path.join(MODEL_DIR, "residual_model.joblib")
BUNDLE_META_PATH = os.path.join(MODEL_DIR, "residual_model_meta.json")
MODEL_CARD_PATH = os.path.join(MODEL_DIR, "模型卡.md")
CLEAN_CSV = os.path.join(ROOT, "data", "clean", "cars_analysis.csv")

# 精简特征集（对比 src/15_ml_final.py 的 90 维哑变量版本）
NUM_FEATURES = [
    "age_months",       # 车龄（月）── 第一驱动因子
    "log_mileage_wan",  # 里程（对数）
    "mileage_per_year", # 年均里程（万km/年）── 使用强度
    "log_price_wan",    # 新车指导价（对数）── 分母效应
    "appearance_score", # 外观评分
    "claim_times",      # 理赔次数
    "transfer_times",   # 过户次数
]
CAT_FEATURES = ["series_cat"]
FEATURES = NUM_FEATURES + CAT_FEATURES
UNKNOWN_SERIES = "其他/未知车系"

# 车系阈值：样本量 >= 该值才把车系当作独立类别，否则并入品牌档
SERIES_MIN_N = 10
# 品牌阈值：样本量 >= 该值才保留品牌档，否则归入全局未知档
BRAND_MIN_N = 20

# 展示用分位数（90% 预测区间）
PI_Q = (0.05, 0.95)


def _r(x, nd: int = 4):
    """安全四舍五入，None/NaN 原样返回。"""
    if x is None:
        return None
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(xf):
        return None
    return round(xf, nd)


# ------------------------------------------------------------------
# 数据加载与特征工程
# ------------------------------------------------------------------
def load_clean(path: str | None = None) -> pd.DataFrame:
    """读入清洗后的 726 条新能源车源详情表。"""
    path = path or CLEAN_CSV
    df = pd.read_csv(path, encoding="utf-8-sig")
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """把原始车源表加工成模型输入矩阵 X（含 series_cat 列）。

    ⚠️ 本函数只做**逐行、确定性**的机械加工：
       里程→万公里、指导价→万元、log 变换、年均里程、缺失兜底。
       它**不做任何依赖全表样本量的判断**，因此预测单辆车时可以逐行调用。

    ⚠️ 车系分档（样本量不足则并入品牌档）**不在这里做**，见 build_bucket_map()。
       曾经的写法把「样本量 <10 就并档」的判断放进本函数，
       导致推理时单一行的样本量恒为 1 → 任何车系都被并成「未知档」，
       训练时说好的「ES6 独立建模」在推理侧永远不生效（静默失效）。
    """
    d = df.copy()

    # 里程统一换算成「万公里」
    if "mileage_wan" not in d.columns:
        if "mileage" in d.columns:
            d["mileage_wan"] = pd.to_numeric(d["mileage"], errors="coerce") / 10000.0
        else:
            d["mileage_wan"] = np.nan

    # 指导价统一换算成「万元」
    if "price_wan" not in d.columns:
        if "guide_price" in d.columns:
            d["price_wan"] = pd.to_numeric(d["guide_price"], errors="coerce") / 10000.0
        else:
            d["price_wan"] = np.nan

    # 数值化（保证推理时传入字符串也不炸）
    for c in ["age_months", "mileage_wan", "price_wan", "appearance_score",
              "claim_times", "transfer_times"]:
        d[c] = pd.to_numeric(d.get(c), errors="coerce")

    # 对数变换：里程与指导价都是右偏重尾，log 后线性模型的斜率才是"每万公里"的稳定含义
    d["log_mileage_wan"] = np.log(d["mileage_wan"].clip(lower=0) + 1.0)
    d["log_price_wan"] = np.log(d["price_wan"].clip(lower=0) + 1.0)

    # 年均里程：用当月车龄折算，不足 6 个月按 6 个月算，避免新车出现天文数字
    age_months_safe = d["age_months"].clip(lower=6)
    d["mileage_per_year"] = d["mileage_wan"] / (age_months_safe / 12.0)

    # 缺失统一填 0 交回给 LightGBM（树模型对 0 与 NaN 的处理不同，显式统一口径）
    for c in ["claim_times", "transfer_times"]:
        d[c] = d[c].fillna(0.0)
    for c in ["appearance_score"]:
        d[c] = d[c].fillna(d[c].median() if d[c].notna().any() else 90.0)

    # 只做中性占位，分档规则由 build_bucket_map() 在训练集上一次性确定
    series = d["series"] if "series" in d.columns else pd.Series(
        [UNKNOWN_SERIES] * len(d), index=d.index)
    d["series_cat_raw"] = series.astype("object").where(
        series.notna(), UNKNOWN_SERIES).astype(str)

    return d[[c for c in FEATURES if c != "series_cat"] + ["series_cat_raw"]].copy()


def build_bucket_map(df_raw: pd.DataFrame) -> dict:
    """在**全部训练样本**上一次性确定「车系 → 建模档」的映射。

    规则：样本量 >= SERIES_MIN_N 的车系独立成档；
          其余并入「品牌·其他车系」档（若连品牌都没有则进全局未知档）。
    这张表随模型一起持久化，推理时必须用它，不能在单行上重新统计。
    """
    s = df_raw["series"] if "series" in df_raw.columns else None
    if s is None:
        return {}
    s = s.astype("object").where(s.notna(), UNKNOWN_SERIES).astype(str)
    vc = s.value_counts()
    major = set(vc[vc >= SERIES_MIN_N].index)
    mapping = {}
    for idx, name in s.items():
        if name in major:
            mapping[name] = name
        else:
            brand = df_raw["brand"].get(idx) if "brand" in df_raw.columns else None
            if isinstance(brand, str) and brand.strip():
                mapping[name] = f"{brand.strip()}·其他车系"
            else:
                mapping[name] = UNKNOWN_SERIES
    return mapping


def apply_bucket(df_feat_or_raw: pd.DataFrame, series_map: dict,
                 series_vocab: list[str]) -> pd.DataFrame:
    """把 build_features 的输出套上训练时确定的档位映射与固定词表。"""
    d = df_feat_or_raw.copy()
    if "series_cat_raw" in d.columns:
        raw = d["series_cat_raw"].astype(str)
    else:
        raw = d["series"].astype(str) if "series" in d.columns else pd.Series(
            [UNKNOWN_SERIES] * len(d), index=d.index)
    mapped = [series_map.get(x, UNKNOWN_SERIES) for x in raw]
    vocab = list(series_vocab)
    d["series_cat"] = pd.Categorical(
        [x if x in vocab else UNKNOWN_SERIES for x in mapped],
        categories=vocab)
    return d.drop(columns=["series_cat_raw"], errors="ignore")


def prepare_matrix(df: pd.DataFrame, series_map: dict, series_vocab: list[str]):
    """训练/推理共用入口：原始表 → 模型输入矩阵。"""
    return apply_bucket(build_features(df), series_map, series_vocab)


# ------------------------------------------------------------------
# 训练
# ------------------------------------------------------------------
def make_lgbm(**overrides):
    """主模型：LightGBM，原生类别特征 + 分位数头用于预测区间。"""
    import lightgbm as lgb
    params = dict(
        n_estimators=400,
        learning_rate=0.04,
        num_leaves=15,
        max_depth=4,
        min_child_samples=12,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=SEED,
        verbose=-1,
    )
    params.update(overrides)
    return lgb.LGBMRegressor(**params)


def make_quantile(alpha: float, **overrides):
    import lightgbm as lgb
    params = dict(
        objective="quantile", alpha=alpha,
        n_estimators=300, learning_rate=0.05, num_leaves=15, max_depth=4,
        min_child_samples=20, reg_lambda=1.0,
        random_state=SEED, verbose=-1,
    )
    params.update(overrides)
    return lgb.LGBMRegressor(**params)


def make_ridge():
    """线性对照模型：用于「1 年后残值」外推——树模型不能外推车龄趋势。"""
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer

    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), NUM_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"),
         CAT_FEATURES),
    ])
    return Pipeline([("pre", pre), ("r", Ridge(alpha=1.0))])


def fit_full(df: pd.DataFrame, series_map: dict, series_vocab: list[str]):
    """在全部样本上拟合主模型 / 分位数头 / 线性对照模型。"""
    X = prepare_matrix(df, series_map, series_vocab)
    y = pd.to_numeric(df[TARGET], errors="coerce").values

    ok = np.isfinite(y) & X[NUM_FEATURES].notna().all(axis=1).values
    X, y = X[ok], y[ok]

    point = make_lgbm()
    point.fit(X, y, categorical_feature=CAT_FEATURES)
    bands = {}
    for a in PI_Q:
        m = make_quantile(a)
        m.fit(X, y, categorical_feature=CAT_FEATURES)
        bands[a] = m
    ridge = make_ridge()
    ridge.fit(X, y)
    return {"point": point, "quantile": bands, "ridge": ridge, "n_fit": int(len(y))}


def group_kfold_oof(df: pd.DataFrame, series_map: dict, series_vocab: list[str],
                    n_splits: int = 5, kind: str = "lgbm", groups=None):
    """按车系整组切分的 OOF 预测（无泄漏）。

    返回 (oof_pred, resid)。group 口径 = 车系 → 回答「预测一个没见过的车系」。

    ⚠️ groups 必须与模型实际使用的类别列**同一个口径**。
    如果模型用的是并档后的 series_cat（24 档），分组却按原始 56 个车系切，
    被并档的稀有小车系会同时出现在训练与测试折里 —— 这是泄漏，指标会虚高。
    """
    from sklearn.model_selection import GroupKFold
    X = prepare_matrix(df, series_map, series_vocab)
    y = pd.to_numeric(df[TARGET], errors="coerce").values
    if groups is None:
        groups = df["series"].astype(str).values
    groups = np.asarray(groups)

    oof = np.full(len(y), np.nan)
    gkf = GroupKFold(n_splits=n_splits)
    for tr, te in gkf.split(X, y, groups):
        if kind == "lgbm":
            m = make_lgbm()
            m.fit(X.iloc[tr], y[tr], categorical_feature=CAT_FEATURES)
        else:
            m = make_ridge()
            m.fit(X.iloc[tr], y[tr])
        oof[te] = m.predict(X.iloc[te])
    resid = y - oof
    return oof, resid


def random_kfold_oof(df: pd.DataFrame, series_map: dict, series_vocab: list[str],
                     n_splits: int = 5):
    """随机切分 OOF —— 回答「给熟悉的车系估价」，会高估能力，仅作对照。"""
    from sklearn.model_selection import KFold
    X = prepare_matrix(df, series_map, series_vocab)
    y = pd.to_numeric(df[TARGET], errors="coerce").values
    oof = np.full(len(y), np.nan)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    for tr, te in kf.split(X):
        m = make_lgbm()
        m.fit(X.iloc[tr], y[tr], categorical_feature=CAT_FEATURES)
        oof[te] = m.predict(X.iloc[te])
    return oof, y - oof


# ------------------------------------------------------------------
# 推理
# ------------------------------------------------------------------
class InputCase(dict):
    """一辆待估价的车。字段缺失时按训练集中位数兜底，并在 warnings 里说明。"""

    _FIELDS = ("series", "age_months", "mileage_wan", "price_wan",
               "appearance_score", "claim_times", "transfer_times")

    def __init__(self, series: str | None = None, age_months: float | None = None,
                 mileage_wan: float | None = None, price_wan: float | None = None,
                 appearance_score: float | None = None,
                 claim_times: float | None = None,
                 transfer_times: float | None = None):
        super().__init__(
            series=series, age_months=age_months, mileage_wan=mileage_wan,
            price_wan=price_wan, appearance_score=appearance_score,
            claim_times=claim_times, transfer_times=transfer_times,
        )

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([{k: self.get(k) for k in self._FIELDS}])


def load_bundle(path: str | None = None) -> dict:
    import joblib
    path = path or BUNDLE_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"未找到模型文件 {path}\n请先运行： python src/16_train_final.py")
    b = joblib.load(path)
    b["_path"] = path
    return b


def _prepare_row(bundle: dict, case) -> tuple[pd.DataFrame, list[str]]:
    """单行推理：加工特征 + 未知车系/缺失值兜底，返回 (X, warnings)。"""
    warns: list[str] = []
    med = bundle["medians"]

    if isinstance(case, dict) and not isinstance(case, InputCase):
        case = InputCase(**{k: case.get(k) for k in InputCase._FIELDS})

    row = case.to_frame()
    for c in ["age_months", "mileage_wan", "price_wan", "appearance_score"]:
        if row[c].isna().all():
            row[c] = med[c]
            warns.append(f"未提供「{c}」，已按训练集中位数 {med[c]:.3g} 兜底")

    X = prepare_matrix(row, bundle["series_map"], bundle["series_vocab"])
    got = str(X["series_cat"].iloc[0])
    raw = case.get("series")
    smap = bundle["series_map"]
    if got == UNKNOWN_SERIES:
        if raw is None:
            warns.append("未提供车系，已按「其他/未知车系」档预测（精度最低）")
        else:
            warns.append(
                f"车系「{raw}」不在训练样本中，已退化为「{UNKNOWN_SERIES}」档。"
                f"该档仍会用车龄/里程/指导价给出预测，但**不含该车系的特异信息**，"
                f"等价于「同期同价位同级车的平均表现」，"
                f"不能用来判断该车系本身保值与否")
    elif raw is not None and str(raw) not in smap:
        warns.append(
            f"车系「{raw}」未在训练样本中出现（属新品/新势力），"
            f"已按品牌档「{got}」预测；"
            f"该档聚合了同品牌其他样本不足的车系，**不能代表该车系本身**")
    elif raw and got != str(raw):
        warns.append(
            f"车系「{raw}」训练样本不足 {SERIES_MIN_N} 条，已并入「{got}」档"
            f"（该档 {bundle['series_info'].get(got, {}).get('n', 0)} 条）；"
            f"预测反映的是该品牌同价位车型的平均水平")

    if case.get("age_months") is not None:
        try:
            a = float(case.get("age_months"))
            lo, hi = bundle["train_range"]["age_months"]
            if a < lo or a > hi:
                warns.append(f"车龄 {a:.0f} 月超出训练范围 [{lo:.0f}, {hi:.0f}] 月，"
                             f"树模型在外推区间会趋于平坦，结论不可靠")
        except (TypeError, ValueError):
            pass
    return X, warns


def predict_one(bundle: dict, case, explain: bool = False) -> dict:
    """对一辆车出预测，含 90% 预测区间与（可选）预测过程分解。

    返回 dict 字段：
        residual_rate / price_wan           点预测
        pi90 / pi90_price                   90% 预测区间（分位数头，非对称）
        band_type                           区间类型说明
        decomposition                       explain=True 时的特征贡献分解
        category                           该车落在哪个车系档 / 训练样本数
    """
    X, warns = _prepare_row(bundle, case)
    rr = float(bundle["point"].predict(X)[0])
    rr = float(np.clip(rr, 0.02, 1.05))

    q_lo = float(bundle["quantile"][PI_Q[0]].predict(X)[0])
    q_hi = float(bundle["quantile"][PI_Q[1]].predict(X)[0])
    # 分位数头在样本稀疏处可能交叉，做单调修正
    q_lo, q_hi = min(q_lo, rr), max(q_hi, rr)

    price_wan = case.get("price_wan")
    if price_wan is None or not np.isfinite(float(price_wan)):
        # 用训练集中位数作为显示基准，但明确标注这不是该车的真实指导价
        price_wan = bundle["medians"]["price_wan"]
        price_known = False
    else:
        price_wan = float(price_wan)
        price_known = True

    ser = str(X["series_cat"].iloc[0])
    info = bundle["series_info"].get(ser, {})

    # 区间口径：**直接用分位数头的输出**，不做任何按车系的事后加宽。
    # 早期版本对样本量小的车系乘了一个 sqrt(1+10/n) 的"加宽系数"，
    # 但那个系数是拍的、没有校准依据，而且会让 Python 与页面 JS（用分位数头原值）
    # 给出不同的区间——同一份模型两套口径，属于不可复现的隐患，故删除。
    # 分位数头本身已在 GroupKFold OOF 上标定（覆盖率见模型卡）。
    band = [max(q_lo, 0.0), q_hi]

    out = {
        "input": {k: case.get(k) for k in InputCase._FIELDS},
        "residual_rate": _r(rr, 4),
        "price_wan": _r(rr * price_wan, 2) if price_known else None,
        "guide_price_wan": _r(price_wan, 2) if price_known else None,
        "pi90": [_r(band[0], 4), _r(band[1], 4)],
        "pi90_price": None,
        # 未舍入原值：给一致性校验用。
        # 对外展示的字段是舍入过的（4 位小数），拿它去校验其他实现会引入
        # 约 5e-5 的假差异——本项目就在这儿误判过一次，务必用 raw 比对。
        "_raw": {"rate": rr, "pi90": [band[0], band[1]]},
        "band_type": "90% 预测区间（分位数头，含个体差异；非均值的置信区间）",
        "category": {"series_bucket": ser, "train_n": info.get("n", 0),
                     "status": info.get("status", "未知")},
        "method": "LightGBM（精简特征 + 原生类别车系）",
        "model_oof": bundle["oof_metrics"],
        "warnings": warns,
    }
    if price_known:
        out["pi90_price"] = [_r(out["pi90"][0] * price_wan, 2),
                             _r(out["pi90"][1] * price_wan, 2)]

    if explain:
        out["decomposition"] = decompose(bundle, case, X)
    return out


def decompose(bundle: dict, case, X: pd.DataFrame) -> dict:
    """预测过程分解：把这辆车的预测值拆成「基准 + 各因素加减」。

    做法是单因素反事实：把某个特征换成训练集基准值（车龄/里程/指导价取中位数，
    车况取众数），其余保持不变，看预测值移动多少。这比 SHAP 更好解释，
    而且不引入额外依赖；代价是不满足严格可加性，因此额外报一个 residual 项。
    """
    base_case = {k: case.get(k) for k in InputCase._FIELDS}
    ref = bundle["reference"]
    f = float(bundle["point"].predict(X)[0])

    items = []
    specs = [
        ("age_months", "车龄", ref["age_months"],
         lambda v: f"{v:.0f} 个月（{v/12:.1f} 年）"),
        ("mileage_wan", "里程", ref["mileage_wan"],
         lambda v: f"{v:.2f} 万公里"),
        ("price_wan", "新车指导价", ref["price_wan"],
         lambda v: f"{v:.2f} 万元"),
        ("appearance_score", "外观评分", ref["appearance_score"],
         lambda v: f"{v:.0f} 分"),
    ]
    for key, label, ref_v, fmt in specs:
        v = base_case.get(key)
        if v is None or not np.isfinite(float(v)):
            continue
        alt = dict(base_case)
        alt[key] = ref_v
        Xa, _ = _prepare_row(bundle, InputCase(**alt))
        fa = float(bundle["point"].predict(Xa)[0])
        items.append({
            "factor": label,
            "value": fmt(float(v)),
            "reference": fmt(float(ref_v)),
            "effect_pp": _r((f - fa) * 100, 2),
        })

    # 车系档贡献：把它换成全局未知档，衡量「这个车系标签值多少个点」
    if base_case.get("series") and str(X["series_cat"].iloc[0]) != UNKNOWN_SERIES:
        alt = dict(base_case); alt["series"] = UNKNOWN_SERIES
        Xa, _ = _prepare_row(bundle, InputCase(**alt))
        fa = float(bundle["point"].predict(Xa)[0])
        items.append({
            "factor": "车系", "value": str(X["series_cat"].iloc[0]),
            "reference": UNKNOWN_SERIES,
            "effect_pp": _r((f - fa) * 100, 2),
        })

    # 基准：全部取基准值时的预测
    Xr, _ = _prepare_row(bundle, InputCase(**{**base_case,
                                              "age_months": ref["age_months"],
                                              "mileage_wan": ref["mileage_wan"],
                                              "price_wan": ref["price_wan"],
                                              "appearance_score": ref["appearance_score"]}))
    f_ref = float(bundle["point"].predict(Xr)[0])
    explained = sum(i["effect_pp"] for i in items) / 100.0
    return {
        "baseline_rate": _r(f_ref, 4),
        "baseline_desc": (f"车龄 {ref['age_months']:.0f} 月 / 里程 "
                          f"{ref['mileage_wan']:.1f} 万 / 指导价 "
                          f"{ref['price_wan']:.1f} 万 / 外观 "
                          f"{ref['appearance_score']:.0f} 分的平均车"),
        "items": items,
        "final_rate": _r(f, 4),
        "residual_pp": _r((f - f_ref - explained) * 100, 2),
        "note": "单因素反事实分解；effect_pp 为该因素相对基准值的贡献（百分点）",
    }


def project_next_year(bundle: dict, case, mileage_add_wan: float = 1.0) -> dict:
    """「这辆车 1 年后残值会跌多少」——双轨对照。

    轨 1（树模型）：把车龄 +12 月、里程 + 年行驶里程，直接让 LightGBM 预测。
        风险：树模型是分段常数，在车龄区间边缘/之外不会继续下降。
    轨 2（线性斜率）：用 Ridge 学到的车龄系数做外推。
        依据：轨 A 时间外推实测——树模型 R² 转负，Ridge 反而 R² 0.338。
    两轨并列呈现，让读者自己看到差异；差异大 = 该区间树模型不可信的信号。
    """
    x = case.get("age_months")
    if x is None or not np.isfinite(float(x)):
        return {"ok": False, "reason": "需要提供车龄才能做 1 年外推"}
    x = float(x)
    m = case.get("mileage_wan")
    m = float(m) if (m is not None and np.isfinite(float(m))) else bundle["medians"]["mileage_wan"]

    now = predict_one(bundle, case)
    fut_case = InputCase(**{**{k: case.get(k) for k in InputCase._FIELDS},
                            "age_months": x + 12,
                            "mileage_wan": m + mileage_add_wan})
    fut = predict_one(bundle, fut_case)

    X_now, _ = _prepare_row(bundle, case)
    X_fut, _ = _prepare_row(bundle, fut_case)
    tree_now = float(bundle["point"].predict(X_now)[0])
    tree_fut = float(bundle["point"].predict(X_fut)[0])

    rid_now = float(bundle["ridge"].predict(X_now)[0])
    rid_fut = float(bundle["ridge"].predict(X_fut)[0])

    gp = case.get("price_wan")
    gp = float(gp) if (gp is not None and np.isfinite(float(gp))) else bundle["medians"]["price_wan"]

    max_age = bundle["train_range"]["age_months"][1]
    return {
        "ok": True,
        "assumptions": [
            f"未来一年行驶 {mileage_add_wan:.1f} 万公里（可调）",
            "二手车市场整体价格水平不变（本数据为单期快照，无时间趋势，无法建模市场涨跌）",
            "车辆无新增重大事故 / 无电池衰减突变",
            f"目标时点车龄 {x + 12:.0f} 月，训练集车龄上限 {max_age:.0f} 月"
            + ("（在范围内）" if x + 12 <= max_age else "（已超出，属真实外推）"),
        ],
        "tree": {
            "name": "LightGBM 树模型",
            "rate_now": _r(tree_now, 4), "rate_1y": _r(tree_fut, 4),
            "drop_pp": _r((tree_now - tree_fut) * 100, 2),
            "price_now_wan": _r(tree_now * gp, 2),
            "price_1y_wan": _r(tree_fut * gp, 2),
            "value_drop_wan": _r((tree_now - tree_fut) * gp, 2),
        },
        "linear": {
            "name": "Ridge 线性斜率（外推轨）",
            "rate_now": _r(rid_now, 4), "rate_1y": _r(rid_fut, 4),
            "drop_pp": _r((rid_now - rid_fut) * 100, 2),
            "price_now_wan": _r(rid_now * gp, 2),
            "price_1y_wan": _r(rid_fut * gp, 2),
            "value_drop_wan": _r((rid_now - rid_fut) * gp, 2),
        },
        "row_disagreement_pp": _r(abs((tree_now - tree_fut) - (rid_now - rid_fut)) * 100, 2),
        "empirical_reference": {
            "slope_controlled_pp_per_year":
                bundle["empirical"].get("slope_controlled_pp_per_year"),
            "slope_simple_pp_per_year":
                bundle["empirical"].get("slope_simple_pp_per_year"),
            "note": bundle["empirical"].get("note"),
        },
    }
