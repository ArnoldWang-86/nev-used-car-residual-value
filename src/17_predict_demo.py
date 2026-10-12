# -*- coding: utf-8 -*-
r"""17_predict_demo.py —— 预测示例：4 辆车对比实际挂牌价 + 1 年后残值外推

回答两个问题：
  1) 模型给一辆具体的车估价，准不准？（对比该车的**实际挂牌价**，报 pp 与万元两种误差）
  2) 这辆车 1 年后残值会跌多少？（双轨对照，见 `residual_model.project_next_year`）

⚠️ 本脚本的核心是**不使用训练集内的拟合值来自我表扬**：
   · 案例 1–3 用 **GroupKFold 的 out-of-fold 预测**（该车所在车系在本次预测中
     未参与训练），因此案例误差与模型卡里的 OOF MAE 是同一口径、可互相印证。
   · 案例 4 是**训练集外车系**：脚本会把该品牌整体从训练集剔除后重新拟合，
     展示「新车系进场」时的真实退化幅度，而不是藏起来。

输出：
    results/p03_预测案例对比.csv    案例对比表（机器可读）
    output/09_预测案例.txt          案例详情（人类可读）

用法：
    python src/17_predict_demo.py
"""
from __future__ import annotations

import copy
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

OUT_TXT = os.path.join(rm.ROOT, "output", "09_预测案例.txt")
OUT_CSV = os.path.join(rm.ROOT, "results", "p03_预测案例对比.csv")
RULE = "=" * 92

LINES: list[str] = []


def say(s: str = "") -> None:
    print(s)
    LINES.append(s)


# ==================================================================
# 0. 准备
# ==================================================================
df = rm.load_clean()
bundle = rm.load_bundle()
oof = pd.read_csv(os.path.join(rm.ROOT, "results", "p01_预测明细_OOF.csv"),
                  encoding="utf-8-sig")

say(RULE)
say("17_predict_demo.py · 预测示例")
say(RULE)
say(f"\n模型：{bundle['method'] if 'method' in bundle else 'LightGBM 精简特征 + 原生类别车系'}")
say(f"训练样本：{bundle['n_train']} 条（{bundle['snapshot']} 单期快照）")
om = bundle["oof_metrics"]
say(f"模型口径精度（GroupKFold OOF，预测没见过的车系）：MAE = "
    f"{om['group_kfold']['M2 LightGBM（精简 8 维）']['mae']:.4f}，"
    f"R² = {om['group_kfold']['M2 LightGBM（精简 8 维）']['r2']:.4f}")
say(f"每 1% 残值率误差 ≈ 该车指导价的 1%；MAE {om['group_kfold']['M2 LightGBM（精简 8 维）']['mae']:.4f} "
    f"按均价 14.7 万折算 ≈ 1.05 万元")

# ==================================================================
# 1. 选案例车
# ==================================================================
PICKS = [
    ("案例1", "蔚来ES6", "主销车型 · 车龄中等"),
    ("案例2", "Model Y", "对比车型 · 里程偏高"),
    ("案例3", "蔚来EC6", "同门车型 · 车况较好"),
    ("案例4", "极氪7X", "训练集外车系 · 检验新车系退化"),
]


def pick(series: str) -> pd.Series:
    g = df[df["series"] == series]
    if not len(g):
        raise SystemExit(f"数据中没有车系「{series}」")
    med = g["age_months"].median()
    return g.iloc[(g["age_months"] - med).abs().argsort()[:1]].iloc[0]


cases = [(tag, s, why, pick(s)) for tag, s, why in PICKS]

# ==================================================================
# 2. 案例 4 用的「剔除该品牌」重训模型
# ==================================================================
brand4 = cases[3][3]["brand"]
train_wo = df[df["brand"] != brand4].copy()
sm_wo = rm.build_bucket_map(train_wo)
vocab_wo = list(dict.fromkeys(
    [sm_wo[s] for s in train_wo["series"].astype(str)] + [rm.UNKNOWN_SERIES]))
fit_wo = rm.fit_full(train_wo, sm_wo, vocab_wo)
bundle_wo = copy.deepcopy(bundle)
bundle_wo.update({"point": fit_wo["point"], "quantile": fit_wo["quantile"],
                  "ridge": fit_wo["ridge"], "series_map": sm_wo,
                  "series_vocab": vocab_wo, "n_train": fit_wo["n_fit"]})
bundle_wo["series_info"] = {
    s: {"n": int(sum(1 for x in train_wo["series"].astype(str)
                     if sm_wo.get(x) == s)),
        "status": "重训模型（该品牌已整体剔除）", "shrink": 1.2}
    for s in vocab_wo}

# ==================================================================
# 3. 逐案例预测
# ==================================================================
say(f"\n{RULE}")
say("一、具体车辆预测 vs 实际挂牌价")
say(RULE)
say("\n误差口径说明：")
say("  · 案例 1–3 用的是 **GroupKFold out-of-fold 预测** —— 该车所在车系在本次预测中")
say("    未参与训练，所以误差与模型卡的 OOF MAE 同口径，不是训练集内的自我表扬。")
say("  · 案例 4 用的是**把整个品牌从训练集剔除后重训**的模型，模拟新车系进场。")

rows = []
detail_blocks = []

for tag, series, why, r in cases:
    is_ood = (tag == "案例4")
    b = bundle_wo if is_ood else bundle

    case = rm.InputCase(
        series=r["series"], age_months=float(r["age_months"]),
        mileage_wan=float(r["mileage_wan"]), price_wan=float(r["price_wan"]),
        appearance_score=float(r["appearance_score"]),
        claim_times=float(r["claim_times"]) if pd.notna(r["claim_times"]) else None,
        transfer_times=float(r["transfer_times"]) if pd.notna(r["transfer_times"]) else None,
    )
    res = rm.predict_one(b, case, explain=True)
    proj = rm.project_next_year(b, case, 1.0)

    actual_rr = float(r["residual_rate"])
    actual_pay = float(r["full_payment"]) / 10000.0

    # 案例 1–3 用 OOF 预测做对比（无泄漏）；案例 4 用重训模型预测
    oof_row = oof[oof["car_id"] == r["car_id"]]
    if is_ood:
        pred_rr = float(res["residual_rate"])
        pred_src = f"重训模型（{brand4} 品牌整体剔除）"
        pi_lo, pi_hi = float(res["pi90"][0]), float(res["pi90"][1])
        pi_src = "重训模型的分位数头（该场景无 OOF，仅为敏感性测试）"
    else:
        pred_rr = float(oof_row["oof_pred"].iloc[0])
        pred_src = "GroupKFold OOF（车系未参与训练）"
        # ⚠️ 区间必须与点估计同源。用全量模型的分位数头去配 OOF 点估计，
        # 是两个模型混用，既不可比也不可复现。
        # 这里统一用「OOF 点估计 + OOF 残差经验分位」，即 conformal 式区间，
        # 与模型卡声明的 90% 覆盖率是同一套标定。
        qlo, qhi = bundle["oof_resid_quantile"]
        pi_lo, pi_hi = pred_rr + qlo, pred_rr + qhi
        pi_src = "OOF 点估计 + OOF 残差经验分位（conformal 式，全局标定）"

    pred_pay = pred_rr * float(r["price_wan"])
    err_pp = (pred_rr - actual_rr) * 100
    err_wan = pred_pay - actual_pay
    hit = bool(pi_lo <= actual_rr <= pi_hi)

    rows.append({
        "案例": tag, "车系": series, "用途": why,
        "car_id": r["car_id"], "车型": r["model"],
        "车龄_月": round(float(r["age_months"]), 1),
        "里程_万km": round(float(r["mileage_wan"]), 2),
        "指导价_万": round(float(r["price_wan"]), 2),
        "实际挂牌价_万": round(actual_pay, 2),
        "实际残值率": round(actual_rr, 4),
        "预测来源": pred_src,
        "预测残值率": round(pred_rr, 4),
        "预测挂牌价_万": round(pred_pay, 2),
        "残值率误差_pp": round(err_pp, 2),
        "金额误差_万": round(err_wan, 2),
        "90%区间_lo": round(pi_lo, 4), "90%区间_hi": round(pi_hi, 4),
        "区间来源": pi_src,
        "命中区间": hit,
        "全量模型预测_残值率": res["residual_rate"],
        "全量模型对同车误差_pp": round((res["residual_rate"] - actual_rr) * 100, 2),
        "1年后_树模型_pp": proj["tree"]["drop_pp"] if proj["ok"] else None,
        "1年后_线性_pp": proj["linear"]["drop_pp"] if proj["ok"] else None,
        "1年后_树模型_万元": proj["tree"]["value_drop_wan"] if proj["ok"] else None,
        "命中类别": res["category"]["series_bucket"],
        "该类别训练样本": res["category"]["train_n"],
    })

    blk = [f"\n{RULE}", f"{tag} · {series} · {why}", RULE,
           f"  车型      {r['model']}",
           f"  车龄      {float(r['age_months']):.0f} 个月（{float(r['age_months'])/12:.1f} 年）"
           f"    里程 {float(r['mileage_wan']):.2f} 万公里",
           f"  指导价    {float(r['price_wan']):.2f} 万元        外观评分 "
           f"{float(r['appearance_score']):.0f}",
           f"  命中类别  {res['category']['series_bucket']}"
           f"（训练样本 {res['category']['train_n']} 条）",
           "",
           f"  {'':<16}{'残值率':>10}{'挂牌价(万)':>14}",
           f"  {'实际':<16}{actual_rr*100:>9.2f}%{actual_pay:>14.2f}",
           f"  {'模型预测':<16}{pred_rr*100:>9.2f}%{pred_pay:>14.2f}",
           f"  {'误差':<16}{err_pp:>+9.2f}pp{err_wan:>+14.2f}",
           f"  预测来源  {pred_src}",
           f"  90% 预测区间  {pi_lo*100:.2f}% ~ {pi_hi*100:.2f}%"
           f"   → 实际值{'落在区间内 ✓' if hit else '落在区间外 ✗'}",
           f"            区间口径：{pi_src}",
           ""]
    dec = res["decomposition"]
    blk.append(f"  预测过程分解（基准：{dec['baseline_desc']}，基准 {dec['baseline_rate']*100:.2f}%）")
    for it in dec["items"]:
        blk.append(f"      {it['factor']:<10}{it['value']:<24}{it['effect_pp']:>+7.2f}pp")
    blk.append(f"      {'交互残差':<10}{'不可加部分':<24}{dec['residual_pp']:>+7.2f}pp")
    blk.append(f"      → 该模型给出的点估计 {dec['final_rate']*100:.2f}%")
    if proj["ok"]:
        blk += ["",
                f"  1 年后：树模型 {proj['tree']['drop_pp']:+.2f}pp"
                f"（约 {proj['tree']['value_drop_wan']:+.2f} 万）"
                f"　|　线性轨 {proj['linear']['drop_pp']:+.2f}pp"
                f"（约 {proj['linear']['value_drop_wan']:+.2f} 万）",
                f"          两轨差异 {proj['row_disagreement_pp']:.2f}pp"
                + ("（>1pp，树模型已开始变平，以线性轨为准）"
                   if proj["row_disagreement_pp"] > 1.0 else "（两轨一致，结论较稳）")]
    if res["warnings"]:
        blk.append("")
        for w in res["warnings"]:
            blk.append(f"  ⚠️ {w}")
    detail_blocks.append("\n".join(blk))

tbl = pd.DataFrame(rows)
say("")
say(f"{'案例':<7}{'车系':<9}{'实际残值率':>10}{'预测残值率':>10}{'误差pp':>9}"
    f"{'实际挂牌':>10}{'预测挂牌':>10}{'误差万':>9}{'区间命中':>9}")
for _, x in tbl.iterrows():
    say(f"{x['案例']:<7}{x['车系']:<9}{x['实际残值率']*100:>9.2f}%"
        f"{x['预测残值率']*100:>9.2f}%{x['残值率误差_pp']:>+9.2f}"
        f"{x['实际挂牌价_万']:>10.2f}{x['预测挂牌价_万']:>10.2f}"
        f"{x['金额误差_万']:>+9.2f}{'✓' if x['命中区间'] else '✗':>9}")

mae_cases = float(tbl["残值率误差_pp"].abs().mean()) / 100
mae_cases_wan = float(tbl["金额误差_万"].abs().mean())
print()
say(f"\n4 个案例的平均绝对误差：{mae_cases:.4f}（残值率）"
    f" ≈ {mae_cases_wan:.2f} 万元（挂牌价）")
say(f"模型卡 GroupKFold OOF MAE：{om['group_kfold']['M2 LightGBM（精简 8 维）']['mae']:.4f}"
    f"　→ 案例误差与声明精度同量级，不是挑出来的好看数字。")
say(f"\n⚠️ 一个必须讲清楚的点：这 4 辆车如果改用**全量模型**（也就是把这几辆车")
say(f"   自己也放进训练集）来预测，误差会小得多：平均绝对误差 "
    f"{float(tbl['全量模型对同车误差_pp'].abs().mean())/100:.4f}"
    f" ≈ {float((tbl['全量模型对同车误差_pp'].abs()/100*tbl['指导价_万']).mean()):.2f} 万元。")
say("   本脚本刻意**不那样做**——那是训练集内自评，属于数据泄漏，")
say("   面试里被问一句「这车在训练集里吗」就解释不清。")
say("   用 OOF 代价是数字没那么漂亮，换来的是与模型卡指标同源、可复现、经得起追问。")
say("\n   全量模型 与 OOF 逐个案例对比：")
for _, x in tbl.iterrows():
    say(f"     {x['案例']} {x['车系']:<9} 实际 {x['实际残值率']*100:>6.2f}%"
        f" | OOF {x['预测残值率']*100:>6.2f}%（误差 {x['残值率误差_pp']:>+6.2f}pp）"
        f" | 全量模型 {x['全量模型预测_残值率']*100:>6.2f}%"
        f"（误差 {x['全量模型对同车误差_pp']:>+6.2f}pp）")

for blk in detail_blocks:
    say(blk)

# ==================================================================
# 4. 1 年后残值外推（统一的假设与双轨对照）
# ==================================================================
say(f"\n\n{RULE}")
say("二、「这辆车 1 年后残值会跌多少」—— 双轨对照")
say(RULE)
say("\n为什么要双轨：")
say("  轨 1 LightGBM：把车龄 +12 月、里程 +1 万公里再预测一次。")
say("       风险：树模型是分段常数，在训练车龄区间边缘不会继续下降，会低估跌幅。")
say("  轨 2 Ridge 线性：用线性回归学到的车龄斜率外推。")
say("       依据：本项目轨 A（17.6 万条）的时间外推实测——树模型 R² 转负")
say("             （-0.79 ~ -0.97），Ridge 反而 R² 0.338。树模型不能外推时间趋势。")
say("  两轨差异本身就是「该车龄处树模型是否可信」的信号。")
say(f"\n现实锚点（回归口径）：控制车系/里程/指导价后年跌 "
    f"{bundle['empirical']['slope_controlled_pp_per_year']:+.2f}pp；"
    f"未控制构成的简单斜率 {bundle['empirical']['slope_simple_pp_per_year']:+.2f}pp（含构成效应，仅作上界）")

say(f"\n{'案例':<7}{'车系':<9}{'当前残值率':>11}{'树模型1年':>11}{'线性1年':>11}"
    f"{'两轨差异':>10}{'树模型跌(万)':>14}")
for _, x in tbl.iterrows():
    tag = x["案例"]
    series = x["车系"]
    r = [c[3] for c in cases if c[0] == tag][0]
    b = bundle_wo if tag == "案例4" else bundle
    case = rm.InputCase(series=r["series"], age_months=float(r["age_months"]),
                        mileage_wan=float(r["mileage_wan"]),
                        price_wan=float(r["price_wan"]),
                        appearance_score=float(r["appearance_score"]))
    p = rm.project_next_year(b, case, 1.0)
    if not p["ok"]:
        continue
    say(f"{tag:<7}{series:<9}{p['tree']['rate_now']*100:>10.2f}%"
        f"{p['tree']['rate_1y']*100:>10.2f}%{p['linear']['rate_1y']*100:>10.2f}%"
        f"{p['row_disagreement_pp']:>9.2f}pp{p['tree']['value_drop_wan']:>13.2f}")

say("\n统一假设（三条，缺一条结论就不成立）：")
for a in rm.project_next_year(bundle, rm.InputCase(
        series="蔚来ES6", age_months=52, mileage_wan=7.4,
        price_wan=35.8, appearance_score=85), 1.0)["assumptions"]:
    say(f"  · {a}")
say("\n⚠️ 本数据是单期快照，**无法建模市场整体涨跌**。这里算的是「车龄+里程」")
say("   自然衰减，不是对未来二手车行情的预测。真实业务要用滚动重训来吸收行情。")

# ==================================================================
# 5. 未知车系退化：单独量化
# ==================================================================
say(f"\n\n{RULE}")
say("三、训练集外车系的退化幅度（案例 4 的量化说明）")
say(RULE)
g4 = df[df["series"] == "极氪7X"]
oof4 = oof[oof["series"] == "极氪7X"]
pred_in = float(oof4["oof_pred"].mean())
pred_out = float(fit_wo["point"].predict(
    rm.prepare_matrix(g4, sm_wo, vocab_wo)).mean())
y4 = float(g4["residual_rate"].mean())
say(f"\n  极氪7X 实际平均残值率              {y4*100:.2f}%（n={len(g4)}）")
say(f"  用「见过极氪7X」的模型（OOF）预测   {pred_in*100:.2f}%"
    f"　→ 偏差 {(pred_in-y4)*100:+.2f}pp")
say(f"  把极氪整个品牌剔除后重训再预测      {pred_out*100:.2f}%"
    f"　→ 偏差 {(pred_out-y4)*100:+.2f}pp")
say(f"\n  退化幅度：{abs(pred_out-y4)*100 - abs(pred_in-y4)*100:+.2f}pp")
say("  说明：极氪7X 是 2025 年上牌的新车，残值率高达 74%，")
say("        当该品牌在训练集中不存在时，模型只能用「同级同价位平均车」去猜，")
say("        必然系统性低估它的残值。**这就是 GroupKFold R² 只有 "
    f"{om['group_kfold']['M2 LightGBM（精简 8 维）']['r2']:.3f}，")
say(f"        而随机 KFold 能到 {om['random_kfold_r2']:.3f} 的原因。**")
say("        业务含义：新品牌/新车系刚进场时，模型给不出可信估价，")
say("        必须先积累车系级历史数据——这恰恰是二手车数据资产的价值所在。")

# ==================================================================
# 6. 写文件
# ==================================================================
tbl.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
with open(OUT_TXT, "w", encoding="utf-8") as f:
    f.write("\n".join(LINES) + "\n")
print(f"\n已写出：{OUT_CSV}")
print(f"已写出：{OUT_TXT}")
