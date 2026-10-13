# -*- coding: utf-8 -*-
r"""predict.py —— 残值预测推理接口（命令行 + Python 函数）

输入：车系 / 车龄 / 里程 / 新车指导价（+ 可选车况）
输出：残值率、预测挂牌价、90% 预测区间、预测过程分解、（可选）1 年后残值

用法
----
1) 单车估价（人类可读）：

    python src/predict.py --series 蔚来ES6 --age-years 3.5 --mileage-wan 6.8 \
        --guide-price-wan 33.8 --explain

2) 只给月龄也行：

    python src/predict.py --series Model Y --age-months 42 --mileage-wan 9.0 \
        --guide-price-wan 26.4

3) 批量（CSV 需含列 series, age_months, mileage_wan, price_wan，
   可选 appearance_score / claim_times / transfer_times）：

    python src/predict.py --csv cases.csv --out results/p02_批量预测.csv

4) 1 年后残值会跌多少：

    python src/predict.py --series 蔚来ES6 --age-years 3.5 --mileage-wan 6.8 \
        --guide-price-wan 33.8 --project-1y

5) JSON 输出（供程序消费 / 看板构建）：

    python src/predict.py ... --json

在 Python 里调用：

    import sys; sys.path.insert(0, "src")
    from residual_model import load_bundle, predict_one, InputCase
    b = load_bundle()
    predict_one(b, InputCase(series="蔚来ES6", age_months=42,
                             mileage_wan=6.8, price_wan=33.8), explain=True)

模型来源：src/16_train_final.py 训练并保存到
output/models/residual_model.joblib（随交付物提供，无需重训即可直接用）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import residual_model as rm  # noqa: E402

BAR = "-" * 76


# ------------------------------------------------------------------
# 打印
# ------------------------------------------------------------------
def _fmt_wan(x) -> str:
    return "—" if x is None else f"{x:.2f} 万"


def print_result(res: dict, bundle: dict, show_explain: bool = False,
                 projection: dict | None = None) -> None:
    inp = res["input"]
    age_m = inp.get("age_months")
    age_txt = (f"{float(age_m):.0f} 个月（{float(age_m)/12:.1f} 年）"
               if age_m is not None and np.isfinite(float(age_m)) else "—")

    print(BAR)
    print("车辆信息")
    print(BAR)
    print(f"  车系        {inp.get('series') or '（未提供）'}")
    print(f"  车龄        {age_txt}")
    print(f"  里程        " + (f"{float(inp['mileage_wan']):.2f} 万公里"
                              if inp.get("mileage_wan") is not None else "—"))
    print(f"  新车指导价  " + _fmt_wan(inp.get("price_wan")))
    if inp.get("appearance_score") is not None:
        print(f"  外观评分    {float(inp['appearance_score']):.0f}")

    cat = res["category"]
    print(f"\n  命中类别    {cat['series_bucket']}"
          f"（训练样本 {cat['train_n']} 条 · {cat['status']}）")

    print(f"\n{BAR}\n预测结果\n{BAR}")
    print(f"  残值率（预测）      {res['residual_rate']*100:.2f}%")
    if res["price_wan"] is not None:
        print(f"  预测挂牌价          {res['price_wan']:.2f} 万元"
              f"   = {res['residual_rate']*100:.2f}% × "
              f"{res['guide_price_wan']:.2f} 万指导价")
        print(f"  90% 预测区间        {res['pi90'][0]*100:.2f}% ~ "
              f"{res['pi90'][1]*100:.2f}%"
              f"   即 {res['pi90_price'][0]:.2f} ~ {res['pi90_price'][1]:.2f} 万元")
    else:
        print(f"  90% 预测区间        {res['pi90'][0]*100:.2f}% ~ "
              f"{res['pi90'][1]*100:.2f}%")
        print("  （未提供指导价，无法折算成金额）")
    print(f"\n  模型                {res['method']}")
    om = res["model_oof"]["group_kfold"]["M2 LightGBM（精简 8 维）"]
    print(f"  模型参考精度        GroupKFold OOF MAE={om['mae']:.4f} "
          f"R²={om['r2']:.4f}（针对没见过的车系的检验）")
    print(f"  区间含义            {res['band_type']}")

    if show_explain and res.get("decomposition"):
        dec = res["decomposition"]
        print(f"\n{BAR}\n预测过程分解（单因素反事实）\n{BAR}")
        print(f"  基准：{dec['baseline_desc']}")
        print(f"        基准残值率 = {dec['baseline_rate']*100:.2f}%")
        print(f"\n  {'因素':<14}{'该车取值':<26}{'相对基准':>10}")
        for it in dec["items"]:
            print(f"  {it['factor']:<14}{it['value']:<26}"
                  f"{it['effect_pp']:>+9.2f}pp")
        print(f"  {'（交互残差）':<14}{'不可加部分':<26}"
              f"{dec['residual_pp']:>+9.2f}pp")
        print(f"\n  → 最终预测残值率 = {dec['final_rate']*100:.2f}%")
        print(f"  注：{dec['note']}")

    if projection and projection.get("ok"):
        p = projection
        print(f"\n{BAR}\n1 年后残值变化（双轨对照）\n{BAR}")
        for k in ("tree", "linear"):
            t = p[k]
            print(f"  {t['name']:<26} {t['rate_now']*100:.2f}% → "
                  f"{t['rate_1y']*100:.2f}%   跌 {t['drop_pp']:+.2f}pp"
                  f"   约 {t['value_drop_wan']:+.2f} 万元")
        print(f"\n  两轨给出的年跌幅差异：{p['row_disagreement_pp']:.2f}pp")
        if p["row_disagreement_pp"] > 1.0:
            print("  ⚠️ 差异 >1pp，说明该车龄处树模型已经开始变平（分段常数、")
            print("     学不到区间外的继续下降），此时应以线性轨为准。")
        else:
            print("  差异小，说明该车龄处两轨一致，结论较稳。")
        er = p["empirical_reference"]
        print(f"\n  现实锚点：控制车系/里程/指导价后的年跌 "
              f"{er['slope_controlled_pp_per_year']:+.2f}pp"
              f"（简单斜率 {er['slope_simple_pp_per_year']:+.2f}pp 含构成效应，仅作上界）")
        print(f"\n  假设条件：")
        for a in p["assumptions"]:
            print(f"    · {a}")

    if res["warnings"]:
        print(f"\n{BAR}\n⚠️  提示\n{BAR}")
        for w in res["warnings"]:
            print(f"  · {w}")

    print(f"\n  口径提醒：预测的是「合理挂牌价」，挂牌价系统性高于成交价；")
    print(f"           本数据为 {bundle['snapshot']} 单期快照，模型不含市场行情涨跌。")
    print(BAR)


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="新能源二手车残值预测（LightGBM + 原生类别车系）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("用法\n----")[-1])
    p.add_argument("--series", help="车系，如 蔚来ES6 / Model Y / 小米SU7")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--age-years", type=float, help="车龄（年），如 3.5")
    g.add_argument("--age-months", type=float, help="车龄（月），如 42")
    p.add_argument("--mileage-wan", type=float, help="里程（万公里），如 6.8")
    p.add_argument("--guide-price-wan", type=float, help="新车指导价（万元），如 33.8")
    p.add_argument("--appearance-score", type=float, default=None,
                   help="外观评分（80/85/90/95/99），可不填")
    p.add_argument("--claim-times", type=float, default=None, help="理赔次数，可不填")
    p.add_argument("--transfer-times", type=float, default=None, help="过户次数，可不填")
    p.add_argument("--explain", action="store_true", help="展示预测过程分解")
    p.add_argument("--project-1y", action="store_true", help="展示 1 年后残值变化")
    p.add_argument("--mileage-add-wan", type=float, default=1.0,
                   help="未来一年预计新增里程（万公里），默认 1.0")
    p.add_argument("--csv", help="批量预测：输入 CSV 路径")
    p.add_argument("--out", help="批量预测：输出 CSV 路径")
    p.add_argument("--json", action="store_true", help="以 JSON 输出结果")
    p.add_argument("--model", help="模型文件路径（默认 output/models/residual_model.joblib）")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    bundle = rm.load_bundle(args.model)

    # ---------- 批量 ----------
    if args.csv:
        src = pd.read_csv(args.csv, encoding="utf-8-sig")
        need = {"series", "age_months", "mileage_wan", "price_wan"}
        missing = need - set(src.columns)
        if missing:
            print(f"[错误] CSV 缺少必需列：{sorted(missing)}", file=sys.stderr)
            return 2
        rows = []
        for _, r in src.iterrows():
            case = rm.InputCase(
                series=r.get("series"),
                age_months=r.get("age_months"),
                mileage_wan=r.get("mileage_wan"),
                price_wan=r.get("price_wan"),
                appearance_score=r.get("appearance_score"),
                claim_times=r.get("claim_times"),
                transfer_times=r.get("transfer_times"),
            )
            res = rm.predict_one(bundle, case)
            rows.append({
                "series": r.get("series"),
                "age_months": r.get("age_months"),
                "mileage_wan": r.get("mileage_wan"),
                "guide_price_wan": res["guide_price_wan"],
                "residual_rate_pred": res["residual_rate"],
                "price_wan_pred": res["price_wan"],
                "pi90_lo": res["pi90"][0], "pi90_hi": res["pi90"][1],
                "pi90_price_lo": (res["pi90_price"] or [None, None])[0],
                "pi90_price_hi": (res["pi90_price"] or [None, None])[1],
                "series_bucket": res["category"]["series_bucket"],
                "train_n": res["category"]["train_n"],
                "warnings": " | ".join(res["warnings"]),
            })
        out = pd.DataFrame(rows)
        out_path = args.out or os.path.join(rm.ROOT, "results", "p02_批量预测.csv")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        out.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"已预测 {len(out)} 辆车，结果写出：{out_path}")
        if args.json:
            print(json.dumps(rows, ensure_ascii=False, indent=1))
        return 0

    # ---------- 单车 ----------
    if args.age_months is not None:
        age_m = args.age_months
    elif args.age_years is not None:
        age_m = args.age_years * 12.0
    else:
        age_m = None

    if args.series is None and age_m is None and args.mileage_wan is None:
        build_parser().print_help()
        print("\n[提示] 至少提供 --series / --age-years / --mileage-wan 之一。")
        return 0

    case = rm.InputCase(series=args.series, age_months=age_m,
                        mileage_wan=args.mileage_wan,
                        price_wan=args.guide_price_wan,
                        appearance_score=args.appearance_score,
                        claim_times=args.claim_times,
                        transfer_times=args.transfer_times)
    res = rm.predict_one(bundle, case, explain=args.explain)
    proj = (rm.project_next_year(bundle, case, args.mileage_add_wan)
            if args.project_1y else None)

    if args.json:
        print(json.dumps({"prediction": res, "projection_1y": proj},
                         ensure_ascii=False, indent=1))
    else:
        print_result(res, bundle, show_explain=args.explain, projection=proj)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
