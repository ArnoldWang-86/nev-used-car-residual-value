# -*- coding: utf-8 -*-
r"""20_verify_page.py —— 校验页面内 JS 推理与 Python 模型是否逐点一致

做法：**从实际生成的 HTML 里抽取 JS**（不是我另写一份），
注入一个假的 echarts 与最小 DOM，在 Node 下跑 120 辆真实车辆，
再把结果与 Python 端 `residual_model.predict_one` 对比。

为什么必须这么做：页面是纯前端交付物，一旦 JS 侧与 Python 侧口径漂移，
页面上每个数字都会错，而且不会报错——正是本项目已经踩过一次的那类坑
（`build_features` 曾在单行上重新统计样本量，导致所有车系静默退化为未知档）。

用法：
    python src/20_verify_page.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import residual_model as rm  # noqa: E402

PAGE = os.path.join(rm.ROOT, "5_交付物", "残值预测演示.html")
RULE = "=" * 84
print(RULE)
print("20_verify_page.py · 页面 JS 与 Python 模型一致性校验")
print(RULE)

with open(PAGE, encoding="utf-8") as f:
    html = f.read()
print(f"\n页面：{PAGE}（{os.path.getsize(PAGE)/1024/1024:.2f} MB）")

# ---- 抽取实际发布的 JS ----
import re

# 页面里有 3 个 <script> 块：①ECharts ②数据资产 ③页面功能代码。
# 必须按块分别提取——直接取「const ASSETS 到 DOM 绑定」会横跨两个块，
# 中间夹着的 </script> 会把功能代码整段切掉（实测就是这么被坑的）。
blocks = re.findall(r"<script>(.*?)</script>", html, flags=re.S)
print(f"\n页面共 {len(blocks)} 个 <script> 块，长度：" +
      "、".join(f"{len(b):,}" for b in blocks))
b_data = next(b for b in blocks if b.lstrip().startswith("const ASSETS"))
b_logic = next(b for b in blocks if "function predict(" in b)
js_core = b_data.rstrip() + "\n" + b_logic.rstrip()
assert "function predict(" in js_core and "function compute(" in js_core
print(f"抽取 JS：{len(js_core):,} 字符"
      f"（数据块 {len(b_data):,} + 功能块 {len(b_logic):,}）")

# ---- 测试样本：真实车辆 ----
df = rm.load_clean()
bundle = rm.load_bundle()
sample = df.sample(120, random_state=rm.SEED)
cases = []
for _, r in sample.iterrows():
    cases.append({
        "series": str(r["series"]),
        "age_months": float(r["age_months"]),
        "mileage_wan": float(r["mileage_wan"]),
        "price_wan": float(r["price_wan"]),
        "appearance_score": float(r["appearance_score"]),
        # ⚠️ 必须带上这两个字段：训练数据里 60.2% / 45.9% 的车非零。
        #    漏传会让 JS 侧按 0 计算，与 Python 产生 0.02~0.14 的差异——
        #    这正是本校验脚本第一次跑出来的失败原因。
        "claim_times": float(r["claim_times"]) if pd.notna(r["claim_times"]) else 0.0,
        "transfer_times": float(r["transfer_times"]) if pd.notna(r["transfer_times"]) else 0.0,
    })

# 把页面 JS 包进函数作用域，再把要用的函数导出——
# 直接写在模块顶层会因为 const 重复声明或与 Node 全局冲突而失败
harness = """
// 最小 DOM / ECharts 桩：功能块末尾会执行 init() 并初始化图表，
// 这些副作用与要校验的预测逻辑无关，桩掉即可。
// ⚠️ 桩要给出**合理的数值**：如果 input 的 value 一律返回字符串，
//    render() 里 +el('f-age').value 会得到 NaN，进而让下标变 NaN，
//    报出来的错看起来像页面 bug，其实是桩的锅。
const IVAL = {"f-age":42, "f-mil":6.8, "f-price":33.8, "f-appr":85,
              "f-series":"蔚来ES6"};
function fakeEl(id){
  const o = { style:{}, classList:{add(){},remove(){}}, dataset:{},
    textContent:'', innerHTML:'', children:[],
    value: (id && IVAL[id] !== undefined) ? IVAL[id] : '' };
  o.appendChild = () => {}; o.addEventListener = () => {};
  o.querySelectorAll = () => [];
  return o;
}
globalThis.document = {
  getElementById: (id) => fakeEl(id),
  createElement: () => fakeEl(),
  querySelectorAll: () => [],
  addEventListener: () => {}
};
globalThis.window = { addEventListener(){}, innerWidth:1400, innerHeight:900 };
function chartStub(){ return { setOption(){}, resize(){}, dispose(){} }; }
globalThis.echarts = { init: chartStub, getInstanceByDom: () => null };
const __page = (function(){
""" + js_core + """
  return { predict, seriesBucket, compute, UNKNOWN, SMAP, SCODE, VOCAB,
           medians: A.medians };
})();

function runCase(c){
  const bucket = __page.seriesBucket(c.series);
  const known = (c.series === bucket) && (bucket !== __page.UNKNOWN);
  // ⚠️ 必须复刻 build_features 的缺失值填补：页面本身只接受已填好的输入，
  //    如果这里把 NaN 直接喂给模型，会走 default_left 分支，
  //    与 Python 端（先用中位数填补）就不一致了——错的是桩，不是页面。
  const med = __page.medians;
  let appr = c.appearance_score;
  if (appr === null || appr === undefined || !isFinite(appr)) appr = med.appearance_score;
  const raw = { series_cat: bucket, age_months: c.age_months,
                mileage_wan: c.mileage_wan, price_wan: c.price_wan,
                appearance_score: appr,
                claim_times: c.claim_times, transfer_times: c.transfer_times };
  const p = __page.predict(raw);
  const x = __page.compute(raw);          // 覆盖分解逻辑，验证不抛错
  return { bucket: bucket, known: known, rate: p.rate, lo: p.lo, hi: p.hi,
           ridge: p.ridge,
           decomp_base: x.base, decomp_age: x.ageEff, decomp_mil: x.milEff,
           decomp_price: x.priceEff, decomp_series: x.seriesEff,
           decomp_inter: x.inter,
           fut_tree: x.treeFut.rate, fut_ridge: x.ridgeFut };
}
const cases = JSON.parse(process.argv[2]);
process.stdout.write(JSON.stringify(cases.map(runCase)));
"""

with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False,
                                 encoding="utf-8") as f:
    f.write(harness)
    js_path = f.name
print(f"临时 harness：{js_path}（{os.path.getsize(js_path)/1024:.0f} KB）")
try:
    proc = subprocess.run(
        ["node", js_path, json.dumps(cases, ensure_ascii=False)],
        capture_output=True, text=True, encoding="utf-8")
except Exception:
    raise
if proc.returncode != 0:
    print("[失败] Node 执行出错：")
    print(proc.stderr[-3000:])
    print(f"\n[诊断] 保留 harness 以便排查：{js_path}")
    print(f"       内容中出现 </script> 的次数：{harness.count('</script>')}")
    raise SystemExit(1)
os.unlink(js_path)
js_out = json.loads(proc.stdout)
print(f"Node 返回 {len(js_out)} 条预测")

# ---- Python 端同口径 ----
diffs, band_diffs, bucket_bad = [], [], 0
for c, j in zip(cases, js_out):
    case = rm.InputCase(series=c["series"], age_months=c["age_months"],
                        mileage_wan=c["mileage_wan"], price_wan=c["price_wan"],
                        appearance_score=c["appearance_score"],
                        claim_times=c["claim_times"],
                        transfer_times=c["transfer_times"])
    res = rm.predict_one(bundle, case)
    # 必须用未舍入原值比对：对外字段是 round(...,4) 的展示值，
    # 拿它比 JS 的完整浮点会凭空产生 ~5e-5 的假差异。
    raw = res["_raw"]
    diffs.append(abs(raw["rate"] - j["rate"]))
    band_diffs.append(max(abs(raw["pi90"][0] - j["lo"]),
                          abs(raw["pi90"][1] - j["hi"])))
    py_bucket = res["category"]["series_bucket"]
    if py_bucket != j["bucket"]:
        bucket_bad += 1

d = np.array(diffs)
b = np.array(band_diffs)
print(f"\n【点估计】最大差 {d.max():.2e}　平均 {d.mean():.2e}")
print(f"【90% 区间】上下界最大差 {b.max():.2e}　平均 {b.mean():.2e}")
print(f"【车系分档】与 Python 不一致的样本数：{bucket_bad} / {len(cases)}")

ok = d.max() < 1e-9 and b.max() < 1e-9 and bucket_bad == 0
if not ok:
    print("\n[失败] 页面 JS 与 Python 模型不一致：")
    for k, (c, j, dd) in enumerate(zip(cases, js_out, diffs)):
        if dd > 1e-9:
            print(f"  {c} → JS {j['rate']:.6f}")
    raise SystemExit(1)
print("\n[通过] 页面预测与 Python 模型逐点一致（<1e-9），车系分档口径一致。")

# 契约校验：车系必须真的进入了模型输入向量。
# 历史上这里踩过一次：导出的 features 漏了 series_cat，
# 页面算不出类别下标，车系编码被静默丢弃、所有车系都退化成未知档。
n_known = sum(1 for j in js_out if j["known"])
rates_by_bucket = {}
for c, j in zip(cases, js_out):
    rates_by_bucket.setdefault(j["bucket"], []).append(j["rate"])
multi = {k: max(v) - min(v) for k, v in rates_by_bucket.items() if len(v) >= 3}
assert multi, "样本中没有任何车系档有 ≥3 条记录，无法校验车系是否生效"
print("车系编码生效性校验（同一车系档内不同车辆应有不同预测）：")
for k, spread in sorted(multi.items(), key=lambda x: -x[1])[:5]:
    print(f"  {k:<18} 预测极差 {spread:.4f}"
          f"{'  ← 极差为 0，车系可能未生效' if spread < 1e-9 else ''}")
assert all(s > 1e-9 for s in multi.values()), \
    "存在同一车系档内所有车预测完全相同 → 车系编码很可能没有进入输入向量"
print(f"（共校验 {n_known} 辆命中独立建模档的车）")

# ---- 顺便校验页面里的分档名与告警逻辑覆盖 ----
known = [s for s in bundle["series_vocab"] if s != rm.UNKNOWN_SERIES]
print(f"\n车系下拉档位 {len(bundle['series_vocab'])} 个"
      f"（独立建模 {len(known)} 个 + 1 个兜底档）")
print(f"页面资产大小 {os.path.getsize(os.path.join(rm.ROOT,'output','predict_assets.json'))/1024:.0f} KB")
print(f"\n{RULE}\n校验完成。\n{RULE}")
