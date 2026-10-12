# -*- coding: utf-8 -*-
r"""19_build_dashboard.py —— 生成自包含的可交互残值预测页面

产物：`5_交付物/残值预测演示.html`（单文件，双击即可打开，无需服务器）

页面做什么：
  1. 输入车系 / 车龄 / 里程 / 新车指导价 / 外观评分
  2. 实时输出：残值率、预测挂牌价、90% 预测区间
  3. **把预测过程摊开**：基准值 → 各因素加减 → 最终值（瀑布图）
  4. 折旧曲线：这辆车在车系曲线上的位置 + 全市场车龄段中位数基准带
  5. 1 年后残值：树模型轨 vs 线性轨对照，并给出两轨差异

技术要点：
  · 页面内置导出的 LightGBM 树结构并用 JS 直接遍历，与 Python 端**逐点一致**
    （导出脚本已断言最大差 0.00e+00）。因此页面上的每个数字都是模型的精确输出，
    不是插值或查表近似。
  · ECharts 与数据资产全部内联，页面不依赖任何外部请求。

用法：
    python src/19_build_dashboard.py
"""
from __future__ import annotations

import json
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSET = os.path.join(ROOT, "output", "predict_assets.json")
ECHARTS_CANDIDATES = [
    os.path.join(ROOT, "5_交付物", "assets", "echarts.min.js"),
    os.path.join(ROOT, "5_交付物", "echarts.min.js"),
]
OUT = os.path.join(ROOT, "5_交付物", "残值预测演示.html")

with open(ASSET, encoding="utf-8") as f:
    assets = json.load(f)

echarts_js = None
for p in ECHARTS_CANDIDATES:
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            echarts_js = f.read()
        print(f"已嵌入 ECharts：{p}")
        break
if echarts_js is None:
    raise SystemExit("未找到 echarts.min.js，请检查 5_交付物/assets/")

DATA_JS = json.dumps(assets, ensure_ascii=False, separators=(",", ":"))

HTML = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>新能源二手车残值预测 · 在线估价</title>
<style>
:root{
  --bg:#0f1217; --card:#171b23; --card2:#1e232d; --line:#2a3140;
  --txt:#e6eaf2; --dim:#9aa5b8; --accent:#4c8dff; --accent2:#22c9a0;
  --warn:#ffb020; --bad:#ff6b6b; --good:#22c9a0;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--txt);
  font-family:"Microsoft YaHei","PingFang SC",system-ui,sans-serif;
  font-size:14px;line-height:1.6}
a{color:var(--accent)}
.wrap{max-width:1440px;margin:0 auto;padding:20px 24px 60px}
header{border-bottom:1px solid var(--line);padding-bottom:14px;margin-bottom:20px}
h1{font-size:21px;margin:0 0 6px}
.sub{color:var(--dim);font-size:13px}
.grid{display:grid;grid-template-columns:330px 1fr;gap:18px;align-items:start}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px}
.card h2{font-size:14px;margin:0 0 12px;color:var(--dim);font-weight:600;
  letter-spacing:.4px}
label{display:block;font-size:12px;color:var(--dim);margin:12px 0 4px}
select,input[type=number]{width:100%;background:var(--card2);color:var(--txt);
  border:1px solid var(--line);border-radius:8px;padding:8px 10px;font-size:14px;
  font-family:inherit}
input[type=range]{width:100%;accent-color:var(--accent);margin-top:2px}
.row{display:flex;gap:8px}
.row>*{flex:1}
.hint{font-size:11px;color:var(--dim);margin-top:3px}
.big{font-size:44px;font-weight:700;line-height:1.1;
  background:linear-gradient(90deg,#4c8dff,#22c9a0);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent}
.kpis{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-bottom:14px}
.kpi{background:var(--card2);border-radius:10px;padding:12px 14px;
  border:1px solid var(--line)}
.kpi .k{font-size:12px;color:var(--dim)}
.kpi .v{font-size:22px;font-weight:700;margin-top:2px}
.kpi .s{font-size:11px;color:var(--dim);margin-top:2px}
.charts{display:grid;grid-template-columns:1fr 1fr;gap:14px}
.chart{background:var(--card2);border:1px solid var(--line);border-radius:10px;
  padding:10px 8px 4px}
.chart .t{font-size:12px;color:var(--dim);padding:0 8px 4px}
.chart .c{height:270px}
.wide{grid-column:1/-1}
.wide .c{height:250px}
.note{font-size:12px;color:var(--dim);margin-top:10px;padding:10px 12px;
  background:var(--card2);border-radius:8px;border-left:3px solid var(--accent)}
.warn{border-left-color:var(--warn);color:#ffd79a}
table{width:100%;border-collapse:collapse;font-size:13px}
th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--line)}
th{color:var(--dim);font-weight:600;font-size:12px}
td.num{text-align:right;font-variant-numeric:tabular-nums}
.pill{display:inline-block;padding:1px 8px;border-radius:20px;font-size:11px;
  border:1px solid var(--line);color:var(--dim)}
.pill.ok{border-color:#1c5c4a;color:var(--good);background:#12241f}
.pill.no{border-color:#5c2a2a;color:var(--bad);background:#241414}
.foot{margin-top:18px;color:var(--dim);font-size:12px}
.legend{font-size:11px;color:var(--dim);margin-top:6px;padding:0 8px}
</style>
</head>
<body>
<div class="wrap">
<header>
  <h1>新能源二手车残值预测 · 在线估价</h1>
  <div class="sub">
    LightGBM · 8 维精简特征 + 原生类别车系 ·
    训练样本 <b id="m-n"></b> 条 · 快照 <b id="m-snap"></b> ·
    GroupKFold OOF MAE <b id="m-mae"></b>，R² <b id="m-r2"></b>
  </div>
</header>

<div class="grid">
  <!-- ============ 输入 ============ -->
  <div class="card">
    <h2>车辆信息</h2>

    <label>车系</label>
    <select id="f-series"></select>
    <div class="hint" id="h-series"></div>

    <label>车龄　<span id="v-age" class="pill"></span></label>
    <input type="range" id="f-age" min="1" max="96" step="1">
    <div class="hint">拖动即实时重算，坐标轴上是模型每次的精确输出</div>

    <label>里程（万公里）　<span id="v-mil" class="pill"></span></label>
    <input type="range" id="f-mil" min="0" max="24" step="0.5">

    <label>新车指导价（万元）</label>
    <div class="row">
      <input type="number" id="f-price-num" step="0.1" min="5" max="120">
    </div>
    <input type="range" id="f-price" min="8" max="90" step="0.5">
    <div class="hint" id="v-price"></div>

    <label>外观评分</label>
    <select id="f-appr">
      <option value="99">99 分</option><option value="95">95 分</option>
      <option value="90">90 分</option><option value="85" selected>85 分</option>
      <option value="80">80 分</option>
    </select>

    <div class="hint" id="h-appr"></div>
    <div class="note">
      理赔次数与过户次数影响很小但不为零（本项目数据里约 60% 的车有理赔记录），
      本页没有这两个输入框，<b>统一按「无理赔、未过户」计算</b>；
      如已知实际记录，请用命令行 <code>python src/predict.py --claim-times N</code>。
    </div>
    <div class="note" style="margin-top:10px">
      页面内置导出的 LightGBM 树结构并用 JS 直接遍历，与 Python 端<b>逐点一致</b>；
      90% 区间也来自导出的分位数头，都是模型精确输出，不是插值估算。
    </div>
  </div>

  <!-- ============ 输出 ============ -->
  <div>
    <div class="card" style="margin-bottom:14px">
      <h2>预测结果</h2>
      <div class="kpis">
        <div class="kpi">
          <div class="k">残值率（预测）</div>
          <div class="v big" id="o-rate">—</div>
          <div class="s">挂牌价 ÷ 新车指导价</div>
        </div>
        <div class="kpi">
          <div class="k">预测挂牌价</div>
          <div class="v" id="o-price">—</div>
          <div class="s" id="o-price-formula">—</div>
        </div>
        <div class="kpi">
          <div class="k">90% 预测区间</div>
          <div class="v" id="o-band" style="font-size:17px">—</div>
          <div class="s" id="o-band-price">单台车的预测区间，非均值置信区间</div>
        </div>
      </div>
      <div id="o-warn"></div>
    </div>

    <div class="charts">
      <div class="chart">
        <div class="t">① 预测过程分解：基准值 → 各因素加减 → 最终预测</div>
        <div class="c" id="c-water"></div>
        <div class="legend">
          单因素反事实：把某一项换成全市场基准值，看预测值移动多少。
          分解不满足严格可加性，差额单列为「交互项」。
        </div>
      </div>

      <div class="chart">
        <div class="t">② 这辆车在车系折旧曲线上的位置</div>
        <div class="c" id="c-curve"></div>
        <div class="legend" id="l-curve"></div>
      </div>

      <div class="chart wide">
        <div class="t">③ 1 年后残值：树模型轨 vs 线性外推轨（两轨差异＝树模型是否已变平）</div>
        <div class="c" id="c-future"></div>
        <div class="legend" id="l-future"></div>
      </div>

      <div class="chart wide">
        <div class="t">④ 横向对比：这辆车 vs 同车系 / 同车龄段 / 全市场</div>
        <div class="c" id="c-bench" style="height:230px"></div>
      </div>
    </div>

    <div class="card" style="margin-top:14px">
      <h2>方法说明与适用边界</h2>
      <table>
        <tr><th style="width:150px">项目</th><th>内容</th></tr>
        <tr><td>目标变量</td><td id="t-target"></td></tr>
        <tr><td>预测引擎</td><td id="t-engine"></td></tr>
        <tr><td>区间口径</td><td id="t-band"></td></tr>
        <tr><td>年折旧锚点</td><td id="t-decay"></td></tr>
        <tr><td>挂牌价 ≠ 成交价</td><td>预测的是<b>合理挂牌价</b>。挂牌价系统性高于成交价，不能直接当收购价用。</td></tr>
        <tr><td>单期快照</td><td id="t-snap"></td></tr>
        <tr><td>车系覆盖</td><td id="t-cov"></td></tr>
        <tr><td>不做因果推断</td><td>所有关系均为相关性描述。</td></tr>
      </table>
      <div class="foot">
        求职作品集项目 · 数据来源：瓜子二手车公开详情页（遵循 robots.txt，仅采集 Allow 路径）·
        模型训练与验证见 <code>src/16_train_final.py</code>，本页面数据由
        <code>src/18_export_dashboard.py</code> 导出。
      </div>
    </div>
  </div>
</div>
</div>

<script>/*__ECHARTS__*/</script>
<script>const ASSETS = /*__DATA__*/;</script>
<script>
/* =====================================================================
   1. 页面内模型：LightGBM 树遍历 + Ridge 线性
   与 Python 端 src/residual_model.py 的口径完全一致
   ===================================================================== */
const A = ASSETS;
const NF = A.features;                    // 模型完整特征名（8 项，末位是 series_cat）
const FIDX = {}; NF.forEach((f,i)=>FIDX[f]=i);
const CATF = A.cat_feature;
const CAT_IDX = A.cat_index;              // 类别特征在输入向量中的下标
const VOCAB = A.series_vocab;
const SCODE = {}; VOCAB.forEach((s,i)=>SCODE[s]=i);
const SMAP = A.series_map;
const UNKNOWN = "其他/未知车系";
// 契约：数值特征必须恰好是「完整特征名去掉类别特征」
if (NF.length !== A.n_features || NF[CAT_IDX] !== CATF) {
  throw new Error('特征表契约不满足：' + JSON.stringify(NF));
}

/** 单棵树遍历：与 Python js_eval 逐行一致 */
function evalModel(m, vec){
  let tot = 0;
  for (let t=0; t<m.trees.length; t++){
    const nd = m.trees[t].n;
    let i = m.trees[t].r;
    for(let guard=0; guard<200; guard++){
      const o = nd[i];
      if (o.v !== undefined){ tot += o.v; break; }
      const x = vec[o.f];
      let left;
      if (x === null || x === undefined || !isFinite(x)){
        left = !!o.dl;
      } else if (o.c){
        left = false;
        const parts = String(o.th).split("||");
        for (const p of parts){
          if (p.endsWith("-")){
            const [lo,hi] = p.slice(0,-1).split("-");
            if (+x >= +lo && +x <= +hi){ left = true; break; }
          } else if (p !== "" && +p === +x){ left = true; break; }
        }
      } else {
        left = (+x) <= o.th;
      }
      i = left ? o.l : o.r;
    }
  }
  return tot;
}

/** Ridge：常数 + 数值点积 + 类别增量 */
function evalRidge(inp){
  const R = A.ridge;
  let v = R.num_const;
  for (let k=0;k<R.num_features.length;k++){
    let x = inp[R.num_features[k]];
    if (x === null || x === undefined || !isFinite(x)) x = R.missing_fill[k];
    v += R.num_slope[k] * x;
  }
  const s = inp[CATF];
  const dropI = R.ohe_drop_idx;
  const kept = R.ohe_categories.filter((_,i)=>i!==dropI).map(String);
  const j = kept.indexOf(String(s));
  if (j >= 0) v += R.ohe_coef[j];
  return v;
}

/** 原始输入 → 模型输入向量（逐行确定性变换，与 build_features 一致）
 *
 * 契约（不要凭印象改）：
 *   raw 必须含 series_cat / age_months / mileage_wan / price_wan / appearance_score；
 *   claim_times 与 transfer_times 若未提供则取 0。
 *   这两个字段在训练数据里 60.2% / 45.9% 的车是非零，**不是可以随手置 0 的**：
 *   页面 UI 没有这两个输入框（用户拿不到理赔/过户记录），所以页面固定按 0 计算，
 *   并已在页面上明确标注「按无理赔、未过户计算」。
 *   校验脚本 20_verify_page.py 会传入真实值，与 Python 逐点对齐。
 */
function featurize(raw){
  const age = raw.age_months, mil = raw.mileage_wan, price = raw.price_wan;
  const log_mil = Math.log(Math.max(mil,0) + 1);
  const log_price = Math.log(Math.max(price,0) + 1);
  const mpy = mil / (Math.max(age,6) / 12);
  const catCode = (SCODE[raw.series_cat] !== undefined) ? SCODE[raw.series_cat]
                                                        : SCODE[UNKNOWN];
  const byName = {
    age_months: age,
    log_mileage_wan: log_mil,
    mileage_per_year: mpy,
    log_price_wan: log_price,
    appearance_score: raw.appearance_score,
    claim_times: (raw.claim_times === undefined || raw.claim_times === null)
                 ? 0 : raw.claim_times,
    transfer_times: (raw.transfer_times === undefined || raw.transfer_times === null)
                 ? 0 : raw.transfer_times,
    series_cat: catCode
  };
  // 按模型的特征名逐位构造，长度天然等于 n_features。
  // 不要用「数值特征 map + 下标赋值/追加」的写法——数组长度与下标一旦对不上，
  // 类别编码会挂到数组外属性上，模型读到 undefined 却不报错（踩过）。
  const vec = NF.map(f => byName[f]);
  for (let i=0;i<vec.length;i++){
    if (vec[i] === undefined || vec[i] === null){
      throw new Error('特征「'+NF[i]+'」没有取值');
    }
  }
  const vals = byName;
  return {vals, vec};
}

function predict(raw){
  const {vals, vec} = featurize(raw);
  let r = evalModel(A.model_point, vec);
  r = Math.min(Math.max(r, 0.02), 1.05);
  let lo = Math.min(evalModel(A.model_quantile["0.05"], vec), r);
  let hi = Math.max(evalModel(A.model_quantile["0.95"], vec), r);
  return {rate:r, lo, hi, vals, vec,
          ridge: Math.min(Math.max(evalRidge(vals), 0.02), 1.05)};
}

/* =====================================================================
   2. 基准数据
   ===================================================================== */
const REF = A.reference;
const MED = A.medians;
const REF_APPLY = {claim_times:0, transfer_times:0};

function seriesBucket(s){
  const b = SMAP[s];
  return (b && SCODE[b] !== undefined) ? b : UNKNOWN;
}

/** 取分解基准曲线（车龄方向，固定价格档与外观档） */
function decompCurve(series, price, appr){
  const ps = A.curve_prices, as = A.curve_appr;
  const p = ps.reduce((a,b)=> Math.abs(b-price)<Math.abs(a-price)?b:a);
  const q = as.reduce((a,b)=> Math.abs(b-appr)<Math.abs(a-appr)?b:a);
  return {curve: A.decomp_curves[series+"|"+p+"|"+q], price:p, appr:q};
}

/** 一元曲线按车龄线性取值（结点为 2 个月，先按结点最近值，避免编造平滑） */
function curveAt(curve, age){
  const ages = A.curve_ages;
  let i = 0, best = 1e9;
  for (let k=0;k<ages.length;k++){ const d=Math.abs(ages[k]-age); if(d<best){best=d;i=k;} }
  return curve[i];
}

/* =====================================================================
   3. 计算一辆车
   ===================================================================== */
function compute(raw){
  const bucket = seriesBucket(raw.series_cat);
  const known = (raw.series_cat === bucket) && (bucket !== UNKNOWN);
  const p = predict({...raw, series_cat: bucket});

  // --- 分解：基准 + 各因素 ---
  const dc = decompCurve(bucket, raw.price_wan, raw.appearance_score);
  const baseAge = curveAt(dc.curve, REF.age_months);
  const curAge = curveAt(dc.curve, raw.age_months);
  const ageEff = curAge - baseAge;

  const milRef = predict({...raw, series_cat: bucket, mileage_wan: REF.mileage_wan,
                          age_months: REF.age_months}).rate;
  const milEff = predict({...raw, series_cat: bucket,
                          age_months: REF.age_months}).rate - milRef;
  const priceRef = predict({...raw, series_cat: bucket,
                            price_wan: REF.price_wan,
                            age_months: REF.age_months,
                            mileage_wan: REF.mileage_wan}).rate;
  const priceEff = predict({...raw, series_cat: bucket,
                            mileage_wan: REF.mileage_wan,
                            age_months: REF.age_months}).rate - priceRef;
  const seriesProbe = predict({...raw, series_cat: UNKNOWN}).rate;
  const seriesEff = known ? (p.rate - seriesProbe) : 0;
  const base = predict({...raw, series_cat: bucket, age_months: REF.age_months,
                        mileage_wan: REF.mileage_wan,
                        price_wan: REF.price_wan,
                        appearance_score: REF.appearance_score}).rate;
  const explained = ageEff + milEff + priceEff + seriesEff;
  const inter = p.rate - base - explained;

  // --- 未来一年 ---
  const futAge = raw.age_months + 12;
  const futMil = raw.mileage_wan + 1.0;
  const treeFut = predict({...raw, series_cat: bucket,
                           age_months: futAge, mileage_wan: futMil});
  const ridgeFut = predict({...raw, series_cat: bucket,
                            age_months: futAge, mileage_wan: futMil}).ridge;

  return {bucket, known, p, dc, base, ageEff, milEff, priceEff, seriesEff, inter,
          treeFut, ridgeFut, futAge, futMil};
}

/* =====================================================================
   4. 图表
   ===================================================================== */
const AX = {axisLine:{lineStyle:{color:'#3a4252'}},
            axisLabel:{color:'#9aa5b8',fontSize:11},
            splitLine:{lineStyle:{color:'#232a36'}}};
const TIP = {backgroundColor:'#1e232d',borderColor:'#2a3140',
             textStyle:{color:'#e6eaf2',fontSize:12}};
const charts = {};
function clamp(v, lo, hi){ return Math.min(Math.max(v, lo), hi); }
function initChart(id){
  const el = document.getElementById(id);
  const c = echarts.init(el, null, {renderer:'canvas'});
  charts[id] = c;
  window.addEventListener('resize', ()=>c.resize());
  return c;
}
['c-water','c-curve','c-future','c-bench'].forEach(initChart);

function drawWater(c, x){
  const items = [
    {n:'基准值\n'+x.baseAgeLabel, v:x.base, base:true},
    {n:'车龄', v:x.ageEff},
    {n:'里程', v:x.milEff},
    {n:'新车指导价', v:x.priceEff},
  ];
  if (x.known) items.push({n:'车系', v:x.seriesEff});
  items.push({n:'交互项', v:x.inter});
  items.push({n:'最终预测', v:x.p.rate, total:true});

  let run = 0; const help = [], val = [];
  items.forEach((it,i)=>{
    if (it.base || it.total){ help.push(0); val.push(it.v); run = it.v; }
    else {
      const s = run; run += it.v;
      help.push(Math.min(s,run)); val.push(Math.abs(it.v));
    }
  });
  c.setOption({
    tooltip:{...TIP, trigger:'axis', axisPointer:{type:'shadow'},
      formatter: ps=>{const i=ps[0].dataIndex; const it=items[i];
        return it.n.replace('\n',' ')+'<br/>'+
          (it.base||it.total ? '' : (it.v>=0?'+':'')+(it.v*100).toFixed(2)+'pp　')
          +'<b>'+(it.v*100).toFixed(2)+'%</b>';}},
    grid:{left:56,right:14,top:16,bottom:52},
    xAxis:{type:'category',data:items.map(i=>i.n),
      ...AX, axisLabel:{...AX.axisLabel,interval:0,fontSize:10,lineHeight:13}},
    yAxis:{type:'value',...AX,axisLabel:{...AX.axisLabel,
      formatter:v=>(v*100).toFixed(0)+'%'}},
    series:[
      {type:'bar',stack:'a',itemStyle:{color:'transparent'},data:help,
       silent:true,barWidth:'52%'},
      {type:'bar',stack:'a',barWidth:'52%',
       label:{show:true,position:'top',color:'#e6eaf2',fontSize:10,
         formatter:p=>{const it=items[p.dataIndex];
           return (it.base||it.total)?(p.value*100).toFixed(1)+'%'
             :((it.v>=0?'+':'')+(it.v*100).toFixed(1));}},
       data:val.map((v,i)=>({value:v,
         itemStyle:{color: items[i].base?'#4c8dff'
           : items[i].total?'#22c9a0'
           : (items[i].v>=0?'#22c9a0':'#ff6b6b'),borderRadius:3}}))}
    ]
  });
}

function drawCurve(c, x, raw){
  const ages = A.curve_ages;
  const seriesCurve = A.curves[x.bucket].map(v=>v*100);
  const decayBands = A.decay_bands;
  const bandY = decayBands.map(b=>b.median*100);
  const mark = [[raw.age_months, x.p.rate*100]];

  c.setOption({
    tooltip:{...TIP,trigger:'axis',
      formatter: ps=>{let s=ps[0].axisValue+' 个月<br/>';
        ps.forEach(p=>{s+=p.marker+p.seriesName+'：'+Number(p.value).toFixed(2)+'%<br/>';});
        return s;}},
    legend:{data:['同车系折旧曲线','全市场车龄段中位数','这辆车'],
      textStyle:{color:'#9aa5b8',fontSize:11},top:0,itemWidth:14,itemHeight:8},
    grid:{left:52,right:16,top:34,bottom:34},
    xAxis:{type:'category',data:ages,...AX,
      name:'车龄（月）',nameTextStyle:{color:'#9aa5b8',fontSize:11}},
    yAxis:{type:'value',...AX,axisLabel:{...AX.axisLabel,
      formatter:v=>v.toFixed(0)+'%'},max:100},
    series:[
      {name:'同车系折旧曲线',type:'line',data:seriesCurve,smooth:false,
       symbol:'none',lineStyle:{color:'#4c8dff',width:2},z:3},
      {name:'全市场车龄段中位数',type:'line',
       data:ages.map(a=>bandY[Math.min(Math.floor(a/12), bandY.length-1)]),
       symbol:'none',lineStyle:{color:'#5a6478',width:1.5,type:'dashed'},z:2},
      {name:'这辆车',type:'scatter',data:mark,symbolSize:13,
       itemStyle:{color:'#22c9a0',borderColor:'#0f1217',borderWidth:2},z:5}
    ]
  });
}

function drawFuture(c, x, raw){
  const ages = [raw.age_months, x.futAge];
  const treeY = [x.p.rate*100, x.treeFut.rate*100];
  const ridgeY = [x.p.ridge*100, x.ridgeFut*100];
  c.setOption({
    tooltip:{...TIP,trigger:'axis',
      formatter: ps=>{let s=ps[0].axisValue+' 个月<br/>';
        ps.forEach(p=>{s+=p.marker+p.seriesName+'：'+Number(p.value).toFixed(2)+'%<br/>';});
        const dT=(treeY[1]-treeY[0]), dR=(ridgeY[1]-ridgeY[0]);
        s+='<br/>树模型跌 '+dT.toFixed(2)+'pp　线性轨跌 '+dR.toFixed(2)+'pp';
        return s;}},
    legend:{data:['树模型（LightGBM）','线性外推轨（Ridge）'],
      textStyle:{color:'#9aa5b8',fontSize:11},top:0,itemWidth:14,itemHeight:8},
    grid:{left:52,right:16,top:34,bottom:34},
    xAxis:{type:'category',data:ages.map(a=>'当前 '+a+' 月'),
      ...AX,axisLabel:{...AX.axisLabel,formatter:(v,i)=>i===0?'当前':'1 年后'}},
    yAxis:{type:'value',...AX,scale:true,axisLabel:{...AX.axisLabel,
      formatter:v=>v.toFixed(0)+'%'}},
    series:[
      {name:'树模型（LightGBM）',type:'line',data:treeY,symbolSize:9,
       lineStyle:{color:'#4c8dff',width:3},itemStyle:{color:'#4c8dff'},
       label:{show:true,color:'#e6eaf2',fontSize:11,
         formatter:p=>p.value.toFixed(1)+'%'}},
      {name:'线性外推轨（Ridge）',type:'line',data:ridgeY,symbolSize:9,
       lineStyle:{color:'#ffb020',width:2,type:'dashed'},
       itemStyle:{color:'#ffb020'},
       label:{show:true,color:'#ffb020',fontSize:11,position:'bottom',
         formatter:p=>p.value.toFixed(1)+'%'}}
    ]
  });
}

function drawBench(c, x){
  const sameSeries = A.curves[x.bucket];
  const ages = A.curve_ages;
  const age = clamp(AGECUR, ages[0], ages[ages.length-1]);
  let i = 0, best = 1e9;
  ages.forEach((a,k)=>{const d=Math.abs(a-age); if(d<best){best=d;i=k;}});
  const names = ['这辆车', '同车系同龄平均'];
  const vals = [x.p.rate*100, sameSeries[i]*100];
  const decayBands = A.decay_bands;
  // ⚠️ 必须夹住下标：车龄 90+ 月时 floor(96/12)=8，而车龄段只有 6 档，
  //    这里曾经越界导致整个页面初始化中断（白屏），是在 Node 桩化校验里抓到的。
  const bi = clamp(Math.floor(age/12), 0, decayBands.length-1);
  names.push('全市场同龄段中位'); vals.push(decayBands[bi].median*100);
  names.push('全市场整体中位'); vals.push(50.0);
  c.setOption({
    tooltip:{...TIP,trigger:'axis',axisPointer:{type:'shadow'},
      formatter:ps=>ps[0].name+'<br/><b>'+Number(ps[0].value).toFixed(2)+'%</b>'},
    grid:{left:110,right:44,top:12,bottom:24},
    xAxis:{type:'value',...AX,axisLabel:{...AX.axisLabel,
      formatter:v=>v.toFixed(0)+'%'}},
    yAxis:{type:'category',data:names.reverse(),...AX,
      axisLabel:{...AX.axisLabel,fontSize:11}},
    series:[{type:'bar',data:vals.reverse().map((v,k)=>({value:v,
      itemStyle:{color:['#5a6478','#5a6478','#8b93a8','#22c9a0'][k],
        borderRadius:3}})),barWidth:'46%',
      label:{show:true,position:'right',color:'#e6eaf2',fontSize:11,
        formatter:p=>p.value.toFixed(2)+'%'}}]
  });
}

/* =====================================================================
   5. 交互
   ===================================================================== */
const el = id => document.getElementById(id);
let AGECUR = 42;

function fmtWan(v){ return v===null||v===undefined ? '—' : v.toFixed(2)+' 万'; }

/** 选中车系后带出该车系的中位输入，避免用户从一个离谱的默认值开始。
 *  指导价滑杆的中点（49 万）对 20 万级的车毫无意义。 */
function seriesMedian(series){
  const g = (A.series_medians || {})[series];
  return g || { age_months: MED.age_months, mileage_wan: MED.mileage_wan,
                price_wan: MED.price_wan };
}

function applySeriesDefaults(series){
  const m = seriesMedian(series);
  el('f-age').value = Math.round(m.age_months);
  el('f-mil').value = Math.round(m.mileage_wan * 2) / 2;
  el('f-price').value = Math.round(m.price_wan * 2) / 2;
  el('f-price-num').value = m.price_wan.toFixed(1);
}

function render(){
  const raw = {
    series_cat: el('f-series').value,
    age_months: +el('f-age').value,
    mileage_wan: +el('f-mil').value,
    price_wan: +el('f-price').value,
    appearance_score: +el('f-appr').value
  };
  AGECUR = raw.age_months;
  const x = compute(raw);

  el('v-age').textContent = raw.age_months + ' 月（' + (raw.age_months/12).toFixed(1) + ' 年）';
  el('v-mil').textContent = raw.mileage_wan.toFixed(1) + ' 万公里';
  el('v-price').textContent = '车系中位 ' +
    seriesMedian(raw.series_cat).price_wan.toFixed(1) + ' 万元';

  const si = A.series_info[x.bucket] || {n:0,status:'—'};
  el('h-series').innerHTML = '命中建模档：<b>'+x.bucket+'</b>（训练样本 '+si.n+
    ' 条 · '+si.status+'）';

  el('o-rate').textContent = (x.p.rate*100).toFixed(2) + '%';
  const pw = x.p.rate * raw.price_wan;
  el('o-price').textContent = fmtWan(pw);
  el('o-price-formula').textContent = (x.p.rate*100).toFixed(2) + '% × ' +
    raw.price_wan.toFixed(2) + ' 万指导价';
  el('o-band').textContent = (x.p.lo*100).toFixed(1) + '% ~ ' +
    (x.p.hi*100).toFixed(1) + '%';
  el('o-band-price').textContent = '即 ' + fmtWan(x.p.lo*raw.price_wan) + ' ~ ' +
    fmtWan(x.p.hi*raw.price_wan);

  // 提示
  const warns = [];
  if (x.bucket === UNKNOWN){
    warns.push('车系「'+raw.series_cat+'」不在训练样本中，已退化为「'+UNKNOWN+
      '」档。预测仍会用车龄/里程/指导价，但<b>不含该车系的特异信息</b>，'+
      '等价于「同期同价位同级车的平均表现」。');
  } else if (!x.known){
    warns.push('车系「'+raw.series_cat+'」训练样本不足，已并入品牌档「'+x.bucket+
      '」。该档聚合了同品牌样本不足的车系，<b>不能代表该车系本身</b>。');
  }
  if (raw.age_months < 6){
    warns.push('车龄不足 6 个月，年均里程按 6 个月折算，已是模型训练中较少的区间');
  }
  el('o-warn').innerHTML = warns.length
    ? '<div class="note warn">⚠️ '+warns.join('<br/><br/>')+'</div>' : '';

  // 图表
  x.baseAgeLabel = '全市场平均车';
  renderCharts(x, raw);
}

function renderCharts(x, raw){
  drawWater(charts['c-water'], x);
  drawCurve(charts['c-curve'], x, raw);
  drawFuture(charts['c-future'], x, raw);
  drawBench(charts['c-bench'], x);

  el('l-curve').innerHTML = '蓝线为「'+x.bucket+'」在基准指导价档上的折旧曲线；'+
    '灰虚线为全市场各车龄段残值率中位数（按 12 个月分档）；'+
    '绿点为这辆车。';
  const dT = (x.treeFut.rate - x.p.rate)*100;
  const dR = (x.ridgeFut*100) - (x.p.ridge*100);
  el('l-future').innerHTML = '树模型年跌 <b>'+dT.toFixed(2)+'pp</b>，'+
    '线性轨年跌 <b>'+dR.toFixed(2)+'pp</b>，两轨差 '+
    Math.abs(dT-dR).toFixed(2) + 'pp。' +
    (Math.abs(dT-dR) > 1
      ? '　差 >1pp：该车龄处树模型已趋平（分段常数、学不到区间外的继续下降），应以线性轨为准。'
      : '　两轨一致，结论较稳。') +
    '　现实锚点：控制车系/里程/指导价后年跌 '+
    Math.abs(A.meta.decay_controlled_pp).toFixed(2) + 'pp。';
}

function init(){
  el('m-n').textContent = A.meta.n_train;
  el('m-snap').textContent = A.meta.snapshot;
  const mc = A.model_comparison['M2 LightGBM（精简 8 维）'] ||
             A.model_comparison['LightGBM'];
  el('m-mae').textContent = (mc.mae).toFixed(4);
  el('m-r2').textContent = (mc.r2).toFixed(4);

  el('t-target').textContent = '残值率 = ' + A.meta.target_definition;
  el('t-engine').textContent = A.meta.model +
    '，训练 '+A.meta.n_train+' 条；本页用导出的树结构在浏览器内直接遍历。';
  el('t-band').textContent = A.meta.interval_note;
  el('t-decay').textContent = '控制车系/里程/指导价后年跌 ' +
    A.meta.decay_controlled_pp.toFixed(2) + 'pp（回归口径）；' +
    '未控制构成的简单斜率为 ' + A.meta.decay_simple_pp.toFixed(2) +
    'pp，含车系构成效应，仅作上界参考。';
  el('t-snap').textContent = '全部数据采集于 ' + A.meta.snapshot +
    '，模型无法预测市场整体涨跌；「1 年后残值」只反映车龄/里程自然衰减。';
  const known = VOCAB.filter(s=>s!==UNKNOWN);
  el('t-cov').textContent = '训练集覆盖 '+known.length+' 个建模档（原始车系经样本量阈值并档）；'+
    '词表外车系统一落到「'+UNKNOWN+'」档并给出上方告警。';

  // 车系下拉：按样本量降序，未知档放最后
  const sel = el('f-series');
  const order = VOCAB.slice().sort((a,b)=>
    (b===UNKNOWN?-1:(A.series_info[b]||{n:0}).n) -
    (a===UNKNOWN?-1:(A.series_info[a]||{n:0}).n));
  order.forEach(s=>{
    const o = document.createElement('option');
    o.value = s;
    const n = (A.series_info[s]||{n:0}).n;
    o.textContent = s + '（n=' + n + '）';
    sel.appendChild(o);
  });
  sel.value = SCODE['蔚来ES6']!==undefined ? '蔚来ES6' : order[0];

  // 交互绑定
  ['f-age','f-mil','f-price','f-appr'].forEach(id=>
    el(id).addEventListener('input', render));
  // 指导价：滑杆与数字框双向同步
  el('f-price').addEventListener('input', ()=>{
    el('f-price-num').value = (+el('f-price').value).toFixed(1);
  });
  el('f-price-num').addEventListener('input', ()=>{
    const v = +el('f-price-num').value;
    if (isFinite(v) && v >= 5 && v <= 120) el('f-price').value = v;
  });
  // 换车系时带出该车系中位数，避免从离谱的默认值开始
  el('f-series').addEventListener('change', ()=>{
    applySeriesDefaults(el('f-series').value);
    render();
  });
  applySeriesDefaults(sel.value);
  render();
}
init();
</script>
</body>
</html>
"""

html = (HTML.replace("/*__ECHARTS__*/", echarts_js)
            .replace("/*__DATA__*/", DATA_JS))
with open(OUT, "w", encoding="utf-8") as f:
    f.write(html)
print(f"已生成页面：{OUT}（{os.path.getsize(OUT)/1024/1024:.2f} MB，自包含，双击可开）")
