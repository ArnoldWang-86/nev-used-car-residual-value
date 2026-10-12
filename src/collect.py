# -*- coding: utf-8 -*-
r"""瓜子二手车 · 合规采集器（新能源残值预测项目）

合规设计（本文件的核心约束，请勿修改）：
  1. 只访问 robots.txt 明确 Allow 的路径：/{城市}/{品牌}/、/{城市}/{品牌}/{车系}/、/car-detail/*.md
  2. robots.txt 有 `Disallow: /*?*`，因此任何含 `?` 的 URL 一律拒绝——
     采集前经过 `assert_path_only()` 硬断言，违反直接抛异常终止。
  3. 限速 2.0 秒/请求，单线程串行，不并发。
  4. 断点续传：已成功的车源号记入状态文件，重跑不重复抓取。

两阶段：
  Phase A  discovery —— 枚举 /{city}/{brand}/ 列表页，抽取车系路径与车源 ID
  Phase B  detail    —— 抓取 /car-detail/{id}.md，解析 YAML 结构化落库

用法：
  python src/collect.py --phase a          # 发现车系与车源 ID
  python src/collect.py --phase b          # 抓详情（可断点续传）
  python src/collect.py --phase b --limit 200
  python src/collect.py --stats            # 查看当前库存
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import sys
import threading
import time
from datetime import datetime, timezone

import requests

# --- 中文 Windows 控制台默认 GBK，强制 UTF-8 输出，避免打印特殊字符时崩溃 ---
try:
    import sys as _sys
    _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    _sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass


# ---------------------------------------------------------------- 路径与常量
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW = os.path.join(ROOT, "data", "raw")
DATA_CLEAN = os.path.join(ROOT, "data", "clean")
LOG_DIR = os.path.join(ROOT, "output")
for d in (DATA_RAW, DATA_CLEAN, LOG_DIR):
    os.makedirs(d, exist_ok=True)

DETAIL_JSONL = os.path.join(DATA_RAW, "detail_raw.jsonl")
FAIL_JSONL = os.path.join(DATA_RAW, "failures.jsonl")
DONE_IDS = os.path.join(DATA_RAW, "_done_ids.txt")
DISCOVERED = os.path.join(DATA_RAW, "discovered_targets.json")

# ---------------------------------------------------------------- 速率阶梯
# 实测教训（两次封禁的真实记录）：
#   2 秒/请求   → 家庭宽带 IP 被封（约 70 个请求后）
#   30 秒/请求  → 机房 IP 仅 8 个请求即被封
# 结论：IP 类型是决定因素，速率只决定早晚。因此采用
# 「安全起步 + 阶梯提速」，先用保守速率探路再提速到目标区间。
RATE_STAGES = [
    (20.0, 30),    # (基准间隔秒, 该档位需先连续成功的请求数)
    (9.0, 0),      # 目标档：9 秒 + ±3 秒抖动 → 实际落在 6~12 秒
]
RATE_JITTER_SEC = 3.0
FALLBACK_RATE = 45.0           # 出现风控信号后的保守速率

RATE_LIMIT_SEC = RATE_STAGES[0][0]   # 运行中会被自适应调整
# 不设单日请求上限：60~75 秒时代设过，现已提速，真正的防线是 detect_block 熔断。
DAILY_MAX_REQUESTS = 0
# 超时给足：列表页 330~400KB，实测多次出现 20~30 秒响应。
# 15 秒会把正常请求误判为失败并触发无谓重试。
TIMEOUT = 40
MAX_RETRY = 3

# 熔断阈值：列表页正常约 330~400 KB，详情 .md 正常 2.9~3.5 KB。
# 被拦截时站点 302 跳到 uc.guazi.com 验证码页，返回约 4 KB 的 HTML，
# 且 requests 跟随重定向后 status_code 仍是 200 —— 只看状态码会完全漏判。
CAPTCHA_HOST = "uc.guazi.com"
MIN_LIST_BYTES = 60_000       # 低于此值几乎必然是拦截页
MIN_MD_BYTES = 1_200


class BlockedError(RuntimeError):
    """检测到站点风控拦截，必须立即全量停止，避免把临时封禁打成长期封禁。"""


# 软熔断用：记录最近若干次成功的响应体积，用于识别"悄悄降质"
_recent_sizes: list[int] = []


def detect_soft_block(size: int, url: str) -> None:
    """软熔断：识别「状态码 200 但内容塌陷」的静默降级。

    风控未必总是跳验证码页 —— 也可能返回一个体积骤减的空壳页。
    这种情况状态码正常、最终 URL 也正常，硬指标全部通过，
    但数据其实是空的。这里用「前 10 次成功响应的中位数」做基线，
    当前体积低于基线 40% 就判定异常。
    """
    if not url.endswith(".md"):
        return
    if len(_recent_sizes) < 10:
        return
    base = sorted(_recent_sizes[-40:])[len(_recent_sizes[-40:]) // 2]
    if base > 0 and size < base * 0.4:
        raise BlockedError(
            f"疑似静默降级：本次响应仅 {size:,} 字节，"
            f"而近期成功响应中位数约 {base:,} 字节。\n"
            f"  触发 URL: {url}\n"
            f"  状态码正常但内容塌陷，按风控处理并立即停止。"
        )

BASE = "https://www.guazi.com"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/128.0 Safari/537.36")
HEADERS = {
    "User-Agent": UA,
    "Accept": "text/html,text/markdown,text/plain,*/*",
    "Accept-Language": "zh-CN,zh;q=0.9",
    "Connection": "close",
}

# 品牌 slug
#
# 采集目标依据岗位业务确定：以蔚来自家为核心，直接竞品为对标。
# 所有 slug 已在瓜子上海站【实测确认】（见 output/brand_slug_check.txt）：
#   蔚来 weilai(车系11) 乐道 ledao(3) 特斯拉 tesila(6) 理想 lixiang(8)
#   小鹏 xpqc(12) 极氪 jike(10) 比亚迪 byd(77) 问界 aito(18) 小米 xiaomiqiche(5)
#
# 实测无效的 slug（勿再加回，会浪费请求）：
#   zeekr(应写 jike) / yinghuochong / firefly(萤火虫) / zhiji(智己) / aion(埃安)
# 腾势 tengshi 在品牌导航中出现但本轮请求超时未判定，列为待验证。
BRANDS_PRIMARY = [
    "weilai",       # 蔚来（岗位核心，优先全量采集）
    "ledao",        # 乐道（蔚来子品牌，BaaS 体系共用）
]
BRANDS_RIVALS = [
    "tesila",       # 特斯拉 Model 3/Y
    "lixiang",      # 理想 L6-L9 / MEGA
    "xpqc",         # 小鹏 P7+/G6/G9
    "jike",         # 极氪 001/007/7X/9X
    "byd",          # 比亚迪 汉/海豹/唐
    "aito",         # 问界 M5/M7/M9
    "xiaomiqiche",  # 小米 SU7/YU7
]
BRANDS_PENDING = ["tengshi"]
BRANDS = BRANDS_PRIMARY + BRANDS_RIVALS

# 城市：从首页导航实测解析出的 89 个 + 主要城市补全
CITIES = [
    "sh", "bj", "gz", "sz", "hz", "cd", "nj", "wh", "tj", "cq",
    "su", "xa", "zz", "cs", "dg", "qd", "sjz", "heb", "dl", "sy",
    "xm", "hf", "fz", "jn", "km", "nc", "gy", "nn", "wuxi", "ningbo",
    "wenzhou", "foshan", "zhuhai", "zhongshan", "huizhou", "jiaxing",
    "shaoxing", "taizhou", "jinhua", "quanzhou", "yantai", "weifang",
    "zibo", "linyi", "luoyang", "xuzhou", "changzhou", "nantong",
    "yangzhou", "tangshan", "baoding", "langfang", "cangzhou", "handan",
    "dazhou", "mianyang", "liuzhou", "guilin", "zunyi", "lanzhou",
    "yinchuan", "wuhu", "bengbu", "anqing", "jiujiang", "ganzhou",
    "yichang", "xiangyang", "zhuzhou", "yueyang", "changde", "shantou",
    "zhanjiang", "jiangmen", "qingyuan", "jieyang", "an shan",
]
CITIES = [c for c in dict.fromkeys(CITIES) if " " not in c]

# 车况相关正则
RE_SOH = re.compile(r"电池健康度\s*([0-9]{1,3}(?:\.[0-9])?)\s*%")
RE_SOH_ALT = re.compile(r"SOH\s*[:：]?\s*([0-9]{1,3}(?:\.[0-9])?)\s*%", re.I)
# 文案实际写法是「75度电池」「100度电池」，不是「75度电」——
# 早期正则写死「度电」，导致覆盖率恒为 0。
RE_KWH_CN = re.compile(r"([0-9]{2,3})\s*度(?:电池|电)")
RE_CHEM = re.compile(r"(三元锂|磷酸铁锂|锰酸锂|钛酸锂)")
RE_CLTC = re.compile(r"(?:CLTC\s*续航|续航)\s*([0-9]{3,4})\s*(?:km|公里)", re.I)
# 车型名里的续航口径，如「2020款 415KM 六座版」
RE_MODEL_RANGE = re.compile(r"([0-9]{3,4})\s*(?:KM|km|公里)")
RE_CLAIM = re.compile(r"理赔\s*([0-9]+)\s*次")
# 结构件损伤：必须区分「结构件受损」与「骨架完好」两种表述。
# 早期版本只要出现「骨架」就判定损伤，把「骨架与三电完好」误判成事故车。
RE_STRUCT_BAD = re.compile(
    r"(纵梁|骨架|结构件|A柱|B柱|C柱|减震器座|底盘)[^。；，]{0,14}"
    r"(切割|焊接|变形|受损|损伤|校正|修复)")
RE_STRUCT_OK = re.compile(r"(骨架|结构件|三电)[^。；，]{0,8}(完好|无损|正常)")
RE_SWAP = re.compile(r"(换电|车电分离|BaaS|电池租赁)", re.I)
RE_DETAIL_ID = re.compile(r"/car-detail/(c\d+)\.(?:html|md)")
RE_SERIES_PATH = re.compile(r'href="(/[a-z0-9_-]+/[a-z0-9_-]+/[a-z0-9_-]+/)"', re.I)


# ---------------------------------------------------------------- 合规断言
def assert_path_only(url: str) -> None:
    """拒绝任何带查询参数的 URL —— 遵循 robots 的 `Disallow: /*?*`。

    这是合规红线，不是可选检查。任何绕过此断言的改动都违反数据使用约定。
    """
    if "?" in url or "#" in url:
        raise AssertionError(
            f"[合规拦截] URL 含查询参数，robots.txt 的 `Disallow: /*?*` 禁止抓取：{url}"
        )
    if not url.startswith(BASE):
        raise AssertionError(f"[合规拦截] 非目标站点域名：{url}")


# ---------------------------------------------------------------- HTTP
_last_req = [0.0]
_req_count = [0]
_ok_streak = [0]        # 当前速率下的连续成功请求数
_stage = [0]            # RATE_STAGES 的当前档位下标
_lock = threading.Lock()

RATE_LIMIT_SEC = RATE_STAGES[0][0]


def _adapt_rate() -> None:
    """阶梯提速：当前档位连续成功达到阈值后，切到下一档更快的速率。"""
    i = _stage[0]
    if i >= len(RATE_STAGES) - 1:
        return
    need = RATE_STAGES[i][1]
    if need and _ok_streak[0] >= need:
        _stage[0] = i + 1
        globals()["RATE_LIMIT_SEC"] = RATE_STAGES[_stage[0]][0]
        log(f"    [提速] 已连续成功 {_ok_streak[0]} 个请求，"
            f"间隔降至 {RATE_LIMIT_SEC:.0f} 秒")
        _ok_streak[0] = 0


def _fallback_rate(reason: str) -> None:
    """出现风控信号时退回最保守速率。"""
    if RATE_LIMIT_SEC < FALLBACK_RATE:
        globals()["RATE_LIMIT_SEC"] = FALLBACK_RATE
        _stage[0] = 0
        _ok_streak[0] = 0
        log(f"    [降速] {reason} → 间隔退回 {FALLBACK_RATE:.0f} 秒")


def _throttle() -> None:
    """全局串行限速：保证任意两次请求之间至少间隔 RATE_LIMIT_SEC + 抖动。

    单日上限默认关闭（DAILY_MAX_REQUESTS = 0）。
    真正拦住失控的是 fetch() 里的风控熔断。
    """
    with _lock:
        if DAILY_MAX_REQUESTS and _req_count[0] >= DAILY_MAX_REQUESTS:
            raise BlockedError(
                f"已达单日请求上限 {DAILY_MAX_REQUESTS} 个，主动停止。\n"
                f"  已抓数据已落盘，重跑同一命令即可断点续传。"
            )
        gap = RATE_LIMIT_SEC + random.uniform(0, RATE_JITTER_SEC)
        wait = gap - (time.time() - _last_req[0])
        if wait > 0:
            time.sleep(wait)
        _last_req[0] = time.time()
        _req_count[0] += 1
        if _req_count[0] % 20 == 0:
            log(f"    [速率] 本次已发 {_req_count[0]} 个请求 "
                f"| 当前间隔 {RATE_LIMIT_SEC:.0f} 秒 | 连续成功 {_ok_streak[0]}")


def detect_block(r: requests.Response, url: str) -> None:
    """识别风控拦截页并抛出 BlockedError。

    站点被触发风控时会 302 跳转到 `uc.guazi.com/.../captcha`，
    requests 跟随重定向后 `status_code` 依然是 200，
    只看状态码会误判成"抓取成功但页面没数据"——这正是上一轮失控的原因。
    因此这里同时检查：最终 URL 域名 + 响应体积。
    """
    final = r.url or ""
    if CAPTCHA_HOST in final or "captcha" in final.lower():
        raise BlockedError(
            f"被风控拦截：跳转到验证码页 {final[:120]}\n"
            f"  触发 URL: {url}\n"
            f"  必须立即停止，等待封禁解除（通常数小时至数天）。"
        )
    size = len(r.content)
    is_md = url.endswith(".md")
    floor = MIN_MD_BYTES if is_md else MIN_LIST_BYTES
    if r.status_code == 200 and size < floor:
        raise BlockedError(
            f"疑似风控拦截：响应仅 {size:,} 字节（正常下限 {floor:,}），"
            f"且状态码为 200。\n  触发 URL: {url}\n"
            f"  必须立即停止。"
        )


def fetch(url: str, session: requests.Session) -> requests.Response | None:
    """限速 + 指数退避抓取。检测到风控直接抛 BlockedError 终止全流程。"""
    assert_path_only(url)
    for attempt in range(MAX_RETRY):
        _throttle()
        try:
            r = session.get(url, headers=HEADERS, timeout=TIMEOUT, allow_redirects=True)
            detect_block(r, url)
            # 走到这里说明没有被拦截，累加连续成功数用于阶梯提速
            _ok_streak[0] += 1
            _adapt_rate()
            if r.status_code == 200:
                detect_soft_block(len(r.content), url)
                _recent_sizes.append(len(r.content))
                return r
            if r.status_code in (429, 503):
                # 限流是明确的风控信号：退回保守速率，并计入失败
                _ok_streak[0] = 0
                _fallback_rate(f"HTTP {r.status_code}")
                wait = 20 * (2 ** attempt) + random.uniform(0, 10)
                log(f"    [限流 {r.status_code}] 退避 {wait:.1f}s")
                time.sleep(wait)
                continue
            if r.status_code == 404:
                return r
            log(f"    [HTTP {r.status_code}] {url}")
        except BlockedError:
            raise
        except Exception as e:  # noqa: BLE001
            _ok_streak[0] = 0
            if attempt == MAX_RETRY - 1:
                log(f"    [失败] {type(e).__name__}: {str(e)[:80]}")
                return None
            time.sleep(5 * (attempt + 1))
    return None


_LOG_FH = [None]


def log(msg: str) -> None:
    """同时输出到控制台与日志文件。

    日志文件每次写入都 flush —— 早期版本依赖 `python ... | Tee-Object` 落盘，
    管道缓冲导致日志长时间不更新，看起来像进程卡死，误导了排查方向。
    """
    line = f"{datetime.now().strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    if _LOG_FH[0] is None:
        path = os.path.join(LOG_DIR, f"collect_{datetime.now():%Y%m%d_%H%M%S}.log")
        _LOG_FH[0] = open(path, "w", encoding="utf-8")
        print(f"[日志] {path}", flush=True)
    _LOG_FH[0].write(line + "\n")
    _LOG_FH[0].flush()


# ---------------------------------------------------------------- 解析
def parse_md(text: str) -> dict:
    """把详情 .md 解析为结构化记录。"""
    d: dict = {}

    # --- YAML front matter ---
    fm = re.match(r"^---\n(.*?)\n---\n", text, re.S)
    head = fm.group(1) if fm else text[:1200]
    m = re.search(r"^title:\s*(.*)$", head, re.M)
    d["page_title"] = m.group(1).strip() if m else None
    m = re.search(r"^generatedAt:\s*'?([^'\n]+)'?", head, re.M)
    d["generated_at"] = m.group(1).strip() if m else None
    # highlights 是 YAML 多行标量，遇到下一个顶层键（如 generatedAt:）即结束。
    # 注意用 [A-Za-z_] 而不是 [a-z_] —— generatedAt 含大写，漏了会把整整一段吞进来。
    m = re.search(r"^highlights:\s*>?-?\s*\n?(.*?)(?=\n[A-Za-z_][A-Za-z0-9_]*:|\Z)",
                  head, re.S | re.M)
    highlights = re.sub(r"\s+", " ", m.group(1)).strip() if m else ""
    d["highlights"] = highlights

    # --- 正文结构字段 ---
    def g(key: str) -> str | None:
        # 行首锚定 + 只允许一个可选空格，避免 `^id:` 误匹配 `guide_price:` 这类
        # 「以 id 结尾的键名」——这曾导致 12 条记录里 2 条 car_id 解析成
        # "uide_price"，进而在去重时被丢掉。
        mm = re.search(rf"^{re.escape(key)}:[ \t]*(.*)$", text, re.M)
        return mm.group(1).strip() if mm else None

    # 车源号：详情页固定写作 `id:c173358342181138`（无空格），单独处理
    mc = re.search(r"^id:[ \t]*(c\d+)", text, re.M)
    d["car_id"] = mc.group(1) if mc else ""
    d["manufacturer"] = g("manufacturer")
    d["production_type"] = g("production_type")
    d["brand"] = g("brand")
    d["series"] = g("series")
    d["model"] = g("model")

    # --- 价格 ---
    fp = g("full_payment")
    gp = g("guide_price")
    d["full_payment"] = _num(fp)
    d["guide_price"] = _num(gp)

    # --- 上牌与使用 ---
    d["first_register"] = g("first_register")
    d["mileage"] = _num(g("mileage"))
    d["transfer_times"] = _num(g("transfer_times"))
    d["city_name"] = g("city")

    # --- 能源/技术 ---
    d["energy_type"] = g("type")
    d["engine"] = g("engine")
    d["transmission"] = g("transmission")
    d["emission_standard"] = g("emission_standard")
    d["drive_mode"] = g("drive_mode")

    # --- 车况 ---
    d["appearance_score"] = _num(g("appearance_score"))
    d["condition_grade"] = g("condition_grade")
    d["condition_desc"] = g("condition_desc")

    # --- 从文本抽取的派生字段 ---
    # 这些信息只存在于平台生成的自然语言里（highlights / description / 正文），
    # 因此对【整篇文本】做抽取，而不只搜 highlights ——
    # 实测「电池健康度87%」可能出现在 description 或正文任意位置。
    d["battery_kwh"] = _extract_kwh_model(d.get("model") or "")
    d["model_range_km"] = _first_float(RE_MODEL_RANGE.search(d.get("model") or ""))
    d["v_battery_kwh"] = _first_float(RE_KWH_CN.search(text))
    d["soh_pct"] = (_first_float(RE_SOH.search(text))
                    or _first_float(RE_SOH_ALT.search(text)))
    mc = RE_CHEM.search(text)
    d["battery_chem"] = mc.group(1) if mc else None
    d["cltc_km"] = _first_float(RE_CLTC.search(text))
    d["claim_times"] = (_first_float(RE_CLAIM.search(d.get("condition_desc") or ""))
                        or _first_float(RE_CLAIM.search(text)))
    # 结构件判定：先排除「骨架完好」这类否定表述，再判损伤关键词。
    # 早期版本只要文本含「骨架」就标损伤，把「骨架与三电完好」误判成事故车。
    has_bad = bool(RE_STRUCT_BAD.search(text))
    has_ok = bool(RE_STRUCT_OK.search(text))
    d["has_structural_repair"] = bool(has_bad and not has_ok)
    d["is_battery_swap"] = bool(RE_SWAP.search(text))

    # --- 目标变量 ---
    if d["full_payment"] and d["guide_price"]:
        d["residual_rate"] = round(d["full_payment"] / d["guide_price"], 6)
    else:
        d["residual_rate"] = None

    # --- 派生特征 ---
    d["age_months"] = _age_months(d.get("first_register"))
    if d["age_months"] and d["mileage"]:
        d["mileage_per_year"] = round(d["mileage"] / (d["age_months"] / 12.0), 1)
    else:
        d["mileage_per_year"] = None
    d["is_first_owner"] = (d["transfer_times"] == 0) if d["transfer_times"] is not None else None

    return d


def _num(v) -> float | None:
    if v is None:
        return None
    m = re.search(r"-?[0-9]+(?:\.[0-9]+)?", str(v).replace(",", ""))
    return float(m.group()) if m else None


def _first_float(m) -> float | None:
    return float(m.group(1)) if m else None


def _extract_kwh_model(model: str) -> float | None:
    """从 model 字段抽电池容量，如 '2023款 100kWh' -> 100.0

    注意：只有部分品牌把容量写进车型名（蔚来写 75kWh/100kWh，特斯拉不写）。
    抽不到是正常情况，覆盖率会被如实统计，不做任何猜测性填充。
    """
    m = re.search(r"([0-9]{2,3}(?:\.[0-9])?)\s*kWh", model, re.I)
    return float(m.group(1)) if m else None


def _age_months(first_register: str | None) -> int | None:
    if not first_register:
        return None
    m = re.match(r"(\d{4})-(\d{2})", str(first_register))
    if not m:
        return None
    y, mo = int(m.group(1)), int(m.group(2))
    now = datetime.now()
    return (now.year - y) * 12 + (now.month - mo)


def probe(session: requests.Session) -> bool:
    """单请求探测：判断 IP 级封禁是否已解除。

    只发 1 个请求，绝不做循环重试 —— 被封状态下反复请求会把临时封禁拉长。
    返回 True 表示已解封。
    """
    url = f"{BASE}/sh/weilai/"
    # 探测必须绕过 fetch 的限速/重试逻辑，这里手工控制
    try:
        r = session.get(url, headers=HEADERS, timeout=TIMEOUT, allow_redirects=True)
    except Exception as e:  # noqa: BLE001
        log(f"[探测] 请求失败 {type(e).__name__}: {str(e)[:100]}")
        return False
    size = len(r.content)
    final = r.url or ""
    blocked = (CAPTCHA_HOST in final) or ("captcha" in final.lower()) or size < MIN_LIST_BYTES
    log("=" * 66)
    log(f"[探测] {url}")
    log(f"       状态码 {r.status_code} | 体积 {size:,} 字节 | 最终 URL {final[:90]}")
    if blocked:
        log("       结论：仍被拦截（跳验证码 / 体积异常）")
        log("       请等待数小时后再探测，切勿连续试探。")
        return False
    ids = set(RE_DETAIL_ID.findall(r.text))
    log(f"       结论：已解封，列表页解析到 {len(ids)} 条车源")
    return True


def run_guarded(fn, *args, **kwargs):
    """执行采集步骤，捕获风控熔断并给出明确收尾。"""
    try:
        return fn(*args, **kwargs)
    except BlockedError as e:
        log("")
        log("!" * 66)
        log(f"[熔断] {e}")
        log("!" * 66)
        log("已立即停止全部采集。已抓到的数据已落盘，可断点续传。")
        return None


# ---------------------------------------------------------------- 状态
def load_done_ids() -> set:
    if not os.path.exists(DONE_IDS):
        return set()
    with open(DONE_IDS, encoding="utf-8") as f:
        return {ln.strip() for ln in f if ln.strip()}


def append_done(car_id: str) -> None:
    with open(DONE_IDS, "a", encoding="utf-8") as f:
        f.write(car_id + "\n")


def append_jsonl(path: str, obj: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------- Phase A
def phase_a(session: requests.Session, cities: list[str], brands: list[str]) -> dict:
    """枚举列表页，收集「车系路径」与「车源 ID」。"""
    log("=" * 76)
    log("Phase A · 发现车系路径与车源 ID")

    series_paths: set = set()
    car_ids: set = set()
    combo_log = []

    for city in cities:
        for brand in brands:
            url = f"{BASE}/{city}/{brand}/"
            r = fetch(url, session)
            if r is None:
                combo_log.append({"url": url, "status": None, "series": 0, "ids": 0})
                continue
            if r.status_code == 404:
                combo_log.append({"url": url, "status": 404, "series": 0, "ids": 0})
                continue
            if r.status_code != 200:
                combo_log.append({"url": url, "status": r.status_code, "series": 0, "ids": 0})
                continue

            t = r.text
            ids = set(RE_DETAIL_ID.findall(t))
            sp = {x for x in RE_SERIES_PATH.findall(t) if x.lower().startswith(f"/{city}/")}
            series_paths |= sp
            car_ids |= ids
            combo_log.append({"url": url, "status": 200, "series": len(sp), "ids": len(ids)})
            log(f"  [{city}/{brand}] 车源 {len(ids):>3} 条 | 车系路径 {len(sp):>2} | "
                f"累计 ID {len(car_ids)}")

    # 再抓一遍车系路径（车系页会带来品牌页没有的车源）
    log(f"\n  进入车系路径抓取，共 {len(series_paths)} 个")
    for i, sp in enumerate(sorted(series_paths), 1):
        url = BASE + sp
        r = fetch(url, session)
        if r is None or r.status_code != 200:
            continue
        ids = set(RE_DETAIL_ID.findall(r.text))
        new = ids - car_ids
        car_ids |= ids
        # 车系页里还可能有更细的车系路径
        sp2 = {x for x in RE_SERIES_PATH.findall(r.text) if x.lower().startswith(f"/{sp.split('/')[1]}/")}
        series_paths |= sp2
        if new:
            log(f"  [{i}/{len(series_paths)}] {sp} 新增 {len(new)} 条，累计 {len(car_ids)}")

    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "cities": cities, "brands": brands,
        "series_paths": sorted(series_paths),
        "car_ids": sorted(car_ids),
        "combos": combo_log,
    }
    with open(DISCOVERED, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    log(f"\nPhase A 完成：车系路径 {len(series_paths)} 个，车源 ID {len(car_ids)} 个")
    log(f"已保存 {DISCOVERED}")
    return out


# ---------------------------------------------------------------- Phase B
def phase_b(session: requests.Session, limit: int | None = None) -> int:
    log("=" * 76)
    log("Phase B · 抓取车辆详情 .md")

    if not os.path.exists(DISCOVERED):
        log("  未找到 discovered_targets.json，请先跑 --phase a")
        return 0
    with open(DISCOVERED, encoding="utf-8") as f:
        disc = json.load(f)
    car_ids = disc.get("car_ids", [])
    if not car_ids:
        log("  车源 ID 为空")
        return 0

    done = load_done_ids()
    todo = [c for c in car_ids if c not in done]
    if limit:
        todo = todo[:limit]
    log(f"  待抓 {len(todo)} 条（已完成 {len(done)} 条）")

    ok = 0
    for i, cid in enumerate(todo, 1):
        url = f"{BASE}/car-detail/{cid}.md"
        r = fetch(url, session)
        if r is None or r.status_code != 200:
            append_jsonl(FAIL_JSONL, {"car_id": cid, "url": url,
                                      "status": None if r is None else r.status_code,
                                      "ts": datetime.now().isoformat()})
            continue

        rec = parse_md(r.text)
        if not rec.get("car_id"):
            # car_id 是面板追踪与去重的主键，缺失意味着解析规则与站点结构漂移
            log(f"    [警告] {cid} 解析不出 car_id，跳过（页面结构可能已变）")
            append_jsonl(FAIL_JSONL, {"car_id": cid, "url": url, "status": 200,
                                      "reason": "car_id_parse_failed",
                                      "ts": datetime.now().isoformat()})
            continue
        rec["snapshot_date"] = datetime.now().strftime("%Y-%m-%d")
        rec["crawled_at"] = datetime.now().isoformat(timespec="seconds")
        rec["source_url"] = url
        append_jsonl(DETAIL_JSONL, rec)
        append_done(cid)
        ok += 1

        if i % 20 == 0 or i == len(todo):
            log(f"  [{i}/{len(todo)}] 成功 {ok} | 最新 {rec.get('brand')} "
                f"{rec.get('series')} 残值率 "
                f"{rec.get('residual_rate') if rec.get('residual_rate') is None else round(rec['residual_rate'], 3)}")

    log(f"\nPhase B 完成：本次成功 {ok} 条")
    export_csv()
    return ok


# ---------------------------------------------------------------- 导出
def export_csv() -> None:
    """把 JSONL 汇总为 CSV（按 car_id 去重，保留最新一条）。"""
    if not os.path.exists(DETAIL_JSONL):
        return
    rows: dict[str, dict] = {}
    with open(DETAIL_JSONL, encoding="utf-8") as f:
        for ln in f:
            try:
                o = json.loads(ln)
            except json.JSONDecodeError:
                continue
            cid = o.get("car_id")
            if cid:
                rows[cid] = o
    if not rows:
        return
    flat = []
    for o in rows.values():
        o = dict(o)
        o.pop("highlights", None)      # 原文太长，另存
        flat.append(o)
    cols = sorted({k for o in flat for k in o})
    path = os.path.join(DATA_CLEAN, "cars.csv")
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=cols)
        wtr.writeheader()
        for o in flat:
            wtr.writerow(o)
    log(f"  已导出 {path}（{len(flat)} 条唯一车源）")


def stats() -> None:
    if not os.path.exists(DETAIL_JSONL):
        log("尚无数据")
        return
    recs = []
    with open(DETAIL_JSONL, encoding="utf-8") as f:
        for ln in f:
            try:
                recs.append(json.loads(ln))
            except json.JSONDecodeError:
                pass
    n = len(recs)
    if not n:
        log("尚无数据")
        return
    ev = [r for r in recs if r.get("energy_type") == "新能源"]
    rr = [r["residual_rate"] for r in recs if r.get("residual_rate")]
    soh = [r for r in recs if r.get("soh_pct")]
    kwh = [r for r in recs if r.get("battery_kwh")]
    swap = [r for r in recs if r.get("is_battery_swap")]
    brands: dict[str, int] = {}
    for r in recs:
        brands[r.get("brand") or "?"] = brands.get(r.get("brand") or "?", 0) + 1
    log("=" * 60)
    log(f"总记录        {n}")
    log(f"新能源        {len(ev)}  ({len(ev)/n:.1%})")
    log(f"有残值率      {len(rr)}  ({len(rr)/n:.1%})")
    log(f"有电池容量    {len(kwh)}  ({len(kwh)/n:.1%})")
    log(f"有 SOH        {len(soh)}  ({len(soh)/n:.1%})   <- 关键覆盖率")
    log(f"换电/BaaS     {len(swap)}  ({len(swap)/n:.1%})")
    if rr:
        import statistics as st
        log(f"残值率 中位数 {st.median(rr):.3f}  均值 {st.mean(rr):.3f}  "
            f"区间 [{min(rr):.3f}, {max(rr):.3f}]")
    log("品牌分布 Top10: " + ", ".join(
        f"{k}={v}" for k, v in sorted(brands.items(), key=lambda x: -x[1])[:10]))


# ---------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description="瓜子二手车合规采集器")
    ap.add_argument("--phase", choices=["a", "b"], help="a=发现  b=抓详情")
    ap.add_argument("--limit", type=int, default=None, help="Phase B 限制条数")
    ap.add_argument("--cities", default=None, help="逗号分隔城市 slug，覆盖默认")
    ap.add_argument("--brands", default=None, help="逗号分隔品牌 slug，覆盖默认")
    ap.add_argument("--stats", action="store_true", help="查看当前库存统计")
    ap.add_argument("--probe", action="store_true",
                    help="单请求探测封禁是否解除（不发循环请求）")
    a = ap.parse_args()

    if a.stats:
        stats()
        return

    session = requests.Session()
    cities = a.cities.split(",") if a.cities else CITIES
    brands = a.brands.split(",") if a.brands else BRANDS

    if a.probe:
        ok = probe(session)
        sys.exit(0 if ok else 2)

    if a.phase == "a":
        run_guarded(phase_a, session, cities, brands)
    elif a.phase == "b":
        run_guarded(phase_b, session, a.limit)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
