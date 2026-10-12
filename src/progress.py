# -*- coding: utf-8 -*-
r"""progress.py —— 采集进度与剩余时间实时看板

只读脚本：不发起任何网络请求，不修改采集状态，可在采集运行时安全并行执行。

它读取：
  - output/collect_*.log        采集日志（取最新一份）
  - data/raw/discovered_targets.json   Phase A 发现的车源 ID 清单
  - data/raw/_done_ids.txt      已成功抓取的详情 ID
  - data/raw/detail_raw.jsonl   已抓取记录（用于真实速率与字段覆盖率统计）

用法：
    python src/progress.py            # 看一次
    python src/progress.py --watch    # 每 30 秒刷新
    python src/progress.py --interval 15
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import time
from datetime import datetime, timedelta

# 中文 Windows 控制台默认 GBK，配合 .bat 里的 chcp 65001 强制 UTF-8 输出，
# 否则中文标题与进度条会显示成乱码。
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(ROOT, "output")
RAW = os.path.join(ROOT, "data", "raw")
DISCOVERED = os.path.join(RAW, "discovered_targets.json")
DONE = os.path.join(RAW, "_done_ids.txt")
DETAIL = os.path.join(RAW, "detail_raw.jsonl")
FAILS = os.path.join(RAW, "failures.jsonl")

# 与 collect.py 保持一致的限速参数（若改动请同步）
RATE_LIMIT_SEC = 60.0
RATE_JITTER_SEC = 15.0
DAILY_MAX = 0          # 0 表示不限；与 collect.py 的 DAILY_MAX_REQUESTS 保持一致
PHASE_A_OVERHEAD = 15.0

BAR_W = 34


def human(seconds: float | None) -> str:
    if seconds is None or seconds != seconds or seconds < 0:
        return "—"
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}小时{m}分"
    if m:
        return f"{m}分{s}秒"
    return f"{s}秒"


def bar(done: int, total: int, width: int = BAR_W) -> str:
    if total <= 0:
        return "[" + " " * width + "]"
    frac = min(1.0, done / total)
    fill = int(round(frac * width))
    return "[" + "#" * fill + "." * (width - fill) + f"] {frac*100:5.1f}%"


def latest_log() -> str | None:
    files = sorted(glob.glob(os.path.join(LOG_DIR, "collect_*.log")),
                   key=os.path.getmtime)
    return files[-1] if files else None


def parse_log(path: str) -> dict:
    """从日志里还原 Phase A 的枚举进度与已发现车源数。"""
    info = {"phase": None, "enum_done": 0, "enum_total": 0, "series_done": 0,
            "series_total": 0, "last_ts": None, "last_line": "",
            "car_ids_seen": 0, "blocked": False, "done_msg": False}
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            lines = f.read().splitlines()
    except OSError:
        return info

    for ln in lines:
        ts = ln[:8]
        if re.match(r"^\d{2}:\d{2}:\d{2}$", ts):
            info["last_ts"] = ts
        info["last_line"] = ln

        if "Phase A" in ln:
            info["phase"] = "A"
        elif "Phase B" in ln:
            info["phase"] = "B"

        # 枚举行： [sh/weilai] 车源  40 条 | 车系路径 11 | 累计 ID 40
        m = re.search(r"\]\s*车源\s+(\d+)\s*条.*累计 ID (\d+)", ln)
        if m:
            info["enum_done"] += 1
            info["car_ids_seen"] = int(m.group(2))

        # 车系行： [3/11] /sh/weilai/weilaiet9/ 新增 6 条，累计 66
        m = re.search(r"\[(\d+)/(\d+)\]\s+(/\S+).*累计 (\d+)", ln)
        if m:
            info["series_done"] = int(m.group(1))
            info["series_total"] = int(m.group(2))
            info["car_ids_seen"] = int(m.group(4))

        # 详情行： [12/200] 成功 12 | 最新 ...
        m = re.search(r"\[(\d+)/(\d+)\]\s+成功\s+(\d+)", ln)
        if m:
            info["detail_done"] = int(m.group(1))
            info["detail_total"] = int(m.group(2))
            info["detail_ok"] = int(m.group(3))

        if "熔断" in ln or "被风控拦截" in ln or "captcha" in ln:
            info["blocked"] = True
        if "Phase A 完成" in ln or "Phase B 完成" in ln:
            info["done_msg"] = True
    return info


def count_lines(path: str) -> int:
    if not os.path.exists(path):
        return 0
    n = 0
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            for _ in f:
                n += 1
    except OSError:
        return 0
    return n


def unique_detail_ids() -> tuple[int, dict]:
    """统计已成功抓取的唯一车源数，以及关键字段覆盖率。"""
    if not os.path.exists(DETAIL):
        return 0, {}
    ids, cols = set(), {}
    keys = ["residual_rate", "battery_kwh", "soh_pct", "is_battery_swap",
            "condition_grade", "first_register"]
    have = {k: 0 for k in keys}
    try:
        with open(DETAIL, encoding="utf-8", errors="replace") as f:
            for ln in f:
                try:
                    o = json.loads(ln)
                except json.JSONDecodeError:
                    continue
                cid = o.get("car_id")
                if cid:
                    ids.add(cid)
                for k in keys:
                    v = o.get(k)
                    if v is not None and v != "" and v is not False:
                        have[k] += 1
    except OSError:
        return 0, {}
    tot = max(1, len(ids))
    cols = {k: have[k] / tot for k in keys}
    return len(ids), cols


def load_discovered() -> dict:
    if not os.path.exists(DISCOVERED):
        return {}
    try:
        with open(DISCOVERED, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def render(once: bool = True) -> None:
    os.system("cls" if os.name == "nt" else "clear")
    now = datetime.now()
    print("=" * 74)
    print(f"  瓜子二手车 · 采集进度看板        {now:%Y-%m-%d %H:%M:%S}")
    print("=" * 74)

    lp = latest_log()
    if not lp:
        print("\n  暂无采集日志。先运行：python src/collect.py --phase a\n")
        return
    info = parse_log(lp)
    print(f"\n  日志：{os.path.basename(lp)}")
    # 用日志文件的 mtime 判断活跃度，而不是解析日志里的时间戳 ——
    # 日志只有 HH:MM:SS 没有日期，跨越午夜后会算出负数间隔。
    try:
        idle = now.timestamp() - os.path.getmtime(lp)
        mark = "  (可能已停止或正在等待限速)" if idle > 180 else ""
        print(f"  最近活动：{human(idle)}前{mark}")
    except OSError:
        pass

    disc = load_discovered()
    car_ids = disc.get("car_ids", [])
    combos = disc.get("combos", [])
    n_discovered = len(car_ids)

    done_ids = count_lines(DONE)
    uniq_detail, cov = unique_detail_ids()
    n_fail = count_lines(FAILS)

    cities = disc.get("cities", [])
    brands = disc.get("brands", [])
    series = disc.get("series_paths", [])

    # ---------------- 阶段判断 ----------------
    if not car_ids:
        phase = "A（枚举列表页）"
    elif uniq_detail == 0 and done_ids == 0:
        phase = "A 已完成 → 待启动 B（抓详情）"
    elif done_ids < n_discovered:
        phase = "B（抓详情中）"
    else:
        phase = "B（已完成）"

    print(f"\n  ── 当前阶段：{phase} " + "─" * max(0, 46 - len(phase)))

    # ---------------- Phase A ----------------
    if combos:
        enum_total = len(combos)
        enum_done = min(info["enum_done"], enum_total)
        print(f"\n  【Phase A · 枚举列表页（{len(cities)} 城 × {len(brands)} 品牌）】")
        print(f"   {bar(enum_done, enum_total)}  {enum_done}/{enum_total} 个列表页")
        if series:
            print(f"   车系路径：{len(series)} 个"
                  + (f"（已抓 {info['series_done']}/{info['series_total']}）"
                     if info["series_total"] else ""))
        print(f"   已发现车源 ID：{n_discovered}")
        if enum_done >= enum_total and not series:
            print("   Phase A 已结束")
        else:
            remain_a = max(0, (enum_total - enum_done)
                           + max(0, len(series) - info["series_done"]))
            eta_a = remain_a * (RATE_LIMIT_SEC + RATE_JITTER_SEC / 2 + PHASE_A_OVERHEAD)
            print(f"   预计剩余请求：{remain_a} 个  →  约 {human(eta_a)}")

    # ---------------- Phase B ----------------
    if n_discovered:
        print(f"\n  【Phase B · 抓取车辆详情】")
        done = uniq_detail
        print(f"   {bar(done, n_discovered)}  {done}/{n_discovered} 条")
        remain = max(0, n_discovered - done)
        per = RATE_LIMIT_SEC + RATE_JITTER_SEC / 2
        eta_b = remain * per
        print(f"   剩余 {remain} 条 × 平均 {per:.0f} 秒/条  →  约 {human(eta_b)}")
        if n_fail:
            print(f"   失败记录：{n_fail} 条（见 data/raw/failures.jsonl）")
        if DAILY_MAX and remain > DAILY_MAX:
            days = remain / DAILY_MAX
            print(f"   [!] 超出单日上限 {DAILY_MAX} 条，按日均上限需 {days:.1f} 天跑完")

        # 真实速率（从日志时间戳估算）
        if info.get("detail_done") and info["detail_done"] > 1:
            print(f"   本次进程进度：{info['detail_done']}/{info.get('detail_total','?')}"
                  f"（成功 {info.get('detail_ok','?')}）")

        # ---------------- 字段覆盖率 ----------------
        if cov:
            print(f"\n  【关键字段覆盖率】（分母 = 已抓 {max(1, done)} 条）")
            labels = {
                "residual_rate": "残值率（建模目标）",
                "first_register": "上牌日期（算车龄）",
                "condition_grade": "车况等级",
                "battery_kwh": "电池容量",
                "soh_pct": "电池健康度 SOH",
                "is_battery_swap": "换电/BaaS 标记",
            }
            for k, label in labels.items():
                v = cov.get(k)
                if v is None:
                    continue
                filled = int(round(v * 20))
                print(f"   {label:<22} {v:>6.1%}  [{'#'*filled}{'.'*(20-filled)}]")

    # ---------------- 告警 ----------------
    if info["blocked"]:
        print("\n  " + "!" * 70)
        print("  [!] 日志中出现「熔断」或验证码字样，采集已被风控拦截。")
        print("      请等待数小时后再用 --probe 探测，不要连续试探。")
        print("  " + "!" * 70)

    print("\n" + "-" * 74)
    print("  提示：本脚本只读，不发任何网络请求。")
    print("       设置限速参数见 src/collect.py 顶部 RATE_LIMIT_SEC。")
    print("-" * 74)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", action="store_true", help="持续刷新")
    ap.add_argument("--interval", type=int, default=30, help="刷新间隔（秒）")
    a = ap.parse_args()

    if not a.watch:
        render()
        return
    try:
        while True:
            render()
            print(f"\n  下次刷新：{a.interval} 秒后（Ctrl+C 退出）")
            time.sleep(a.interval)
    except KeyboardInterrupt:
        print("\n已退出。")


if __name__ == "__main__":
    main()
