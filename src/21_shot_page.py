# -*- coding: utf-8 -*-
r"""21_shot_page.py —— 用无头浏览器实际渲染预测页面并截图 + 抓控制台报错

为什么需要这一步：Node 桩化校验只能证明**计算逻辑**对，
证明不了页面能正常渲染——DOM 结构、ECharts 配置、样式都可能出问题，
而这类问题在交付物里就是"打开是白屏"。

本脚本做三件事：
  1. 打开页面，捕获 console 错误与未捕获异常（有错直接判定失败）
  2. 断言关键 DOM 已有内容（残值率不是初始占位符）
  3. 截图保存到 5_交付物/图片/，供人工确认

用法：
    python src/21_shot_page.py
"""
from __future__ import annotations

import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAGE = os.path.join(ROOT, "5_交付物", "残值预测演示.html")
OUTDIR = os.path.join(ROOT, "5_交付物", "图片")
os.makedirs(OUTDIR, exist_ok=True)

from playwright.sync_api import sync_playwright  # noqa: E402

errors: list[str] = []
console_msgs: list[str] = []

with sync_playwright() as pw:
    # 优先用系统已装的 Edge/Chrome：Playwright 自带的 chromium 需要额外下载，
    # 而本机已有 msedge，直接复用可以少一个环境依赖。
    EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
    CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
    launch_kw = {}
    if os.path.exists(EDGE):
        launch_kw["executable_path"] = EDGE
        print(f"使用系统浏览器：{EDGE}")
    elif os.path.exists(CHROME):
        launch_kw["executable_path"] = CHROME
        print(f"使用系统浏览器：{CHROME}")
    browser = pw.chromium.launch(**launch_kw)
    page = browser.new_page(viewport={"width": 1500, "height": 1400},
                            device_scale_factor=2)
    page.on("console", lambda m: (console_msgs.append(f"[{m.type}] {m.text}"),
                                  errors.append(f"console.{m.type}: {m.text}")
                                  if m.type == "error" else None))
    page.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))
    page.goto("file:///" + PAGE.replace("\\", "/"))
    page.wait_for_timeout(2500)

    rate = page.inner_text("#o-rate")
    price = page.inner_text("#o-price")
    band = page.inner_text("#o-band")
    series_hint = page.inner_text("#h-series")
    print(f"默认状态：残值率 {rate}　预测挂牌价 {price}　区间 {band}")
    print(f"命中档位：{series_hint}")

    shot1 = os.path.join(OUTDIR, "预测页面_默认.png")
    page.screenshot(path=shot1, full_page=True)
    print(f"已截图：{shot1}")

    # 切换车系 + 拉高车龄，验证交互
    page.select_option("#f-series", "Model Y")
    page.evaluate("""() => {
        const a = document.getElementById('f-age'); a.value = 78;
        a.dispatchEvent(new Event('input'));
        const m = document.getElementById('f-mil'); m.value = 16;
        m.dispatchEvent(new Event('input'));
    }""")
    page.wait_for_timeout(1200)
    rate2 = page.inner_text("#o-rate")
    price2 = page.inner_text("#o-price")
    band2 = page.inner_text("#o-band")
    print(f"交互后（Model Y / 78 月 / 16 万公里）：残值率 {rate2}　"
          f"挂牌价 {price2}　区间 {band2}")
    shot2 = os.path.join(OUTDIR, "预测页面_ModelY高龄.png")
    page.screenshot(path=shot2, full_page=True)
    print(f"已截图：{shot2}")

    # 车龄拉到最大，验证边界（之前这里下标越界会导致白屏）
    page.evaluate("""() => {
        const a = document.getElementById('f-age'); a.value = 96;
        a.dispatchEvent(new Event('input'));
    }""")
    page.wait_for_timeout(900)
    rate3 = page.inner_text("#o-rate")
    print(f"边界（96 月）：残值率 {rate3}")
    browser.close()

print(f"\n控制台消息 {len(console_msgs)} 条，错误 {len(errors)} 条")
for e in errors[:20]:
    print("  " + e)

assert rate != "—" and price != "—", "页面未渲染出预测结果（可能是白屏）"
assert errors == [], f"页面存在 JS 错误：{errors[:5]}"
print("\n[通过] 页面渲染正常，无 JS 错误，交互与边界状态均可响应。")
