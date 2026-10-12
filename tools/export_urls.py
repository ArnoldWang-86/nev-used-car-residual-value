# -*- coding: utf-8 -*-
"""导出一份可直接复制到浏览器测试的 URL 清单。"""
import json
import os

# --- 中文 Windows 控制台默认 GBK，强制 UTF-8 输出，避免打印特殊字符时崩溃 ---
try:
    import sys as _sys
    _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    _sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = json.load(open(os.path.join(ROOT, "data", "raw", "discovered_targets.json"),
                   encoding="utf-8"))
OUT = os.path.join(ROOT, "output", "可测试网址清单.txt")

ids = D.get("car_ids", [])
series = D.get("series_paths", [])

L = []
L.append("=" * 78)
L.append("瓜子二手车 · 可测试网址清单")
L.append("=" * 78)
L.append("")
L.append("【合规依据】")
L.append("  robots.txt:  https://www.guazi.com/robots.txt")
L.append("    - User-agent: *  Allow: /")
L.append("    - Allow: /car-detail/*.md   （官方为 AI 开放的详情 Markdown）")
L.append("    - 注意：Disallow: /*?*  因此【带 ? 的网址不要用】")
L.append("")
L.append("【1. 探测是否解封 —— 先试这一个】")
L.append("  https://www.guazi.com/sh/weilai/")
L.append("  正常应返回约 330~400 KB 的列表页；被封会跳到 uc.guazi.com 验证码页。")
L.append("")
L.append("【2. 站点地图（看总量）】")
L.append("  https://www.guazi.com/guazisou/cardetail/pc_cardetail_md_index.xml")
L.append("  下面是 13 个子地图，每个约 10000 条车源 URL：")
L.append("  https://www.guazi.com/guazisou/cardetail/pc_cardetail_md_1.xml")
L.append("")
L.append("【3. 车辆详情 .md（可直接在浏览器打开，看真实数据）】")
L.append("  本次已发现 %d 条车源，下面是前 15 条：" % len(ids))
for cid in ids[:15]:
    L.append(f"    https://www.guazi.com/car-detail/{cid}.md")
L.append("")
L.append("  注意：把 .md 换成 .html 也能打开，是有图片的网页版")
for cid in ids[:3]:
    L.append(f"    https://www.guazi.com/car-detail/{cid}.html")
L.append("")
L.append("【4. 列表页枚举方式（无查询参数，合规）】")
L.append("  城市+品牌：   https://www.guazi.com/{城市}/{品牌}/")
L.append("  城市+品牌+车系：https://www.guazi.com/{城市}/{品牌}/{车系}/")
L.append("")
L.append("  实例：")
L.append("    https://www.guazi.com/sh/weilai/         上海·蔚来")
L.append("    https://www.guazi.com/bj/weilai/         北京·蔚来")
L.append("    https://www.guazi.com/sh/tesila/         上海·特斯拉")
L.append("")
L.append("  本次发现的车系路径（共 %d 个）：" % len(series))
for sp in series:
    L.append(f"    https://www.guazi.com{sp}")
L.append("")
L.append("【5. 已被证明不可用的入口（别再试）】")
L.append("  × 带分页参数的网址：https://www.guazi.com/sh/weilai/?page=2")
L.append("      → robots.txt 的 Disallow: /*?* 明确禁止，合规上不能抓")
L.append("  × https://m.guazi.com/...        移动端，连接超时")
L.append("  × https://es.guazi.com/...       海外站，返回混淆脚本无数据")
L.append("  × https://www.guazisou.com/...   该域名连接被重置")
L.append("")
L.append("=" * 78)
L.append("提示：浏览器直接打开这些网址不会被封 —— 风控针对的是程序化高频请求。")
L.append("=" * 78)

open(OUT, "w", encoding="utf-8").write("\n".join(L))
print("written:", OUT)
print("车源 ID 数:", len(ids), "| 车系路径数:", len(series))
