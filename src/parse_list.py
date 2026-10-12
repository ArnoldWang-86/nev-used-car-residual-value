# -*- coding: utf-8 -*-
r"""从瓜子列表页 HTML 抽取车源结构化列表（本地解析，无网络请求）。

发现：列表页内嵌 Next.js 流式 JSON（self.__next_f.push），其中
      `"carList":[{...}]` 就是完整的车源数组，
      每条含 encryptedClueId（车源ID）、title（车系+年款+配置）等。
      之前抽不出来是因为**转义层数**：JSON 字符串里再嵌 JSON，
      引号被写成 \"，正则按 " 匹配自然全部失效。

本模块被 tools/parse_manual_html.py 与后续批量采集复用。
"""
from __future__ import annotations

import json
import re

# 车源列表容器的候选键名
LIST_KEYS = ["carList", "car_list", "clueList", "list"]


def extract_next_payload(html: str) -> str:
    """取出 self.__next_f.push 的流式负载并反转义。

    关键：负载是 JSON 字符串字面量，需要先把 \\" 还原成 "，
    否则所有字段名匹配都会失败。
    """
    chunks = re.findall(r'self\.__next_f\.push\(\[1,"(.*?)"\]\)', html, re.S)
    if not chunks:
        return ""
    raw = "".join(chunks)
    # 逐层反转义：先 "\  ->  " ，再常见转义
    s = raw.replace('\\"', '"')
    s = s.replace("\\n", "\n").replace("\\t", "\t")
    s = s.replace("\\/", "/")
    # 中文等多字节字符被写成 \uXXXX，统一还原
    try:
        s = s.encode("utf-8").decode("unicode_escape").encode("latin1").decode("utf-8")
    except (UnicodeDecodeError, UnicodeEncodeError):
        # 该路径失败时保留上面的替换结果即可
        pass
    return s


def _balanced_slice(text: str, start: int) -> str | None:
    """从 start 位置的 '[' 开始，返回配平的 JSON 数组字符串。"""
    if start >= len(text) or text[start] != "[":
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def find_car_list(payload: str) -> list[dict]:
    """在负载里定位 carList 数组并解析成 list[dict]。"""
    for key in LIST_KEYS:
        for m in re.finditer(rf'"{key}"\s*:\s*\[', payload):
            arr = _balanced_slice(payload, m.end() - 1)
            if not arr:
                continue
            try:
                data = json.loads(arr)
            except json.JSONDecodeError:
                continue
            if isinstance(data, list) and data and isinstance(data[0], dict):
                # 校验：应含车源标识类字段
                keys0 = " ".join(data[0].keys())
                if any(t in keys0 for t in ("sku", "clue", "Car", "car")):
                    return data
    return []


def flatten_car(item: dict) -> dict:
    """把嵌套的车源对象拍平成一行。键名做归一化，便于落表。"""
    row: dict = {}

    def walk(o, prefix="") -> None:
        if isinstance(o, dict):
            for k, v in o.items():
                walk(v, f"{prefix}{k}.")
        elif isinstance(o, list):
            for it in o:
                walk(it, prefix)
        else:
            if o not in (None, "", []):
                row[prefix.rstrip(".")] = o

    walk(item)
    return row


# 关键字段的取值策略。
# 字段名来自对真实列表页的实测（tools/probe_list_json.py 输出），
# 不是猜测 —— 车源对象共 30 个键，价格/里程/上牌都在 skuBasicArea 与 priceArea 下。
FIELD_ALIASES = {
    "car_id": ["skuBasicArea.encryptedClueId", "encryptedClueId", "clueId"],
    "title": ["skuBasicArea.title", "title"],
    # 挂牌价（用户实际支付口径）
    "price": ["priceArea.purchasePrice.value", "priceArea.purchasePrice.amountMajor"],
    "price_text": ["priceArea.purchasePrice.priceString",
                   "priceArea.purchasePrice.amountMajor"],
    # 优惠后价格
    "discounted": ["priceArea.discountedAmount.value",
                   "priceArea.discountedAmount.amountMajor"],
    "has_coupon": ["priceArea.hasCoupon"],
    # 上牌日期
    "license_date": ["skuBasicArea.licenseDate"],
    # 表显里程
    "road_haul": ["skuBasicArea.roadHaul"],
    # 车源城市
    "city": ["skuBasicArea.carSourceCityDesc"],
    # 认证/标签信息
    "title_tag": ["skuBasicArea.titleTag.text", "bottomArea.bottomTag.text"],
    "approval": ["skuPredictArea.approvalStatus"],
}


def pick(row: dict, field: str):
    for k in FIELD_ALIASES.get(field, []):
        if k in row and row[k] not in (None, ""):
            return row[k]
    return None


def parse_list_html(html: str) -> tuple[list[dict], dict]:
    """解析列表页 HTML，返回 (车源行列表, 诊断信息)。"""
    diag = {"payload_len": 0, "n_raw": 0, "n_rows": 0}
    payload = extract_next_payload(html)
    diag["payload_len"] = len(payload)
    if not payload:
        return [], diag
    items = find_car_list(payload)
    diag["n_raw"] = len(items)
    rows = []
    for it in items:
        flat = flatten_car(it)
        rows.append({
            "car_id": pick(flat, "car_id"),
            "title": pick(flat, "title"),
            "price": pick(flat, "price"),
            "price_text": pick(flat, "price_text"),
            "discounted": pick(flat, "discounted"),
            "has_coupon": pick(flat, "has_coupon"),
            "license_date": pick(flat, "license_date"),
            "road_haul": pick(flat, "road_haul"),
            "city": pick(flat, "city"),
            "title_tag": pick(flat, "title_tag"),
        })
    diag["n_rows"] = len(rows)
    return rows, diag


def all_keys_of_first(html: str, n: int = 3) -> list[str]:
    """诊断用：返回前 n 条车源拍平后的全部键名，便于发现新字段。"""
    payload = extract_next_payload(html)
    items = find_car_list(payload)
    keys: list[str] = []
    for it in items[:n]:
        keys += list(flatten_car(it).keys())
    return sorted(set(keys))
