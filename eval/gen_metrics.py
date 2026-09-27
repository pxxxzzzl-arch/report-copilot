"""生成评测的规则判分（不依赖大模型，结果确定、可复现）。

1. 数值正确（numbers_match）
   从参考答案中抽取"关键数值"，统一换算到基本单位后，逐个在模型答案里找是否有相对误差不超过 tol 的数值。
   - 单位换算：元 / 千元 / 万元 / 亿元（及单独的"亿""万"）都换算成元；% 保留百分数本身；人、股、元/股按原值；
   - 年份（"2024年"）不算数值；
   - 关键数值 = 参考答案括号外的数值：括号里通常是算式或补充说明，例如
       "约91.93%（(170,899,152,276.34-13,789,482,367.98)/170,899,152,276.34）" 的关键数值只有 91.93%；
     括号外一个数值都没有时（如"茅台（168,838,102,514.79元）> 五粮液（…）"），退回使用全部数值；
   - 相互之间误差在 tol 以内的数值视为同一个（"695,376,735.81元（约6.95亿元）"只算一个）。
   比较时不区分数值种类，只比换算后的数值：模型把每股收益写成"2.3068元"而不是"2.3068元/股"也算对。

2. 引用召回（citation_recall）
   gold 的每条证据（主位置或任一 alt）是否被某条"通过核验"的引用指向（doc_id + 页码一致）。
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from common.text import to_halfwidth

# 带千分位的数字，或普通整数 / 小数；前面可带负号
_NUM = r"-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
# 单位按"长的在前"排列，保证"亿元"不会被当成"亿"+"元"
_UNITS = {
    "元/股": ("per_share", 1.0),
    "亿元": ("money", 1e8),
    "万元": ("money", 1e4),
    "千元": ("money", 1e3),
    "元": ("money", 1.0),
    "亿": ("money", 1e8),
    "万": ("money", 1e4),
    "%": ("percent", 1.0),
    "人": ("count", 1.0),
    "股": ("count", 1.0),
    "年": ("year", 1.0),  # 仅用于识别并排除年份
}
_QTY_RE = re.compile(rf"({_NUM})\s*({'|'.join(re.escape(u) for u in _UNITS)})?")
# 最内层的一对括号（全角或半角），反复删除以处理嵌套
_PAREN_RE = re.compile(r"[（(][^（）()]*[）)]")


@dataclass(frozen=True)
class Quantity:
    value: float  # 换算到基本单位后的数值
    kind: str  # money / percent / count / per_share / plain


def extract_quantities(text: str) -> list[Quantity]:
    """抽取文本中的所有数值（已换算单位，已排除年份）。"""
    text = to_halfwidth(text).replace("／", "/")
    out = []
    for m in _QTY_RE.finditer(text):
        num, unit = m.group(1), m.group(2)
        kind, scale = _UNITS.get(unit, ("plain", 1.0)) if unit else ("plain", 1.0)
        if kind == "year":
            continue
        out.append(Quantity(float(num.replace(",", "")) * scale, kind))
    return out


def strip_parentheses(text: str) -> str:
    """删除所有括号及其内容（支持嵌套）。"""
    prev = None
    while prev != text:
        prev, text = text, _PAREN_RE.sub("", text)
    return text


def close(a: float, b: float, tol: float) -> bool:
    """相对误差不超过 tol；两者都为 0 时视为相等。"""
    if a == b:
        return True
    return abs(a - b) <= tol * max(abs(a), abs(b))


def dedupe(qs: Iterable[Quantity], tol: float) -> list[Quantity]:
    out: list[Quantity] = []
    for q in qs:
        if not any(close(q.value, o.value, tol) for o in out):
            out.append(q)
    return out


def key_quantities(reference: str, tol: float = 0.01) -> list[Quantity]:
    outside = extract_quantities(strip_parentheses(to_halfwidth(reference)))
    return dedupe(outside or extract_quantities(reference), tol)


def numbers_match(reference: str, answer: str, tol: float = 0.01) -> tuple[int, int]:
    """返回 (模型答案命中的关键数值个数, 关键数值总数)。总数为 0 表示这道题不适用数值判分。"""
    keys = key_quantities(reference, tol)
    found = extract_quantities(answer)
    hit = sum(1 for k in keys if any(close(k.value, f.value, tol) for f in found))
    return hit, len(keys)


def citation_recall(gold: Sequence[dict], cited: set[tuple[str, int]]) -> float | None:
    """gold 证据中被引用覆盖的比例；gold 为空（拒答题）时返回 None。"""
    if not gold:
        return None
    covered = 0
    for g in gold:
        locs = {(g["doc_id"], g["page"])} | {(a["doc_id"], a["page"]) for a in g.get("alt", [])}
        covered += bool(locs & cited)
    return covered / len(gold)
