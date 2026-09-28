"""表格抽取：从年报"主要会计数据和财务指标"表中抽取核心财务指标（阶段 7，供 Agent 的 get_financial_metric 工具使用）。

为什么单独抽这张表：v3 的失败案例里，表格读错行 / 列（BC-012）、单位换算错（BC-008）占了相当比例。
这 9 个核心指标被问得最多，把它们抽成结构化数据后，Agent 可以精确查询，不必从切碎的表格文本里读数。

抽取流程（每份年报）：
    1. 定位：找到同时包含"主要会计数据"和"营业收入"的第一页 p，处理 p 和 p+1 两页（表格常跨页）；
    2. 用 pdfplumber 按从上到下的顺序取出表格，逐行处理：
       - 表头行：含"2025年""2024年"这类年份单元格。第一列年份必须是本报告年度、第二列是上一年，否则这张表不用；
         表头含"季度"或"非经常性损益"时，说明已经离开主要会计数据表，停止；
       - 续行：没有任何数字的行（pdfplumber 把折行的指标名拆成了两行，如"归属于上市公司股东" + "的净利润"），
         把文字拼到上一行的指标名后面；
       - 数据行：取前两个数字，分别是本年、上年（第三个通常是增减比例，不用）；
       - 跨页续表没有表头时，沿用上一张表的表头状态；
    3. 指标名归一化（去空白、全角转半角、去掉括号里的单位和注释）后，与 METRICS 中的同义名精确匹配；
       同一指标只取第一次出现的值；
    4. 单位：指标名括号里的单位（如"营业收入（元）"）优先，其次是页面上"单位：千元"这类标注；
       金额统一换算成元，每股收益单位为元/股，收益率单位为 %；
    5. 原文校验：原始数字串必须出现在该页 PyMuPDF 抽出的文本中（归一化后比较），否则丢弃并报告。

只处理这一张表，不做通用的表格结构化（范围见 docs/decisions.md D-026）。
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path

from common.text import normalize, to_halfwidth

EXTRACTOR_VERSION = "key-metrics-v1"

# 指标键 -> (中文名, 类型, 归一化后的同义名集合)
#   类型 money：金额，统一换算成元；per_share：元/股；percent：百分比数值（24.91 表示 24.91%）
METRICS: dict[str, tuple[str, str, frozenset[str]]] = {
    "revenue": ("营业收入", "money", frozenset({"营业收入"})),
    "net_profit": ("归属于上市公司股东的净利润", "money", frozenset({"归属于上市公司股东的净利润"})),
    "net_profit_deducted": (
        "归属于上市公司股东的扣除非经常性损益的净利润",
        "money",
        frozenset({"归属于上市公司股东的扣除非经常性损益的净利润", "归属于上市公司股东的扣除非经常性损益后的净利润"}),
    ),
    "operating_cash_flow": ("经营活动产生的现金流量净额", "money", frozenset({"经营活动产生的现金流量净额"})),
    "eps_basic": ("基本每股收益", "per_share", frozenset({"基本每股收益"})),
    "eps_diluted": ("稀释每股收益", "per_share", frozenset({"稀释每股收益"})),
    "roe_weighted": ("加权平均净资产收益率", "percent", frozenset({"加权平均净资产收益率"})),
    "total_assets": ("总资产", "money", frozenset({"总资产", "资产总额"})),
    "equity_parent": ("归属于上市公司股东的净资产", "money", frozenset({"归属于上市公司股东的净资产"})),
}

# 金额单位 -> 换算成元的倍数
MONEY_UNITS = {"元": Decimal(1), "千元": Decimal(1000), "万元": Decimal(10000), "百万元": Decimal(1000000), "亿元": Decimal(100000000)}
CANONICAL_UNIT = {"money": "元", "per_share": "元/股", "percent": "%"}

_NUM_RE = re.compile(r"^-?\d{1,3}(,\d{3})*(\.\d+)?%?$|^-?\d+(\.\d+)?%?$")
_YEAR_RE = re.compile(r"^(\d{4})年(末|度)?$")
_PAREN_RE = re.compile(r"[(（][^()（）]*[)）]")
_UNIT_IN_LABEL_RE = re.compile(r"[(（](元|千元|万元|百万元|亿元)[)）]")
_PAGE_UNIT_RE = re.compile(r"单位[:：]\s*(元|千元|万元|百万元|亿元)")


@dataclass
class MetricValue:
    doc_id: str
    fiscal_year: int  # 数值所属的会计年度（本年或上年）
    metric: str  # METRICS 的键
    value: Decimal  # 换算后的数值（金额为元）
    unit: str  # 换算后的单位：元 / 元/股 / %
    raw: str  # 原文中的数字串，如 "423,701,834" 或 "24.91%"
    raw_unit: str  # 原文单位，如 "千元"
    page: int  # 物理页码
    label: str  # 原文指标名（拼接折行后）


def parse_number(cell: str | None) -> Decimal | None:
    """把单元格解析成数字；不是纯数字（如"减少3.49个百分点""-""2025年"）时返回 None。"""
    if cell is None:
        return None
    s = normalize(cell)
    if not s or not _NUM_RE.match(s):
        return None
    try:
        return Decimal(s.rstrip("%").replace(",", ""))
    except InvalidOperation:  # 正则已保证格式，这里只是兜底
        return None


def canonical_label(label: str) -> str:
    """指标名归一化：去空白、全角转半角、去掉括号及其内容（单位、注释编号等）。"""
    s = normalize(label)
    prev = None
    while prev != s:  # 反复去除，处理 "(%)(1)" 这类连续括号
        prev, s = s, _PAREN_RE.sub("", s)
    return s


def match_metric(label: str) -> str | None:
    c = canonical_label(label)
    for key, (_, _, names) in METRICS.items():
        if c in names:
            return key
    return None


def label_unit(label: str) -> str | None:
    m = _UNIT_IN_LABEL_RE.search(normalize(label))
    return m.group(1) if m else None


def page_unit(page_text: str) -> str | None:
    """页面上"主要会计数据"之后第一个"单位：X"标注；表格前没有标注时返回 None。"""
    i = page_text.find("主要会计数据")
    m = _PAGE_UNIT_RE.search(page_text, max(i, 0))
    return m.group(1) if m else None


def merge_rows(rows: Sequence[Sequence[str | None]]) -> list[tuple[str, list[str]]]:
    """把 pdfplumber 的原始行整理成 [(指标名, [数字串...])]。

    没有数字的行视为上一行指标名的折行续写，文字拼到上一行后面；
    表头行（含年份单元格）单独成行，数字列表为空、指标名为 "<header>|2025年|2024年..."。
    """
    out: list[tuple[str, list[str]]] = []
    for row in rows:
        cells = [c for c in row if c not in (None, "")]
        years = [c for c in cells if _YEAR_RE.match(normalize(c))]
        if len(years) >= 2:
            out.append(("<header>|" + "|".join(normalize(c) for c in cells), []))
            continue
        nums = [normalize(c) for c in cells if parse_number(c) is not None]
        # 指标名只取第一个数字之前的文字；数字之后的文字是"减少3.49个百分点"这类增减说明，不属于指标名
        first_num = next((i for i, c in enumerate(cells) if parse_number(c) is not None), len(cells))
        text = "".join(cells[:first_num])
        if not nums:
            # 续行：拼到上一条数据行（不拼到表头）
            if out and not out[-1][0].startswith("<header>") and text:
                label, vals = out[-1]
                out[-1] = (label + text, vals)
            elif text:
                out.append((text, []))
            continue
        out.append((text, nums))
    return out


def header_years(header: str) -> list[int]:
    return [int(m.group(1)) for c in header.split("|")[1:] if (m := _YEAR_RE.match(c))]


def extract_from_tables(
    tables: Sequence[tuple[int, Sequence[Sequence[str | None]]]],
    report_year: int,
    doc_id: str,
    unit_of_page: dict[int, str | None],
) -> tuple[list[MetricValue], list[str]]:
    """从按页面顺序排列的表格 [(页码, 行)] 中抽取指标，返回 (结果, 问题说明)。纯函数，便于单元测试。"""
    results: dict[tuple[int, str], MetricValue] = {}
    issues: list[str] = []
    active = False  # 当前是否处在"主要会计数据"表内（年份顺序已确认）
    for page, rows in tables:
        for label, nums in merge_rows(rows):
            if label.startswith("<header>"):
                if "季度" in label or "非经常性损益" in label:
                    return list(results.values()), issues
                ys = header_years(label)
                active = ys[:2] == [report_year, report_year - 1]
                continue
            if "季度" in label:  # 分季度表的表头有时不含年份
                return list(results.values()), issues
            key = match_metric(label) if active else None
            if key is None:
                continue
            if len(nums) < 2:
                issues.append(f"{doc_id} p{page} {label}: 数字不足两个 {nums}")
                continue
            kind = METRICS[key][1]
            raw_unit = CANONICAL_UNIT[kind]
            mult = Decimal(1)
            if kind == "money":
                raw_unit = label_unit(label) or unit_of_page.get(page) or ""
                if raw_unit not in MONEY_UNITS:
                    issues.append(f"{doc_id} p{page} {label}: 无法确定金额单位")
                    continue
                mult = MONEY_UNITS[raw_unit]
            for fy, raw in zip((report_year, report_year - 1), nums[:2]):
                if (fy, key) in results:
                    continue
                if kind != "percent" and raw.endswith("%"):
                    issues.append(f"{doc_id} p{page} {label}: 金额列出现百分号 {raw}")
                    continue
                value = parse_number(raw) * mult  # type: ignore[operator]
                results[(fy, key)] = MetricValue(
                    doc_id, fy, key, value, CANONICAL_UNIT[kind], raw, raw_unit, page, "".join(label.split())  # fmt: skip
                )
    return list(results.values()), issues


def contains_number(text: str, raw: str) -> bool:
    """text 中是否出现完整的数字 raw：前后不能紧挨数字、逗号或小数点。
    只用子串判断不够，"1" 会在 "423,701,834" 里被找到。
    注意这里只做全角转半角、不去空白：去掉空白会把表格里相邻两列的数字粘在一起，边界就判断不了了。"""
    pattern = r"(?<![\d.,])" + re.escape(normalize(raw)) + r"(?![\d.,]*\d)"
    return re.search(pattern, to_halfwidth(text)) is not None


def verify_on_page(values: Sequence[MetricValue], page_texts: dict[int, str]) -> tuple[list[MetricValue], list[str]]:
    """原文校验：原始数字串必须作为完整数字出现在所标注页的 PyMuPDF 文本中（归一化后比较）。"""
    ok, issues = [], []
    for v in values:
        if contains_number(page_texts.get(v.page, ""), v.raw):
            ok.append(v)
        else:
            issues.append(f"{v.doc_id} p{v.page} {v.metric} {v.raw}: 原文校验失败，已丢弃")
    return ok, issues


def extract_key_metrics(pdf_path: Path, doc_id: str, report_year: int) -> tuple[list[MetricValue], list[str]]:
    """读取一份年报 PDF，抽取并校验核心指标。"""
    import pdfplumber
    import pymupdf

    with pymupdf.open(pdf_path) as doc:
        texts = {i + 1: p.get_text() for i, p in enumerate(doc)}
    start = next((p for p, t in texts.items() if "主要会计数据" in t and "营业收入" in t), None)
    if start is None:
        return [], [f"{doc_id}: 找不到主要会计数据表"]
    pages = [p for p in (start, start + 1) if p in texts]

    tables: list[tuple[int, list]] = []
    with pdfplumber.open(pdf_path) as pdf:
        for p in pages:
            found = sorted(pdf.pages[p - 1].find_tables(), key=lambda t: t.bbox[1])  # 按表格上边缘从上到下
            tables += [(p, t.extract()) for t in found]
    # 单位标注在表格上方，跨页续表沿用起始页的标注
    units = {p: page_unit(texts[start]) for p in pages}
    values, issues = extract_from_tables(tables, report_year, doc_id, units)
    values, verify_issues = verify_on_page(values, {p: texts[p] for p in pages})
    return values, issues + verify_issues
