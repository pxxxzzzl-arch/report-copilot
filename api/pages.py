"""年报页面截图 + 引用高亮（阶段 9，前端点击引用时调用）。

定位摘录在页面上的位置，按顺序尝试三种方法（docs/decisions.md D-030）：
    1. exact   PyMuPDF 的 page.search_for 整段搜索：G007 的引用中 203/210 条能直接命中；
    2. chars   逐字对齐：取出页面上每个字符及其坐标，按与引用校验相同的规则归一化（去空白、全角转半角）后拼成文本，
               在其中找归一化后的摘录，再把命中的字符按行合并成矩形。能处理摘录跨行、PDF 里多了空格等 search_for 找不到的情况；
    3. number  摘录中的长数字串（带千分位或小数点）：get_financial_metric 的证据是工具生成的句子，句子本身不在 PDF 上，
               但其中的原始数字在；只接受前后不紧挨数字的完整匹配，避免高亮到更长数字的一部分。
都找不到时返回原页面、不画框（method=none），前端据此提示"未能定位"。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from common.text import normalize

_NUMBER_RE = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+\.\d+")
_DIGITISH = set("0123456789,.")

HIGHLIGHT_FILL = (1.0, 0.85, 0.0)  # 黄色填充（半透明）
HIGHLIGHT_BORDER = (0.9, 0.5, 0.0)  # 橙色边框


@dataclass(frozen=True)
class Located:
    method: str  # exact / chars / number / none
    rects: tuple[tuple[float, float, float, float], ...]  # PDF 坐标（点），每行一个矩形


@dataclass
class PageChars:
    """页面上的字符：text 为归一化后的文本（去空白、全角转半角），boxes[i] 是 text[i] 的 (行号, 字符框)；
    gaps 记录"前面有间隔"的字符下标（原文中有空白，或换了 span / 行）。

    为什么要单独记 gaps：去掉空白后，表格里相邻两列的数字会粘在一起（"19,181,768,363.65 10,648,..." 变成
    "…363.6510,648…"），判断"完整数字"时就分不清边界了。摘录匹配用去空白的 text（与引用校验一致），
    数字边界判断再加上 gaps。
    """

    text: str
    boxes: list[tuple[int, tuple[float, float, float, float]]]
    gaps: set[int]


def page_chars(page) -> PageChars:
    parts: list[str] = []
    boxes: list[tuple[int, tuple[float, float, float, float]]] = []
    gaps: set[int] = set()
    line_no = 0
    for block in page.get_text("rawdict")["blocks"]:
        for line in block.get("lines", []):
            line_no += 1
            for span in line["spans"]:
                gaps.add(len(parts))  # 换 span / 换行视为间隔
                for ch in span["chars"]:
                    n = normalize(ch["c"])  # 空白归一化后为空串；全角转半角是一对一的
                    if not n:
                        gaps.add(len(parts))
                    for c in n:
                        parts.append(c)
                        boxes.append((line_no, tuple(ch["bbox"])))
    return PageChars("".join(parts), boxes, gaps)


def merge_by_line(boxes: list[tuple[int, tuple[float, float, float, float]]]) -> tuple[tuple[float, float, float, float], ...]:
    """同一行的字符框合并成一个矩形（跨行的摘录得到多个矩形，不会画出一个横跨整栏的大框）。"""
    lines: dict[int, list[float]] = {}
    for ln, (x0, y0, x1, y1) in boxes:
        r = lines.setdefault(ln, [x0, y0, x1, y1])
        r[0], r[1], r[2], r[3] = min(r[0], x0), min(r[1], y0), max(r[2], x1), max(r[3], y1)
    return tuple(tuple(r) for r in lines.values())


def find_all(pc: PageChars, sub: str, whole_number: bool = False) -> list[int]:
    """sub 在页面文本中所有出现的起点；whole_number=True 时要求前后是间隔，或者不是数字、逗号、小数点。"""
    text, out = pc.text, []
    i = text.find(sub)
    while i >= 0 and sub:
        j = i + len(sub)
        left_ok = i == 0 or i in pc.gaps or text[i - 1] not in _DIGITISH
        right_ok = j == len(text) or j in pc.gaps or text[j] not in _DIGITISH
        if not whole_number or (left_ok and right_ok):
            out.append(i)
        i = text.find(sub, i + 1)
    return out


def locate_quote(page, quote: str) -> Located:
    quote = quote.strip()
    if not quote:
        return Located("none", ())
    hits = page.search_for(quote)
    if hits:
        return Located("exact", tuple(tuple(r) for r in hits))

    pc = page_chars(page)
    q = normalize(quote)
    starts = find_all(pc, q)
    if starts:  # 同一摘录在页面上出现多次时全部高亮
        return Located("chars", sum((merge_by_line(pc.boxes[s : s + len(q)]) for s in starts), ()))

    rects: tuple = ()
    for num in dict.fromkeys(_NUMBER_RE.findall(q)):  # 去重并保持顺序
        if len(num) >= 4:
            for s in find_all(pc, num, whole_number=True):
                rects += merge_by_line(pc.boxes[s : s + len(num)])
    return Located("number", rects) if rects else Located("none", ())


@lru_cache(maxsize=64)
def render_page_png(pdf_path: Path, page_no: int, quote: str, scale: float) -> tuple[bytes, str, int]:
    """渲染一页为 PNG 并画出高亮框，返回 (PNG 字节, 定位方法, 矩形数)。

    每次打开一个新的文档对象、只在内存里画框，不修改 PDF 文件；结果按参数缓存（同一引用反复查看不重复渲染）。
    """
    import pymupdf

    with pymupdf.open(pdf_path) as doc:
        page = doc[page_no - 1]
        loc = locate_quote(page, quote) if quote else Located("none", ())
        for r in loc.rects:
            rect = pymupdf.Rect(r) + (-1.5, -1.5, 1.5, 1.5)  # 稍微外扩，框不压字
            page.draw_rect(rect, color=HIGHLIGHT_BORDER, fill=HIGHLIGHT_FILL, fill_opacity=0.3, width=1.2)
        pix = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale))
        return pix.tobytes("png"), loc.method, len(loc.rects)
