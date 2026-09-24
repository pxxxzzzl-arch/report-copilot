"""PDF 预检：在正式解析前，逐页摸清每份年报的"体质"。

回答三个问题，决定后续解析策略：
    1. 哪些页有可用的文字层？          -> 直接用 PyMuPDF 抽文本
    2. 哪些页是扫描图片、没有文字层？  -> 需要 OCR
    3. 哪些页文字层是乱码？            -> 字体缺少 ToUnicode 映射，抽出来的文本不能用，也需要 OCR

每页计算的指标：
    n_chars         去掉空白后的字符数
    n_images        页内图片个数
    image_ratio     图片覆盖面积 / 页面面积（多张图片重叠的部分只算一次）
    bad_char_ratio  乱码字符占比
    kind            text / scanned / blank / garbled

页码约定：page 从 1 开始，是 PDF 的物理页序号（不是年报页脚印的页码），全项目统一。

用法：
    uv run python -m ingest.precheck                 # 预检 manifest.csv 中的全部文档
    uv run python -m ingest.precheck --doc 600519_2024

输出：
    data/parsed/precheck/{doc_id}.jsonl   逐页明细（不提交 Git）
    data/precheck_summary.csv             每文档一行的汇总（提交 Git）
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import pymupdf

from common.log import setup_logging, timed, trace_context
from config import PrecheckSettings, get_settings
from ingest.download import read_manifest

logger = logging.getLogger(__name__)

PageKind = Literal["text", "scanned", "blank", "garbled"]
Rect = tuple[float, float, float, float]  # (x0, y0, x1, y1)，PDF 坐标，单位 pt

SUMMARY_FIELDS = (
    "doc_id", "n_pages", "n_text", "n_scanned", "n_blank", "n_garbled",
    "avg_chars", "needs_ocr", "ocr_pages",
)  # fmt: skip


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------


@dataclass
class PageStat:
    page: int  # 物理页码，从 1 开始
    n_chars: int
    n_images: int
    image_ratio: float
    bad_char_ratio: float
    kind: PageKind


@dataclass
class DocReport:
    doc_id: str
    pages: list[PageStat] = field(default_factory=list)

    def count(self, kind: PageKind) -> int:
        return sum(p.kind == kind for p in self.pages)

    @property
    def ocr_pages(self) -> list[int]:
        """需要 OCR 的页：扫描页 + 乱码页。"""
        return [p.page for p in self.pages if p.kind in ("scanned", "garbled")]

    def summary(self) -> dict:
        n = len(self.pages)
        return {
            "doc_id": self.doc_id,
            "n_pages": n,
            "n_text": self.count("text"),
            "n_scanned": self.count("scanned"),
            "n_blank": self.count("blank"),
            "n_garbled": self.count("garbled"),
            "avg_chars": round(sum(p.n_chars for p in self.pages) / n, 1) if n else 0.0,
            "needs_ocr": bool(self.ocr_pages),
            # 页码列表压缩成 "1-3;7" 形式，避免 CSV 单元格过长
            "ocr_pages": compress_ranges(self.ocr_pages),
        }


# ---------------------------------------------------------------------------
# 纯函数：几何与文本指标
# ---------------------------------------------------------------------------


def clip_rect(r: Rect, bound: Rect) -> Rect | None:
    """把矩形裁剪到 bound 范围内；裁剪后面积为 0 则返回 None。

    图片的 bbox 可能超出页面（出血设计），不裁剪会让占比超过 100%。
    """
    x0, y0 = max(r[0], bound[0]), max(r[1], bound[1])
    x1, y1 = min(r[2], bound[2]), min(r[3], bound[3])
    return (x0, y0, x1, y1) if x1 > x0 and y1 > y0 else None


def union_area(rects: list[Rect]) -> float:
    """多个轴对齐矩形的并集面积（重叠部分只算一次）。

    算法：坐标压缩 + 扫描竖条。
        1. 把所有矩形的左右边 x 坐标排序去重，相邻两个 x 之间形成一个"竖条"；
        2. 在每个竖条内，完整覆盖该竖条的矩形各贡献一个 y 区间；
        3. 合并这些 y 区间得到被覆盖的总高度，乘以竖条宽度就是该竖条的覆盖面积；
        4. 所有竖条累加。
    复杂度 O(n² log n)。一页图片通常不超过几十张，足够快，且逻辑直观、便于验证。
    """
    rects = [r for r in rects if r[2] > r[0] and r[3] > r[1]]
    xs = sorted({x for r in rects for x in (r[0], r[2])})
    total = 0.0
    for left, right in zip(xs, xs[1:]):
        # 覆盖整个竖条 [left, right] 的矩形。由于 left/right 都来自矩形边界，
        # 一个矩形要么完全覆盖某竖条，要么完全不覆盖，不存在"部分覆盖"。
        spans = sorted((r[1], r[3]) for r in rects if r[0] <= left and r[2] >= right)
        covered = 0.0
        cur_start = cur_end = None
        for y0, y1 in spans:  # 经典的区间合并
            if cur_end is None or y0 > cur_end:
                if cur_end is not None:
                    covered += cur_end - cur_start
                cur_start, cur_end = y0, y1
            else:
                cur_end = max(cur_end, y1)
        if cur_end is not None:
            covered += cur_end - cur_start
        total += covered * (right - left)
    return total


def strip_whitespace(text: str) -> str:
    """去掉所有空白字符（空格、换行、制表符、全角空格等）。"""
    return "".join(text.split())


def is_bad_char(ch: str) -> bool:
    """判断单个字符是否属于乱码特征字符。

    - U+FFFD：解码失败时的替换符 "�"
    - Co（私用区）：字体缺少 ToUnicode 映射时，字形常被映射到这里
    - Cc（控制字符）/ Cs（代理项）/ Cn（未分配码位）：正常文本中不应出现
    调用前应先去掉空白，否则换行符（也属于 Cc）会被误判。
    """
    return ch == "�" or unicodedata.category(ch) in ("Co", "Cc", "Cs", "Cn")


def bad_char_ratio(text: str) -> float:
    """乱码字符占比（按去除空白后的字符计算）；空文本返回 0。"""
    chars = strip_whitespace(text)
    if not chars:
        return 0.0
    return sum(is_bad_char(c) for c in chars) / len(chars)


def classify_page(n_chars: int, image_ratio: float, bad_ratio: float, cfg: PrecheckSettings) -> PageKind:
    """根据指标给页面分类。

    判定顺序：
        文字很少：图片占比高 -> scanned（文字在图片里，需要 OCR）；否则 -> blank（空白页、分隔页）
        文字足够：乱码比例高 -> garbled（文字层不可用，需要 OCR）；否则 -> text
    """
    if n_chars < cfg.min_text_chars:
        return "scanned" if image_ratio >= cfg.scan_image_ratio else "blank"
    if bad_ratio > cfg.garbled_ratio:
        return "garbled"
    return "text"


def compress_ranges(pages: list[int]) -> str:
    """把页码列表压缩为区间字符串：[1, 2, 3, 7, 9, 10] -> "1-3;7;9-10"。"""
    nums = sorted(set(pages))
    parts: list[str] = []
    i = 0
    while i < len(nums):
        j = i
        # 向后延伸，直到遇到不连续的页码
        while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
            j += 1
        parts.append(str(nums[i]) if i == j else f"{nums[i]}-{nums[j]}")
        i = j + 1
    return ";".join(parts)


# ---------------------------------------------------------------------------
# PDF 读取
# ---------------------------------------------------------------------------


def precheck_page(page: pymupdf.Page, page_no: int, cfg: PrecheckSettings) -> PageStat:
    """计算单页指标。page_no 为从 1 开始的物理页码。"""
    text = page.get_text("text")
    n_chars = len(strip_whitespace(text))

    bound: Rect = tuple(page.rect)  # 页面范围
    page_area = page.rect.width * page.rect.height
    # get_image_info 返回页面上每一次图片"放置"的位置（同一图片放两处算两次）
    img_rects = [clip_rect(tuple(info["bbox"]), bound) for info in page.get_image_info()]
    img_rects = [r for r in img_rects if r is not None]
    image_ratio = union_area(img_rects) / page_area if page_area > 0 else 0.0

    bad = bad_char_ratio(text)
    return PageStat(
        page=page_no,
        n_chars=n_chars,
        n_images=len(img_rects),
        image_ratio=round(image_ratio, 4),
        bad_char_ratio=round(bad, 4),
        kind=classify_page(n_chars, image_ratio, bad, cfg),
    )


def precheck_pdf(path: Path, doc_id: str, cfg: PrecheckSettings) -> DocReport:
    """逐页预检一份 PDF。"""
    report = DocReport(doc_id=doc_id)
    with pymupdf.open(path) as doc:
        for i, page in enumerate(doc, start=1):
            report.pages.append(precheck_page(page, i, cfg))
    return report


# ---------------------------------------------------------------------------
# 输出
# ---------------------------------------------------------------------------


def write_page_details(report: DocReport, out_dir: Path) -> Path:
    """逐页明细写成 JSONL（一行一页），便于用 jq / pandas 分析。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{report.doc_id}.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for p in report.pages:
            f.write(json.dumps(asdict(p), ensure_ascii=False) + "\n")
    return path


def write_summary(path: Path, summaries: dict[str, dict]) -> None:
    """合并写出汇总 CSV：已有的其他文档保留，本次预检的文档覆盖。"""
    rows: dict[str, dict] = {}
    if path.exists():
        with path.open(encoding="utf-8", newline="") as f:
            rows = {r["doc_id"]: r for r in csv.DictReader(f)}
    rows.update(summaries)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        w.writeheader()
        for doc_id in sorted(rows):
            w.writerow(rows[doc_id])


def run(doc_ids: list[str] | None = None) -> dict[str, dict]:
    """预检 manifest 中的文档（doc_ids 为空时预检全部），返回 {doc_id: 汇总}。"""
    settings = get_settings()
    manifest = read_manifest(settings.paths.manifest_path)
    targets = doc_ids or sorted(manifest)
    out_dir = settings.paths.parsed_dir / "precheck"

    summaries: dict[str, dict] = {}
    for doc_id in targets:
        row = manifest.get(doc_id)
        if row is None:
            logger.error("doc not in manifest", extra={"doc_id": doc_id})
            continue
        with timed("precheck", logger, doc_id=doc_id) as info:
            report = precheck_pdf(settings.paths.raw_dir / row.file_name, doc_id, settings.precheck)
            write_page_details(report, out_dir)
            summaries[doc_id] = report.summary()
            info.update({k: v for k, v in summaries[doc_id].items() if k != "doc_id"})
    write_summary(settings.paths.precheck_summary_path, summaries)
    return summaries


def _print_table(summaries: dict[str, dict]) -> None:
    """在终端打印对齐的汇总表，方便人工查看。"""
    cols = SUMMARY_FIELDS
    widths = {c: max(len(c), *(len(str(s[c])) for s in summaries.values())) for c in cols}
    print("  ".join(c.ljust(widths[c]) for c in cols))
    for s in summaries.values():
        print("  ".join(str(s[c]).ljust(widths[c]) for c in cols))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="PDF 预检：逐页统计字符数、图片占比、乱码比例并分类")
    p.add_argument("--doc", action="append", help="只预检指定 doc_id，可重复；默认预检 manifest 中全部文档")
    args = p.parse_args(argv)

    settings = get_settings()
    setup_logging(settings.log.level, settings.log.json_format)
    with trace_context(), timed("precheck_all", logger) as info:
        summaries = run(args.doc)
        info["n_docs"] = len(summaries)
    if summaries:
        _print_table(summaries)


if __name__ == "__main__":
    main()
