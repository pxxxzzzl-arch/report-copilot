"""PDF 解析：把每份年报转成统一中间格式（AGENTS.md 4.1），每个"块"一条记录。

v0（parser_version = pymupdf-blocks-v0）刻意保持朴素，作为消融实验的基线：
    - 用 PyMuPDF 的 get_text("blocks", sort=True) 取文本块，按从上到下、从左到右的顺序排列；
    - 所有文本块的 type 都记为 "text"；不识别标题（title_path 为空）、不抽取表格、不清理页眉页脚；
    - 图片块只保留位置信息（type="image"，text 为空），留给以后做 OCR 或引用截图；
    - 不处理符号字体字符（如 Wingdings 的勾选框，见 docs/decisions.md D-008）。
以上每一项都是阶段 5 可以单独改进、单独评测的变量。

v1（pymupdf-blocks-v1-nohf）= v0 + 去页眉页脚（experiments/badcases.md BC-004）：
    页眉（"××公司2025年年度报告全文"）和页码几乎出现在每一页，会混进大量 chunk，成为重复噪声。
    判定规则见 remove_headers_footers：只看页面顶部/底部的带状区域，并要求在很多页上重复出现，
    所以正文里偶尔出现在页面顶部的内容（如"合并财务报表主要项目注释（续）"，约 10% 的页）不会被误删。

用法：
    uv run python -m ingest.parse                 # 解析 manifest 中的全部文档
    uv run python -m ingest.parse --doc 600519_2024
输出：data/parsed/blocks/{parser}/{doc_id}.jsonl（parser 取自 config.parse.parser）
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import pymupdf

from common.log import setup_logging, timed, trace_context
from config import ParseSettings, get_settings
from ingest.download import read_manifest

logger = logging.getLogger("ingest.parse")

# 配置中的解析器短名 -> 写入 chunk.parser_version 的完整版本号
PARSER_VERSIONS = {"v0": "pymupdf-blocks-v0", "v1": "pymupdf-blocks-v1-nohf"}
BlockType = Literal["text", "title", "table", "image"]

# PyMuPDF get_text("blocks") 返回元组 (x0, y0, x1, y1, text, block_no, block_type)，block_type：0 文本，1 图片
_PYMUPDF_IMAGE_BLOCK = 1


@dataclass
class Block:
    """统一中间格式的一条记录，字段与 AGENTS.md 4.1 一致。"""

    doc_id: str
    page: int  # 物理页码，从 1 开始
    block_id: str  # 页内编号，如 "p45_b3"
    type: BlockType
    text: str
    bbox: list[float]  # [x0, y0, x1, y1]，PDF 坐标（pt），原点在左上角
    title_path: list[str] = field(default_factory=list)
    table_id: str | None = None
    ocr: bool = False
    ocr_conf: float | None = None


def parse_page(page: pymupdf.Page, doc_id: str, page_no: int) -> list[Block]:
    """解析一页，返回按阅读顺序排列的块。去掉只含空白的文本块。"""
    blocks: list[Block] = []
    # sort=True：按块的纵坐标、再按横坐标排序，近似阅读顺序（单栏排版的年报基本准确）
    for x0, y0, x1, y1, text, _no, btype in page.get_text("blocks", sort=True):
        bbox = [round(v, 2) for v in (x0, y0, x1, y1)]
        if btype == _PYMUPDF_IMAGE_BLOCK:
            kind: BlockType = "image"
            text = ""
        else:
            kind = "text"
            text = text.strip()
            if not text:
                continue
        blocks.append(Block(doc_id, page_no, f"p{page_no}_b{len(blocks)}", kind, text, bbox))
    return blocks


def parse_pdf(path: Path, doc_id: str, parser: str = "v0", cfg: ParseSettings | None = None) -> list[Block]:
    """解析整份 PDF。parser="v1" 时在 v0 的结果上去掉页眉页脚。"""
    with pymupdf.open(path) as doc:
        blocks = [b for page_no, page in enumerate(doc, start=1) for b in parse_page(page, doc_id, page_no)]
        page_heights = {page_no: page.rect.height for page_no, page in enumerate(doc, start=1)}
    if parser == "v1":
        cfg = cfg or ParseSettings()
        blocks = remove_headers_footers(blocks, page_heights, cfg.header_band, cfg.header_min_page_ratio)
    return blocks


def header_key(text: str) -> str:
    """页眉页脚比对用的归一化文本：去掉空白，每一串数字替换成 #。

    这样 "8 / 143" 与 "9 / 143"、"2024年年度报告" 的不同页都会得到同一个 key。
    """
    return re.sub(r"\d+", "#", "".join(text.split()))


def remove_headers_footers(
    blocks: list[Block], page_heights: dict[int, float], band: float, min_page_ratio: float
) -> list[Block]:
    """删除页眉页脚块。

    1. 候选：文本块完全位于页面顶部 band 比例（y1 <= band*高）或底部 band 比例（y0 >= (1-band)*高）区域内；
    2. 统计每个候选 key（header_key）出现在多少个不同页面上；
    3. 出现页数 / 总页数 >= min_page_ratio 的 key 判为页眉页脚，删除这些候选块。
    不在带状区域内的块永远不会被删除，即使它的文字与页眉相同（例如正文里提到公司全称）。
    """
    n_pages = len(page_heights)
    if n_pages == 0:
        return blocks

    def in_band(b: Block) -> bool:
        h = page_heights[b.page]
        return b.type == "text" and (b.bbox[3] <= band * h or b.bbox[1] >= (1 - band) * h)

    pages_by_key: dict[str, set[int]] = defaultdict(set)
    for b in blocks:
        if in_band(b):
            pages_by_key[header_key(b.text)].add(b.page)
    repeated = {k for k, pages in pages_by_key.items() if len(pages) / n_pages >= min_page_ratio}
    return [b for b in blocks if not (in_band(b) and header_key(b.text) in repeated)]


def write_blocks(blocks: list[Block], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for b in blocks:
            f.write(json.dumps(asdict(b), ensure_ascii=False) + "\n")


def read_blocks(path: Path) -> list[Block]:
    with path.open(encoding="utf-8") as f:
        return [Block(**json.loads(line)) for line in f]


def blocks_path(parsed_dir: Path, parser: str, doc_id: str) -> Path:
    """不同解析器的结果分目录存放：data/parsed/blocks/{parser}/{doc_id}.jsonl。"""
    return parsed_dir / "blocks" / parser / f"{doc_id}.jsonl"


def run(doc_ids: list[str] | None = None) -> dict[str, int]:
    """解析指定文档（默认全部），返回 {doc_id: 块数}。"""
    settings = get_settings()
    manifest = read_manifest(settings.paths.manifest_path)
    parser = settings.parse.parser
    counts: dict[str, int] = {}
    for doc_id in doc_ids or sorted(manifest):
        with timed("parse", logger, doc_id=doc_id, parser=PARSER_VERSIONS[parser]) as info:
            blocks = parse_pdf(settings.paths.raw_dir / manifest[doc_id].file_name, doc_id, parser, settings.parse)
            write_blocks(blocks, blocks_path(settings.paths.parsed_dir, parser, doc_id))
            counts[doc_id] = info["n_blocks"] = len(blocks)
    return counts


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="PDF 解析为统一中间格式")
    p.add_argument("--doc", action="append", help="只解析指定 doc_id，可重复")
    args = p.parse_args(argv)
    settings = get_settings()
    setup_logging(settings.log.level, settings.log.json_format)
    with trace_context(), timed("parse_all", logger) as info:
        info["n_blocks"] = sum(run(args.doc).values())


if __name__ == "__main__":
    main()
