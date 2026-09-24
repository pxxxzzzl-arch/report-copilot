"""切分：把解析得到的块（Block）切成用于检索的 Chunk（字段见 AGENTS.md 4.2）。

v0 只有一种策略 fixed（固定字符长度 + 重叠）：
    1. 把一份文档的文本块按阅读顺序用换行拼成一个长字符串，同时记下每个块在长字符串中的起止位置；
    2. 用长度为 chunk_size 的窗口从头滑到尾，每次前进 chunk_size - overlap 个字符；
    3. 每个窗口就是一个 chunk；与窗口有重叠的块决定它的 page_start / page_end 和 bboxes。
    窗口不管页边界和句子边界，所以一个 chunk 可能跨页，也可能把一个数字从中间切开。
    这是朴素基线的已知弱点，也是阶段 5 消融实验要改进的地方。

用法：
    uv run python -m ingest.chunk        # 切分全部已解析文档，配置取自 config.chunk
输出：data/parsed/chunks/{chunk_key}.jsonl，例如 v0_fixed_c500_o50.jsonl
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

from common.log import setup_logging, timed, trace_context
from config import ChunkSettings, get_settings
from ingest.download import ManifestRow, read_manifest
from ingest.parse import PARSER_VERSIONS, Block, blocks_path, read_blocks

logger = logging.getLogger("ingest.chunk")

BLOCK_SEPARATOR = "\n"  # 拼接块时的分隔符，计入字符数


@dataclass
class Chunk:
    """检索单元，字段与 AGENTS.md 4.2 一致；满足 eval.metrics.ChunkLike。"""

    chunk_id: str
    doc_id: str
    company: str
    stock_code: str
    year: int
    chunk_index: int  # 文档内序号，从 0 开始
    text: str
    text_for_embedding: str  # 标题路径 + 正文；v0 没有标题路径，与 text 相同
    title_path: list[str]
    page_start: int
    page_end: int
    bboxes: list[dict]  # [{"page": 12, "bbox": [x0, y0, x1, y1]}, ...]，用于引用高亮
    chunk_type: str  # v0 统一为 "text"
    table_id: str | None
    content_hash: str  # 文本的 sha256 前 16 位，用于去重和判断内容是否变化
    parser_version: str
    chunker_name: str  # 切分标识（解析器 + 切分器配置），如 v0_fixed_c500_o50
    embed_model: str = ""  # 写入索引时填写
    extra: dict = field(default_factory=dict)  # 预留给后续策略的附加信息


def make_chunk_id(doc_id: str, chunker_key: str, index: int) -> str:
    return f"{doc_id}__{chunker_key}__{index:05d}"


def text_for_embedding(title_path: list[str], text: str) -> str:
    """向量化用文本：有标题路径时把它放在正文前面，让向量带上章节语境。"""
    return (" > ".join(title_path) + "\n" + text) if title_path else text


def window_spans(total: int, size: int, overlap: int) -> list[tuple[int, int]]:
    """固定窗口的 [start, end) 列表。

    例：total=10, size=4, overlap=1 -> 步长 3 -> [(0,4), (3,7), (6,10)]
    最后一个窗口到达末尾即停止，不再产生只含重叠部分的尾巴窗口。
    """
    if total <= 0:
        return []
    step = size - overlap
    spans = []
    start = 0
    while True:
        end = min(start + size, total)
        spans.append((start, end))
        if end >= total:
            return spans
        start += step


def chunk_document(
    blocks: list[Block], meta: ManifestRow, cfg: ChunkSettings, chunk_key: str, parser_version: str
) -> list[Chunk]:
    """对一份文档做固定长度切分。只使用文本块（图片块没有文字）。

    chunk_key（如 v0_fixed_c500_o50）写入 chunk_id 和 chunker_name；parser_version 写入 parser_version。
    """
    text_blocks = [b for b in blocks if b.type != "image" and b.text]
    # 拼接长字符串，并记录每个块的起止偏移 [starts[i], ends[i])
    starts: list[int] = []
    ends: list[int] = []
    parts: list[str] = []
    pos = 0
    for b in text_blocks:
        starts.append(pos)
        parts.append(b.text)
        pos += len(b.text)
        ends.append(pos)
        pos += len(BLOCK_SEPARATOR)
    full = BLOCK_SEPARATOR.join(parts)

    chunks: list[Chunk] = []
    for start, end in window_spans(len(full), cfg.chunk_size, cfg.overlap):
        text = full[start:end]
        if not text.strip():
            continue
        # 与窗口 [start, end) 有重叠的块：满足 starts[i] < end 且 ends[i] > start。
        # starts 与 ends 都单调递增，用二分查找定位区间，避免每个窗口都遍历全部块。
        lo = bisect.bisect_right(ends, start)  # 第一个 ends[i] > start 的块
        hi = bisect.bisect_left(starts, end)  # 第一个 starts[i] >= end 的块
        covered = text_blocks[lo:hi]
        if not covered:  # 窗口只落在分隔符上（理论上不会出现，防御性处理）
            continue
        idx = len(chunks)
        chunks.append(
            Chunk(
                chunk_id=make_chunk_id(meta.doc_id, chunk_key, idx),
                doc_id=meta.doc_id,
                company=meta.company,
                stock_code=meta.stock_code,
                year=meta.year,
                chunk_index=idx,
                text=text,
                text_for_embedding=text_for_embedding([], text),
                title_path=[],
                page_start=min(b.page for b in covered),
                page_end=max(b.page for b in covered),
                bboxes=[{"page": b.page, "bbox": b.bbox} for b in covered],
                chunk_type="text",
                table_id=None,
                content_hash=hashlib.sha256(text.encode("utf-8")).hexdigest()[:16],
                parser_version=parser_version,
                chunker_name=chunk_key,
            )
        )
    return chunks


def write_chunks(chunks: list[Chunk], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")


def read_chunks(path: Path) -> list[Chunk]:
    with path.open(encoding="utf-8") as f:
        return [Chunk(**json.loads(line)) for line in f]


def chunks_path(parsed_dir: Path, chunk_key: str) -> Path:
    return parsed_dir / "chunks" / f"{chunk_key}.jsonl"


def run() -> list[Chunk]:
    """切分 manifest 中全部文档（需要先运行 ingest.parse），写出并返回全部 chunk。"""
    settings = get_settings()
    cfg, key, parser = settings.chunk, settings.chunk_key(), settings.parse.parser
    manifest = read_manifest(settings.paths.manifest_path)
    all_chunks: list[Chunk] = []
    for doc_id in sorted(manifest):
        with timed("chunk", logger, doc_id=doc_id, chunker=key) as info:
            blocks = read_blocks(blocks_path(settings.paths.parsed_dir, parser, doc_id))
            chunks = chunk_document(blocks, manifest[doc_id], cfg, key, PARSER_VERSIONS[parser])
            info["n_chunks"] = len(chunks)
        all_chunks.extend(chunks)
    write_chunks(all_chunks, chunks_path(settings.paths.parsed_dir, key))
    return all_chunks


def main(argv: list[str] | None = None) -> None:
    argparse.ArgumentParser(description="按 config.chunk 的配置切分全部已解析文档").parse_args(argv)
    settings = get_settings()
    setup_logging(settings.log.level, settings.log.json_format)
    with trace_context(), timed("chunk_all", logger, chunker=settings.chunk_key()) as info:
        info["n_chunks"] = len(run())


if __name__ == "__main__":
    main()
