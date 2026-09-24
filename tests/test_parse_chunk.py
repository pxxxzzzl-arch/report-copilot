"""ingest/parse.py 与 ingest/chunk.py 的单元测试。"""

from pathlib import Path

import pymupdf
import pytest

from config import ChunkSettings
from ingest.chunk import chunk_document, make_chunk_id, read_chunks, text_for_embedding, window_spans, write_chunks
from ingest.download import ManifestRow
from ingest.parse import Block, parse_pdf, read_blocks, write_blocks

KEY, PV = "v0_fixed_c500_o50", "pymupdf-blocks-v0"  # 切分标识、解析器版本
META = ManifestRow("000001_2024", "测试公司", "000001", 2024, "x.pdf", 2, "SZSE", "", "", "")


def _block(page: int, text: str, i: int = 0) -> Block:
    return Block("000001_2024", page, f"p{page}_b{i}", "text", text, [0, 0, 10, 10])


# ---------------- window_spans（手算） ----------------


@pytest.mark.parametrize(
    ("total", "size", "overlap", "expected"),
    [
        (10, 4, 1, [(0, 4), (3, 7), (6, 10)]),  # 步长 3，最后一个窗口正好到末尾
        (11, 4, 1, [(0, 4), (3, 7), (6, 10), (9, 11)]),  # 末尾剩 2 个字符，单独成一个短窗口
        (3, 4, 1, [(0, 3)]),  # 文本比窗口短
        (8, 4, 0, [(0, 4), (4, 8)]),  # 无重叠
        (0, 4, 1, []),
    ],
)
def test_window_spans(total: int, size: int, overlap: int, expected: list) -> None:
    assert window_spans(total, size, overlap) == expected


def test_chunk_settings_rejects_bad_overlap() -> None:
    with pytest.raises(ValueError):
        ChunkSettings(chunk_size=10, overlap=10)
    assert ChunkSettings(chunk_size=500, overlap=50).key == "fixed_c500_o50"


# ---------------- chunk_document ----------------


def test_chunk_across_pages_tracks_page_range() -> None:
    # 拼接后："aaaa\nbbbb"（长度 9），第 1 块在第 1 页、第 2 块在第 2 页
    blocks = [_block(1, "aaaa"), _block(2, "bbbb")]
    cfg = ChunkSettings(chunk_size=6, overlap=2)  # 窗口 (0,6)="aaaa\nb"、(4,9)="\nbbbb"
    c0, c1 = chunk_document(blocks, META, cfg, KEY, PV)
    assert (c0.text, c0.page_start, c0.page_end) == ("aaaa\nb", 1, 2)  # 跨页
    assert (c1.text, c1.page_start, c1.page_end) == ("\nbbbb", 2, 2)  # 只含第 2 块（分隔符不算第 1 块）
    assert [b["page"] for b in c0.bboxes] == [1, 2]


def test_chunk_fields() -> None:
    cfg = ChunkSettings(chunk_size=500, overlap=50)
    (c,) = chunk_document([_block(3, "营业收入 1,234.56 元")], META, cfg, KEY, PV)
    assert c.chunk_id == "000001_2024__v0_fixed_c500_o50__00000" == make_chunk_id("000001_2024", KEY, 0)
    assert (c.company, c.stock_code, c.year, c.chunk_index) == ("测试公司", "000001", 2024, 0)
    assert c.text_for_embedding == c.text  # v0 无标题路径
    assert c.chunker_name == KEY and c.parser_version == PV
    assert len(c.content_hash) == 16


def test_chunk_skips_image_blocks_and_counts_consecutively() -> None:
    img = Block("000001_2024", 1, "p1_b1", "image", "", [0, 0, 5, 5])
    blocks = [_block(1, "x" * 10), img, _block(2, "y" * 10, 2)]
    chunks = chunk_document(blocks, META, ChunkSettings(chunk_size=8, overlap=0), KEY, PV)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    assert "".join(c.text for c in chunks) == "x" * 10 + "\n" + "y" * 10  # 无重叠时拼回原文


def test_overlap_keeps_boundary_number_intact_in_one_chunk() -> None:
    # 数字 "1,234,567.89" 横跨第一个窗口的边界；有足够重叠时，第二个窗口能完整包含它
    text = "a" * 15 + "1,234,567.89" + "b" * 10
    no_overlap = chunk_document([_block(1, text)], META, ChunkSettings(chunk_size=20, overlap=0), KEY, PV)
    with_overlap = chunk_document([_block(1, text)], META, ChunkSettings(chunk_size=20, overlap=12), KEY, PV)
    assert not any("1,234,567.89" in c.text for c in no_overlap)  # 被切断
    assert any("1,234,567.89" in c.text for c in with_overlap)


def test_empty_document() -> None:
    assert chunk_document([], META, ChunkSettings(), KEY, PV) == []


def test_text_for_embedding_with_title() -> None:
    assert text_for_embedding(["第三节", "四、主营业务"], "正文") == "第三节 > 四、主营业务\n正文"


# ---------------- parse + 读写 ----------------


def test_parse_pdf_blocks_in_reading_order(tmp_path: Path) -> None:
    doc = pymupdf.open()
    p1 = doc.new_page()
    p1.insert_text((72, 300), "second block", fontsize=11)  # 位置靠下
    p1.insert_text((72, 100), "first block", fontsize=11)  # 位置靠上，但后插入
    doc.new_page()  # 空白页不产生块
    p3 = doc.new_page()
    p3.insert_text((72, 72), "营业收入", fontname="china-s", fontsize=11)
    doc.save(tmp_path / "t.pdf")
    doc.close()

    blocks = parse_pdf(tmp_path / "t.pdf", "000001_2024")
    assert [(b.page, b.text) for b in blocks] == [(1, "first block"), (1, "second block"), (3, "营业收入")]
    assert [b.block_id for b in blocks] == ["p1_b0", "p1_b1", "p3_b0"]
    assert all(b.type == "text" and b.title_path == [] for b in blocks)
    assert blocks[0].bbox[1] < blocks[1].bbox[1]  # y0 更小 = 更靠上


def test_blocks_and_chunks_roundtrip(tmp_path: Path) -> None:
    blocks = [_block(1, "营业收入"), _block(2, "净利润", 1)]
    write_blocks(blocks, tmp_path / "b.jsonl")
    assert read_blocks(tmp_path / "b.jsonl") == blocks
    chunks = chunk_document(blocks, META, ChunkSettings(chunk_size=5, overlap=1), KEY, PV)
    write_chunks(chunks, tmp_path / "c.jsonl")
    assert read_chunks(tmp_path / "c.jsonl") == chunks
