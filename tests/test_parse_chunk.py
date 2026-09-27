"""ingest/parse.py 与 ingest/chunk.py 的单元测试。"""

from pathlib import Path

import pymupdf
import pytest

from config import ChunkSettings
from ingest.chunk import (
    chunk_document,
    make_chunk_id,
    pack_blocks,
    read_chunks,
    text_for_embedding,
    window_spans,
    write_chunks,
)
from ingest.download import ManifestRow
from ingest.parse import Block, header_key, parse_pdf, read_blocks, remove_headers_footers, write_blocks

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


# ---------------- v1：去页眉页脚 ----------------


def _b(page: int, text: str, y0: float, y1: float) -> Block:
    return Block("d", page, f"p{page}_{text}", "text", text, [0, y0, 100, y1])


def test_header_key() -> None:
    assert header_key("8 / 143") == header_key("9/143") == "#/#"
    assert header_key("贵州茅台2024 年年度报告") == "贵州茅台#年年度报告"


def test_remove_headers_footers() -> None:
    heights = {p: 1000.0 for p in range(1, 11)}  # 10 页，每页高 1000；band=0.1 -> 顶部 y1<=100，底部 y0>=900
    blocks = []
    for p in range(1, 11):
        blocks.append(_b(p, "某公司2024年年度报告", 20, 40))  # 每页顶部：页眉
        blocks.append(_b(p, f"{p} / 10", 950, 970))  # 每页底部：页码
        blocks.append(_b(p, f"正文第{p}页", 300, 400))  # 正文
    blocks.append(_b(3, "附注（续）", 50, 70))  # 只在 2 页顶部出现（20% < 30%），不是页眉
    blocks.append(_b(4, "附注（续）", 50, 70))
    blocks.append(_b(5, "某公司2024年年度报告", 500, 520))  # 与页眉同文，但在页面中部：不删
    kept = remove_headers_footers(blocks, heights, band=0.1, min_page_ratio=0.3)
    texts = [b.text for b in kept]
    assert "某公司2024年年度报告" in texts and texts.count("某公司2024年年度报告") == 1  # 只剩页面中部那个
    assert not any("/ 10" in t for t in texts)  # 页码全部删除
    assert texts.count("附注（续）") == 2  # 低于阈值，保留
    assert sum(t.startswith("正文") for t in texts) == 10


def test_parse_pdf_v1_removes_repeated_header(tmp_path: Path) -> None:
    doc = pymupdf.open()
    for i in range(3):
        page = doc.new_page()  # 默认 A4：高 842pt
        page.insert_text((72, 30), "ACME annual report", fontsize=9)  # 顶部页眉
        page.insert_text((72, 400), f"body text {i}", fontsize=11)
    doc.save(tmp_path / "h.pdf")
    doc.close()
    v0 = [b.text for b in parse_pdf(tmp_path / "h.pdf", "d", "v0")]
    v1 = [b.text for b in parse_pdf(tmp_path / "h.pdf", "d", "v1")]
    assert v0.count("ACME annual report") == 3
    assert v1 == ["body text 0", "body text 1", "body text 2"]


# ---------------- block 切分（手算） ----------------


def test_pack_blocks_hand_calculated() -> None:
    # size=10：块长 4、4、4。块1+块2 = 4 + 1（分隔符）+ 4 = 9 <= 10；再加块3 = 14 > 10，另起一个 chunk
    blocks = [_block(1, "aaaa", 0), _block(1, "bbbb", 1), _block(2, "cccc", 2)]
    groups = pack_blocks(blocks, size=10, overlap=2)
    assert [text for text, _ in groups] == ["aaaa\nbbbb", "cccc"]
    assert [[b.text for b in bs] for _, bs in groups] == [["aaaa", "bbbb"], ["cccc"]]


def test_pack_blocks_exact_fit_and_oversized() -> None:
    # 恰好等于 size 时仍装得下：3 + 1 + 6 = 10
    assert [t for t, _ in pack_blocks([_block(1, "abc"), _block(1, "defghi", 1)], size=10, overlap=0)] == ["abc\ndefghi"]
    # 超长块（长 12 > 10）单独按窗口拆分：步长 10-2=8 -> [0,10)、[8,12)；前后的块不与它合并
    blocks = [_block(1, "xx"), _block(1, "0123456789AB", 1), _block(1, "yy", 2)]
    assert [t for t, _ in pack_blocks(blocks, size=10, overlap=2)] == ["xx", "0123456789", "89AB", "yy"]


def test_block_chunker_never_splits_a_block() -> None:
    # 与 fixed 的"数字被切断"用例对照：数字在自己的块里，block 策略不会把它切开
    blocks = [_block(1, "前一段"), _block(1, "1,234,567.89"), _block(2, "后一段")]
    chunks = chunk_document(blocks, META, ChunkSettings(chunker="block", chunk_size=20, overlap=0), KEY, PV)
    assert any("1,234,567.89" in c.text for c in chunks)
    assert [(c.page_start, c.page_end) for c in chunks] == [(1, 2)]  # 3+1+12+1+3 = 20，恰好装进一个 chunk
