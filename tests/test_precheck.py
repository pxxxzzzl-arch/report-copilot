"""ingest/precheck.py 的单元测试。

几何与文本指标都用手算可验证的用例；端到端测试用 PyMuPDF 现场生成 PDF，不依赖真实年报。
"""

import csv
import json
from pathlib import Path

import pymupdf
import pytest

from config import PrecheckSettings, get_settings
from ingest.download import ManifestRow, write_manifest
from ingest.precheck import (
    bad_char_ratio,
    classify_page,
    clip_rect,
    compress_ranges,
    precheck_pdf,
    run,
    union_area,
)

CFG = PrecheckSettings(min_text_chars=30, scan_image_ratio=0.5, garbled_ratio=0.1)


# ---------------- union_area（手算用例） ----------------


def test_union_area_empty() -> None:
    assert union_area([]) == 0


def test_union_area_disjoint() -> None:
    # 2x2 + 3x1 = 4 + 3 = 7
    assert union_area([(0, 0, 2, 2), (10, 10, 13, 11)]) == 7


def test_union_area_partial_overlap() -> None:
    # 两个 2x2 正方形，重叠 1x1：4 + 4 - 1 = 7
    assert union_area([(0, 0, 2, 2), (1, 1, 3, 3)]) == 7


def test_union_area_contained() -> None:
    # 小矩形完全在大矩形内，只算大矩形：10x10 = 100
    assert union_area([(0, 0, 10, 10), (2, 2, 5, 5)]) == 100


def test_union_area_duplicates() -> None:
    # 同一位置放了三次同一张图，只算一次
    assert union_area([(0, 0, 4, 5)] * 3) == 20


def test_union_area_cross_shape() -> None:
    # 十字形：横条 6x2 + 竖条 2x6 - 中心重叠 2x2 = 12 + 12 - 4 = 20
    assert union_area([(0, 2, 6, 4), (2, 0, 4, 6)]) == 20


def test_union_area_ignores_degenerate() -> None:
    assert union_area([(0, 0, 0, 5), (1, 1, 3, 1), (0, 0, 1, 1)]) == 1


def test_clip_rect() -> None:
    page = (0, 0, 100, 100)
    assert clip_rect((-10, -10, 50, 50), page) == (0, 0, 50, 50)  # 出血图裁剪到页面内
    assert clip_rect((200, 200, 300, 300), page) is None  # 完全在页面外


# ---------------- 文本指标 ----------------


def test_bad_char_ratio() -> None:
    assert bad_char_ratio("") == 0
    assert bad_char_ratio("营业收入 1,234.56 元\n") == 0  # 正常中英文、数字、标点、换行
    # 去掉空白后 4 个字符：2 个私用区字符 + 1 个替换符 + 1 个正常字 -> 3/4
    assert bad_char_ratio(" \n�收") == pytest.approx(0.75)


def test_bad_char_ratio_whitespace_not_counted() -> None:
    # 换行符属于控制字符 Cc，但它是空白，不能算作乱码
    assert bad_char_ratio("甲\n\n\n乙\t\r丙") == 0


# ---------------- 页面分类（边界值） ----------------


@pytest.mark.parametrize(
    ("n_chars", "image_ratio", "bad", "kind"),
    [
        (500, 0.0, 0.0, "text"),
        (500, 0.9, 0.0, "text"),  # 文字多、图片也多（图文混排页）仍是 text
        (30, 0.0, 0.0, "text"),  # 恰好等于 min_text_chars 时视为有文字
        (29, 0.0, 0.0, "blank"),
        (29, 0.5, 0.0, "scanned"),  # 恰好等于 scan_image_ratio 时视为扫描页
        (0, 0.49, 0.0, "blank"),
        (500, 0.0, 0.1, "text"),  # 恰好等于 garbled_ratio 不算乱码（严格大于才算）
        (500, 0.0, 0.11, "garbled"),
        (5, 0.0, 1.0, "blank"),  # 文字太少时不判断乱码
    ],
)
def test_classify_page(n_chars: int, image_ratio: float, bad: float, kind: str) -> None:
    assert classify_page(n_chars, image_ratio, bad, CFG) == kind


@pytest.mark.parametrize(
    ("pages", "expected"),
    [([], ""), ([5], "5"), ([1, 2, 3, 7, 9, 10], "1-3;7;9-10"), ([3, 1, 2, 2], "1-3")],
)
def test_compress_ranges(pages: list[int], expected: str) -> None:
    assert compress_ranges(pages) == expected


# ---------------- 端到端：现场生成 PDF ----------------


def _make_pdf(path: Path) -> None:
    """生成 4 页 PDF：
    1. 文字页（中英文）
    2. 扫描页：整页一张图，没有文字层
    3. 空白页
    4. 同一张图在页面顶部 1/4 区域放了两次（验证重叠只算一次：占比应为 0.25，而不是 0.5）
    """
    doc = pymupdf.open()
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 20, 20), False)
    pix.clear_with(200)

    p1 = doc.new_page()
    p1.insert_text((72, 72), "Revenue of the company in 2024 was 1,234,567.89 yuan.", fontsize=11)
    p1.insert_text((72, 100), "营业收入同比增长百分之十五", fontname="china-s", fontsize=11)

    p2 = doc.new_page()
    p2.insert_image(p2.rect, pixmap=pix)

    doc.new_page()

    p4 = doc.new_page()
    top_quarter = pymupdf.Rect(0, 0, p4.rect.width, p4.rect.height / 4)
    p4.insert_image(top_quarter, pixmap=pix, keep_proportion=False)
    p4.insert_image(top_quarter, pixmap=pix, keep_proportion=False)

    doc.save(path)
    doc.close()


def test_precheck_pdf_end_to_end(tmp_path: Path) -> None:
    pdf = tmp_path / "t.pdf"
    _make_pdf(pdf)
    report = precheck_pdf(pdf, "000001_2024", CFG)

    kinds = [p.kind for p in report.pages]
    assert kinds == ["text", "scanned", "blank", "blank"]
    assert [p.page for p in report.pages] == [1, 2, 3, 4]  # 物理页码从 1 开始

    p1, p2, _, p4 = report.pages
    assert p1.n_chars > 50
    assert p1.bad_char_ratio == 0
    assert p2.image_ratio == pytest.approx(1.0)
    assert p4.n_images == 2
    assert p4.image_ratio == pytest.approx(0.25)  # 两次放置完全重叠，只算一次

    s = report.summary()
    assert (s["n_pages"], s["n_text"], s["n_scanned"], s["n_blank"], s["n_garbled"]) == (4, 1, 1, 2, 0)
    assert s["needs_ocr"] is True
    assert s["ocr_pages"] == "2"


def test_run_writes_details_and_summary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """通过环境变量把所有路径指向临时目录，验证 run() 的完整输出。"""
    raw = tmp_path / "raw"
    raw.mkdir()
    _make_pdf(raw / "000001_2024.pdf")
    manifest = tmp_path / "manifest.csv"
    summary = tmp_path / "summary.csv"
    write_manifest(
        manifest,
        {
            "000001_2024": ManifestRow(
                doc_id="000001_2024", company="测试", stock_code="000001", year=2024,
                file_name="000001_2024.pdf", n_pages=4, exchange="SZSE", publish_date="2025-03-01",
                source_url="http://example/x.PDF", sha256="0" * 64,
            )  # fmt: skip
        },
    )
    monkeypatch.setenv("PATHS__RAW_DIR", str(raw))
    monkeypatch.setenv("PATHS__PARSED_DIR", str(tmp_path / "parsed"))
    monkeypatch.setenv("PATHS__MANIFEST_PATH", str(manifest))
    monkeypatch.setenv("PATHS__PRECHECK_SUMMARY_PATH", str(summary))
    get_settings.cache_clear()
    try:
        result = run()
    finally:
        get_settings.cache_clear()  # 恢复，避免影响其他测试

    assert result["000001_2024"]["n_scanned"] == 1
    lines = (tmp_path / "parsed" / "precheck" / "000001_2024.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["kind"] for line in lines] == ["text", "scanned", "blank", "blank"]
    with summary.open(encoding="utf-8") as f:
        (row,) = list(csv.DictReader(f))
    assert row["doc_id"] == "000001_2024" and row["needs_ocr"] == "True"
