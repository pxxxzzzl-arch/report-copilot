"""api/pages.py 的测试：在测试里用 PyMuPDF 生成一个小 PDF（中文、跨行、表格里相邻的数字列），不依赖真实年报。"""

from pathlib import Path

import pymupdf
import pytest

from api.pages import find_all, locate_quote, page_chars, render_page_png


@pytest.fixture
def pdf(tmp_path: Path) -> Path:
    doc = pymupdf.open()
    page = doc.new_page()
    # 内置中文字体 china-s；四行：普通句子、被换行切开的一句话（两行）、表格一行（两列数字之间只有空格）
    page.insert_text((72, 72), "公司主营国窖1573、泸州老窖特曲等系列白酒", fontname="china-s", fontsize=11)
    page.insert_text((72, 90), "报告期内研发投入合计为", fontname="china-s", fontsize=11)
    page.insert_text((72, 108), "695,376,735.81元，同比增长", fontname="china-s", fontsize=11)
    # 这一行不能超出页宽（595 点），否则行尾会被截掉
    page.insert_text((72, 126), "现金流量净额（元） 19,181,768,363.65 10,648,364,935.46", fontname="china-s", fontsize=11)
    path = tmp_path / "t.pdf"
    doc.save(path)
    return path


def _page(path: Path):
    return pymupdf.open(path)[0]


def test_exact_match(pdf: Path) -> None:
    loc = locate_quote(_page(pdf), "国窖1573、泸州老窖特曲")
    assert loc.method == "exact" and len(loc.rects) == 1


def test_cross_line_quote_uses_char_alignment(pdf: Path) -> None:
    # 摘录跨过换行（模型摘录时两行之间没有空格）：search_for 找不到，逐字对齐能找到，并按行给出两个矩形
    page = _page(pdf)
    quote = "研发投入合计为695,376,735.81元"
    assert not page.search_for(quote)
    loc = locate_quote(page, quote)
    assert loc.method == "chars" and len(loc.rects) == 2
    ys = sorted(r[1] for r in loc.rects)
    assert ys[1] - ys[0] > 10  # 两个矩形分别在两行上


def test_tool_sentence_falls_back_to_number(pdf: Path) -> None:
    # 财务指标工具生成的句子不在 PDF 上，但其中的原始数字在：只高亮这个数字
    loc = locate_quote(_page(pdf), "现金流量净额（元），2024年为 19,181,768,363.65（原文单位：元）")
    assert loc.method == "number" and len(loc.rects) == 1


def test_number_boundary_with_adjacent_columns(pdf: Path) -> None:
    pc = page_chars(_page(pdf))
    # 去空白后两列数字粘在一起（…363.6510,648…），靠 gaps 仍能判断出完整数字的边界
    assert "363.6510,648" in pc.text
    assert len(find_all(pc, "19,181,768,363.65", whole_number=True)) == 1
    assert len(find_all(pc, "10,648,364,935.46", whole_number=True)) == 1
    # 更长数字的一部分不算完整数字
    assert find_all(pc, "181,768,363.65", whole_number=True) == []
    assert find_all(pc, "181,768,363.65") != []


def test_not_found(pdf: Path) -> None:
    assert locate_quote(_page(pdf), "完全不存在的内容 88,888.88").method == "none"
    assert locate_quote(_page(pdf), "   ").method == "none"


def test_render_png(pdf: Path) -> None:
    png, method, n = render_page_png(pdf, 1, "国窖1573", 1.0)
    assert png[:8] == b"\x89PNG\r\n\x1a\n" and method == "exact" and n == 1
    assert render_page_png(pdf, 1, "国窖1573", 1.0)[0] is png  # 相同参数命中缓存
    assert render_page_png(pdf, 1, "", 1.0)[1:] == ("none", 0)
    # 只在内存里画框，不修改 PDF 文件
    assert not pymupdf.open(pdf)[0].get_drawings()
