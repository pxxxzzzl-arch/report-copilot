"""ingest/download.py 的单元测试（不连网，只测纯函数与文件处理）。"""

import hashlib
import io
from datetime import date, datetime
from pathlib import Path

import pytest

from ingest.download import (
    Announcement,
    ManifestRow,
    StockInfo,
    build_query_form,
    clean_title,
    extract_report_year,
    parse_announcements,
    publish_window,
    read_manifest,
    save_pdf_stream,
    select_annual_reports,
    write_manifest,
)


def _ann(title: str, day: int = 1, url: str = "finalpage/x.PDF") -> Announcement:
    return Announcement(title=title, adjunct_url=url, publish_time=datetime(2025, 4, day), size_kb=100)


# ---------------- 标题解析 ----------------


@pytest.mark.parametrize(
    ("title", "year"),
    [
        ("贵州茅台2024年年度报告", 2024),
        ("2025年年度报告", 2025),
        ("招商银行股份有限公司2025年度报告", 2025),  # 没有重复的"年"字
        ("宁德时代 2024 年年度报告", 2024),  # 带空格
        ("2024年年度报告（更新后）", 2024),
        ("2024年半年度报告", None),  # 半年报不能被当作年报
        ("关于召开2024年度股东大会的通知", None),
        ("2024年第三季度报告", None),
    ],
)
def test_extract_report_year(title: str, year: int | None) -> None:
    assert extract_report_year(title) == year


def test_clean_title_removes_highlight_tags() -> None:
    assert clean_title(" <em>贵州茅台</em>2024年年度报告 ") == "贵州茅台2024年年度报告"


# ---------------- 公告筛选 ----------------


def test_select_excludes_summary_english_cancelled_hk() -> None:
    anns = [
        _ann("2024年年度报告摘要"),
        _ann("2024年年度报告（英文版）"),
        _ann("2024年年度报告（已取消）"),
        _ann("港股公告：2024年年报"),
        _ann("2024年年度报告", url="finalpage/good.PDF"),
    ]
    chosen = select_annual_reports(anns, {2024})
    assert chosen[2024].adjunct_url == "finalpage/good.PDF"


def test_select_prefers_revised_over_newer_original() -> None:
    """修订版优先级高于发布时间：即使原版发布得更晚，也选修订版。"""
    anns = [
        _ann("2024年年度报告（更新后）", day=1, url="revised.PDF"),
        _ann("2024年年度报告", day=20, url="original.PDF"),
    ]
    assert select_annual_reports(anns, {2024})[2024].adjunct_url == "revised.PDF"


def test_select_takes_latest_when_same_priority() -> None:
    anns = [_ann("2024年年度报告", day=1, url="old.PDF"), _ann("2024年年度报告", day=9, url="new.PDF")]
    assert select_annual_reports(anns, {2024})[2024].adjunct_url == "new.PDF"


def test_select_skips_before_correction_and_non_pdf() -> None:
    anns = [_ann("2024年年度报告（更正前）", url="a.PDF"), _ann("2024年年度报告", url="b.html")]
    assert select_annual_reports(anns, {2024}) == {}


def test_select_groups_by_year_and_filters_target_years() -> None:
    anns = [_ann("2023年年度报告", url="23.PDF"), _ann("2024年年度报告", url="24.pdf"), _ann("2025年年度报告", url="25.PDF")]
    chosen = select_annual_reports(anns, {2024, 2025})
    assert set(chosen) == {2024, 2025}
    assert chosen[2024].adjunct_url == "24.pdf"  # 小写扩展名也接受


# ---------------- 接口参数与返回解析 ----------------


def test_stock_exchange_and_column() -> None:
    assert StockInfo("600519", "贵州茅台", "gssh0600519").column == "sse"
    assert StockInfo("688981", "中芯国际", "x").exchange == "SSE"  # 科创板属于上交所
    assert StockInfo("000858", "五粮液", "x").column == "szse"
    assert StockInfo("300750", "宁德时代", "x").exchange == "SZSE"  # 创业板属于深交所
    with pytest.raises(ValueError):
        _ = StockInfo("830799", "北交所某股", "x").exchange


def test_build_query_form() -> None:
    form = build_query_form(StockInfo("600519", "贵州茅台", "gssh0600519"), date(2025, 1, 1), date(2026, 12, 31), 2)
    assert form["stock"] == "600519,gssh0600519"
    assert form["column"] == "sse"
    assert form["seDate"] == "2025-01-01~2026-12-31"
    assert form["pageNum"] == "2"
    assert form["category"] == "category_ndbg_szsh"


def test_publish_window() -> None:
    assert publish_window({2024, 2025}) == (date(2025, 1, 1), date(2026, 12, 31))


def test_parse_announcements() -> None:
    payload = {
        "announcements": [
            {
                "announcementTitle": "<em>贵州茅台</em>2024年年度报告",
                "adjunctUrl": "finalpage/2025-04-03/1.PDF",
                "announcementTime": 1743609600000,  # 2025-04-03 00:00 北京时间
                "adjunctSize": 3542,
            }
        ],
        "hasMore": False,
    }
    (ann,) = parse_announcements(payload)
    assert ann.title == "贵州茅台2024年年度报告"
    assert ann.size_kb == 3542
    assert ann.publish_time == datetime.fromtimestamp(1743609600)


def test_parse_announcements_null() -> None:
    # 没有结果时接口返回 "announcements": null
    assert parse_announcements({"announcements": None}) == []


# ---------------- 文件保存 ----------------


def test_save_pdf_stream_ok(tmp_path: Path) -> None:
    content = b"%PDF-1.7\n" + b"x" * 200_000  # 超过一个读取块，覆盖分块逻辑
    dest = tmp_path / "raw" / "a.pdf"
    sha = save_pdf_stream(io.BytesIO(content), dest)
    assert dest.read_bytes() == content
    assert sha == hashlib.sha256(content).hexdigest()
    assert not (tmp_path / "raw" / "a.pdf.part").exists()


def test_save_pdf_stream_rejects_non_pdf(tmp_path: Path) -> None:
    dest = tmp_path / "a.pdf"
    with pytest.raises(ValueError, match="不是 PDF"):
        save_pdf_stream(io.BytesIO(b"<html>error</html>"), dest)
    # 失败后既没有目标文件，也没有残留的临时文件
    assert list(tmp_path.iterdir()) == []


# ---------------- manifest ----------------


def _row(doc_id: str, code: str, year: int) -> ManifestRow:
    return ManifestRow(
        doc_id=doc_id, company="五粮液", stock_code=code, year=year, file_name=f"{doc_id}.pdf",
        n_pages=200, exchange="SZSE", publish_date="2025-04-26",
        source_url="http://static.cninfo.com.cn/x.PDF", sha256="ab" * 32,
    )  # fmt: skip


def test_manifest_roundtrip_keeps_leading_zero(tmp_path: Path) -> None:
    path = tmp_path / "manifest.csv"
    rows = {"000858_2025": _row("000858_2025", "000858", 2025), "000858_2024": _row("000858_2024", "000858", 2024)}
    write_manifest(path, rows)
    back = read_manifest(path)
    assert back == rows
    assert back["000858_2024"].stock_code == "000858"  # 前导 0 没丢
    # 按 doc_id 排序输出
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("doc_id,company,stock_code,year")
    assert lines[1].startswith("000858_2024") and lines[2].startswith("000858_2025")


def test_read_manifest_missing_file(tmp_path: Path) -> None:
    assert read_manifest(tmp_path / "nope.csv") == {}
