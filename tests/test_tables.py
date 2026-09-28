"""ingest/tables.py 的单元测试：用合成的表格行（模仿 pdfplumber 的真实输出形态）验证抽取逻辑。"""

from decimal import Decimal

import pytest

from ingest.tables import (
    canonical_label,
    extract_from_tables,
    match_metric,
    merge_rows,
    page_unit,
    parse_number,
    verify_on_page,
)


@pytest.mark.parametrize(
    ("cell", "expected"),
    [
        ("168,838,102,514.79", Decimal("168838102514.79")),
        ("-1.21", Decimal("-1.21")),
        ("24.91%", Decimal("24.91")),
        ("423,701,834", Decimal("423701834")),
        ("减少3.49个百分点", None),
        ("2025年", None),
        ("-", None),
        ("", None),
        (None, None),
        ("1,23,4", None),  # 千分位格式不对
    ],
)
def test_parse_number(cell, expected) -> None:
    assert parse_number(cell) == expected


def test_canonical_label_strips_units_and_notes() -> None:
    assert canonical_label("加权平均净资产收益率（%）(1)") == "加权平均净资产收益率"
    assert canonical_label("基本每股收益（元/\n股）") == "基本每股收益"
    assert match_metric("资产总额") == "total_assets"
    # 相近但不同的指标不能误匹配
    assert match_metric("扣除非经常性损益后的基本每股收益（元／股）") is None


def test_merge_rows_joins_wrapped_labels() -> None:
    rows = [
        ["", "项目", "", "2025年", None, "2024年", "本年比上年增减", "2023年"],
        ["", "归属于上市公司股东", "", "72,201,282", None, "50,744,682", "42.28%", "44,121,248"],
        [None, "的净利润", None, None, None, None, None, None],
        ["加权平均净资产收益率（%）", "32.53", "36.02", "减少3.49个百分点", "34.19"],
    ]
    out = merge_rows(rows)
    assert out[0][0].startswith("<header>")
    assert out[1] == ("归属于上市公司股东的净利润", ["72,201,282", "50,744,682", "42.28%", "44,121,248"])
    # 数字之后的增减说明不能拼进指标名
    assert out[2] == ("加权平均净资产收益率（%）", ["32.53", "36.02", "34.19"])


def _header(y: int, extra: str = "本期比上年同期增减(%)"):
    return ["主要会计数据", f"{y}年", f"{y - 1}年", extra, f"{y - 2}年"]


def test_extract_units_years_and_stop() -> None:
    tables = [
        (6, [["会计师事务所", "名称", "某某会计师事务所"]]),  # 表头之前的无关表格：不在范围内
        (6, [
            _header(2025),
            ["营业收入", "1,000.50", "900.00", "11.17", "800.00"],
            ["总资产（千元）", "2,000", "1,500", "33.3", "1,000"],
        ]),
        (7, [  # 跨页续表，没有表头：沿用上一张表的状态
            ["基本每股收益（元/股）", "1.23", "1.00", "23.0", "0.9"],
            ["加权平均净资产收益率", "24.91%", "24.13%", "0.78%", "24.04%"],
        ]),
        (7, [["", "第一季度", "第二季度", "第三季度", "第四季度"], ["营业收入", "1", "2", "3", "4"]]),  # 季度表：停止
    ]  # fmt: skip
    values, issues = extract_from_tables(tables, 2025, "d", {6: "元", 7: "元"})
    got = {(v.fiscal_year, v.metric): v for v in values}
    assert issues == []
    assert got[(2025, "revenue")].value == Decimal("1000.50") and got[(2024, "revenue")].value == Decimal("900.00")
    # 指标名里的单位优先于页面标注：2,000 千元 = 2,000,000 元
    ta = got[(2025, "total_assets")]
    assert ta.value == Decimal("2000000") and ta.raw == "2,000" and ta.raw_unit == "千元" and ta.page == 6
    assert got[(2025, "eps_basic")].unit == "元/股" and got[(2025, "eps_basic")].page == 7
    assert got[(2024, "roe_weighted")].value == Decimal("24.13") and got[(2024, "roe_weighted")].unit == "%"
    assert len(values) == 8  # 季度表中的营业收入没有覆盖主表的值


def test_extract_rejects_wrong_year_order_and_unknown_unit() -> None:
    wrong_order = [(1, [["项目", "2023年", "2024年", "增减"], ["营业收入", "1.00", "2.00", "1%"]])]
    assert extract_from_tables(wrong_order, 2024, "d", {1: "元"}) == ([], [])
    no_unit = [(1, [_header(2024), ["营业收入", "1.00", "2.00", "1", "3"]])]
    values, issues = extract_from_tables(no_unit, 2024, "d", {1: None})
    assert values == [] and "无法确定金额单位" in issues[0]


def test_page_unit_and_verify_on_page() -> None:
    text = "单位：万元 其他表\n主要会计数据\n单位：千元 币种：人民币\n营业收入 423,701,834"
    assert page_unit(text) == "千元"  # 取"主要会计数据"之后的第一个单位标注
    values, _ = extract_from_tables([(3, [_header(2025), ["营业收入", "423,701,834", "1", "2", "3"]])], 2025, "d", {3: "千元"})
    ok, issues = verify_on_page(values, {3: text})
    # 上年数 "1" 虽然是 "423,701,834" 的子串，但不是页面上的完整数字，必须被丢弃
    assert [v.fiscal_year for v in ok] == [2025] and "原文校验失败" in issues[0]


@pytest.mark.parametrize(
    ("text", "raw", "ok"),
    [
        ("营业收入 423,701,834 元", "423,701,834", True),
        ("营业收入 423,701,834", "1", False),
        ("净资产收益率 24.91%。", "24.91%", True),
        ("每股 1.234 元", "1.23", False),  # 小数被截断的情况
        ("合计 12,345.67", "2,345.67", False),
        ("增长 3.2,下降 1.0", "3.2", True),  # 逗号后面不是数字，仍算完整数字
        ("423,701,834 362,012,554 17.04%", "362,012,554", True),  # 表格相邻两列：空白是边界，不能被去掉
        ("营业收入 ４２３", "423", True),  # 全角数字
    ],
)
def test_contains_number(text: str, raw: str, ok: bool) -> None:
    from ingest.tables import contains_number

    assert contains_number(text, raw) is ok
