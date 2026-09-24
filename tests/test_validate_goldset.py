"""eval/validate_goldset.py 的单元测试。

用内存中的假 manifest 和假页面文本代替真实 PDF，每个用例只构造一个错误，便于定位。
"""

import json
from pathlib import Path

import pymupdf
import pytest

from common.text import normalize
from eval.validate_goldset import PdfPageTexts, check_item, find_span_pages, load_goldset, validate
from ingest.download import ManifestRow

# 两份文档：A 公司 2024、2025 年报，各 3 页
_PAGES = {
    "000001_2024": ["封面", "营业收入 1,234,567.89 元", "合并利润表 营业收入 1,234,567.89"],
    "000001_2025": ["封面", "营业收入 2,000,000.00 元 上年同期 1,234,567.89", "董事长 张三"],
}


def _row(doc_id: str) -> ManifestRow:
    return ManifestRow(doc_id, "A公司", "000001", int(doc_id[-4:]), f"{doc_id}.pdf", 3, "SZSE", "", "", "")


MANIFEST = {d: _row(d) for d in _PAGES}


def page_texts(doc_id: str) -> list[str]:
    return [normalize(t) for t in _PAGES[doc_id]]


def _item(**kw) -> dict:
    base = {
        "id": "q001",
        "type": "table_number",
        "question": "A公司2024年营业收入是多少？",
        "answer": "1,234,567.89元",
        "gold": [{"doc_id": "000001_2024", "page": 2, "span": "1,234,567.89"}],
    }
    base.update(kw)
    return base


def _errors(item: dict) -> list[str]:
    return [i.msg for i in check_item(item, MANIFEST, page_texts) if i.level == "error"]


def _warnings(item: dict) -> list[str]:
    return [i.msg for i in check_item(item, MANIFEST, page_texts) if i.level == "warning"]


# ---------------- 正常用例 ----------------


def test_valid_item_with_alt_has_no_issue() -> None:
    gold = [
        {
            "doc_id": "000001_2024",
            "page": 2,
            "span": "1,234,567.89",
            "alt": [{"doc_id": "000001_2024", "page": 3}, {"doc_id": "000001_2025", "page": 2}],
        }
    ]
    assert check_item(_item(gold=gold), MANIFEST, page_texts) == []


def test_span_matched_after_normalization() -> None:
    # 标注里有空格和全角字符，归一化后仍能匹配
    gold = [{"doc_id": "000001_2025", "page": 3, "span": "董事长　张三"}]
    assert _errors(_item(type="fact", gold=gold)) == []


def test_out_of_scope_valid() -> None:
    assert _errors(_item(type="out_of_scope", gold=[], answer="年报未披露，无法回答")) == []


# ---------------- 格式错误 ----------------


def test_missing_field() -> None:
    item = _item()
    del item["answer"]
    assert any("缺少字段" in m for m in _errors(item))


@pytest.mark.parametrize("bad_id", ["1", "q1", "Q001", "q0001"])
def test_bad_id(bad_id: str) -> None:
    assert any("id 应形如" in m for m in _errors(_item(id=bad_id)))


def test_unknown_type() -> None:
    assert any("未知题型" in m for m in _errors(_item(type="opinion")))


def test_empty_answer() -> None:
    assert any("不能为空" in m for m in _errors(_item(answer="  ")))


# ---------------- gold 规则 ----------------


def test_out_of_scope_must_have_empty_gold() -> None:
    assert any("必须为空" in m for m in _errors(_item(type="out_of_scope")))


def test_non_oos_must_have_gold() -> None:
    assert any("不能为空" in m for m in _errors(_item(gold=[])))


@pytest.mark.parametrize("qtype", ["multi_hop", "calc", "compare"])
def test_multi_evidence_needs_two_gold(qtype: str) -> None:
    assert any("至少需要 2 条" in m for m in _errors(_item(type=qtype)))


def test_compare_single_doc_warns() -> None:
    gold = [
        {"doc_id": "000001_2024", "page": 2, "span": "1,234,567.89"},
        {"doc_id": "000001_2024", "page": 1, "span": "封面"},
    ]
    assert any("只来自 1 份文档" in m for m in _warnings(_item(type="compare", gold=gold)))


# ---------------- 位置与原文 ----------------


def test_unknown_doc() -> None:
    gold = [{"doc_id": "999999_2024", "page": 1, "span": "封面"}]
    assert any("不在 manifest" in m for m in _errors(_item(gold=gold)))


@pytest.mark.parametrize("page", [0, 4, "2"])
def test_page_out_of_range_or_not_int(page) -> None:
    gold = [{"doc_id": "000001_2024", "page": page, "span": "封面"}]
    assert any("page 越界" in m for m in _errors(_item(gold=gold)))


def test_span_not_on_page() -> None:
    gold = [{"doc_id": "000001_2024", "page": 1, "span": "1,234,567.89"}]  # 在第 2、3 页，不在第 1 页
    assert any("找不到 span" in m for m in _errors(_item(gold=gold)))


def test_alt_span_must_exist_too() -> None:
    gold = [{"doc_id": "000001_2024", "page": 2, "span": "1,234,567.89", "alt": [{"doc_id": "000001_2025", "page": 3}]}]
    assert any("alt[0]" in m and "找不到 span" in m for m in _errors(_item(gold=gold)))


def test_alt_duplicate_of_primary() -> None:
    gold = [{"doc_id": "000001_2024", "page": 2, "span": "1,234,567.89", "alt": [{"doc_id": "000001_2024", "page": 2}]}]
    assert any("重复" in m for m in _errors(_item(gold=gold)))


def test_span_elsewhere_not_in_alt_warns() -> None:
    # 第 3 页也有该数字，但没列入 alt
    assert any("还出现在" in m and "[3]" in m for m in _warnings(_item()))


def test_short_span_warns() -> None:
    gold = [{"doc_id": "000001_2024", "page": 1, "span": "封面"}]
    assert any("过短" in m for m in _warnings(_item(type="fact", gold=gold)))


def test_find_span_pages() -> None:
    assert find_span_pages("1,234,567.89", page_texts("000001_2024")) == [2, 3]
    assert find_span_pages("", page_texts("000001_2024")) == []  # 空 span 不应匹配所有页


# ---------------- 整体 ----------------


def test_validate_duplicate_ids_and_doc_coverage() -> None:
    report = validate([_item(), _item()], MANIFEST, page_texts)
    assert any("id 重复" in i.msg for i in report.errors)
    assert report.type_counts["table_number"] == 2
    assert report.doc_counts["000001_2024"] == 2
    # 000001_2025 没有题目引用 -> 覆盖不足的 warning
    assert any("000001_2025" in i.msg for i in report.warnings)


def test_load_goldset_reports_bad_line(tmp_path: Path) -> None:
    path = tmp_path / "g.jsonl"
    path.write_text(json.dumps(_item(), ensure_ascii=False) + "\n\n{bad json\n", encoding="utf-8")
    items, issues = load_goldset(path)
    assert len(items) == 1
    assert len(issues) == 1 and issues[0].qid == "line 3"


def test_pdf_page_texts_reads_and_normalizes(tmp_path: Path) -> None:
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), "Revenue  1,234.56", fontsize=11)
    doc.new_page()
    doc.save(tmp_path / "x.pdf")
    doc.close()
    loader = PdfPageTexts({"x": ManifestRow("x", "", "", 0, "x.pdf", 2, "", "", "", "")}, tmp_path)
    assert loader("x") == ["Revenue1,234.56", ""]
    assert loader("x") is loader("x")  # 第二次走缓存


# ---------------- 改写题（v2） ----------------


def _para(**kw) -> dict:
    base = {**_item(), "id": "q101", "question": "A公司2024年卖了多少钱？", "variant": "paraphrase", "source_id": "q001"}
    base.update(kw)
    return base


def _variant_errors(items: list[dict]) -> list[str]:
    return [i.msg for i in validate(items, MANIFEST, page_texts).issues if i.level == "error"]


def test_valid_paraphrase() -> None:
    assert _variant_errors([_item(), _para()]) == []


def test_paraphrase_must_match_source() -> None:
    assert any("answer 必须与原题" in m for m in _variant_errors([_item(), _para(answer="别的答案")]))
    assert any("source_id" in m for m in _variant_errors([_item(), _para(source_id="q999")]))
    assert any("与原题相同" in m for m in _variant_errors([_item(), _para(question=_item()["question"])]))
    assert any("未知 variant" in m for m in _variant_errors([_item(variant="other")]))


def test_paraphrase_leaking_span_warns() -> None:
    items = [_item(), _para(question="A公司2024年营业收入是1,234,567.89元吗？")]
    warnings = [i.msg for i in validate(items, MANIFEST, page_texts).warnings]
    assert any("gold 原文片段" in m for m in warnings)
