"""common/text.py 的单元测试。"""

import pytest

from common.text import normalize, to_halfwidth


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("（１２３，ＡＢＣ）", "(123,ABC)"),
        ("！～", "!~"),  # 全角区首尾两个字符 U+FF01、U+FF5E
        ("营业收入。、", "营业收入。、"),  # 中文句号、顿号不在全角 ASCII 区，保持不变
        ("１,２３４", "1,234"),
    ],
)
def test_to_halfwidth(raw: str, expected: str) -> None:
    assert to_halfwidth(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("", ""),
        ("  \n\t ", ""),
        ("1, 234,\n567.89", "1,234,567.89"),  # PDF 抽取常见的断行、空格
        ("营业　收入", "营业收入"),  # 全角空格
        ("研发\xa0投入", "研发投入"),  # 不间断空格
        ("净利润（元）：１２３", "净利润(元):123"),
    ],
)
def test_normalize(raw: str, expected: str) -> None:
    assert normalize(raw) == expected


def test_normalize_idempotent() -> None:
    s = " 营业收入（元）\n１,２３４ "
    assert normalize(normalize(s)) == normalize(s)
