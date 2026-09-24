"""文本归一化：评测命中判定、评测集校验、引用校验共用同一个实现。

规则（AGENTS.md 4.4）：
    1. 全角转半角：全角 ASCII 区 U+FF01~U+FF5E 映射到 U+0021~U+007E，全角空格 U+3000 映射为普通空格；
    2. 去掉所有空白字符（空格、换行、制表符、不间断空格等）。
    数字里的千分位逗号保持原样（全角逗号"，"会变成半角","，这是全角转半角的正常结果）。

为什么要归一化：PDF 抽出的文本会在任意位置断行、插入空格，全半角也不统一；
不归一化的话，"1,234.56" 和 "1, 234.56" 或 "1，234.56" 会被判为不相等。
"""

from __future__ import annotations

# 全角字符与对应半角字符的码位差：0xFF01 - 0x0021 = 0xFEE0
_FULLWIDTH_OFFSET = 0xFEE0

# 预先构建转换表，str.translate 一次完成所有替换，比逐字符判断快得多
_FULL_TO_HALF = {code: code - _FULLWIDTH_OFFSET for code in range(0xFF01, 0xFF5F)}
_FULL_TO_HALF[0x3000] = 0x20  # 全角空格 -> 半角空格（随后会被当作空白去掉）


def to_halfwidth(text: str) -> str:
    """全角转半角：'（１２３，ＡＢＣ）' -> '(123,ABC)'。"""
    return text.translate(_FULL_TO_HALF)


def normalize(text: str) -> str:
    """归一化：全角转半角后去掉所有空白。

    >>> normalize("营业收入 １,２３４.５６\\n元")
    '营业收入1,234.56元'
    """
    # str.split() 不带参数时按任意空白切分（含 \\n \\t \\xa0 \\u3000 等），再拼接即去掉全部空白
    return "".join(to_halfwidth(text).split())
