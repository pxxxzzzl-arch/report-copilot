"""查询理解：从问题中识别公司和年份，转成检索时的元数据过滤条件。

为什么需要（见 experiments/badcases.md BC-001、BC-002）：
    稠密向量很难区分"哪家公司、哪一年"——公司名反复出现的子公司名单会挤到前排，
    结构相同的上一年年报也会排在前面。公司、年份是结构化条件，应该交给过滤而不是语义匹配。

规则（刻意保持简单、可解释）：
    公司  问题中出现 manifest 里的公司简称、股票代码，或配置中的别名（如"茅台"）即算识别到；
          可以识别出多家（比较题）
    年份  匹配"四位数字 + 年"，如"2024年""2024 年度""2025年报"；可以识别出多个
    过滤  识别到公司 -> stock_code 必须在其中；识别到年份 -> year 必须在其中；都没识别到 -> 不过滤
          年份按"严格口径"过滤：问 2024 年就只查 2024 年报（见 docs/decisions.md D-014）
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ingest.download import ManifestRow

# 四位年份后紧跟"年"（中间允许空格），避免把金额、股数里的四位数字误认成年份
_YEAR_RE = re.compile(r"((?:19|20)\d{2})\s*年")


@dataclass(frozen=True)
class QueryFilter:
    """检索过滤条件；空元组表示该维度不限制。"""

    stock_codes: tuple[str, ...] = ()
    years: tuple[int, ...] = ()

    @property
    def is_empty(self) -> bool:
        return not self.stock_codes and not self.years

    def matches(self, stock_code: str, year: int) -> bool:
        """判断一个文档是否满足条件（给不经过 Qdrant 的召回通道用，如 BM25）。"""
        return (not self.stock_codes or stock_code in self.stock_codes) and (not self.years or year in self.years)


class QueryParser:
    def __init__(self, manifest: dict[str, ManifestRow], aliases: dict[str, str]) -> None:
        # 名称 -> 股票代码：公司简称、股票代码本身、额外别名
        names: dict[str, str] = {}
        for row in manifest.values():
            names[row.company] = row.stock_code
            names[row.stock_code] = row.stock_code
        names.update(aliases)
        self._names = names

    def parse(self, query: str) -> QueryFilter:
        # "贵州茅台"和"茅台"会同时匹配，但映射到同一个代码，用集合去重
        codes = sorted({code for name, code in self._names.items() if name in query})
        years = sorted({int(y) for y in _YEAR_RE.findall(query)})
        return QueryFilter(stock_codes=tuple(codes), years=tuple(years))
