"""用于检验评测流程本身的合成检索器（不是检索方案）。

思路：用"已知答案"的输入检验测量工具。
    oracle       直接返回 gold 所在页的整页文本 -> 各项指标应接近满分；
                 达不到满分说明命中判定、评测集标注或页面文本之间有不一致。
    random_page  从全部页面中随机取 -> 各项指标应接近 0，给出"瞎猜"的下限。

两者都以"整页"作为一个 chunk（page_start == page_end == 该页）。
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass

from ingest.download import ManifestRow

PageTextLoader = Callable[[str], list[str]]


@dataclass(frozen=True)
class PageChunk:
    """以整页为单位的 chunk，满足 eval.metrics.ChunkLike。"""

    chunk_id: str
    doc_id: str
    page_start: int
    page_end: int
    text: str


def _page_chunk(doc_id: str, page: int, page_texts: PageTextLoader) -> PageChunk:
    return PageChunk(f"{doc_id}_p{page}", doc_id, page, page, page_texts(doc_id)[page - 1])


class OracleRetriever:
    """按问题原文查出 gold，按 gold 顺序返回各条 gold 主位置所在的页（同一页只返回一次）。"""

    name = "oracle"
    config = {"retriever": "oracle", "chunker": "page"}

    def __init__(self, items: list[dict], page_texts: PageTextLoader) -> None:
        self._gold_by_question = {it["question"]: it["gold"] for it in items}
        self._page_texts = page_texts

    def search(self, query: str, top_k: int) -> list[PageChunk]:
        pages: list[tuple[str, int]] = []
        for g in self._gold_by_question.get(query, []):
            loc = (g["doc_id"], g["page"])
            if loc not in pages:  # 多条 gold 在同一页时只返回一次
                pages.append(loc)
        return [_page_chunk(d, p, self._page_texts) for d, p in pages[:top_k]]


class RandomPageRetriever:
    """从所有文档的所有页中随机抽取 top_k 页。

    随机数种子 = 全局 seed + 问题原文，所以同一问题每次结果都相同（可复现），不同问题互不相关。
    """

    name = "random_page"

    def __init__(self, manifest: dict[str, ManifestRow], page_texts: PageTextLoader, seed: int = 0) -> None:
        self._all_pages = [(d, p) for d in sorted(manifest) for p in range(1, manifest[d].n_pages + 1)]
        self._page_texts = page_texts
        self._seed = seed
        self.config = {"retriever": "random_page", "chunker": "page", "seed": seed}

    def search(self, query: str, top_k: int) -> list[PageChunk]:
        rng = random.Random(f"{self._seed}:{query}")
        return [_page_chunk(d, p, self._page_texts) for d, p in rng.sample(self._all_pages, top_k)]
