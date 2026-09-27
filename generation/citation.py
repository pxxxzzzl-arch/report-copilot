"""引用校验：模型给出的引用是不是真的。

对每条引用 (ref, quote)：
    1. 编号有效：1 <= ref <= 上下文中的证据条数；
    2. 摘录可核验：normalize(quote) 出现在第 ref 条证据（chunk）的 normalize(text) 中
       —— 模型只能从证据里逐字摘录，改写或编造的摘录都不算；
    3. 定位页码：在该 chunk 的页码范围内，找出 PDF 原文包含这段摘录的具体页；
       chunk 跨页、摘录恰好也跨页时找不到单独一页，page 为 None，此时用 chunk 的页码范围。
定位到的 (doc_id, page) 用于：评测"引用是否覆盖 gold 所在页"，以及阶段 9 在前端高亮原文。
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from common.text import normalize
from generation.answer import Citation
from ingest.chunk import Chunk

# 按 doc_id 返回每页归一化后的文本（下标 0 为第 1 页），与 eval.validate_goldset.PdfPageTexts 相同
PageTextLoader = Callable[[str], list[str]]


@dataclass
class CitationCheck:
    ref: int
    quote: str
    valid_ref: bool
    verified: bool  # 摘录确实出现在对应证据中
    doc_id: str | None = None
    page: int | None = None  # 定位到的具体页（物理页码）；无法定位到单页时为 None
    page_start: int | None = None
    page_end: int | None = None

    def locations(self) -> set[tuple[str, int]]:
        """这条引用指向的 (doc_id, page) 集合；未通过核验的引用不指向任何位置。"""
        if not self.verified or self.doc_id is None:
            return set()
        if self.page is not None:
            return {(self.doc_id, self.page)}
        return {(self.doc_id, p) for p in range(self.page_start, self.page_end + 1)}


def locate_page(quote_norm: str, chunk: Chunk, page_texts: PageTextLoader) -> int | None:
    """在 chunk 的页码范围内找出包含摘录的第一页。"""
    pages = page_texts(chunk.doc_id)
    for p in range(chunk.page_start, chunk.page_end + 1):
        if quote_norm in pages[p - 1]:
            return p
    return None


def check_citations(
    citations: Sequence[Citation], chunks: Sequence[Chunk], page_texts: PageTextLoader
) -> list[CitationCheck]:
    out = []
    for c in citations:
        if not 1 <= c.ref <= len(chunks):
            out.append(CitationCheck(c.ref, c.quote, valid_ref=False, verified=False))
            continue
        chunk = chunks[c.ref - 1]
        q = normalize(c.quote)
        verified = bool(q) and q in normalize(chunk.text)
        page = locate_page(q, chunk, page_texts) if verified else None
        out.append(
            CitationCheck(
                c.ref, c.quote, valid_ref=True, verified=verified, doc_id=chunk.doc_id, page=page,
                page_start=chunk.page_start, page_end=chunk.page_end,
            )  # fmt: skip
        )
    return out
