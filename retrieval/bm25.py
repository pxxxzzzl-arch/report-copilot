"""BM25 关键词检索（手写实现，AGENTS.md 要求核心检索算法可讲解）。

为什么需要（experiments/badcases.md BC-003）：
    表格 chunk 以数字为主，指标名只有几个字，稠密向量对"研发投入合计""归属于上市公司股东的净利润"
    这类精确术语不敏感；BM25 按词精确匹配，正好补上这块短板。

打分公式（Robertson / Lucene 变体）：对查询中的每个词 t（去重）累加
    idf(t) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgdl))
    idf(t) = ln(1 + (N - df + 0.5) / (df + 0.5))       —— 加 1 保证 idf 恒为正（常见词不会产生负分）
    tf     该词在 chunk 中出现的次数；dl 为 chunk 的词数；avgdl 为全部 chunk 的平均词数
    k1     词频饱和速度：tf 越大得分越高，但增长越来越慢（k1 越小饱和越快）
    b      长度归一化强度：b=1 完全按长度惩罚长 chunk，b=0 不惩罚

分词：jieba 搜索模式（同时输出长词和其中的子词，如"归属于""归属""属于"，提高召回），
      去掉标点和少量虚词；全角转半角、英文转小写。查询和 chunk 用同一个分词函数。
"""

from __future__ import annotations

import heapq
import logging
import math
import re
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import replace

import jieba

from common.text import to_halfwidth
from ingest.chunk import Chunk
from retrieval.query import QueryFilter

jieba.setLogLevel(logging.WARNING)  # 屏蔽 jieba 加载词典时的调试输出
logger = logging.getLogger("retrieval.bm25")

# 只含标点、符号、空白的 token 不参与检索
_MEANINGFUL = re.compile(r"[0-9A-Za-z一-鿿]")
# 高频虚词和疑问词：出现在几乎所有问题里，对检索没有区分作用
STOPWORDS = frozenset({"的", "是", "了", "和", "与", "及", "在", "为", "多少", "什么", "哪些", "哪家", "哪个", "如何", "是否", "分别"})

Tokenizer = Callable[[str], list[str]]


def tokenize(text: str) -> list[str]:
    tokens = jieba.lcut_for_search(to_halfwidth(text).lower())
    return [t for t in tokens if t not in STOPWORDS and _MEANINGFUL.search(t)]


class BM25Index:
    """倒排索引 + BM25 打分。只依赖"每个文档的词列表"，与 chunk 结构无关，便于手算测试。"""

    def __init__(self, docs_tokens: Sequence[Sequence[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.n_docs = len(docs_tokens)
        self.doc_len = [len(toks) for toks in docs_tokens]
        self.avgdl = sum(self.doc_len) / self.n_docs if self.n_docs else 0.0
        # 倒排表：词 -> [(文档下标, 词频), ...]；打分时只遍历包含查询词的文档
        self.postings: dict[str, list[tuple[int, int]]] = {}
        for i, toks in enumerate(docs_tokens):
            for term, tf in Counter(toks).items():
                self.postings.setdefault(term, []).append((i, tf))

    def idf(self, term: str) -> float:
        df = len(self.postings.get(term, ()))
        return math.log(1 + (self.n_docs - df + 0.5) / (df + 0.5))

    def scores(self, query_tokens: Sequence[str]) -> dict[int, float]:
        """返回 {文档下标: BM25 分数}，只包含至少命中一个查询词的文档。查询词去重后计分。"""
        out: dict[int, float] = {}
        for term in set(query_tokens):
            postings = self.postings.get(term)
            if not postings:
                continue
            idf = self.idf(term)
            for i, tf in postings:
                norm = self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avgdl)
                out[i] = out.get(i, 0.0) + idf * tf * (self.k1 + 1) / (tf + norm)
        return out


class BM25Retriever:
    """BM25 召回通道，满足 retrieval.search.Source 协议。"""

    def __init__(self, chunks: Sequence[Chunk], k1: float, b: float, tokenizer: Tokenizer = tokenize) -> None:
        self.chunks = list(chunks)
        self.tokenizer = tokenizer
        self.index = BM25Index([tokenizer(c.text_for_embedding) for c in self.chunks], k1, b)

    def search(self, query: str, top_k: int, flt: QueryFilter | None = None) -> list[Chunk]:
        scores = self.index.scores(self.tokenizer(query))
        if flt is not None and not flt.is_empty:
            scores = {i: s for i, s in scores.items() if flt.matches(self.chunks[i].stock_code, self.chunks[i].year)}
        # 同分按 chunk 下标排序，保证结果确定
        top = heapq.nlargest(top_k, scores.items(), key=lambda kv: (kv[1], -kv[0]))
        return [replace(self.chunks[i], extra={**self.chunks[i].extra, "bm25_score": s}) for i, s in top]
