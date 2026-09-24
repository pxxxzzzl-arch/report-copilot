"""RRF（Reciprocal Rank Fusion，倒数排名融合）：把多路召回的排序结果合成一个排序。

公式：一个文档的融合分 = Σ_每一路 1 / (k + 该文档在这一路中的排名)，排名从 1 开始；
      某一路没有召回这个文档，这一路就不贡献分数。

为什么用排名而不用原始分数：
    dense 的余弦相似度、BM25 分数、sparse 内积的量纲和分布完全不同，直接相加没有意义；
    RRF 只看排名，天然不需要归一化，也不需要调权重。
k 的作用：k 越大，头部排名之间的分差越小（越"民主"）；k=60 是原论文（Cormack et al., 2009）的默认值。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

from ingest.chunk import Chunk


def rrf_scores(ranked_ids: Sequence[Sequence[str]], k: int) -> dict[str, float]:
    """输入每一路的 id 排序列表，返回每个 id 的 RRF 分数。"""
    scores: dict[str, float] = {}
    for ids in ranked_ids:
        for rank, doc_id in enumerate(ids, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return scores


def rrf_fuse(ranked_lists: Sequence[Sequence[Chunk]], k: int) -> list[Chunk]:
    """融合多路 chunk 排序，返回按 RRF 分数降序排列的去重 chunk 列表。

    同分时的次序：先比在各路中的最好排名（越靠前越优先），再比 chunk_id，保证结果确定、可复现。
    融合分写入 chunk.extra["rrf_score"]。
    """
    scores = rrf_scores([[c.chunk_id for c in lst] for lst in ranked_lists], k)
    best_rank: dict[str, int] = {}
    chunk_by_id: dict[str, Chunk] = {}
    for lst in ranked_lists:
        for rank, c in enumerate(lst, start=1):
            best_rank[c.chunk_id] = min(rank, best_rank.get(c.chunk_id, rank))
            chunk_by_id.setdefault(c.chunk_id, c)
    order = sorted(scores, key=lambda cid: (-scores[cid], best_rank[cid], cid))
    return [replace(chunk_by_id[cid], extra={**chunk_by_id[cid].extra, "rrf_score": scores[cid]}) for cid in order]
