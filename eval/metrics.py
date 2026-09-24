"""检索评测指标：命中判定 + Hit@K / Recall@K / Precision@K / MRR@K / NDCG@K。

核心思路：先把"检索结果"转成一个命中矩阵，所有指标都只基于它计算，与检索器实现无关。

    hits[i] = 排名第 i+1 的 chunk 命中了哪些 gold（gold 下标的集合，可能为空）

命中判定（AGENTS.md 4.4，eval/goldset_schema.md 第 3 节）：
    chunk 命中某条 gold，当且仅当该 gold 的某个位置（主位置或任一 alt）满足：
        1. doc_id 相同；
        2. 该位置的 page 落在 chunk 的 [page_start, page_end] 内；
        3. normalize(chunk.text) 包含 normalize(span)。

一道题可以有多条 gold（多跳、比较、计算题），每条是一份必需证据，Recall 的分母是 gold 条数。
所有指标都先逐题计算，再对题目取平均（macro average）。
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from common.text import normalize

HitMatrix = list[frozenset[int]]
METRIC_NAMES = ("hit", "recall", "precision", "mrr", "ndcg")


class ChunkLike(Protocol):
    """评测只依赖检索结果的这几个属性；阶段 4 的 Chunk 只要具备它们即可直接评测。"""

    chunk_id: str
    doc_id: str
    page_start: int
    page_end: int
    text: str


@dataclass(frozen=True)
class GoldEvidence:
    """一条 gold 证据：一个归一化后的 span + 若干个等价位置 (doc_id, page)。"""

    span_norm: str
    locations: tuple[tuple[str, int], ...]  # 第 0 个是主位置，其余来自 alt

    @classmethod
    def from_dict(cls, g: dict) -> GoldEvidence:
        locs = [(g["doc_id"], g["page"])] + [(a["doc_id"], a["page"]) for a in g.get("alt", [])]
        return cls(span_norm=normalize(g["span"]), locations=tuple(locs))


# ---------------------------------------------------------------------------
# 命中判定
# ---------------------------------------------------------------------------


def chunk_hits_gold(chunk: ChunkLike, ev: GoldEvidence, chunk_text_norm: str | None = None) -> bool:
    """判断一个 chunk 是否命中一条 gold。

    chunk_text_norm 可传入预先归一化好的文本，避免同一个 chunk 与多条 gold 比较时重复归一化。
    """
    in_location = any(
        doc_id == chunk.doc_id and chunk.page_start <= page <= chunk.page_end for doc_id, page in ev.locations
    )
    if not in_location:  # 先做廉价的位置判断，不满足就不必做字符串查找
        return False
    text = chunk_text_norm if chunk_text_norm is not None else normalize(chunk.text)
    return ev.span_norm in text


def hit_matrix(chunks: Sequence[ChunkLike], evidences: Sequence[GoldEvidence]) -> HitMatrix:
    """为每个排名位置计算它命中的 gold 下标集合。"""
    matrix: HitMatrix = []
    for chunk in chunks:
        text = normalize(chunk.text)
        matrix.append(frozenset(gi for gi, ev in enumerate(evidences) if chunk_hits_gold(chunk, ev, text)))
    return matrix


# ---------------------------------------------------------------------------
# 指标（输入都是命中矩阵；k 大于结果数时，缺的位置视为未命中）
# ---------------------------------------------------------------------------


def hit_at_k(hits: HitMatrix, k: int) -> float:
    """前 k 个结果中至少命中一条 gold 记为 1，否则为 0。"""
    return 1.0 if any(hits[:k]) else 0.0


def recall_at_k(hits: HitMatrix, n_gold: int, k: int) -> float:
    """前 k 个结果覆盖的 gold 条数 / gold 总条数。同一条 gold 被多个 chunk 命中只算一次。"""
    covered = set().union(*hits[:k]) if hits[:k] else set()
    return len(covered) / n_gold


def precision_at_k(hits: HitMatrix, k: int) -> float:
    """前 k 个结果中"命中了任意 gold"的个数 / k。

    分母固定为 k（而非实际返回数）：检索器少返回结果不应让精确率虚高。
    """
    return sum(1 for h in hits[:k] if h) / k


def mrr_at_k(hits: HitMatrix, k: int) -> float:
    """第一个命中结果的排名倒数（排名从 1 开始）；前 k 个都没命中记为 0。"""
    for rank, h in enumerate(hits[:k], start=1):
        if h:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(hits: HitMatrix, n_gold: int, k: int) -> float:
    """按 gold 去重的 NDCG@k。

    增益：排名 i 的 chunk 的增益 = 它"新覆盖"的 gold 条数（排在它前面已覆盖的 gold 不再计分）。
          这样同一页被切成多个 chunk、重复占据前几名时，不会被重复奖励。
    折损：第 i 名（从 1 开始）除以 log2(i + 1)。
    理想值 IDCG：假设前 min(k, n_gold) 个位置各覆盖 1 条新 gold。
    截断：一个 chunk 可能同时覆盖多条 gold（例如两条 gold 在同一页），DCG 可能超过上述理想值，
          这种情况下已经是最优，结果截断为 1。
    """
    covered: set[int] = set()
    dcg = 0.0
    for rank, h in enumerate(hits[:k], start=1):
        new = h - covered
        dcg += len(new) / math.log2(rank + 1)
        covered |= new
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(k, n_gold) + 1))
    return min(dcg / idcg, 1.0)


def score_question(hits: HitMatrix, n_gold: int, ks: Sequence[int]) -> dict[str, float]:
    """计算一道题在各个 k 下的全部指标，键名形如 "recall@5"。"""
    if n_gold < 1:
        raise ValueError("没有 gold 的题（out_of_scope）不参与检索指标")
    scores: dict[str, float] = {}
    for k in ks:
        scores[f"hit@{k}"] = hit_at_k(hits, k)
        scores[f"recall@{k}"] = recall_at_k(hits, n_gold, k)
        scores[f"precision@{k}"] = precision_at_k(hits, k)
        scores[f"mrr@{k}"] = mrr_at_k(hits, k)
        scores[f"ndcg@{k}"] = ndcg_at_k(hits, n_gold, k)
    return scores


def mean_scores(rows: Sequence[dict[str, float]]) -> dict[str, float]:
    """逐项取平均（macro average）。rows 为空时返回空字典。"""
    if not rows:
        return {}
    return {key: sum(r[key] for r in rows) / len(rows) for key in rows[0]}
