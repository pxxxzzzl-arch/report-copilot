"""稠密检索：问题 -> 向量 -> 在 Qdrant 中按余弦相似度取 top_k -> 还原成 Chunk。

可选的过滤条件（公司、年份）在 Qdrant 端执行：先过滤再找最近邻，而不是先取 top_k 再过滤，
否则过滤后剩下的结果可能不足 top_k。
检索结果的相似度分数写在 chunk.extra["score"] 中，供后续融合 / 重排和排障使用。
"""

from __future__ import annotations

from dataclasses import replace

from qdrant_client import QdrantClient, models

from config import Settings
from ingest.chunk import Chunk
from ingest.index import DENSE_VECTOR
from ingest.parse import PARSER_VERSIONS
from retrieval.embedding import Embedder
from retrieval.query import QueryFilter


class DenseRetriever:
    """满足 eval.run_eval.Retriever 协议。"""

    name = "dense"

    def __init__(self, client: QdrantClient, collection: str, embedder: Embedder, settings: Settings) -> None:
        self.client = client
        self.collection = collection
        self.embedder = embedder
        # 写入 log.csv 对应列；未涉及的列（fusion、k_rrf、reranker、top_n）留空
        self.config = {
            "parser": PARSER_VERSIONS[settings.parse.parser],
            "chunker": settings.chunk.chunker,
            "chunk_size": settings.chunk.chunk_size,
            "overlap": settings.chunk.overlap,
            "retriever": f"dense({embedder.name})",
            # 以下不在 log.csv 的列中，只记入逐题明细的元信息，便于复现
            "embed_device": getattr(embedder, "device", ""),
            "collection": collection,
        }

    def search(self, query: str, top_k: int, flt: QueryFilter | None = None) -> list[Chunk]:
        vec = self.embedder.encode_queries([query])[0]
        hits = self.client.query_points(
            self.collection,
            query=vec.tolist(),
            using=DENSE_VECTOR,
            query_filter=to_qdrant_filter(flt),
            limit=top_k,
            with_payload=True,
        ).points
        return [replace(Chunk(**h.payload), extra={**h.payload.get("extra", {}), "score": h.score}) for h in hits]


def to_qdrant_filter(flt: QueryFilter | None) -> models.Filter | None:
    """把 QueryFilter 转成 Qdrant 的 payload 过滤条件；没有条件时返回 None（不过滤）。

    must 中的条件是"与"关系；MatchAny 表示字段值属于给定集合（集合内部是"或"关系）。
    """
    if flt is None or flt.is_empty:
        return None
    must = []
    if flt.stock_codes:
        must.append(models.FieldCondition(key="stock_code", match=models.MatchAny(any=list(flt.stock_codes))))
    if flt.years:
        must.append(models.FieldCondition(key="year", match=models.MatchAny(any=list(flt.years))))
    return models.Filter(must=must)
