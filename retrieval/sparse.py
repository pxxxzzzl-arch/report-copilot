"""稀疏检索：bge-m3 的 lexical weights（学出来的词权重）在 Qdrant 中做内积检索。

与 BM25 的区别：
    BM25    词权重由统计量决定（词频、逆文档频率、文档长度），分词靠 jieba
    sparse  词权重由模型根据上下文学出来（同一个词在不同句子里权重不同），分词用模型自己的 tokenizer
两者都是"按词匹配"，对"研发投入合计"这类精确术语敏感；sparse 还能给同义表达的公共字词分配权重。
"""

from __future__ import annotations

from dataclasses import replace

from qdrant_client import QdrantClient
from qdrant_client.models import SparseVector

from ingest.chunk import Chunk
from ingest.index import SPARSE_VECTOR
from retrieval.dense import to_qdrant_filter
from retrieval.query import QueryFilter


class SparseRetriever:
    """满足 retrieval.search.Source 协议。embedder 需提供 encode_queries_sparse（BgeM3Embedder）。"""

    def __init__(self, client: QdrantClient, collection: str, embedder) -> None:
        self.client = client
        self.collection = collection
        self.embedder = embedder

    def search(self, query: str, top_k: int, flt: QueryFilter | None = None) -> list[Chunk]:
        (weights,) = self.embedder.encode_queries_sparse([query])
        if not weights:  # 查询里没有任何有权重的 token（极少见），直接返回空
            return []
        hits = self.client.query_points(
            self.collection,
            query=SparseVector(indices=list(weights.keys()), values=list(weights.values())),
            using=SPARSE_VECTOR,
            query_filter=to_qdrant_filter(flt),
            limit=top_k,
            with_payload=True,
        ).points
        return [replace(Chunk(**h.payload), extra={**h.payload.get("extra", {}), "sparse_score": h.score}) for h in hits]
