"""稠密检索：问题 -> 向量 -> 在 Qdrant 中按余弦相似度取 top_k -> 还原成 Chunk。

检索结果的相似度分数写在 chunk.extra["score"] 中，供后续融合 / 重排和排障使用。
"""

from __future__ import annotations

from dataclasses import replace

from qdrant_client import QdrantClient

from config import Settings
from ingest.chunk import Chunk
from ingest.index import DENSE_VECTOR
from ingest.parse import PARSER_VERSIONS
from retrieval.embedding import Embedder


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

    def search(self, query: str, top_k: int) -> list[Chunk]:
        vec = self.embedder.encode_queries([query])[0]
        hits = self.client.query_points(
            self.collection, query=vec.tolist(), using=DENSE_VECTOR, limit=top_k, with_payload=True
        ).points
        return [replace(Chunk(**h.payload), extra={**h.payload.get("extra", {}), "score": h.score}) for h in hits]
