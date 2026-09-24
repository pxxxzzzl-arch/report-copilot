"""可配置的检索流水线：阶段 5 的消融实验都通过它运行，一次只切换一个开关（config.retrieval）。

    问题
     │ 1. 查询理解（use_filter）：识别公司、年份 -> 过滤条件
     ▼
     │ 2. 多路召回（sources）：dense / bm25 / sparse，各取 top_n 个候选，都带同一个过滤条件
     ▼
     │ 3. 融合（多于一路时）：RRF
     ▼
     │ 4. 重排（rerank）：对融合后的前 top_n 个候选重新打分
     ▼
    取前 top_k 返回

只有一路召回且不重排时，直接取 top_k，与 E001 的 DenseRetriever 行为完全一致。
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Protocol

from config import Settings
from ingest.chunk import Chunk
from ingest.parse import PARSER_VERSIONS
from retrieval.fusion import rrf_fuse
from retrieval.query import QueryFilter, QueryParser

logger = logging.getLogger("retrieval.search")


class Source(Protocol):
    """一路召回。"""

    def search(self, query: str, top_k: int, flt: QueryFilter | None = None) -> list[Chunk]: ...


class Reranker(Protocol):
    name: str

    def rerank(self, query: str, chunks: Sequence[Chunk]) -> list[Chunk]: ...


class SearchPipeline:
    """满足 eval.run_eval.Retriever 协议。"""

    name = "pipeline"

    def __init__(
        self,
        settings: Settings,
        sources: dict[str, Source],
        parser: QueryParser | None = None,
        reranker: Reranker | None = None,
        source_labels: dict[str, str] | None = None,
    ) -> None:
        self.cfg = settings.retrieval
        self.sources = sources
        self.parser = parser
        self.reranker = reranker
        multi = len(sources) > 1
        labels = source_labels or {}
        parts = (["filter"] if parser else []) + [labels.get(n, n) for n in sources]
        # 写入 log.csv 对应列；没用到的组件对应的列留空
        self.config = {
            "parser": PARSER_VERSIONS[settings.parse.parser],
            "chunker": settings.chunk.chunker,
            "chunk_size": settings.chunk.chunk_size,
            "overlap": settings.chunk.overlap,
            "retriever": "+".join(parts),
            "fusion": self.cfg.fusion if multi else "",
            "k_rrf": self.cfg.k_rrf if multi else "",
            "reranker": reranker.name if reranker else "",
            "top_n": self.cfg.top_n if (multi or reranker) else "",
            "collection": settings.collection_name(),
            "embed_device": settings.embedding.device,
        }

    def search(self, query: str, top_k: int) -> list[Chunk]:
        flt = self.parser.parse(query) if self.parser else None
        if flt is not None:
            logger.info("query parsed", extra={"stock_codes": list(flt.stock_codes), "years": list(flt.years)})

        # 需要融合或重排时，每路多取一些候选；否则直接取 top_k
        n = self.cfg.top_n if (len(self.sources) > 1 or self.reranker) else top_k
        lists = [src.search(query, n, flt) for src in self.sources.values()]
        results = lists[0] if len(lists) == 1 else rrf_fuse(lists, self.cfg.k_rrf)

        if self.reranker:
            results = self.reranker.rerank(query, results[: self.cfg.top_n])
        return results[:top_k]


def build_pipeline(settings: Settings) -> SearchPipeline:
    """按配置组装真实组件（会加载模型、连接 Qdrant）。"""
    from qdrant_client import QdrantClient

    from ingest.chunk import chunks_path, read_chunks
    from ingest.download import read_manifest
    from retrieval.bm25 import BM25Retriever
    from retrieval.dense import DenseRetriever
    from retrieval.embedding import build_embedder

    cfg = settings.retrieval
    client = QdrantClient(url=settings.store.qdrant_url)
    sources: dict[str, Source] = {}
    labels: dict[str, str] = {}
    for name in cfg.sources:
        if name == "dense":
            embedder = build_embedder(settings.embedding, settings.index.max_length)
            sources[name] = DenseRetriever(client, settings.collection_name(), embedder, settings)
            labels[name] = f"dense({embedder.name})"
        elif name == "bm25":
            # BM25 与 dense 使用同一份 chunk（同一个切分标识），保证两路召回的候选可比、可融合
            chunks = read_chunks(chunks_path(settings.paths.parsed_dir, settings.chunk_key()))
            sources[name] = BM25Retriever(chunks, cfg.bm25_k1, cfg.bm25_b)
            labels[name] = "bm25(jieba)"
        else:
            raise NotImplementedError(f"召回通道 {name} 尚未实现")
    parser = QueryParser(read_manifest(settings.paths.manifest_path), cfg.company_aliases) if cfg.use_filter else None
    if cfg.rerank:
        raise NotImplementedError("重排尚未实现")
    return SearchPipeline(settings, sources, parser=parser, source_labels=labels)
