"""重排：用交叉编码器（cross-encoder）对召回的候选重新打分排序。

与召回阶段的区别：
    召回（dense / sparse / BM25）  问题和 chunk 分别编码或分别分词，再比较 —— 快，可以在全库里找候选
    重排（cross-encoder）           把"问题 + chunk"拼成一段文本一起过模型，逐词交互后输出一个相关性分数
                                    —— 准，但每个候选都要做一次前向，只能用于少量候选（这里是融合后的前 top_n 个）
模型：BAAI/bge-reranker-v2-m3（与 bge-m3 同源，支持中文），本地通过 FlagEmbedding 推理。

provider=api（托管的重排接口）为预留开关，部署到无 GPU 服务器时再接入（见 docs/decisions.md D-018）。
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import replace

from config import RerankerSettings
from ingest.chunk import Chunk

logger = logging.getLogger("retrieval.rerank")


class BgeReranker:
    """满足 retrieval.search.Reranker 协议。"""

    def __init__(self, cfg: RerankerSettings) -> None:
        # 延迟导入：只有真正需要重排时才加载 torch / FlagEmbedding
        from FlagEmbedding import FlagReranker

        self.name = cfg.model_name
        self.batch_size = cfg.batch_size
        self.max_length = cfg.max_length
        self._model = FlagReranker(
            cfg.model_source, use_fp16=cfg.device != "cpu", devices=cfg.device, max_length=cfg.max_length
        )
        logger.info("reranker loaded", extra={"model": cfg.model_source, "device": cfg.device})

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        """返回每个 (问题, 文本) 对的相关性分数（未经 sigmoid 的 logit，只用于排序）。"""
        if not texts:
            return []
        scores = self._model.compute_score(
            [(query, t) for t in texts], batch_size=self.batch_size, max_length=self.max_length
        )
        return [float(scores)] if isinstance(scores, (int, float)) else [float(s) for s in scores]

    def rerank(self, query: str, chunks: Sequence[Chunk]) -> list[Chunk]:
        return rerank_by_scores(chunks, self.score(query, [c.text_for_embedding for c in chunks]))


def rerank_by_scores(chunks: Sequence[Chunk], scores: Sequence[float]) -> list[Chunk]:
    """按分数降序重排；同分时保持原来的顺序（sorted 是稳定排序）。分数写入 extra["rerank_score"]。"""
    order = sorted(range(len(chunks)), key=lambda i: -scores[i])
    return [replace(chunks[i], extra={**chunks[i].extra, "rerank_score": scores[i]}) for i in order]


def build_reranker(cfg: RerankerSettings) -> BgeReranker:
    if cfg.provider == "local":
        return BgeReranker(cfg)
    raise NotImplementedError("RERANKER__PROVIDER=api 尚未实现（预留开关，部署阶段按需接入）")
