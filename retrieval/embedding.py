"""向量化：把文本编码为稠密向量。建索引（ingest/index.py）和检索（retrieval/dense.py）共用。

接口 Embedder：
    name                       写入 chunk.embed_model，区分不同模型建出的索引
    dim                        向量维度（bge-m3 为 1024）
    encode_documents(texts)    文档侧编码
    encode_queries(texts)      查询侧编码
    返回 float32 的二维数组 (n, dim)，每行已做 L2 归一化，因此内积 = 余弦相似度。

bge-m3 的查询不需要加指令前缀（与 bge-v1.5 不同），所以两个方法内部实现相同；
仍然分成两个方法，是为了以后换成需要区分查询和文档的模型时，调用方不用改。

provider=openai_compat 的实现留到需要时再补（AGENTS.md：预留切换开关）。
"""

from __future__ import annotations

import logging
from typing import Protocol

import numpy as np

from config import EmbeddingSettings

logger = logging.getLogger("retrieval.embedding")


class Embedder(Protocol):
    name: str
    dim: int

    def encode_documents(self, texts: list[str]) -> np.ndarray: ...

    def encode_queries(self, texts: list[str]) -> np.ndarray: ...


class BgeM3Embedder:
    """通过 FlagEmbedding 调用 bge-m3，只取 dense 向量（sparse 在阶段 5 加入）。"""

    dim = 1024

    def __init__(self, cfg: EmbeddingSettings, max_length: int) -> None:
        # 延迟导入：FlagEmbedding 会连带加载 torch / transformers，只在真正需要模型时才付出这个开销
        from FlagEmbedding import BGEM3FlagModel

        self.name = cfg.model_name
        self.device = cfg.device
        self.batch_size = cfg.batch_size
        self.max_length = max_length
        # fp16 只在 GPU（mps / cuda）上启用；CPU 上 fp16 反而更慢，且部分算子不支持
        self._model = BGEM3FlagModel(cfg.model_source, use_fp16=cfg.device != "cpu", devices=cfg.device)
        logger.info("embedder loaded", extra={"model": cfg.model_source, "device": cfg.device})

    def _encode(self, texts: list[str]) -> np.ndarray:
        out = self._model.encode(
            texts,
            batch_size=self.batch_size,
            max_length=self.max_length,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        vecs = np.asarray(out["dense_vecs"], dtype=np.float32)
        return vecs.reshape(len(texts), self.dim)  # 单条输入时库可能返回一维数组，统一成二维

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts)

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts)


def build_embedder(cfg: EmbeddingSettings, max_length: int) -> Embedder:
    """按配置创建向量化器。"""
    if cfg.provider == "local":
        return BgeM3Embedder(cfg, max_length)
    raise NotImplementedError("EMBEDDING__PROVIDER=openai_compat 尚未实现（预留开关，需要时再接入）")
