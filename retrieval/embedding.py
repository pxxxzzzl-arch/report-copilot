"""向量化：把文本编码为稠密向量。建索引（ingest/index.py）和检索（retrieval/dense.py）共用。

接口 Embedder：
    name                       写入 chunk.embed_model，区分不同模型建出的索引
    dim                        向量维度（bge-m3 为 1024）
    encode_documents(texts)    文档侧编码
    encode_queries(texts)      查询侧编码
    返回 float32 的二维数组 (n, dim)，每行已做 L2 归一化，因此内积 = 余弦相似度。

bge-m3 的查询不需要加指令前缀（与 bge-v1.5 不同），所以两个方法内部实现相同；
仍然分成两个方法，是为了以后换成需要区分查询和文档的模型时，调用方不用改。

稀疏向量（bge-m3 sparse，阶段 5）：
    encode_documents_with_sparse(texts) -> (dense, sparse)   建索引时一次前向同时得到两种向量
    encode_queries_sparse(texts) -> sparse                    检索时只算查询的稀疏向量
    稀疏向量表示为 {token_id: 权重}：模型为文本中每个 token 学出一个权重（类似"学出来的 BM25 词权重"）。

provider=openai_compat 的实现留到需要时再补（AGENTS.md：预留切换开关）。
"""

from __future__ import annotations

import logging
from typing import Protocol

import numpy as np

from config import EmbeddingSettings

logger = logging.getLogger("retrieval.embedding")


SparseVec = dict[int, float]


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

    def _encode(self, texts: list[str], dense: bool = True, sparse: bool = False) -> dict:
        return self._model.encode(
            texts,
            batch_size=self.batch_size,
            max_length=self.max_length,
            return_dense=dense,
            return_sparse=sparse,
            return_colbert_vecs=False,
        )

    def _dense(self, out: dict, n: int) -> np.ndarray:
        vecs = np.asarray(out["dense_vecs"], dtype=np.float32)
        return vecs.reshape(n, self.dim)  # 单条输入时库可能返回一维数组，统一成二维

    @staticmethod
    def _sparse(out: dict) -> list[SparseVec]:
        # FlagEmbedding 返回 {token_id 字符串: float16 权重}，转成 {int: float} 便于写入 Qdrant
        return [{int(k): float(v) for k, v in w.items()} for w in out["lexical_weights"]]

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        return self._dense(self._encode(texts), len(texts))

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return self._dense(self._encode(texts), len(texts))

    def encode_documents_with_sparse(self, texts: list[str]) -> tuple[np.ndarray, list[SparseVec]]:
        out = self._encode(texts, dense=True, sparse=True)
        return self._dense(out, len(texts)), self._sparse(out)

    def encode_queries_sparse(self, texts: list[str]) -> list[SparseVec]:
        return self._sparse(self._encode(texts, dense=False, sparse=True))


def build_embedder(cfg: EmbeddingSettings, max_length: int) -> Embedder:
    """按配置创建向量化器。"""
    if cfg.provider == "local":
        return BgeM3Embedder(cfg, max_length)
    raise NotImplementedError("EMBEDDING__PROVIDER=openai_compat 尚未实现（预留开关，需要时再接入）")
