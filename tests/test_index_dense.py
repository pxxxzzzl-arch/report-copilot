"""ingest/index.py 与 retrieval/dense.py 的单元测试。

用 Qdrant 内存模式 + 确定性的假向量化器：不需要 Docker，也不需要下载模型。
"""

import numpy as np
import pytest
from qdrant_client import QdrantClient

from config import Settings
from ingest.chunk import Chunk
from ingest.index import ensure_collection, index_chunks, point_id
from retrieval.dense import DenseRetriever

DIM = 8
KEYWORDS = ["营业收入", "净利润", "研发投入", "现金流", "员工", "分红", "审计", "注册地址"]


class KeywordEmbedder:
    """假向量化器：第 i 维 = 文本中是否含第 i 个关键词，再做 L2 归一化。

    含相同关键词的文本向量方向一致，余弦相似度最高，便于构造"检索应该返回谁"的用例。
    """

    name = "fake-kw"
    dim = DIM

    def __init__(self) -> None:
        self.calls = 0

    def _encode(self, texts: list[str]) -> np.ndarray:
        self.calls += 1
        vecs = np.array([[1.0 if k in t else 0.0 for k in KEYWORDS] for t in texts], dtype=np.float32)
        vecs += 1e-3  # 避免全零向量（余弦相似度无定义）
        return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)

    encode_documents = _encode
    encode_queries = _encode


def _chunk(i: int, text: str, page: int = 1) -> Chunk:
    return Chunk(
        chunk_id=f"000001_2024__fixed_c500_o50__{i:05d}", doc_id="000001_2024", company="测试", stock_code="000001",
        year=2024, chunk_index=i, text=text, text_for_embedding=text, title_path=[], page_start=page, page_end=page,
        bboxes=[{"page": page, "bbox": [0, 0, 1, 1]}], chunk_type="text", table_id=None, content_hash="h",
        parser_version="pymupdf-blocks-v0", chunker_name="fixed_c500_o50",
    )  # fmt: skip


CHUNKS = [
    _chunk(0, "本期营业收入为 1,234 元", 3),
    _chunk(1, "研发投入合计 567 元", 11),
    _chunk(2, "在职员工的数量合计 25,132", 34),
]


@pytest.fixture
def client() -> QdrantClient:
    return QdrantClient(":memory:")


def test_point_id_deterministic() -> None:
    assert point_id("a") == point_id("a") != point_id("b")


def test_ensure_collection(client: QdrantClient) -> None:
    assert ensure_collection(client, "c", DIM) is True  # 新建
    assert ensure_collection(client, "c", DIM) is False  # 已存在
    assert ensure_collection(client, "c", DIM, recreate=True) is True  # 删除后重建


def test_index_and_search_roundtrip(client: QdrantClient) -> None:
    emb = KeywordEmbedder()
    ensure_collection(client, "c", DIM)
    assert index_chunks(client, "c", CHUNKS, emb, batch=2) == 3
    assert emb.calls == 2  # 3 个 chunk、每批 2 个 -> 2 次编码

    retriever = DenseRetriever(client, "c", emb, Settings(_env_file=None))
    top = retriever.search("公司研发投入是多少？", top_k=2)
    assert len(top) == 2
    assert top[0].chunk_id == CHUNKS[1].chunk_id  # 含"研发投入"的 chunk 排第一
    assert top[0].page_start == 11 and top[0].text == CHUNKS[1].text  # payload 完整还原
    assert top[0].embed_model == "fake-kw"  # 建索引时写入了模型名
    assert top[0].extra["score"] >= top[1].extra["score"]  # 分数降序


def test_reindex_is_idempotent(client: QdrantClient) -> None:
    emb = KeywordEmbedder()
    ensure_collection(client, "c", DIM)
    index_chunks(client, "c", CHUNKS, emb, batch=10)
    index_chunks(client, "c", CHUNKS, emb, batch=10)  # 同样的 chunk 再写一遍
    assert client.count("c", exact=True).count == 3  # 覆盖而不是新增


def test_dense_retriever_config() -> None:
    s = Settings(_env_file=None, chunk={"chunk_size": 300, "overlap": 30})
    r = DenseRetriever(QdrantClient(":memory:"), "c", KeywordEmbedder(), s)
    assert r.config["chunk_size"] == 300 and r.config["overlap"] == 30
    assert r.config["retriever"] == "dense(fake-kw)"
