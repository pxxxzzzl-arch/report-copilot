"""建索引：对 chunk 做向量化，写入 Qdrant。

集合结构：
    名称      config.Settings.collection_name()，如 report_chunks__fixed_c500_o50__bge-m3
              —— 切分配置或向量模型不同，就是不同的集合，消融实验的索引可以并存
    向量      命名向量 "dense"（1024 维，余弦距离）；阶段 5 加入 sparse 时再增加命名向量 "sparse"
    点 ID     uuid5(chunk_id)：同一个 chunk 每次得到同一个 ID，重复写入是覆盖而不是新增（幂等）
    payload   Chunk 的全部字段，检索时可以直接还原成 Chunk

用法：
    uv run python -m ingest.index               # 集合已存在且点数与 chunk 数一致时跳过
    uv run python -m ingest.index --recreate    # 删除后重建
前置：ingest.parse、ingest.chunk 已运行；Qdrant 已启动（deploy/docker-compose.dev.yml）
"""

from __future__ import annotations

import argparse
import logging
import uuid
from dataclasses import asdict, replace

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from common.log import setup_logging, timed, trace_context
from config import get_settings
from ingest.chunk import Chunk, chunks_path, read_chunks
from retrieval.embedding import Embedder, build_embedder

logger = logging.getLogger("ingest.index")

DENSE_VECTOR = "dense"  # 命名向量的名字，检索时通过 using="dense" 指定


def point_id(chunk_id: str) -> str:
    """由 chunk_id 生成确定性的 UUID（Qdrant 的点 ID 只能是无符号整数或 UUID）。"""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def ensure_collection(client: QdrantClient, name: str, dim: int, recreate: bool = False) -> bool:
    """确保集合存在；返回 True 表示新建了集合。"""
    exists = client.collection_exists(name)
    if exists and recreate:
        client.delete_collection(name)
        exists = False
    if not exists:
        client.create_collection(name, vectors_config={DENSE_VECTOR: VectorParams(size=dim, distance=Distance.COSINE)})
        return True
    return False


def index_chunks(client: QdrantClient, collection: str, chunks: list[Chunk], embedder: Embedder, batch: int) -> int:
    """分批向量化并写入，返回写入的点数。"""
    total = 0
    for i in range(0, len(chunks), batch):
        part = [replace(c, embed_model=embedder.name) for c in chunks[i : i + batch]]
        with timed("embed_upsert", logger, batch_start=i, batch_size=len(part)):
            vecs = embedder.encode_documents([c.text_for_embedding for c in part])
            points = [
                PointStruct(id=point_id(c.chunk_id), vector={DENSE_VECTOR: v.tolist()}, payload=asdict(c))
                for c, v in zip(part, vecs)
            ]
            # wait=True：写入完成后才返回，保证随后的检索能查到
            client.upsert(collection, points=points, wait=True)
        total += len(points)
    return total


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="向量化 chunk 并写入 Qdrant")
    p.add_argument("--recreate", action="store_true", help="删除已有集合后重建")
    args = p.parse_args(argv)

    settings = get_settings()
    setup_logging(settings.log.level, settings.log.json_format)
    collection = settings.collection_name()
    chunks = read_chunks(chunks_path(settings.paths.parsed_dir, settings.chunk))
    client = QdrantClient(url=settings.store.qdrant_url)

    with trace_context(), timed("index_all", logger, collection=collection, n_chunks=len(chunks)) as info:
        embedder = build_embedder(settings.embedding, settings.index.max_length)
        created = ensure_collection(client, collection, embedder.dim, recreate=args.recreate)
        existing = client.count(collection, exact=True).count
        if not created and existing == len(chunks):
            info["skipped"] = True  # 已经建好，避免重复花几分钟做向量化
            logger.info("collection up to date, skip", extra={"collection": collection, "points": existing})
            return
        info["n_points"] = index_chunks(client, collection, chunks, embedder, settings.index.upsert_batch)


if __name__ == "__main__":
    main()
