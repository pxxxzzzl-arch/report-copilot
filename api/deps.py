"""依赖容器与请求级依赖：鉴权、限流、并发名额。

AppState 在服务启动时创建一次（加载嵌入模型、建连接池、编译图），所有请求共享；
测试时可以注入假的 AppState（假大模型、内存限流器），不需要真实的模型和数据库。
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Protocol

from fastapi import Request

from api.errors import APIError
from config import Settings

logger = logging.getLogger("api.deps")


# ---------------- 限流 ----------------


class RateLimiter(Protocol):
    def hit(self, client: str) -> tuple[bool, int]:
        """记一次请求，返回 (是否放行, 需要等待的秒数)。"""
        ...


def _window(now: float, window_s: int) -> tuple[int, int]:
    """固定窗口：返回 (窗口编号, 距窗口结束的秒数)。"""
    idx = int(now // window_s)
    return idx, max(1, int((idx + 1) * window_s - now))


class RedisRateLimiter:
    """固定窗口计数：键 = rl:{客户端}:{窗口编号}，INCR 计数、EXPIRE 让旧窗口自动过期。

    为什么选固定窗口：实现只需两条命令、行为容易解释；缺点是窗口交界处最多可能放行 2 倍请求，
    对"防止单个客户端刷接口、控制大模型费用"这个目标足够。需要更平滑时可换成滑动窗口或令牌桶。

    Redis 不可用时放行（fail open）并记警告：限流是保护措施，不应该因为它出故障而让整个服务不可用；
    /readyz 会把 Redis 标记为异常，运维能及时发现。
    """

    def __init__(self, client, limit: int, window_s: int, prefix: str = "rl") -> None:
        self.client, self.limit, self.window_s, self.prefix = client, limit, window_s, prefix

    def hit(self, client: str) -> tuple[bool, int]:
        idx, retry_after = _window(time.time(), self.window_s)
        key = f"{self.prefix}:{client}:{idx}"
        try:
            pipe = self.client.pipeline()  # 两条命令一次往返；MULTI/EXEC 保证原子执行
            pipe.incr(key)
            pipe.expire(key, self.window_s + 1)
            count, _ = pipe.execute()
        except Exception as e:  # noqa: BLE001
            logger.warning("rate limiter unavailable, allowing request", extra={"error": repr(e)})
            return True, 0
        return count <= self.limit, retry_after


class MemoryRateLimiter:
    """进程内实现（测试用；单进程开发时也可以用）。"""

    def __init__(self, limit: int, window_s: int, clock: Callable[[], float] = time.time) -> None:
        self.limit, self.window_s, self.clock = limit, window_s, clock
        self.counts: dict[tuple[str, int], int] = {}
        self._lock = threading.Lock()

    def hit(self, client: str) -> tuple[bool, int]:
        idx, retry_after = _window(self.clock(), self.window_s)
        with self._lock:
            self.counts[(client, idx)] = n = self.counts.get((client, idx), 0) + 1
        return n <= self.limit, retry_after


# ---------------- 并发名额 ----------------


class Slot:
    """一个并发名额；release 可以重复调用（只有第一次生效），避免异常路径下重复释放。"""

    def __init__(self, sem: threading.BoundedSemaphore) -> None:
        self._sem, self._released = sem, False
        self._lock = threading.Lock()

    def release(self) -> None:
        with self._lock:
            if not self._released:
                self._released = True
                self._sem.release()


class ConcurrencyGate:
    """限制同时运行的 Agent 数量。满了直接拒绝（503），不排队：
    一次问答要几秒，排队只会让所有请求一起变慢；让客户端稍后重试更可控。"""

    def __init__(self, max_runs: int) -> None:
        self._sem = threading.BoundedSemaphore(max_runs)

    def try_acquire(self) -> Slot | None:
        return Slot(self._sem) if self._sem.acquire(blocking=False) else None


# ---------------- 依赖容器 ----------------


@dataclass
class AppState:
    settings: Settings
    graph: object  # 编译好的 LangGraph 图（带 checkpointer）
    limiter: RateLimiter
    gate: ConcurrencyGate
    # 就绪检查：组件名 -> 检查函数（抛异常表示不可用）
    checks: dict[str, Callable[[], None]] = field(default_factory=dict)
    closers: list[Callable[[], None]] = field(default_factory=list)  # 关闭服务时依次调用

    def close(self) -> None:
        for fn in reversed(self.closers):
            try:
                fn()
            except Exception as e:  # noqa: BLE001
                logger.warning("close failed", extra={"error": repr(e)})


def get_state(request: Request) -> AppState:
    return request.app.state.core


def client_id(request: Request, state: AppState) -> str:
    """限流的客户端标识：配置了 API Key 时按 Key（只用哈希，不把 Key 本身写进 Redis），否则按 IP。"""
    key = request.headers.get("x-api-key")
    if state.settings.api.keys and key:
        return "key-" + hashlib.sha256(key.encode()).hexdigest()[:16]
    return "ip-" + (request.client.host if request.client else "unknown")


def require_api_key(request: Request) -> None:
    """鉴权：未配置 API__KEYS 时不校验（本地开发）；配置后必须携带其中之一。"""
    keys = get_state(request).settings.api.keys
    if not keys:
        return
    given = request.headers.get("x-api-key", "")
    # compare_digest：比较耗时与内容无关，防止通过响应时间逐位猜出密钥
    if not any(hmac.compare_digest(given.encode(), k.get_secret_value().encode()) for k in keys):
        raise APIError(401, "unauthorized", "缺少或错误的 API Key（请求头 X-API-Key）")


def rate_limit(request: Request) -> None:
    state = get_state(request)
    allowed, retry_after = state.limiter.hit(client_id(request, state))
    if not allowed:
        raise APIError(429, "rate_limited", f"请求过于频繁，请 {retry_after} 秒后重试",
                       headers={"Retry-After": str(retry_after)})  # fmt: skip


def acquire_slot(request: Request) -> Iterator[Slot]:
    """占用一个并发名额。

    写成 yield 依赖、在 finally 里归还：FastAPI 先解析依赖、再校验请求体，
    如果用普通依赖，请求体校验失败（422）时名额已经占了却没人归还。yield 依赖的清理由框架保证执行，
    对流式响应则在事件流结束后执行。接口里也会提前调用 release（可重复调用，只有第一次生效）。
    """
    slot = get_state(request).gate.try_acquire()
    if slot is None:
        raise APIError(503, "busy", "服务繁忙，请稍后重试", headers={"Retry-After": "5"})
    try:
        yield slot
    finally:
        slot.release()


def build_app_state(settings: Settings) -> AppState:
    """用真实组件组装：加载嵌入模型与检索流水线、Postgres 连接池、PostgresSaver、Redis。启动时调用一次。"""
    import psycopg
    import redis
    from langgraph.checkpoint.postgres import PostgresSaver
    from psycopg.rows import dict_row
    from psycopg_pool import ConnectionPool
    from qdrant_client import QdrantClient

    from agent.fin_db import PgFinStore
    from agent.graph import build_deps, build_graph
    from retrieval.search import build_pipeline

    dsn = settings.store.postgres_dsn.get_secret_value()
    n = settings.api.max_concurrent_runs
    closers: list[Callable[[], None]] = []

    store = PgFinStore(dsn, max_size=n)
    closers.append(store.close)
    # PostgresSaver 使用连接池时，连接必须是 autocommit + dict_row（LangGraph 文档要求），prepare_threshold=0 关闭预编译语句
    cp_pool = ConnectionPool(dsn, max_size=n + 2, open=True,
                             kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row})  # fmt: skip
    closers.append(cp_pool.close)
    saver = PostgresSaver(cp_pool)
    saver.setup()  # 首次运行建 checkpoint 表；之后是空操作

    pipeline = build_pipeline(settings)
    audit = store.log_tool_call if settings.agent.log_tool_calls else None
    deps = build_deps(settings, pipeline.search, store, audit=audit, llm_cache=settings.api.llm_cache)
    graph = build_graph(deps, checkpointer=saver)

    r = redis.Redis.from_url(settings.store.redis_url.get_secret_value(), socket_timeout=1, socket_connect_timeout=1)
    closers.append(r.close)
    limiter = RedisRateLimiter(r, settings.api.rate_limit, settings.api.rate_window_s)

    qdrant = QdrantClient(url=settings.store.qdrant_url, timeout=2)

    def check_postgres() -> None:
        with psycopg.connect(dsn, connect_timeout=2) as conn:
            conn.execute("SELECT 1")

    checks = {
        "postgres": check_postgres,
        "redis": lambda: r.ping(),
        "qdrant": lambda: qdrant.get_collection(settings.collection_name()),
    }
    return AppState(settings, graph, limiter, ConcurrencyGate(n), checks, closers)
