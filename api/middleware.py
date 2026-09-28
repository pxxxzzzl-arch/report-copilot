"""请求追踪中间件：为每个请求绑定 trace_id，响应头返回 X-Trace-Id，请求结束时记一条访问日志（含耗时）。

为什么写成纯 ASGI 中间件，而不用 Starlette 的 BaseHTTPMiddleware：
    BaseHTTPMiddleware 会把响应体包进自己的流里，与流式响应（SSE）配合时有已知问题（缓冲、取消传播）；
    纯 ASGI 中间件只在 send 时插入一个响应头，对响应体原样放行，SSE 事件能及时到达客户端。

trace_id 的来源：客户端传了合法的 X-Request-ID（8~64 位字母、数字、_、-）就沿用，便于跨服务串联；
否则新生成。不合法的值直接丢弃，避免把任意字符串写进日志（日志注入）。
trace_id 存在 contextvars 里：同步接口和 SSE 生成器在线程池中运行时，上下文会随之复制过去，日志里都能带上它。
"""

from __future__ import annotations

import logging
import re
import time

from common.log import new_trace_id, trace_context

logger = logging.getLogger("api.access")

_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9_-]{8,64}$")


class TraceMiddleware:
    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":  # lifespan 等事件直接放行
            await self.app(scope, receive, send)
            return

        incoming = dict(scope.get("headers") or []).get(b"x-request-id", b"").decode("latin-1")
        trace_id = incoming if _REQUEST_ID_RE.match(incoming) else new_trace_id()
        status = 500  # 应用在发出响应头之前就抛异常时，按 500 记录
        start = time.perf_counter()

        async def send_with_trace(message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
                message.setdefault("headers", [])
                message["headers"] = [*message["headers"], (b"x-trace-id", trace_id.encode())]
            await send(message)

        with trace_context(trace_id):
            try:
                await self.app(scope, receive, send_with_trace)
            finally:
                # 流式响应要等整个事件流结束才会走到这里，所以耗时是完整的响应时间
                logger.info(
                    "request",
                    extra={"method": scope["method"], "path": scope["path"], "status": status,
                           "elapsed_ms": round((time.perf_counter() - start) * 1000, 1)},
                )  # fmt: skip
