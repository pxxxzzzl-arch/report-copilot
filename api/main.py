"""FastAPI 服务（阶段 8）：把阶段 7 的 Agent 以 HTTP 接口提供出来。

启动（检索配置与评测相同，例如 E013；首次启动会加载 bge-m3，约需十几秒）：
    env PARSE__PARSER=v1 EMBEDDING__DEVICE=mps RETRIEVAL__USE_FILTER=true RETRIEVAL__STRIP_ENTITIES=true \\
        RETRIEVAL__SOURCES='["dense","bm25","sparse"]' \\
        uv run uvicorn api.main:app --host 127.0.0.1 --port 8000 --no-access-log
    （--no-access-log：访问日志由 api/middleware.py 输出结构化 JSON，带 trace_id 和耗时，不需要 uvicorn 再打一份）
    接口文档：http://127.0.0.1:8000/docs

接口：
    POST /v1/chat          一次性返回答案
    POST /v1/chat/stream   SSE：start → rewrite → retrieve → decide → (tool ... → decide)* → answer → done；出错时发 error
    GET  /v1/threads/{id}  对话历史
    GET  /v1/documents     语料清单
    GET  /v1/documents/{doc_id}/pages/{page}/image?q=摘录   页面截图（PNG），摘录位置画高亮框（阶段 9）
    GET  /healthz          存活检查（进程在就返回 200）
    GET  /readyz           就绪检查（Postgres、Redis、Qdrant 都可用才返回 200）

运行模型：Agent 是同步代码（大模型调用层、嵌入模型都是同步的），接口用普通 def 定义，
由 FastAPI 放到线程池执行，不阻塞事件循环；同时运行的 Agent 数由并发名额限制（config.api.max_concurrent_runs）。
"""

from __future__ import annotations

import logging
import re
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Query
from fastapi.responses import JSONResponse, Response
from fastapi.sse import EventSourceResponse, ServerSentEvent

from api.deps import AppState, Slot, acquire_slot, build_app_state, get_state, rate_limit, require_api_key
from api.errors import APIError, classify_exception, error_body, install_error_handlers
from api.middleware import TraceMiddleware
from api.schemas import THREAD_ID_PATTERN, ChatRequest, ChatResponse, DocumentOut, ErrorResponse, ThreadOut
from common.log import get_trace_id, setup_logging
from config import Settings, get_settings

logger = logging.getLogger("api.main")

# 各错误码在 /docs 中的说明
ERROR_RESPONSES = {
    s: {"model": ErrorResponse} for s in (401, 422, 429, 502, 503)
}  # fmt: skip


def to_response(state: dict, question: str, thread_id: str, latency_ms: float) -> ChatResponse:
    """把一轮的最终状态整理成接口响应。"""
    a = state["answer"]
    return ChatResponse(
        thread_id=thread_id, trace_id=get_trace_id(), question=question, standalone=state["standalone"],
        answer=a["answer"], refused=a["refused"], format_ok=a["format_ok"], citations=state["citations"],
        evidence=state["evidence"], tool_calls=state["tool_log"], steps=state["steps"],
        latency_ms=round(latency_ms, 1),
    )  # fmt: skip


def create_app(settings: Settings | None = None, state_factory: Callable[[Settings], AppState] | None = None) -> FastAPI:
    """应用工厂。state_factory 用于测试时注入假的依赖；默认用 build_app_state 加载真实组件。"""
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        setup_logging(settings.log.level, settings.log.json_format)
        app.state.core = (state_factory or build_app_state)(settings)
        logger.info("service ready")
        yield
        app.state.core.close()

    app = FastAPI(title="report-copilot", version="0.8.0", lifespan=lifespan)
    app.add_middleware(TraceMiddleware)
    install_error_handlers(app)

    def check_question(req: ChatRequest) -> None:
        if len(req.question) > settings.api.max_question_chars:
            raise APIError(422, "invalid_request", f"问题过长（最多 {settings.api.max_question_chars} 个字符）")

    guards = [Depends(require_api_key), Depends(rate_limit)]  # 顺序：先鉴权，再限流（未通过鉴权的请求不占限流额度）

    @app.get("/healthz")
    def healthz() -> dict:
        return {"status": "ok"}

    @app.get("/readyz")
    def readyz(core: AppState = Depends(get_state)) -> JSONResponse:
        results = {}
        for name, check in core.checks.items():
            try:
                check()
                results[name] = "ok"
            except Exception as e:  # noqa: BLE001
                logger.warning("readiness check failed", extra={"component": name, "error": repr(e)})
                results[name] = "unavailable"  # 不返回异常详情（可能含地址等内部信息）
        ok = all(v == "ok" for v in results.values())
        return JSONResponse({"status": "ok" if ok else "degraded", "components": results}, status_code=200 if ok else 503)

    @app.post("/v1/chat", response_model=ChatResponse, dependencies=guards, responses=ERROR_RESPONSES)
    def chat(req: ChatRequest, core: AppState = Depends(get_state), slot: Slot = Depends(acquire_slot)) -> ChatResponse:
        from agent.graph import stream_turn

        try:
            check_question(req)
            thread_id = req.thread_id or uuid.uuid4().hex
            t0 = time.perf_counter()
            final = None
            for ev in stream_turn(core.graph, req.question, settings.agent.max_steps, thread_id):
                if ev["event"] == "final":
                    final = ev["state"]
            return to_response(final, req.question, thread_id, (time.perf_counter() - t0) * 1000)
        except Exception as e:
            err = classify_exception(e)
            if err is not e:
                logger.error("chat failed", exc_info=e, extra={"code": err.code})
            raise err from None
        finally:
            slot.release()

    @app.post("/v1/chat/stream", response_class=EventSourceResponse, dependencies=guards, responses=ERROR_RESPONSES)
    def chat_stream(
        req: ChatRequest, core: AppState = Depends(get_state), slot: Slot = Depends(acquire_slot)
    ) -> Iterator[ServerSentEvent]:
        """SSE 流式接口。鉴权、限流、并发名额在依赖中完成，失败时直接返回对应的 HTTP 状态码；
        事件流开始之后（状态码已是 200）的错误，以 error 事件发出，随后结束。"""
        from agent.graph import stream_turn

        t0 = time.perf_counter()
        try:
            check_question(req)
            thread_id = req.thread_id or uuid.uuid4().hex
            yield ServerSentEvent(event="start", data={"thread_id": thread_id, "trace_id": get_trace_id()})
            for ev in stream_turn(core.graph, req.question, settings.agent.max_steps, thread_id):
                name = ev.pop("event")
                if name == "final":
                    resp = to_response(ev["state"], req.question, thread_id, (time.perf_counter() - t0) * 1000)
                    yield ServerSentEvent(event="answer", data=resp)
                else:
                    yield ServerSentEvent(event=name, data=ev)
            yield ServerSentEvent(event="done", data={"latency_ms": round((time.perf_counter() - t0) * 1000, 1)})
        except Exception as e:  # noqa: BLE001
            err = classify_exception(e)
            if err is not e:
                logger.error("chat stream failed", exc_info=e, extra={"code": err.code})
            yield ServerSentEvent(event="error", data=error_body(err.code, err.message))
        finally:
            # 事件流结束就归还名额；客户端中途断开时，由 acquire_slot 依赖的清理兜底归还
            slot.release()

    @app.get("/v1/threads/{thread_id}", response_model=ThreadOut, dependencies=[Depends(require_api_key)],
             responses={404: {"model": ErrorResponse}})  # fmt: skip
    def get_thread(thread_id: str, core: AppState = Depends(get_state)) -> ThreadOut:
        if not re.match(THREAD_ID_PATTERN, thread_id):
            raise APIError(422, "invalid_request", "thread_id 格式不正确")
        values = core.graph.get_state({"configurable": {"thread_id": thread_id}}).values
        if not values:
            raise APIError(404, "not_found", "对话不存在")
        return ThreadOut(thread_id=thread_id, history=values.get("history", []))

    @app.get("/v1/documents", response_model=list[DocumentOut], dependencies=[Depends(require_api_key)])
    def list_documents(core: AppState = Depends(get_state)) -> list[DocumentOut]:
        return [DocumentOut(doc_id=r.doc_id, company=r.company, stock_code=r.stock_code, year=r.year, n_pages=r.n_pages)
                for r in sorted(core.manifest.values(), key=lambda r: r.doc_id)]  # fmt: skip

    @app.get("/v1/documents/{doc_id}/pages/{page}/image", dependencies=[Depends(require_api_key)],
             responses={200: {"content": {"image/png": {}}}, 404: {"model": ErrorResponse}, 422: {"model": ErrorResponse}})  # fmt: skip
    def page_image(
        doc_id: str, page: int, q: str = Query(default="", description="要高亮的摘录"),
        scale: float | None = Query(default=None), core: AppState = Depends(get_state),
    ) -> Response:
        """渲染一页并高亮摘录。响应头 X-Highlight-Method（exact / chars / number / none）、X-Highlight-Count 告诉前端是否定位成功。

        安全：doc_id 只能是 manifest 中登记的文档，文件路径由服务端根据 manifest 拼出，用户输入不参与路径拼接（防路径穿越）。
        """
        from api.pages import render_page_png

        row = core.manifest.get(doc_id)
        if row is None:
            raise APIError(404, "not_found", "文档不存在")
        if not 1 <= page <= row.n_pages:
            raise APIError(422, "invalid_request", f"页码超出范围（1~{row.n_pages}）")
        api_cfg = core.settings.api
        if len(q) > api_cfg.max_quote_chars:
            raise APIError(422, "invalid_request", f"摘录过长（最多 {api_cfg.max_quote_chars} 个字符）")
        scale = api_cfg.page_scale if scale is None else scale
        if not 0.5 <= scale <= api_cfg.page_scale_max:
            raise APIError(422, "invalid_request", f"scale 超出范围（0.5~{api_cfg.page_scale_max}）")
        png, method, n = render_page_png(core.settings.paths.raw_dir / row.file_name, page, q, round(scale, 2))
        return Response(png, media_type="image/png", headers={
            "X-Highlight-Method": method, "X-Highlight-Count": str(n), "Cache-Control": "private, max-age=3600"})  # fmt: skip

    return app


# uvicorn api.main:app 的入口；延迟到 lifespan 才加载模型，导入本模块本身很快
app = create_app()
