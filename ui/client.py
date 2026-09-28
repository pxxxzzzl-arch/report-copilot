"""前端使用的 API 客户端。与页面代码（ui/app.py）分开，便于单元测试。

前端只通过 HTTP 访问 API，不直接读 PDF、不直接连数据库：部署时两者可以分开，PDF 只需要放在服务端（D-030）。
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

import httpx


class ApiError(Exception):
    """API 返回的错误（统一格式 {"error": {code, message, trace_id}}），或无法连接。"""

    def __init__(self, code: str, message: str, trace_id: str | None = None, status: int | None = None) -> None:
        super().__init__(message)
        self.code, self.message, self.trace_id, self.status = code, message, trace_id, status

    def __str__(self) -> str:
        return f"{self.message}（{self.code}" + (f"，trace_id={self.trace_id}）" if self.trace_id else "）")


@dataclass
class PageImage:
    png: bytes
    method: str  # 高亮定位方法：exact / chars / number / none
    n_rects: int


def parse_sse_lines(lines: Iterable[str]) -> Iterator[tuple[str, dict]]:
    """把 SSE 文本流解析成 (事件名, 数据)。

    格式：每个事件由若干 "字段: 值" 行组成，空行结束一个事件；以冒号开头的是注释（服务端的保活心跳），忽略。
    本项目的 data 都是单行 JSON；多行 data 按规范用换行拼接后再解析。
    """
    event, data = "message", []
    for raw in lines:
        line = raw.rstrip("\r\n")
        if not line:  # 空行：一个事件结束
            if data:
                yield event, json.loads("\n".join(data))
            event, data = "message", []
        elif line.startswith(":"):
            continue
        else:
            name, _, value = line.partition(":")
            value = value[1:] if value.startswith(" ") else value
            if name == "event":
                event = value
            elif name == "data":
                data.append(value)
    if data:  # 流结束时没有以空行收尾的最后一个事件
        yield event, json.loads("\n".join(data))


def _raise_for_error(resp: httpx.Response) -> None:
    if resp.status_code < 400:
        return
    try:
        err = resp.json()["error"]
        raise ApiError(err["code"], err["message"], err.get("trace_id"), resp.status_code)
    except (ValueError, KeyError, TypeError):
        raise ApiError("http_error", f"HTTP {resp.status_code}", resp.headers.get("x-trace-id"), resp.status_code) from None


class ApiClient:
    def __init__(self, base_url: str, api_key: str = "", timeout_s: float = 120.0, http: httpx.Client | None = None) -> None:
        """http 仅供测试注入（例如 httpx.Client(transport=httpx.MockTransport(...))）。"""
        headers = {"X-API-Key": api_key} if api_key else {}
        # trust_env=False：不读取系统代理设置。API 是内部服务，走代理反而会连不上（本机开发时 shell 里常设了 HTTPS_PROXY）
        self.http = http or httpx.Client(base_url=base_url, headers=headers, timeout=timeout_s, trust_env=False)

    def _get(self, path: str, **params) -> httpx.Response:
        try:
            resp = self.http.get(path, params=params or None)
        except httpx.HTTPError as e:
            raise ApiError("connection_error", f"无法连接 API 服务：{type(e).__name__}") from None
        _raise_for_error(resp)
        return resp

    def ready(self) -> dict:
        """就绪状态；服务不可用时返回 {"status": "unreachable"}，不抛异常（侧边栏展示用）。"""
        try:
            return self.http.get("/readyz").json()
        except (httpx.HTTPError, ValueError):
            return {"status": "unreachable", "components": {}}

    def documents(self) -> list[dict]:
        return self._get("/v1/documents").json()

    def thread(self, thread_id: str) -> dict:
        return self._get(f"/v1/threads/{thread_id}").json()

    def page_image(self, doc_id: str, page: int, quote: str = "", scale: float | None = None) -> PageImage:
        params = {"q": quote} if quote else {}
        if scale is not None:
            params["scale"] = scale
        resp = self._get(f"/v1/documents/{doc_id}/pages/{page}/image", **params)
        return PageImage(resp.content, resp.headers.get("x-highlight-method", "none"), int(resp.headers.get("x-highlight-count", 0)))

    def stream_chat(self, question: str, thread_id: str | None = None) -> Iterator[tuple[str, dict]]:
        """流式问答，逐个产出 (事件名, 数据)。事件流开始前的错误（鉴权、限流、繁忙、参数错误）抛 ApiError；
        开始后的错误以 ("error", {...}) 事件产出，由调用方展示。"""
        body = {"question": question} | ({"thread_id": thread_id} if thread_id else {})
        try:
            with self.http.stream("POST", "/v1/chat/stream", json=body) as resp:
                if resp.status_code >= 400:
                    resp.read()
                    _raise_for_error(resp)
                yield from parse_sse_lines(resp.iter_lines())
        except httpx.HTTPError as e:
            raise ApiError("connection_error", f"与 API 服务的连接中断：{type(e).__name__}") from None


# ---------------- 展示辅助 ----------------

TOOL_NAMES = {"get_financial_metric": "查询财务指标", "search_reports": "补充检索", "calculator": "计算"}


def describe_event(event: str, data: dict, question: str = "") -> str | None:
    """把一个 SSE 事件转成给用户看的一行进度说明；不需要展示的事件返回 None。"""
    from ingest.tables import METRICS  # 指标键 -> 中文名（纯 Python 字典，不会加载 PDF 相关依赖）

    if event == "rewrite":
        s = data.get("standalone", "")
        return f"理解追问：{s}" if s and s != question else None
    if event == "retrieve":
        return f"检索到 {data['n_evidence']} 段年报原文"
    if event == "decide":
        if not data["tool_calls"]:
            return "证据已足够，正在作答…"
        names = "、".join(TOOL_NAMES.get(t, t) for t in data["tool_calls"])
        return f"第 {data['step']} 步：需要{names}"
    if event == "tool":
        args = data.get("args") or {}
        name = data.get("tool", "")
        if name == "get_financial_metric":
            metric = METRICS.get(args.get("metric", ""), (args.get("metric", ""),))[0]
            detail = f"{args.get('company', '')} {args.get('year', '')} 年 {metric}"
        elif name == "search_reports":
            detail = " ".join(str(args[k]) for k in ("company", "year", "query") if args.get(k))
        elif name == "calculator":
            detail = args.get("expression", "")
        else:
            detail = ""
        status = f"（得到 {data['n_results']} 条证据）" if data.get("ok") and data.get("n_results") else ""
        if not data.get("ok"):
            status = f"（失败：{data.get('error')}）"
        return f"· {TOOL_NAMES.get(name, name)}：{detail}{status}"
    if event == "error":
        return f"出错：{data['error']['message']}"
    return None
