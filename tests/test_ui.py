"""ui/* 的测试：SSE 解析、API 客户端（httpx.MockTransport，不起服务）、进度文案，以及用 AppTest 对页面做冒烟测试。"""

import json

import httpx
import pymupdf
import pytest
from streamlit.testing.v1 import AppTest

from config import PROJECT_ROOT
from ui.client import ApiClient, ApiError, PageImage, describe_event, parse_sse_lines

# ---------------- SSE 解析 ----------------


def test_parse_sse_lines() -> None:
    lines = [
        "event: start", 'data: {"thread_id": "t"}', "",
        ": ping",  # 保活注释
        "", 'data: {"a": 1}', "",  # 没有 event 字段时事件名为 message
        "event: multi", 'data: {"x":', "data: 2}", "",  # 多行 data 用换行拼接
        "event: tail", 'data: {"end": true}',  # 流结束时没有空行收尾
    ]  # fmt: skip
    assert list(parse_sse_lines(lines)) == [
        ("start", {"thread_id": "t"}), ("message", {"a": 1}), ("multi", {"x": 2}), ("tail", {"end": True}),
    ]  # fmt: skip


# ---------------- API 客户端 ----------------


def _client(handler) -> ApiClient:
    return ApiClient("http://api", http=httpx.Client(base_url="http://api", transport=httpx.MockTransport(handler)))


def test_stream_chat_parses_events() -> None:
    body = 'event: start\r\ndata: {"thread_id": "t1"}\r\n\r\n: ping\r\n\r\nevent: done\r\ndata: {"latency_ms": 1}\r\n\r\n'

    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.path == "/v1/chat/stream" and json.loads(req.content) == {"question": "q", "thread_id": "t1"}
        return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})

    assert [e for e, _ in _client(handler).stream_chat("q", "t1")] == ["start", "done"]


def test_errors_are_raised_with_code_and_trace() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"error": {"code": "rate_limited", "message": "请求过于频繁", "trace_id": "abc"}})

    with pytest.raises(ApiError) as e:
        list(_client(handler).stream_chat("q"))
    assert e.value.code == "rate_limited" and e.value.status == 429 and "trace_id=abc" in str(e.value)
    with pytest.raises(ApiError):
        _client(handler).documents()


def test_page_image_and_ready() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/readyz":
            raise httpx.ConnectError("refused")
        assert req.url.params["q"] == "摘录"
        return httpx.Response(200, content=b"PNG", headers={"x-highlight-method": "chars", "x-highlight-count": "2"})

    c = _client(handler)
    assert c.page_image("d", 3, "摘录") == PageImage(b"PNG", "chars", 2)
    assert c.ready()["status"] == "unreachable"  # 连不上时不抛异常，侧边栏显示"无法连接"


def test_describe_event() -> None:
    assert describe_event("rewrite", {"standalone": "q"}, "q") is None  # 没有改写时不显示
    assert describe_event("rewrite", {"standalone": "五粮液2024年营业收入"}, "那五粮液呢？") == "理解追问：五粮液2024年营业收入"
    assert describe_event("decide", {"step": 1, "tool_calls": []}) == "证据已足够，正在作答…"
    two = {"step": 1, "tool_calls": ["get_financial_metric", "get_financial_metric", "calculator"]}
    assert describe_event("decide", two) == "第 1 步：需要查询财务指标（2 次）、计算"
    tool = {"tool": "get_financial_metric", "args": {"company": "茅台", "year": 2024, "metric": "revenue"},
            "ok": True, "n_results": 2}  # fmt: skip
    assert describe_event("tool", tool) == "· 查询财务指标：茅台 2024 年 营业收入（得到 2 条证据）"
    bad = {"tool": "calculator", "args": {"expression": "1/0"}, "ok": False, "error": "除数为 0"}
    assert describe_event("tool", bad) == "· 计算：1/0（失败：除数为 0）"


# ---------------- 页面冒烟测试（AppTest） ----------------


def _png() -> bytes:
    doc = pymupdf.open()
    doc.new_page(width=50, height=50)
    return doc[0].get_pixmap().tobytes("png")


class FakeClient:
    """假 API 客户端：返回一轮带引用的答案（一条通过核验、一条未通过）。"""

    def __init__(self) -> None:
        self.asked: list[tuple[str, str | None]] = []
        self.pages: list[tuple] = []

    def ready(self):
        return {"status": "ok", "components": {"postgres": "ok"}}

    def documents(self):
        return [{"doc_id": "600519_2025", "company": "贵州茅台", "stock_code": "600519", "year": 2025, "n_pages": 200}]

    def stream_chat(self, question, thread_id=None):
        self.asked.append((question, thread_id))
        yield "start", {"thread_id": "t-1", "trace_id": "x"}
        yield "retrieve", {"n_evidence": 10}
        yield "decide", {"step": 1, "tool_calls": []}
        ev = {"id": 1, "source": "retrieve", "doc_id": "600519_2025", "company": "贵州茅台", "year": 2025,
              "page_start": 6, "page_end": 6, "text": "营业收入 168,838,102,514.79"}
        cites = [
            {"ref": 1, "quote": "168,838,102,514.79", "valid_ref": True, "verified": True, "doc_id": "600519_2025",
             "page": 6, "page_start": 6, "page_end": 6},
            {"ref": 1, "quote": "编造的摘录", "valid_ref": True, "verified": False, "doc_id": "600519_2025",
             "page": None, "page_start": 6, "page_end": 6},
        ]
        yield "answer", {"thread_id": "t-1", "trace_id": "x", "question": question, "standalone": question,
                         "answer": "营业收入为168,838,102,514.79元[1]", "refused": False, "format_ok": True,
                         "citations": cites, "evidence": [ev], "tool_calls": [], "steps": 1, "latency_ms": 1200.0}
        yield "done", {"latency_ms": 1200.0}  # fmt: skip

    def page_image(self, doc_id, page, quote="", scale=None):
        self.pages.append((doc_id, page, quote))
        return PageImage(_png(), "exact", 1)


def test_app_ask_and_open_citation() -> None:
    fake = FakeClient()
    at = AppTest.from_file(str(PROJECT_ROOT / "ui" / "app.py"), default_timeout=20)  # 相对路径会按测试文件所在目录解析
    at.session_state["client"] = fake
    at.run()
    assert not at.exception

    at.chat_input[0].set_value("贵州茅台2025年营业收入是多少？").run()
    assert not at.exception
    assert fake.asked == [("贵州茅台2025年营业收入是多少？", None)]
    assert at.session_state["thread_id"] == "t-1"
    assert any("168,838,102,514.79元[1]" in m.value for m in at.markdown)
    assert any("未通过核验" in c.value for c in at.caption)  # 未通过核验的引用只显示灰色提示

    buttons = [b for b in at.button if b.label == "查看原文"]
    assert len(buttons) == 1  # 只有通过核验的引用可以查看原文
    buttons[0].click().run()
    assert not at.exception
    assert at.session_state["viewer"]["page"] == 6
    assert fake.pages[-1] == ("600519_2025", 6, "168,838,102,514.79")

    # 第二次提问沿用同一对话编号（多轮）
    at.chat_input[0].set_value("那2024年呢？").run()
    assert fake.asked[-1] == ("那2024年呢？", "t-1")
