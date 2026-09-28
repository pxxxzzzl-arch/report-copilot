"""api/* 的测试：TestClient + 真实的 LangGraph 图（假大模型按脚本返回、假检索、内存 checkpointer 和内存限流器）。

不加载模型、不连数据库；最后一个测试需要本地 Redis，连不上时自动跳过。
"""

import json
import uuid

import httpx
import openai
import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import SecretStr

from agent.graph import build_graph
from api.deps import AppState, ConcurrencyGate, MemoryRateLimiter, RedisRateLimiter
from api.main import create_app
from config import ApiSettings, get_settings
from tests.test_agent import ScriptedLLM, _deps, _row, _tool_turn
from generation.llm import ToolTurn

ANSWER = {"refused": False, "answer": "毛利600元[1][2]，每股收益68.64元[3]",
          "citations": [{"ref": 1, "quote": "营业收入 1,000 元"}, {"ref": 3, "quote": "68.64"}]}  # fmt: skip


def scripted_llm() -> ScriptedLLM:
    """一轮带工具调用的脚本：查指标 + 计算 -> 证据足够 -> 作答。"""
    return ScriptedLLM(
        turns=[
            _tool_turn(("get_financial_metric", {"company": "茅台", "year": 2024, "metric": "eps_basic"}),
                       ("calculator", {"expression": "1000-400"})),
            ToolTurn("证据已足够"),
        ],
        jsons=[ANSWER],
    )  # fmt: skip


class FailingLLM:
    def chat_tools(self, *a, **k):
        raise openai.APIConnectionError(request=httpx.Request("POST", "http://llm.invalid"))


def make_client(llm=None, audit=None, checks=None, **api) -> tuple[TestClient, AppState]:
    settings = get_settings().model_copy(update={"api": ApiSettings(**api)})
    deps = _deps(llm or scripted_llm(), rows=[_row(2025, "68.64", "68.64", 6)], audit=audit)
    state = AppState(settings, build_graph(deps, checkpointer=InMemorySaver()),
                     MemoryRateLimiter(settings.api.rate_limit, settings.api.rate_window_s),
                     ConcurrencyGate(settings.api.max_concurrent_runs), checks or {})  # fmt: skip
    return TestClient(create_app(settings, state_factory=lambda s: state)), state


def parse_sse(text: str) -> list[tuple[str, dict]]:
    """把 SSE 响应体解析成 [(事件名, 数据)]；跳过以冒号开头的保活注释。"""
    events = []
    for block in text.strip().split("\n\n"):
        fields = dict(line.split(": ", 1) for line in block.splitlines() if line and not line.startswith(":"))
        if fields:
            events.append((fields.get("event", "message"), json.loads(fields["data"])))
    return events


# ---------------- 健康检查 ----------------


def test_healthz_and_readyz() -> None:
    def broken():
        raise ConnectionError("redis://secret@host")

    client, _ = make_client(checks={"postgres": lambda: None, "redis": broken})
    with client:
        assert client.get("/healthz").json() == {"status": "ok"}
        r = client.get("/readyz")
        assert r.status_code == 503 and r.json()["components"] == {"postgres": "ok", "redis": "unavailable"}
        assert "secret" not in r.text  # 不泄露异常详情


# ---------------- 一次性问答 ----------------


def test_chat_returns_answer_citations_and_trace() -> None:
    audit = []
    client, _ = make_client(audit=audit.append)
    with client:
        r = client.post("/v1/chat", json={"question": "茅台毛利是多少？"}, headers={"X-Request-ID": "req-12345678"})
    assert r.status_code == 200
    body = r.json()
    # 合法的 X-Request-ID 被沿用为 trace_id：响应头、响应体、工具审计记录（在线程池中产生）三处一致
    assert r.headers["x-trace-id"] == body["trace_id"] == "req-12345678"
    assert {a["trace_id"] for a in audit} == {"req-12345678"}
    assert body["answer"].startswith("毛利600元") and not body["refused"] and body["steps"] == 2
    assert [c["verified"] for c in body["citations"]] == [True, True]
    assert body["citations"][1]["doc_id"] == "600519_2025" and body["citations"][1]["page"] == 6
    assert [e["source"] for e in body["evidence"]] == ["retrieve", "retrieve", "get_financial_metric"]
    assert [t["tool"] for t in body["tool_calls"]] == ["get_financial_metric", "calculator"]
    assert len(body["thread_id"]) == 32  # 未传 thread_id 时新建


def test_invalid_request_id_is_replaced() -> None:
    client, _ = make_client()
    with client:
        r = client.get("/healthz", headers={"X-Request-ID": "bad id\n<script>"})
    assert r.headers["x-trace-id"] != "bad id\n<script>" and len(r.headers["x-trace-id"]) == 16


@pytest.mark.parametrize(
    "payload",
    [{"question": ""}, {"question": "x" * 501}, {"question": "q", "thread_id": "../etc/passwd"}, {}],
)
def test_validation_errors_use_unified_format(payload) -> None:
    client, state = make_client(max_concurrent_runs=1)
    with client:
        r = client.post("/v1/chat", json=payload)
        assert r.status_code == 422
        err = r.json()["error"]
        assert err["code"] == "invalid_request" and err["trace_id"] == r.headers["x-trace-id"]
        # 校验失败后并发名额已归还：名额只有 1 个，仍然可以占到
        slot = state.gate.try_acquire()
        assert slot is not None
        slot.release()


def test_upstream_error_maps_to_502() -> None:
    client, state = make_client(llm=FailingLLM(), max_concurrent_runs=1)
    with client:
        r = client.post("/v1/chat", json={"question": "q"})
        assert r.status_code == 502 and r.json()["error"]["code"] == "upstream_error"
        assert "llm.invalid" not in r.text  # 不泄露上游地址
        assert state.gate.try_acquire() is not None  # 出错后名额已归还


# ---------------- 鉴权、限流、并发 ----------------


def test_api_key_checked_before_rate_limit() -> None:
    client, _ = make_client(keys=[SecretStr("k-1"), SecretStr("k-2")], rate_limit=1)
    with client:
        assert client.post("/v1/chat", json={"question": "q"}).json()["error"]["code"] == "unauthorized"
        assert client.post("/v1/chat", json={"question": "q"}, headers={"X-API-Key": "wrong"}).status_code == 401
        # 未通过鉴权的请求不占限流额度：正确的 Key 第一次请求仍然成功
        assert client.post("/v1/chat", json={"question": "q"}, headers={"X-API-Key": "k-2"}).status_code == 200
        r = client.post("/v1/chat", json={"question": "q"}, headers={"X-API-Key": "k-2"})
        assert r.status_code == 429 and int(r.headers["retry-after"]) >= 1


def test_busy_when_no_slot() -> None:
    client, state = make_client(max_concurrent_runs=1)
    with client:
        held = state.gate.try_acquire()
        r = client.post("/v1/chat/stream", json={"question": "q"})
        assert r.status_code == 503 and r.json()["error"]["code"] == "busy"  # 事件流开始前就拒绝，是正常的 HTTP 状态码
        held.release()


# ---------------- SSE ----------------


def test_stream_event_sequence() -> None:
    client, state = make_client(max_concurrent_runs=1)
    with client:
        r = client.post("/v1/chat/stream", json={"question": "茅台毛利是多少？", "thread_id": "t-1"})
        assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
        events = parse_sse(r.text)
        assert [e for e, _ in events] == ["start", "rewrite", "retrieve", "decide", "tool", "tool", "decide", "answer", "done"]
        data = dict(events)
        assert data["start"]["thread_id"] == "t-1" and data["start"]["trace_id"] == r.headers["x-trace-id"]
        assert data["retrieve"] == {"n_evidence": 2}
        assert events[3][1] == {"step": 1, "tool_calls": ["get_financial_metric", "calculator"]}
        assert events[5][1]["tool"] == "calculator" and events[5][1]["ok"]
        assert data["answer"]["answer"].startswith("毛利600元") and data["answer"]["thread_id"] == "t-1"
        assert state.gate.try_acquire() is not None  # 事件流结束后名额已归还


def test_stream_error_event() -> None:
    client, state = make_client(llm=FailingLLM(), max_concurrent_runs=1)
    with client:
        r = client.post("/v1/chat/stream", json={"question": "q"})
        events = parse_sse(r.text)
        assert [e for e, _ in events][-1] == "error" and "answer" not in dict(events)
        assert events[-1][1]["error"]["code"] == "upstream_error"
        assert state.gate.try_acquire() is not None


# ---------------- 多轮对话 ----------------


def test_multi_turn_and_thread_history() -> None:
    llm = ScriptedLLM(turns=[], jsons=[
        {"refused": False, "answer": "茅台营业收入1,000元[1]", "citations": [{"ref": 1, "quote": "1,000"}]},
        {"question": "五粮液2024年营业收入是多少？"},
        {"refused": True, "answer": "证据不足", "citations": []},
    ])  # fmt: skip
    client, _ = make_client(llm=llm)
    with client:
        first = client.post("/v1/chat", json={"question": "茅台2024年营业收入是多少？"}).json()
        second = client.post("/v1/chat", json={"question": "那五粮液呢？", "thread_id": first["thread_id"]}).json()
        assert second["standalone"] == "五粮液2024年营业收入是多少？"
        hist = client.get(f"/v1/threads/{first['thread_id']}").json()["history"]
        assert [h["question"] for h in hist] == ["茅台2024年营业收入是多少？", "五粮液2024年营业收入是多少？"]
        assert client.get("/v1/threads/not-exist").json()["error"]["code"] == "not_found"


# ---------------- Redis 限流（集成） ----------------


def test_redis_rate_limiter() -> None:
    redis = pytest.importorskip("redis")
    r = redis.Redis.from_url(get_settings().store.redis_url.get_secret_value(), socket_connect_timeout=1)
    try:
        r.ping()
    except redis.exceptions.RedisError:
        pytest.skip("本地 Redis 不可用")
    prefix = f"test-rl-{uuid.uuid4().hex[:8]}"
    limiter = RedisRateLimiter(r, limit=2, window_s=60, prefix=prefix)
    try:
        assert [limiter.hit("c")[0] for _ in range(3)] == [True, True, False]
        assert limiter.hit("other")[0]  # 不同客户端各自计数
        assert 0 < r.ttl(next(iter(r.scan_iter(f"{prefix}:c:*")))) <= 61  # 计数键会过期，不会越积越多
    finally:
        for k in r.scan_iter(f"{prefix}:*"):
            r.delete(k)
        r.close()


def test_redis_down_fails_open() -> None:
    class Down:
        def pipeline(self):
            raise ConnectionError("down")

    assert RedisRateLimiter(Down(), limit=1, window_s=60).hit("c") == (True, 0)
