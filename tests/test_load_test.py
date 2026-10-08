"""bench/* 的测试：题目分组、发请求与计时（httpx.MockTransport，不起服务）、服务端日志汇总、运行汇总（手算用例）。"""

import json
import threading
from collections import Counter

import httpx

from bench.load_test import attach_server_stats, make_groups, run_burst, run_closed_loop, run_one, summarize
from bench.server_log import aggregate
from config import get_settings
from eval.run_eval import load_frozen_goldset
from generation.llm import Price
from ui.client import ApiClient


def _sse(*events: tuple[str, dict]) -> str:
    return "".join(f"event: {e}\r\ndata: {json.dumps(d)}\r\n\r\n" for e, d in events)


OK_STREAM = _sse(
    ("start", {"thread_id": "t1", "trace_id": "tr1"}),
    ("retrieve", {"n": 10}),
    ("answer", {"refused": False, "format_ok": True, "tool_calls": [{"tool": "calculator"}], "steps": 2}),
    ("done", {"latency_ms": 12.0}),
)


def _client(handler) -> ApiClient:
    return ApiClient("http://api", http=httpx.Client(base_url="http://api", transport=httpx.MockTransport(handler)))


def _ok_handler(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, text=OK_STREAM, headers={"content-type": "text/event-stream"})


def _item(i: int) -> dict:
    return {"id": f"q{i:03d}", "type": "fact", "variant": "original", "question": f"问题{i}"}


# ---------------- 分组 ----------------


def test_make_groups_small_case_by_hand() -> None:
    """2 组；fact 有 2 对（每对原题 + 改写），calc 有 2 道单题：每组恰好 1 对 fact + 1 道 calc，且改写与原题同组。"""
    items = [
        {"id": "q1", "type": "fact"}, {"id": "q101", "type": "fact", "source_id": "q1"},
        {"id": "q2", "type": "fact"}, {"id": "q102", "type": "fact", "source_id": "q2"},
        {"id": "q3", "type": "calc"}, {"id": "q4", "type": "calc"},
    ]  # fmt: skip
    groups = make_groups(items, n_groups=2, seed=0)
    assert sorted(len(g) for g in groups) == [3, 3]
    for g in groups:
        ids = {it["id"] for it in g}
        assert Counter(it["type"] for it in g) == {"fact": 2, "calc": 1}
        assert ("q1" in ids) == ("q101" in ids) and ("q2" in ids) == ("q102" in ids)
    assert make_groups(items, n_groups=2, seed=0) == groups  # 固定种子可复现


def test_make_groups_on_goldset_v2() -> None:
    """v2 的 120 题切成 4 组：每组 30 题、互不重叠、覆盖全部；每组题型分布相同；改写题与原题同组。"""
    s = get_settings()
    items = load_frozen_goldset(s.eval.goldset_file("v2"), s.eval.goldsets["v2"].sha256)
    groups = make_groups(items)
    assert [len(g) for g in groups] == [30, 30, 30, 30]
    all_ids = [it["id"] for g in groups for it in g]
    assert len(all_ids) == len(set(all_ids)) == 120
    dists = [Counter(it["type"] for it in g) for g in groups]
    assert all(d == dists[0] for d in dists)
    assert dists[0] == {"table_number": 8, "fact": 6, "multi_hop": 4, "calc": 4, "compare": 4, "out_of_scope": 4}
    where = {it["id"]: gi for gi, g in enumerate(groups) for it in g}
    assert all(where[it["id"]] == where[it["source_id"]] for it in items if it.get("source_id"))


# ---------------- 发请求 ----------------


def test_run_one_success_records_timings() -> None:
    rec = run_one(_client(_ok_handler), _item(1), t_run0=0.0)
    assert rec["status"] == "ok" and rec["trace_id"] == "tr1" and rec["tools"] == ["calculator"]
    assert 0 <= rec["start_ms"] <= rec["retrieved_ms"] <= rec["answer_ms"] <= rec["total_ms"]
    assert list(rec["events_ms"]) == ["retrieve"]
    assert rec["server_ms"] == 12.0


def test_run_one_rejected_before_stream() -> None:
    def handler(request):
        return httpx.Response(503, json={"error": {"code": "busy", "message": "并发已满", "trace_id": "tr9"}})

    rec = run_one(_client(handler), _item(1), t_run0=0.0)
    assert rec["status"] == "error" and rec["error_code"] == "busy" and rec["http_status"] == 503
    assert rec["trace_id"] == "tr9" and "start_ms" not in rec


def test_run_one_error_event_after_start() -> None:
    body = _sse(("start", {"trace_id": "tr2"}), ("error", {"error": {"code": "upstream_error", "message": "x"}}))

    def handler(request):
        return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})

    rec = run_one(_client(handler), _item(1), t_run0=0.0)
    assert rec["status"] == "error" and rec["error_code"] == "upstream_error" and "answer_ms" not in rec


def test_closed_loop_asks_each_question_once_with_bounded_concurrency() -> None:
    in_flight, peak, lock = 0, 0, threading.Lock()

    def handler(request):
        nonlocal in_flight, peak
        with lock:
            in_flight += 1
            peak = max(peak, in_flight)
        threading.Event().wait(0.02)
        with lock:
            in_flight -= 1
        return _ok_handler(request)

    items = [_item(i) for i in range(7)]
    records, wall = run_closed_loop(lambda: _client(handler), items, concurrency=3)
    assert sorted(r["id"] for r in records) == [it["id"] for it in items]
    assert all(r["status"] == "ok" for r in records) and peak <= 3 and wall > 0


def test_burst_rounds_and_verify() -> None:
    records, _ = run_burst(lambda: _client(_ok_handler), [_item(i) for i in range(5)], size=4, rounds=2, verify=3)
    assert Counter(r["round"] for r in records) == {0: 4, 1: 4, "verify": 3}
    # 题目循环使用：8 + 3 = 11 个请求依次取 5 道题
    assert [r["id"] for r in records if r["round"] == "verify"] == ["q003", "q004", "q000"]


# ---------------- 服务端日志 ----------------


def test_aggregate_server_log_by_hand() -> None:
    lines = [
        "INFO:     Uvicorn running",  # 不是 JSON，跳过
        json.dumps({"trace_id": "a", "stage": "lock", "lock": "search", "wait_ms": 100.0, "hold_ms": 300.0}),
        json.dumps({"trace_id": "a", "stage": "agent.retrieve", "elapsed_ms": 401.0}),
        json.dumps({"trace_id": "a", "msg": "llm tool call", "latency_ms": 1000.0,
                    "prompt_tokens": 1000, "completion_tokens": 100, "cache_hit_tokens": 400}),
        json.dumps({"trace_id": "a", "stage": "lock", "lock": "search", "wait_ms": 50.0, "hold_ms": 200.0}),
        json.dumps({"trace_id": "a", "stage": "lock", "lock": "page_texts", "wait_ms": 9.0, "hold_ms": 9.0}),
        json.dumps({"trace_id": "a", "msg": "llm call", "latency_ms": 500.0,
                    "prompt_tokens": 2000, "completion_tokens": 50, "cache_hit_tokens": 0}),
        json.dumps({"trace_id": "b", "msg": "llm call", "latency_ms": 700.0,
                    "prompt_tokens": 10, "completion_tokens": 1, "cache_hit_tokens": 0}),
        json.dumps({"trace_id": "-", "msg": "llm call", "latency_ms": 1.0}),  # 没有 trace_id 的日志不归属任何请求
    ]  # fmt: skip
    out = aggregate(lines, Price(input_cache_hit=1.0, input_cache_miss=10.0, output=100.0))
    a = out["a"]
    assert (a["n_llm"], a["llm_ms"]) == (2, 1500.0)
    assert (a["n_search"], a["search_wait_ms"], a["search_hold_ms"], a["retrieve_ms"]) == (2, 150.0, 500.0, 401.0)
    assert (a["prompt_tokens"], a["completion_tokens"], a["cache_hit_tokens"]) == (3000, 150, 400)
    # 费用：命中 400×1 + 未命中 2600×10 + 输出 150×100 = 400 + 26000 + 15000 = 41400，除以 100 万
    assert abs(a["cost_usd"] - 0.0414) < 1e-12
    assert set(out) == {"a", "b"}


# ---------------- 汇总 ----------------


def test_summarize_by_hand() -> None:
    """4 个成功（答案 1000/2000/3000/4000ms）+ 1 个 503；墙钟 30 秒。

    最近秩法：p50 = 第 ceil(0.5×4)=2 个 = 2000；p90 = 第 ceil(3.6)=4 个 = 4000。吞吐 = 4 / 30 × 60 = 8 题/分钟。
    """
    ok = [{"status": "ok", "answer_ms": float(t), "retrieved_ms": t / 10, "trace_id": f"t{t}"}
          for t in (3000, 1000, 4000, 2000)]  # fmt: skip
    bad = {"status": "error", "error_code": "busy", "http_status": 503, "total_ms": 5.0}
    records = ok + [bad]
    server = {f"t{t}": {"search_wait_ms": w, "search_hold_ms": 300.0, "llm_ms": 2000.0, "n_llm": n,
                        "prompt_tokens": 1000, "cache_hit_tokens": 250, "cost_usd": 0.001}
              for t, w, n in ((1000, 0.0, 2), (2000, 100.0, 2), (3000, 200.0, 3), (4000, 900.0, 3))}  # fmt: skip
    assert attach_server_stats(records, server) == 4
    s = summarize(records, wall_s=30.0)
    assert (s["n_requests"], s["n_ok"], s["errors"]) == (5, 4, {"busy": 1})
    assert s["throughput_qpm"] == 8.0
    assert (s["answer_p50_ms"], s["answer_p90_ms"], s["answer_max_ms"]) == (2000.0, 4000.0, 4000.0)
    assert (s["retrieved_p50_ms"], s["retrieved_p90_ms"]) == (200.0, 400.0)
    assert (s["search_wait_p50_ms"], s["search_wait_p90_ms"]) == (100.0, 900.0)
    assert (s["llm_calls_mean"], s["cache_hit_ratio"], s["cost_usd"]) == (2.5, 0.25, 0.004)
    assert (s["rejected_p50_ms"], s["rejected_max_ms"]) == (5.0, 5.0)
    assert s["by_round"] is None
