"""从 api 服务的 JSON 日志中按 trace_id 汇总每个请求在服务端的耗时构成（阶段 11 压测，D-032）。

用到的日志（每行一个 JSON 对象，见 common/log.py）：
    - stage="lock"、lock="search"：检索锁的排队时间 wait_ms 与执行时间 hold_ms（agent/graph.py locked）。
      初始检索和 search_reports 工具共用这把锁，一个请求可能有多条，按请求求和；
    - stage="agent.retrieve"：初始检索节点的总耗时（含排队）；
    - msg="llm call" / "llm tool call"（generation/llm.py）：每次大模型调用的耗时与 token 用量。

日志从 api 容器读取：docker logs --since <开始时间> <容器名>。
"""

from __future__ import annotations

import json
import subprocess
from collections import defaultdict
from collections.abc import Iterable

from generation.llm import Price, Usage

LLM_MSGS = {"llm call", "llm tool call"}


def _empty() -> dict:
    return {"n_llm": 0, "llm_ms": 0.0, "prompt_tokens": 0, "completion_tokens": 0, "cache_hit_tokens": 0,
            "n_search": 0, "search_wait_ms": 0.0, "search_hold_ms": 0.0, "retrieve_ms": None}  # fmt: skip


def aggregate(lines: Iterable[str], price: Price | None = None) -> dict[str, dict]:
    """trace_id -> 该请求的服务端统计。不是 JSON 的行（例如 uvicorn 的启动信息）直接跳过。

    price 给定时，按 token 用量估算该请求的大模型费用 cost_usd（与评测相同的计价方法）。
    """
    out: dict[str, dict] = defaultdict(_empty)
    for line in lines:
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        tid = rec.get("trace_id") if isinstance(rec, dict) else None
        if not tid or tid == "-":
            continue
        s = out[tid]
        if rec.get("msg") in LLM_MSGS:
            s["n_llm"] += 1
            s["llm_ms"] += rec.get("latency_ms") or 0
            for k in ("prompt_tokens", "completion_tokens", "cache_hit_tokens"):
                s[k] += rec.get(k) or 0
        elif rec.get("stage") == "lock" and rec.get("lock") == "search":
            s["n_search"] += 1
            s["search_wait_ms"] += rec.get("wait_ms") or 0
            s["search_hold_ms"] += rec.get("hold_ms") or 0
        elif rec.get("stage") == "agent.retrieve":
            s["retrieve_ms"] = rec.get("elapsed_ms")
    if price is not None:
        for s in out.values():
            s["cost_usd"] = price.cost(Usage(s["prompt_tokens"], s["completion_tokens"], s["cache_hit_tokens"]))
    return dict(out)


def read_container_logs(container: str, since: str) -> list[str]:
    """读取容器自 since（RFC 3339 时间）以来的日志。应用日志写在 stderr，所以合并两个输出流。"""
    proc = subprocess.run(["docker", "logs", "--since", since, container],
                          capture_output=True, text=True, check=True)  # fmt: skip
    return (proc.stdout + proc.stderr).splitlines()
