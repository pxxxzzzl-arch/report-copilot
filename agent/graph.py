"""LangGraph Agent：固定骨架 + 工具循环（docs/decisions.md D-026）。

    START
      │
    rewrite    多轮对话：结合历史把追问改写成完整问题（"那五粮液呢？" -> "五粮液2025年营业收入是多少？"）；
      │        没有历史时不调用模型。同时清空上一轮留下的每轮字段
    retrieve   用阶段 5 的检索流水线（E013）取前 context_k 个 chunk，作为证据 [1]..[k]
      │
    agent  ⇄  tools      模型判断证据是否足够：不够就调用工具（可并行多个），工具结果登记为新证据后回到 agent；
      │                  模型不再调用工具，或达到 max_steps 次，进入作答
    answer     JSON 模式作答（沿用 answer-v1 的规则和格式 + 计算结果使用规则），只能引用证据编号
      │
    verify     引用校验：摘录是否逐字出现在所引证据中、定位页码（复用 generation/citation.py），
      │        并把"问题 + 回答"追加到跨轮历史
     END

为什么先固定检索、而不是让模型从零决定：起点与 G002 完全相同，简单题上 Agent 至少不比 G002 差；
工具只在需要时补充。模型调用次数也更少、更可控。
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from agent.state import AgentState, Evidence
from agent.tools import TOOL_SPECS, ToolBox
from common.log import get_trace_id, timed
from config import AgentSettings
from generation.answer import Answer, Citation, parse_answer
from generation.citation import PageTextLoader, check_citations
from generation.llm import LLMClient, Usage
from generation.prompts import build_agent_answer_messages, build_agent_messages, build_rewrite_messages
from ingest.chunk import Chunk

logger = logging.getLogger("agent.graph")

# 每轮开始时清空的字段（history 跨轮保留，不在此列）
PER_TURN_RESET = {"evidence": [], "messages": [], "steps": 0, "calc_log": [], "tool_log": [], "llm_calls": [],
                  "answer": {}, "citations": []}  # fmt: skip


@dataclass
class AgentDeps:
    """图运行所需的外部依赖。全部通过参数注入，测试时可以换成假的大模型、假的检索和内存存储。"""

    llm: LLMClient
    retrieve: Callable[[str, int], list[Chunk]]  # 初始检索（与 search_reports 共用同一条流水线）
    tools: ToolBox
    page_texts: PageTextLoader
    cfg: AgentSettings
    context_k: int  # 初始检索取几个 chunk（与 G002 相同：generation.context_k）
    answer_max_tokens: int  # 作答阶段最大输出 token（generation.max_tokens）
    audit: Callable[[dict], None] | None = None  # 工具调用审计（写 Postgres）；为 None 时不写


def _call_record(node: str, usage: Usage, latency_ms: float, from_cache: bool) -> dict:
    return {"node": node, "usage": asdict(usage), "latency_ms": latency_ms, "from_cache": from_cache}


def build_graph(deps: AgentDeps, checkpointer=None):
    """组装并编译图。checkpointer 为 None 时不持久化（单轮评测）；多轮对话传入 PostgresSaver。"""
    cfg = deps.cfg

    def rewrite(state: AgentState) -> dict:
        question = state["question"]
        history = (state.get("history") or [])[-cfg.max_history_turns :]
        update = {**PER_TURN_RESET, "standalone": question}
        if not history:
            return update
        with timed("agent.rewrite", logger):
            res = deps.llm.chat_json(build_rewrite_messages(history, question), cfg.rewrite_max_tokens)
        rewritten = (res.data or {}).get("question")
        if isinstance(rewritten, str) and rewritten.strip():
            update["standalone"] = rewritten.strip()
        else:  # 改写失败时用原话继续，不中断对话
            logger.warning("rewrite failed", extra={"error": res.error})
        update["llm_calls"] = [_call_record("rewrite", res.usage, res.latency_ms, res.from_cache)]
        logger.info("question rewritten", extra={"question": question, "standalone": update["standalone"]})
        return update

    def retrieve(state: AgentState) -> dict:
        with timed("agent.retrieve", logger):
            chunks = deps.retrieve(state["standalone"], deps.context_k)
        evidence = [
            Evidence(i, "retrieve", c.doc_id, c.company, c.year, c.page_start, c.page_end, c.text, c.chunk_id)
            for i, c in enumerate(chunks, start=1)
        ]
        return {
            "evidence": [e.to_dict() for e in evidence],
            "messages": build_agent_messages(state["standalone"], evidence),
        }

    def agent(state: AgentState) -> dict:
        with timed("agent.decide", logger, step=state.get("steps", 0) + 1):
            turn = deps.llm.chat_tools(state["messages"], TOOL_SPECS, cfg.tool_max_tokens)
        return {
            "messages": state["messages"] + [turn.assistant_message()],
            "steps": state.get("steps", 0) + 1,
            "llm_calls": state.get("llm_calls", []) + [_call_record("agent", turn.usage, turn.latency_ms, turn.from_cache)],
        }

    def route_after_agent(state: AgentState) -> str:
        last = state["messages"][-1]
        return "tools" if last.get("tool_calls") else "answer"

    def tools(state: AgentState, config: RunnableConfig) -> dict:
        registry = [Evidence.from_dict(d) for d in state["evidence"]]
        messages = list(state["messages"])
        calc_log, tool_log = list(state.get("calc_log", [])), list(state.get("tool_log", []))
        # 按模型给出的顺序逐个执行（检索流水线的嵌入模型不是线程安全的，这里不并发）
        for call in state["messages"][-1]["tool_calls"]:
            fn = call["function"]
            with timed("agent.tool", logger, tool=fn["name"]):
                out = deps.tools.run(fn["name"], fn["arguments"], registry)
            registry += out.new_evidence
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": out.content})
            tool_log.append(out.record)
            if out.calc:
                calc_log.append(out.calc)
            if deps.audit:  # 审计记录带上 trace_id 和对话编号，便于按请求或按对话排查
                thread = (config.get("configurable") or {}).get("thread_id")
                deps.audit({**out.record, "trace_id": get_trace_id(), "thread_id": thread})
        return {"evidence": [e.to_dict() for e in registry], "messages": messages,
                "calc_log": calc_log, "tool_log": tool_log}  # fmt: skip

    def route_after_tools(state: AgentState) -> str:
        # 达到步数上限就停止取证，用已有证据作答（防止模型反复调用工具、陷入循环）
        return "agent" if state["steps"] < cfg.max_steps else "answer"

    def answer(state: AgentState) -> dict:
        evidence = [Evidence.from_dict(d) for d in state["evidence"]]
        calcs = [f"{c['expression']} = {c['result']}" for c in state.get("calc_log", []) if "result" in c]
        with timed("agent.answer", logger, n_evidence=len(evidence), n_calc=len(calcs)):
            res = deps.llm.chat_json(
                build_agent_answer_messages(state["standalone"], evidence, calcs), deps.answer_max_tokens
            )
        ans = parse_answer(res.data, res.error)
        return {
            "answer": asdict(ans),
            "llm_calls": state.get("llm_calls", []) + [_call_record("answer", res.usage, res.latency_ms, res.from_cache)],
        }

    def verify(state: AgentState) -> dict:
        evidence = [Evidence.from_dict(d) for d in state["evidence"]]
        cites = [Citation(**c) for c in state["answer"]["citations"]]
        checks = check_citations(cites, evidence, deps.page_texts)
        a = state["answer"]
        logger.info(
            "agent done",
            extra={"steps": state.get("steps", 0), "n_evidence": len(evidence), "refused": a["refused"],
                   "n_cites": len(checks), "n_verified": sum(c.verified for c in checks)},
        )  # fmt: skip
        return {
            "citations": [asdict(c) for c in checks],
            "history": [{"question": state["standalone"], "answer": a["answer"]}],  # 由 operator.add 追加到历史
        }

    g = StateGraph(AgentState)
    for name, fn in [("rewrite", rewrite), ("retrieve", retrieve), ("agent", agent), ("tools", tools),
                     ("answer", answer), ("verify", verify)]:  # fmt: skip
        g.add_node(name, fn)
    g.add_edge(START, "rewrite")
    g.add_edge("rewrite", "retrieve")
    g.add_edge("retrieve", "agent")
    g.add_conditional_edges("agent", route_after_agent, {"tools": "tools", "answer": "answer"})
    g.add_conditional_edges("tools", route_after_tools, {"agent": "agent", "answer": "answer"})
    g.add_edge("answer", "verify")
    g.add_edge("verify", END)
    return g.compile(checkpointer=checkpointer)


def answer_from_state(state: dict) -> Answer:
    """把图的最终状态还原成 generation.answer.Answer（评测复用阶段 6 的计分代码）。"""
    a = state["answer"]
    return Answer(a["answer"], a["refused"], [Citation(**c) for c in a["citations"]], a["format_ok"],
                  a["n_bad_citations"], a["error"])  # fmt: skip


def _turn_config(max_steps: int, thread_id: str | None) -> dict:
    config: dict = {"configurable": {"thread_id": thread_id}} if thread_id else {}
    # LangGraph 的递归上限按"经过的节点数"计：每次取证循环经过 agent、tools 两个节点，再加 5 个固定节点，留出余量
    config["recursion_limit"] = 2 * max_steps + 10
    return config


def run_turn(graph, question: str, max_steps: int, thread_id: str | None = None) -> dict:
    """执行一轮问答，返回最终状态。thread_id 相同的多次调用共享历史（需要编译时传入 checkpointer）。"""
    return graph.invoke({"question": question}, config=_turn_config(max_steps, thread_id))


def stream_turn(graph, question: str, max_steps: int, thread_id: str | None = None) -> Iterator[dict]:
    """执行一轮问答，边运行边产出进度事件（API 的 SSE 流式接口用它）。

    用 LangGraph 的 stream_mode="updates"：每个节点运行结束时拿到它返回的更新，转成对用户有意义的事件：
        rewrite   改写后的完整问题        retrieve  初始检索到的证据数
        decide    第几步取证、要调用哪些工具（为空表示证据已足够，接下来作答）
        tool      每次工具调用的结果       final     本轮最终状态（与 run_turn 的返回值字段相同，不含 history）
    """
    final: dict = {}
    n_tools = 0
    for update in graph.stream({"question": question}, config=_turn_config(max_steps, thread_id), stream_mode="updates"):
        for node, upd in update.items():
            if not isinstance(upd, dict):
                continue
            # 每轮字段在 rewrite 节点被整体重置，所以把各节点的更新依次合并，就得到本轮的完整状态
            final.update({k: v for k, v in upd.items() if k != "history"})
            if node == "rewrite":
                yield {"event": "rewrite", "standalone": upd["standalone"]}
            elif node == "retrieve":
                yield {"event": "retrieve", "n_evidence": len(upd["evidence"])}
            elif node == "agent":
                calls = upd["messages"][-1].get("tool_calls") or []
                yield {"event": "decide", "step": upd["steps"], "tool_calls": [c["function"]["name"] for c in calls]}
            elif node == "tools":
                for rec in upd["tool_log"][n_tools:]:
                    yield {"event": "tool", **rec}
                n_tools = len(upd["tool_log"])
    yield {"event": "final", "state": final}


def locked(fn: Callable, name: str) -> Callable:
    """给函数加一把锁：嵌入模型和页面文本缓存都不是线程安全的，多个 Agent 并发运行时要串行访问。

    每次调用输出一条 stage="lock" 的日志：wait_ms 是排队等锁的时间，hold_ms 是拿到锁后实际执行的时间。
    并发时检索变慢，靠这两个数区分是在排队还是在计算（阶段 11 压测，D-032）。
    """
    lock = threading.Lock()

    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        with lock:
            t1 = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                t2 = time.perf_counter()
                logger.info("lock released", extra={"stage": "lock", "lock": name,
                                                    "wait_ms": round((t1 - t0) * 1000, 2),
                                                    "hold_ms": round((t2 - t1) * 1000, 2)})  # fmt: skip

    return wrapper


def build_deps(
    settings,
    search: Callable[[str, int], list[Chunk]],
    store,
    audit: Callable[[dict], None] | None = None,
    llm_cache: bool = True,
) -> AgentDeps:
    """用真实组件组装依赖：search 为检索函数（通常是 SearchPipeline.search），store 为财务指标表（可为 None）。

    检索和页面文本在这里加锁，因此返回的依赖可以被多个线程同时使用（评测并发、API 服务）。
    llm_cache：是否使用大模型本地文件缓存（评测开启以保证可复现；API 服务默认关闭，见 config.api.llm_cache）。
    """
    from eval.validate_goldset import PdfPageTexts
    from ingest.download import read_manifest
    from retrieval.query import QueryParser

    manifest = read_manifest(settings.paths.manifest_path)
    parser = QueryParser(manifest, settings.retrieval.company_aliases)
    names = {r.stock_code: r.company for r in manifest.values()}
    search = locked(search, "search")  # 初始检索与 search_reports 共用同一把锁
    toolbox = ToolBox(search, parser, store, names, settings.agent.search_k)
    return AgentDeps(
        llm=LLMClient(settings.llm, settings.generation.cache_dir if llm_cache else None),
        retrieve=search,
        tools=toolbox,
        page_texts=locked(PdfPageTexts(manifest, settings.paths.raw_dir), "page_texts"),
        cfg=settings.agent,
        context_k=settings.generation.context_k,
        answer_max_tokens=settings.generation.max_tokens,
        audit=audit,
    )


def main(argv: list[str] | None = None) -> int:
    """命令行试用：uv run python -m agent.graph "问题" ["追问" ...]（同一次运行内的多个问题属于同一对话）。

    检索配置沿用环境变量（与生成评测相同，例如 E013）。加 --persist 时用 PostgresSaver 保存对话，
    配合 --thread 可以在下一次运行时继续同一个对话。
    """
    import argparse
    import uuid

    from agent.fin_db import PgFinStore
    from common.log import setup_logging, trace_context
    from config import get_settings
    from retrieval.search import build_pipeline

    p = argparse.ArgumentParser(description="Agent 命令行试用")
    p.add_argument("questions", nargs="+")
    p.add_argument("--persist", action="store_true", help="用 Postgres 保存对话状态")
    p.add_argument("--thread", default=None, help="对话编号；不填则新建")
    args = p.parse_args(argv)

    settings = get_settings()
    setup_logging(settings.log.level, settings.log.json_format)
    dsn = settings.store.postgres_dsn.get_secret_value()
    store = PgFinStore(dsn)
    thread = args.thread or uuid.uuid4().hex
    try:
        deps = build_deps(settings, build_pipeline(settings).search, store,
                          audit=store.log_tool_call if settings.agent.log_tool_calls else None)  # fmt: skip
        if args.persist:
            from langgraph.checkpoint.postgres import PostgresSaver

            with PostgresSaver.from_conn_string(dsn) as saver:
                saver.setup()
                _chat(build_graph(deps, saver), args.questions, settings.agent.max_steps, thread, trace_context)
        else:
            from langgraph.checkpoint.memory import InMemorySaver

            _chat(build_graph(deps, InMemorySaver()), args.questions, settings.agent.max_steps, thread, trace_context)
    finally:
        store.close()
    print(f"\n对话编号 thread={thread}")
    return 0


def _chat(graph, questions: list[str], max_steps: int, thread: str, trace_context) -> None:
    for q in questions:
        with trace_context():
            st = run_turn(graph, q, max_steps, thread)
        print(f"\n问：{q}\n改写：{st['standalone']}")
        for t in st["tool_log"]:
            print(f"  工具 {t['tool']} {t['args']} ok={t['ok']} 新证据 {t['n_results']}")
        a = st["answer"]
        print(f"答{'（拒答）' if a['refused'] else ''}：{a['answer']}")
        for c in st["citations"]:
            print(f"  [{c['ref']}] {'✓' if c['verified'] else '✗'} {c['doc_id']} p{c['page']} {c['quote'][:40]}")


if __name__ == "__main__":
    import sys

    sys.exit(main())
