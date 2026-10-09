"""agent/* 与 LLMClient.chat_tools 的单元测试：假大模型按脚本发出工具调用，假检索 + 内存存储，不联网、不连数据库。"""

import json
import threading
import time
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from agent.fin_db import MetricRow
from agent.graph import AgentDeps, answer_from_state, build_graph, locked, run_turn
from agent.state import Evidence
from agent.tools import CalcError, ToolBox, format_decimal, safe_calc
from common.log import trace_context
from config import AgentSettings, LLMSettings
from generation.llm import LLMClient, LLMResult, ToolCall, ToolTurn, Usage
from ingest.download import ManifestRow
from retrieval.query import QueryParser
from tests.test_index_dense import _chunk

# ---------------- 安全计算器 ----------------


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        ("1+2*3", "7"),
        ("(168838102514.79-170899152276.34)/170899152276.34*100", "-1.2060035021"),  # 用 fractions.Fraction 独立核对过
        ("0.1+0.2", "0.3"),  # Decimal：没有二进制浮点误差
        ("-(3-5)", "2"),
        ("10/4", "2.5"),
        (".5+1e3", "1000.5"),  # 小数点开头、科学计数法都是普通十进制写法
    ],
)
def test_safe_calc_values(expr: str, expected: str) -> None:
    assert format_decimal(safe_calc(expr)) == expected


@pytest.mark.parametrize(
    "expr",
    [
        "__import__('os').system('ls')",  # 函数调用
        "x+1",  # 变量名
        "(1).real",  # 属性访问
        "9**9**9",  # 幂运算：可以耗尽内存
        "1,234+1",  # 千分位逗号被解析成元组
        "1/0",
        "",
        "1+" * 200 + "1",  # 过长
        "[1][0]",
        "5%",
        "0/0",  # Decimal 抛 InvalidOperation
        "1e999999*1e999999",  # Decimal 抛 Overflow：修复前未被捕获，整个请求失败（D-035）
        "9e999998*100",
        "0x10+1",  # 十六进制等 Python 字面量
        "1_000+1",
    ],
)
def test_safe_calc_rejects(expr: str) -> None:
    with pytest.raises(CalcError):
        safe_calc(expr)


@pytest.mark.parametrize(
    ("expr", "message"),
    [("1/0", "除数为 0"), ("0/0", "无效"), ("1e999999*1e999999", "数值过大"), ("0x10+1", "十进制")],
)
def test_safe_calc_error_messages(expr: str, message: str) -> None:
    """错误信息会返回给模型，让它改正后重试，所以要说清楚错在哪里。"""
    with pytest.raises(CalcError, match=message):
        safe_calc(expr)


def test_format_decimal() -> None:
    assert format_decimal(Decimal("12.3400")) == "12.34"
    assert format_decimal(Decimal("1E+3")) == "1000"
    assert format_decimal(Decimal("2") / Decimal("3")) == "0.6666666667"


# ---------------- 工具 ----------------

MANIFEST = {
    "600519_2025": ManifestRow("600519_2025", "贵州茅台", "600519", 2025, "a.pdf", 10, "sse", "2026-03-01", "u", "h"),
    "000858_2025": ManifestRow("000858_2025", "五粮液", "000858", 2025, "b.pdf", 10, "szse", "2026-03-01", "u", "h"),
}


class MemStore:
    def __init__(self, rows: list[MetricRow]) -> None:
        self.rows, self.logged = rows, []

    def get_metric(self, stock_code, fiscal_year, metric):
        return [r for r in self.rows if (r.stock_code, r.fiscal_year, r.metric) == (stock_code, fiscal_year, metric)]

    def log_tool_call(self, record):
        self.logged.append(record)


def _row(report_year: int, value: str, raw: str, page: int) -> MetricRow:
    return MetricRow(f"600519_{report_year}", "600519", "贵州茅台", report_year, 2024, "eps_basic", Decimal(value),
                     "元/股", raw, "元/股", page, "基本每股收益（元／股）")  # fmt: skip


def _toolbox(search=None, rows=None) -> ToolBox:
    parser = QueryParser(MANIFEST, {"茅台": "600519"})
    names = {r.stock_code: r.company for r in MANIFEST.values()}
    return ToolBox(search or (lambda q, k: []), parser, MemStore(rows or []), names, search_k=3)


def test_metric_tool_registers_evidence_with_continuing_ids() -> None:
    tb = _toolbox(rows=[_row(2025, "68.64", "68.64", 6), _row(2024, "68.64", "68.64", 5)])
    registry = [Evidence(1, "retrieve", "d", "c", 2025, 1, 1, "x")]
    out = tb.run("get_financial_metric", json.dumps({"company": "茅台", "year": 2024, "metric": "eps_basic"}), registry)
    assert out.record["ok"] and [e.id for e in out.new_evidence] == [2, 3]
    assert out.new_evidence[0].source == "get_financial_metric" and out.new_evidence[0].page_start == 6
    assert '<evidence id="2"' in out.content and "68.64" in out.content
    assert "追溯调整" not in out.content  # 两份年报数值相同，不提示


def test_metric_tool_flags_restatement_and_bad_args() -> None:
    tb = _toolbox(rows=[_row(2025, "4.61", "4.61", 11), _row(2024, "13.84", "13.84", 11)])
    out = tb.run("get_financial_metric", '{"company": "600519", "year": "2024年", "metric": "eps_basic"}', [])
    assert "追溯调整" in out.content  # 年份写成字符串 "2024年" 也接受

    for raw, msg in [
        ("not json", "JSON"),
        ('{"company": "某不存在公司", "year": 2024, "metric": "eps_basic"}', "无法唯一识别公司"),
        ('{"company": "茅台", "year": 2024, "metric": "gross_margin"}', "metric 必须是"),
        ('{"company": "茅台", "year": true, "metric": "eps_basic"}', "year 必须是整数"),
    ]:
        bad = tb.run("get_financial_metric", raw, [])
        assert not bad.record["ok"] and msg in bad.content and bad.new_evidence == []
    assert not tb.run("rm_rf", "{}", []).record["ok"]


def test_search_tool_prefixes_entities_and_dedupes() -> None:
    calls = []
    chunks = [_chunk(0, "a", 3), _chunk(1, "b", 4)]

    def search(q, k):
        calls.append((q, k))
        return chunks

    tb = _toolbox(search=search)
    registry = [Evidence(1, "retrieve", "000001_2024", "测试", 2024, 3, 3, "a", chunks[0].chunk_id)]
    out = tb.run("search_reports", '{"query": "研发投入", "company": "五粮液", "year": 2025}', registry)
    assert calls == [("五粮液2025年 研发投入", 3)]  # 公司和年份写进检索文本，交给查询理解过滤
    assert [e.id for e in out.new_evidence] == [2] and out.new_evidence[0].chunk_id == chunks[1].chunk_id
    assert "已在现有证据中：[1]" in out.content


def test_calculator_tool_record() -> None:
    out = _toolbox().run("calculator", '{"expression": "(2-1)/4"}', [])
    assert out.calc == {"expression": "(2-1)/4", "result": "0.25"} and out.new_evidence == []
    bad = _toolbox().run("calculator", '{"expression": "abs(-1)"}', [])
    assert not bad.record["ok"] and "error" in bad.calc
    # 溢出也要变成一次失败的工具调用、把错误返回给模型，而不是抛出异常让整个请求失败（D-035）
    big = _toolbox().run("calculator", '{"expression": "1e999999*1e999999"}', [])
    assert not big.record["ok"] and "数值过大" in big.content


# ---------------- LLMClient.chat_tools ----------------


class FakeToolCompletions:
    def __init__(self, messages) -> None:
        self.messages, self.requests = list(messages), []

    def create(self, **req):
        self.requests.append(req)
        msg = self.messages.pop(0)
        usage = SimpleNamespace(prompt_tokens=50, completion_tokens=10, prompt_cache_hit_tokens=0)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="tool_calls")], usage=usage)


def test_chat_tools_parses_and_caches(tmp_path: Path) -> None:
    fn = SimpleNamespace(name="calculator", arguments='{"expression": "1+1"}')
    msg = SimpleNamespace(content=None, tool_calls=[SimpleNamespace(id="call_1", function=fn)])
    comp = FakeToolCompletions([msg])
    client = LLMClient(LLMSettings(model="m"), tmp_path, client=SimpleNamespace(chat=SimpleNamespace(completions=comp)))
    turn = client.chat_tools([{"role": "user", "content": "q"}], [{"type": "function"}], 100)
    assert turn.tool_calls == [ToolCall("call_1", "calculator", '{"expression": "1+1"}')] and not turn.from_cache
    req = comp.requests[0]
    assert "tools" in req and "response_format" not in req and req["temperature"] == 0
    assert turn.assistant_message()["tool_calls"][0]["function"]["name"] == "calculator"
    # 第二次相同请求命中缓存，不再调用接口（FakeToolCompletions 已没有可弹出的消息，调用会报错）
    again = client.chat_tools([{"role": "user", "content": "q"}], [{"type": "function"}], 100)
    assert again.from_cache and again.tool_calls == turn.tool_calls


# ---------------- 图 ----------------


class ScriptedLLM:
    """按脚本返回结果的假大模型：chat_tools 依次返回 turns，chat_json 依次返回 jsons。"""

    def __init__(self, turns: list[ToolTurn], jsons: list[dict]) -> None:
        self.turns, self.jsons = list(turns), list(jsons)
        self.tool_requests: list[list[dict]] = []
        self.json_requests: list[list[dict]] = []

    def chat_tools(self, messages, tools, max_tokens):
        self.tool_requests.append(messages)
        return self.turns.pop(0) if self.turns else ToolTurn("证据已足够")

    def chat_json(self, messages, max_tokens):
        self.json_requests.append(messages)
        data = self.jsons.pop(0)
        return LLMResult(json.dumps(data, ensure_ascii=False), data, Usage(10, 5), 1.0)


def _tool_turn(*calls: tuple[str, dict]) -> ToolTurn:
    return ToolTurn("", [ToolCall(f"c{i}", n, json.dumps(a, ensure_ascii=False)) for i, (n, a) in enumerate(calls)])


def _deps(llm, rows=None, max_steps=4, audit=None) -> AgentDeps:
    chunks = [_chunk(0, "营业收入 1,000 元", 3), _chunk(1, "营业成本 400 元", 4)]
    page_texts = {"000001_2024": ["p"] * 2 + ["营业收入1,000元", "营业成本400元"] + ["p"] * 8,
                  "600519_2025": ["p"] * 5 + ["基本每股收益68.64"] + ["p"] * 5}  # fmt: skip
    return AgentDeps(
        llm=llm, retrieve=lambda q, k: chunks[:k], tools=_toolbox(rows=rows),
        page_texts=lambda d: page_texts[d], cfg=AgentSettings(max_steps=max_steps), context_k=2,
        answer_max_tokens=100, audit=audit,
    )  # fmt: skip


def test_graph_tool_loop_then_answer() -> None:
    audit = []
    llm = ScriptedLLM(
        turns=[
            _tool_turn(("get_financial_metric", {"company": "茅台", "year": 2024, "metric": "eps_basic"}),
                       ("calculator", {"expression": "1000-400"})),
            ToolTurn("证据已足够"),
        ],
        jsons=[{"refused": False, "answer": "毛利600元[1][2]，每股收益68.64元[3]",
                "citations": [{"ref": 1, "quote": "营业收入 1,000 元"}, {"ref": 3, "quote": "68.64"},
                              {"ref": 9, "quote": "编造"}]}],
    )  # fmt: skip
    graph = build_graph(_deps(llm, rows=[_row(2025, "68.64", "68.64", 6)], audit=audit.append))
    with trace_context("trace-x"):
        st = run_turn(graph, "问题", max_steps=4)
    assert audit[0]["trace_id"] == "trace-x"  # trace_id 能传进 LangGraph 的节点

    assert st["steps"] == 2 and [e["id"] for e in st["evidence"]] == [1, 2, 3]
    assert st["evidence"][2]["source"] == "get_financial_metric"
    assert [t["tool"] for t in st["tool_log"]] == ["get_financial_metric", "calculator"] and len(audit) == 2
    # 第二次取证请求带上了 assistant 的工具调用和两条工具结果（编号对应）
    msgs = llm.tool_requests[1]
    assert msgs[-3]["role"] == "assistant" and [m["tool_call_id"] for m in msgs[-2:]] == ["c0", "c1"]
    # 作答阶段看到全部 3 条证据和计算结果
    user = llm.json_requests[0][1]["content"]
    assert '<evidence id="3"' in user and "<calculations>\n1000-400 = 600\n</calculations>" in user
    checks = {c["ref"]: c for c in st["citations"]}
    assert checks[1]["verified"] and checks[1]["page"] == 3
    assert checks[3]["verified"] and checks[3]["doc_id"] == "600519_2025" and checks[3]["page"] == 6
    assert not checks[9]["valid_ref"]
    assert answer_from_state(st).answer.startswith("毛利600元")
    assert [c["node"] for c in st["llm_calls"]] == ["agent", "agent", "answer"]


def test_graph_stops_at_max_steps() -> None:
    loop = [_tool_turn(("calculator", {"expression": "1+1"})) for _ in range(10)]
    llm = ScriptedLLM(turns=loop, jsons=[{"refused": True, "answer": "证据不足", "citations": []}])
    st = run_turn(build_graph(_deps(llm, max_steps=2)), "问题", max_steps=2)
    assert st["steps"] == 2 and len(st["tool_log"]) == 2 and st["answer"]["refused"]


def test_graph_multi_turn_rewrite_with_checkpointer() -> None:
    llm = ScriptedLLM(
        turns=[],
        jsons=[
            {"refused": False, "answer": "茅台营业收入1,000元[1]", "citations": [{"ref": 1, "quote": "1,000"}]},
            {"question": "五粮液2024年营业收入是多少？"},  # 第二轮：先改写
            {"refused": True, "answer": "证据不足", "citations": []},
        ],
    )
    graph = build_graph(_deps(llm), checkpointer=InMemorySaver())
    first = run_turn(graph, "茅台2024年营业收入是多少？", max_steps=4, thread_id="t1")
    assert first["standalone"] == "茅台2024年营业收入是多少？" and len(llm.json_requests) == 1  # 首轮不改写

    second = run_turn(graph, "那五粮液呢？", max_steps=4, thread_id="t1")
    assert second["standalone"] == "五粮液2024年营业收入是多少？"
    assert "茅台营业收入1,000元[1]" in llm.json_requests[1][1]["content"]  # 改写时看到了上一轮的回答
    assert [h["question"] for h in second["history"]] == ["茅台2024年营业收入是多少？", "五粮液2024年营业收入是多少？"]
    # 每轮字段被重置：第二轮的证据编号从 1 重新开始，没有残留上一轮的引用校验
    assert [e["id"] for e in second["evidence"]] == [1, 2] and second["citations"] == []
    # 不同 thread 互不影响：t2 没有任何保存的状态
    assert graph.get_state({"configurable": {"thread_id": "t2"}}).values == {}


# ---------------- 锁的等待 / 持有时间日志（D-032） ----------------


def test_locked_logs_wait_and_hold(caplog):
    """两个线程同时调用：后拿到锁的一方 wait_ms 约等于前一方的执行时间；两次的 hold_ms 都约等于执行时间。"""
    started = threading.Event()

    def slow(x):
        started.set()
        time.sleep(0.1)
        return x * 2

    f = locked(slow, "search")
    caplog.set_level("INFO", logger="agent.graph")
    out = []
    t1 = threading.Thread(target=lambda: out.append(f(1)))
    t1.start()
    started.wait()  # 确保第一个线程已经拿到锁、正在执行
    out.append(f(2))
    t1.join()

    assert sorted(out) == [2, 4]
    recs = [r for r in caplog.records if getattr(r, "stage", None) == "lock"]
    assert len(recs) == 2 and all(r.lock == "search" for r in recs)
    assert all(r.hold_ms >= 95 for r in recs)
    first, second = recs  # 先释放的是第一个线程
    assert first.wait_ms < 20
    assert second.wait_ms >= 50  # 第二次调用至少排队到第一个线程执行结束


def test_locked_logs_even_on_error(caplog):
    def boom():
        raise ValueError("x")

    caplog.set_level("INFO", logger="agent.graph")
    with pytest.raises(ValueError):
        locked(boom, "page_texts")()
    assert [r.lock for r in caplog.records if getattr(r, "stage", None) == "lock"] == ["page_texts"]
