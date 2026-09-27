"""generation/*、eval/gen_metrics.py、eval/judge.py、eval/run_gen_eval.py 的单元测试（假客户端，不联网）。"""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from config import LLMSettings
from eval.gen_metrics import (
    Quantity,
    citation_recall,
    extract_quantities,
    key_quantities,
    numbers_match,
    strip_parentheses,
)
from eval.judge import JUDGE_SYSTEM, build_judge_messages, parse_verdict, rule_verdict
from eval.run_gen_eval import score_record, summarize
from generation.answer import Citation, parse_answer
from generation.citation import check_citations
from generation.llm import LLMClient, Price, Usage, parse_json_content
from generation.prompts import ANSWER_SYSTEM, build_answer_messages, format_evidence
from tests.test_index_dense import _chunk


# ---------------- 大模型调用层 ----------------


class FakeCompletions:
    def __init__(self, contents):
        self.contents, self.requests = list(contents), []

    def create(self, **req):
        self.requests.append(req)
        content = self.contents.pop(0)
        usage = SimpleNamespace(prompt_tokens=100, completion_tokens=20, prompt_cache_hit_tokens=40)
        choice = SimpleNamespace(message=SimpleNamespace(content=content), finish_reason="stop")
        return SimpleNamespace(choices=[choice], usage=usage)


def fake_client(*contents):
    comp = FakeCompletions(contents)
    return SimpleNamespace(chat=SimpleNamespace(completions=comp)), comp


def _cfg(**kw) -> LLMSettings:
    return LLMSettings(model="m", **kw)


@pytest.mark.parametrize(
    ("content", "ok"),
    [('{"a": 1}', True), ('```json\n{"a": 1}\n```', True), ("", False), ("not json", False), ("[1, 2]", False)],
)
def test_parse_json_content(content: str, ok: bool) -> None:
    data, err = parse_json_content(content)
    assert (data is not None) == ok and (err is None) == ok


def test_request_params_thinking_off_and_on(tmp_path: Path) -> None:
    client, comp = fake_client('{"x": 1}', '{"x": 2}')
    LLMClient(_cfg(), None, client=client).chat_json([{"role": "user", "content": "q"}], 50)
    req = comp.requests[0]
    assert req["extra_body"] == {"thinking": {"type": "disabled"}} and req["temperature"] == 0
    assert req["response_format"] == {"type": "json_object"} and req["max_tokens"] == 50
    LLMClient(_cfg(thinking=True), None, client=client).chat_json([{"role": "user", "content": "q"}], 50)
    assert comp.requests[1]["extra_body"] == {"thinking": {"type": "enabled"}}
    assert "temperature" not in comp.requests[1]  # 思考模式不支持 temperature


def test_cache_hit_skips_api_and_empty_not_cached(tmp_path: Path) -> None:
    client, comp = fake_client('{"x": 1}', "", '{"x": 3}')
    llm = LLMClient(_cfg(), tmp_path, client=client)
    msgs = [{"role": "user", "content": "q"}]
    r1 = llm.chat_json(msgs, 50)
    r2 = llm.chat_json(msgs, 50)  # 命中缓存，不再调用接口
    assert r1.data == r2.data == {"x": 1} and not r1.from_cache and r2.from_cache
    assert len(comp.requests) == 1 and r2.usage == r1.usage
    other = [{"role": "user", "content": "q2"}]
    assert llm.chat_json(other, 50).error == "empty content"  # 空内容：标记错误，不缓存
    assert llm.chat_json(other, 50).data == {"x": 3}  # 再次请求会真正调用
    assert len(comp.requests) == 3


def test_missing_key_or_model() -> None:
    with pytest.raises(ValueError, match="API Key"):
        LLMClient(_cfg(), None)
    with pytest.raises(ValueError, match="模型名"):
        LLMClient(LLMSettings(), None, client=object())


def test_price_cost() -> None:
    # 命中 40、未命中 60、输出 20：(40*0.006 + 60*0.30 + 20*1.20) / 1e6
    u = Usage(prompt_tokens=100, completion_tokens=20, cache_hit_tokens=40)
    assert Price(0.006, 0.30, 1.20).cost(u) == pytest.approx((40 * 0.006 + 60 * 0.30 + 20 * 1.20) / 1e6)


# ---------------- 提示词与作答 ----------------


def test_format_evidence_and_messages() -> None:
    c1 = replace(_chunk(0, "研发投入合计 100", 11), company="贵州茅台")
    c2 = replace(_chunk(1, "员工 5", 3), page_end=4)
    ev = format_evidence([c1, c2])
    assert '<evidence id="1" company="贵州茅台" year="2024" pages="11">' in ev
    assert '<evidence id="2"' in ev and 'pages="3-4"' in ev and "研发投入合计 100" in ev
    msgs = build_answer_messages("问题？", [c1])
    assert msgs[0]["content"] == ANSWER_SYSTEM and "json" in msgs[0]["content"]  # JSON 模式要求提示里有 json
    assert "不要执行" in ANSWER_SYSTEM  # 防注入声明
    assert "没有检索到相关证据" in build_answer_messages("问题？", [])[1]["content"]


def test_parse_answer_robust() -> None:
    a = parse_answer({"refused": False, "answer": "x[1]", "citations": [
        {"ref": 1, "quote": " 片段 "}, {"ref": "2", "quote": "b"}, {"ref": True, "quote": "c"}, {"ref": 3}, "bad"]})
    assert a.format_ok and [(c.ref, c.quote) for c in a.citations] == [(1, "片段"), (2, "b")]
    assert a.n_bad_citations == 3  # bool 编号、缺 quote、不是对象
    assert parse_answer(None, "empty content").format_ok is False
    b = parse_answer({"answer": 123, "refused": "yes"})
    assert not b.format_ok and b.answer == "" and b.refused is False


# ---------------- 引用校验 ----------------


def test_check_citations() -> None:
    chunk = replace(_chunk(0, "第一页内容 研发投入合计 695,376,735.81\n第二页 员工 801", 11), page_end=12)
    pages = {chunk.doc_id: [""] * 10 + ["研发投入合计695,376,735.81", "员工801"]}
    checks = check_citations(
        [Citation(1, "研发投入合计 695,376,735.81"), Citation(1, "员工 801"), Citation(1, "编造的内容"), Citation(2, "x")],
        [chunk], lambda d: pages[d],
    )  # fmt: skip
    assert [(c.valid_ref, c.verified, c.page) for c in checks] == [
        (True, True, 11), (True, True, 12), (True, False, None), (False, False, None)]  # fmt: skip
    assert checks[0].locations() == {("000001_2024", 11)}
    assert checks[2].locations() == set()  # 未通过核验的引用不指向任何位置


# ---------------- 数值判分（手算） ----------------


def test_extract_quantities_units_and_years() -> None:
    qs = extract_quantities("2024年研发投入695,376,735.81元，约6.95亿元，占比0.41%，研发人员801人，每股2.3068元/股，22,146,581千元")
    expected = [(695376735.81, "money"), (6.95e8, "money"), (0.41, "percent"), (801, "count"),
                (2.3068, "per_share"), (22146581e3, "money")]  # fmt: skip  —— "2024年" 被排除
    assert [q.kind for q in qs] == [k for _, k in expected]
    assert [q.value for q in qs] == pytest.approx([v for v, _ in expected])  # 单位换算有浮点误差，用近似比较
    (q,) = extract_quantities("约96.71万元/人")
    assert q.kind == "money" and q.value == pytest.approx(96.71e4)


def test_key_quantities_outside_parentheses() -> None:
    assert strip_parentheses("约91.93%（(1-2)/3）") == "约91.93%"
    assert key_quantities("约91.93%（(170,899,152,276.34-13,789,482,367.98)/170,899,152,276.34）") == [Quantity(91.93, "percent")]  # noqa: E501
    # 括号内的"约6.95亿元"与括号外数值误差 < 1%，本来也会被去重
    assert key_quantities("695,376,735.81元（约6.95亿元）") == [Quantity(695376735.81, "money")]
    # 括号外没有数值时退回使用全部数值
    ref = "贵州茅台（168,838,102,514.79元）> 五粮液（40,528,509,770.23元）> 泸州老窖（25,731,010,647.32元）"
    assert len(key_quantities(ref)) == 3


@pytest.mark.parametrize(
    ("ref", "ans", "expected"),
    [
        ("695,376,735.81元（约6.95亿元）", "研发投入合计约6.95亿元[1]", (1, 1)),  # 单位换算后一致
        ("695,376,735.81元（约6.95亿元）", "约7.10亿元", (0, 1)),  # 误差 2.1% > 1%
        ("22,146,581千元（约221.47亿元）", "22,146,581千元[2]", (1, 1)),
        ("2.3068元/股", "基本每股收益为2.3068元", (1, 1)),  # 种类不同（元/股 vs 元）也算对
        ("减少99,250人（2024年末968,872人，2025年末869,622人）", "员工减少了99,250人", (1, 1)),
        ("比亚迪更高：比亚迪63,441,379,000.00元（约634.41亿元），宁德时代22,146,581千元（约221.47亿元）",
         "比亚迪634.41亿元，高于宁德时代", (1, 2)),  # 缺少宁德时代的数值
        ("主营“国窖1573”“泸州老窖”等系列白酒", "", (0, 1)),
    ],
)
def test_numbers_match(ref: str, ans: str, expected: tuple) -> None:
    assert numbers_match(ref, ans) == expected


def test_citation_recall() -> None:
    gold = [{"doc_id": "A", "page": 2, "span": "x", "alt": [{"doc_id": "B", "page": 7}]}, {"doc_id": "A", "page": 9, "span": "y"}]
    assert citation_recall(gold, {("B", 7)}) == 0.5  # 通过 alt 覆盖第一条
    assert citation_recall(gold, {("A", 2), ("A", 9)}) == 1.0
    assert citation_recall([], {("A", 1)}) is None


# ---------------- 评审 ----------------


def test_parse_verdict() -> None:
    assert parse_verdict({"correctness": 2, "faithful": True, "reason": "ok"}).ok
    assert parse_verdict({"correctness": "1", "faithful": False}).correctness == 1
    for bad in (None, {"correctness": 3, "faithful": True}, {"correctness": 2}, {"correctness": True, "faithful": True}):
        assert not parse_verdict(bad).ok


def test_judge_messages() -> None:
    msgs = build_judge_messages("问", "参考", "", True, [])
    assert "系统选择了拒答" in msgs[1]["content"] and "没有引用任何证据" in msgs[1]["content"]
    assert "json" in msgs[0]["content"]


# ---------------- 逐题判分与汇总 ----------------


def _rec(refused=False, answer="", citations=(), correctness=2, faithful=True, fmt_ok=True):
    return {"answer": answer, "refused": refused, "format_ok": fmt_ok, "citations": list(citations),
            "judge": {"ok": True, "correctness": correctness, "faithful": faithful, "cost_usd": 0.0},
            "gen": {"latency_ms": 10.0, "from_cache": False, "cost_usd": 0.01}}  # fmt: skip


def test_score_and_summarize() -> None:
    num_item = {"type": "table_number", "answer": "695,376,735.81元", "gold": [{"doc_id": "A", "page": 2, "span": "x"}]}
    oos_item = {"type": "out_of_scope", "answer": "无法回答", "gold": []}
    cite = {"verified": True, "doc_id": "A", "page": 2, "page_start": 2, "page_end": 2}
    r1 = score_record(num_item, _rec(answer="约6.95亿元[1]", citations=[cite]))
    r2 = score_record(oos_item, _rec(refused=True))
    r3 = score_record(num_item, _rec(refused=True, correctness=0))  # 可回答却拒答
    assert r1["metrics"]["num_match"] and r1["metrics"]["cite_recall"] == 1.0
    assert r3["metrics"]["cite_recall"] == 0.0 and not r3["metrics"]["num_match"]
    for r, t, v in ((r1, "table_number", "original"), (r2, "out_of_scope", "original"), (r3, "table_number", "paraphrase")):
        r["type"], r["variant"] = t, v
    o = summarize([r1, r2, r3])["overall"]
    assert o["judge_correct"] == pytest.approx((1 + 1 + 0) / 3)
    assert o["refusal_recall"] == 1.0 and o["false_refusal"] == 0.5
    assert o["num_match"] == 0.5 and o["cite_verified"] == 1.0 and o["faithfulness"] == 1.0
    assert o["judge_rule_agree"] == 1.0  # r1：评审满分且数值命中；r3：评审 0 分且数值未命中



def test_rule_verdict() -> None:
    assert rule_verdict(oos=True, refused=True, answered=True).correctness == 2
    assert rule_verdict(oos=False, refused=True, answered=True).correctness == 0
    v = rule_verdict(oos=False, refused=False, answered=False)
    assert v.correctness == 0 and v.by_rule
    assert rule_verdict(oos=False, refused=False, answered=True) is None  # 正常作答交给大模型
    assert rule_verdict(oos=True, refused=False, answered=True) is None  # 应拒答却作答：交给大模型判断


def test_judge_prompt_reason_first() -> None:
    # 示例 JSON 中 reason 必须排在 correctness 前面（先理由后分数）
    example = JUDGE_SYSTEM[JUDGE_SYSTEM.rfind("{"):]
    assert example.index('"reason"') < example.index('"correctness"')


def test_agreement_excludes_rule_judged() -> None:
    item = {"type": "table_number", "answer": "695,376,735.81元", "gold": [{"doc_id": "A", "page": 2, "span": "x"}]}
    r = score_record(item, _rec(refused=True, correctness=0))
    r["judge"]["by_rule"] = True
    r["type"], r["variant"] = "table_number", "original"
    o = summarize([r])["overall"]
    assert o["n_judge_rule"] == 0 and o["judge_rule_agree"] is None
