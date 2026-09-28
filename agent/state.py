"""Agent 的状态定义。

状态会被 checkpointer 序列化存进 Postgres（多轮对话靠它记住上文），所以只放 JSON 基本类型
（dict / list / str / int / float / bool），不放 Decimal、dataclass 等对象，避免反序列化的兼容和安全问题。

字段分两类：
    跨轮保留：history（每轮结束时追加一条"问题 + 回答"，用 operator.add 作为合并函数）
    每轮重置：其余字段，由每轮的第一个节点 rewrite 清空，避免上一轮的证据、工具记录混进本轮
"""

from __future__ import annotations

import operator
from dataclasses import asdict, dataclass
from typing import Annotated, TypedDict


@dataclass
class Evidence:
    """证据登记表中的一条。字段名与 ingest.chunk.Chunk 保持一致，
    因此可以直接交给 generation.prompts.format_evidence 和 generation.citation.check_citations 使用。"""

    id: int  # 证据编号，从 1 开始连续递增；最终答案用 [id] 引用
    source: str  # retrieve（初始检索）/ search_reports / get_financial_metric
    doc_id: str
    company: str
    year: int  # 年报年度
    page_start: int
    page_end: int
    text: str
    chunk_id: str | None = None  # 来自检索时的 chunk_id；来自财务指标表时为 None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Evidence:
        return cls(**d)


class AgentState(TypedDict, total=False):
    # ---- 跨轮保留 ----
    history: Annotated[list[dict], operator.add]  # [{"question": 完整问题, "answer": 回答文本}]

    # ---- 每轮重置 ----
    question: str  # 本轮用户原话
    standalone: str  # 结合上文改写后的完整问题；单轮对话或没有上文时等于 question
    evidence: list[dict]  # 证据登记表（Evidence.to_dict()）
    messages: list[dict]  # 取证阶段与模型的对话（OpenAI 消息格式，含工具调用和工具结果）
    steps: int  # 取证阶段已调用模型的次数
    calc_log: list[dict]  # 计算器调用记录 {"expression", "result" | "error"}
    tool_log: list[dict]  # 全部工具调用记录 {"tool", "args", "ok", "error", "n_results", "latency_ms"}
    llm_calls: list[dict]  # 本轮每次模型调用的用量 {"node", "usage", "latency_ms", "from_cache"}
    answer: dict  # generation.answer.Answer 的字典形式
    citations: list[dict]  # generation.citation.CitationCheck 的字典形式
