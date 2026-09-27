"""作答：检索结果 -> 提示词 -> 大模型 -> 结构化答案。

模型输出可能不守规矩（字段缺失、类型不对、JSON 损坏、返回空内容），这里全部容错：
    - JSON 无法解析：format_ok=False，answer 为空，按"未作答"计分（既不算正确，也不算拒答）
    - refused 缺失：按 False 处理
    - citations 中 ref 不是整数、quote 不是字符串的条目：丢弃，并计入 n_bad_citations
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field

from generation.llm import LLMClient, LLMResult
from generation.prompts import build_answer_messages
from ingest.chunk import Chunk

logger = logging.getLogger("generation.answer")


@dataclass
class Citation:
    ref: int  # 证据编号，从 1 开始，对应上下文中第 ref 个 chunk
    quote: str  # 模型声称从该证据中逐字摘录的片段


@dataclass
class Answer:
    answer: str
    refused: bool
    citations: list[Citation] = field(default_factory=list)
    format_ok: bool = True  # 模型输出能否解析为符合约定的 JSON
    n_bad_citations: int = 0  # 结构不合法、被丢弃的引用条数
    error: str | None = None


def parse_answer(data: dict | None, error: str | None = None) -> Answer:
    """把模型输出的 JSON 对象整理成 Answer。data 为 None 表示解析失败。"""
    if data is None:
        return Answer(answer="", refused=False, format_ok=False, error=error or "no json")
    answer = data.get("answer")
    refused = data.get("refused", False)
    raw_cites = data.get("citations") or []
    ok = isinstance(answer, str) and isinstance(refused, bool) and isinstance(raw_cites, list)

    cites: list[Citation] = []
    bad = 0
    for c in raw_cites if isinstance(raw_cites, list) else []:
        ref = c.get("ref") if isinstance(c, dict) else None
        quote = c.get("quote") if isinstance(c, dict) else None
        # bool 是 int 的子类，要单独排除；模型偶尔会把编号写成字符串 "2"，也接受
        if isinstance(ref, str) and ref.strip().isdigit():
            ref = int(ref.strip())
        if isinstance(ref, int) and not isinstance(ref, bool) and isinstance(quote, str) and quote.strip():
            cites.append(Citation(ref=ref, quote=quote.strip()))
        else:
            bad += 1
    return Answer(
        answer=answer if isinstance(answer, str) else "",
        refused=refused if isinstance(refused, bool) else False,
        citations=cites,
        format_ok=ok,
        n_bad_citations=bad,
        error=error,
    )


def generate_answer(question: str, chunks: Sequence[Chunk], llm: LLMClient, max_tokens: int) -> tuple[Answer, LLMResult]:
    """对一道题作答，返回 (结构化答案, 原始调用结果)。"""
    result = llm.chat_json(build_answer_messages(question, chunks), max_tokens=max_tokens)
    ans = parse_answer(result.data, result.error)
    if not ans.format_ok:
        logger.warning("answer format error", extra={"error": ans.error})
    return ans, result
