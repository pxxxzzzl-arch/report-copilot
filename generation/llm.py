"""大模型调用层：OpenAI 兼容接口（DeepSeek、vLLM 等）的一层薄封装。

负责：
    - chat_json：JSON 输出（response_format=json_object）与解析；解析失败或返回空内容时不抛异常，而是在结果里标记
      （DeepSeek 文档提示 JSON 模式偶尔会返回空内容）
    - chat_tools：工具调用（OpenAI 格式的 tools / tool_calls，阶段 7 Agent 使用）
    - 关闭 / 开启思考模式（DeepSeek 通过 extra_body={"thinking": {"type": ...}} 控制）
    - 超时与重试（交给 openai SDK：网络错误、429、5xx 按指数退避重试 max_retries 次）
    - 记录 token 用量、费用估算和耗时（结构化日志，带 trace_id）
    - 本地响应缓存：key = sha256(模型 + 完整请求参数)。重跑评测时命中缓存，不重复花钱，结果也完全可复现；
      改了提示词、模型或参数，key 自然不同，不会误用旧结果

不负责：提示词内容（generation/prompts.py）和业务逻辑（generation/answer.py、eval/judge.py）。
安全：API Key 只从配置（环境变量 / .env）读取，不写入日志、缓存或实验明细。
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from config import PROJECT_ROOT, LLMSettings

logger = logging.getLogger("generation.llm")


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cache_hit_tokens: int = 0  # 服务端上下文缓存命中的输入 token（DeepSeek 返回 prompt_cache_hit_tokens）

    @property
    def cache_miss_tokens(self) -> int:
        return self.prompt_tokens - self.cache_hit_tokens


@dataclass
class LLMResult:
    content: str  # 原始文本输出
    data: dict | None  # 解析出的 JSON；解析失败或内容为空时为 None
    usage: Usage = field(default_factory=Usage)
    latency_ms: float = 0.0
    from_cache: bool = False  # 命中本地缓存时为 True（此时 usage 是首次调用时记录的用量）
    error: str | None = None  # 解析失败、返回空内容等问题的说明

    @property
    def ok(self) -> bool:
        return self.data is not None


@dataclass
class ToolCall:
    id: str  # 模型生成的调用编号，回传工具结果时必须带上
    name: str
    arguments: str  # 模型输出的参数 JSON 字符串（可能不合法，由调用方解析和容错）


@dataclass
class ToolTurn:
    """一次工具调用模式的对话结果：模型要么调用工具（tool_calls 非空），要么直接回复文字。"""

    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str | None = None
    usage: Usage = field(default_factory=Usage)
    latency_ms: float = 0.0
    from_cache: bool = False

    def assistant_message(self) -> dict:
        """回传给模型的 assistant 消息（OpenAI 格式）：下一轮请求必须带上它，工具结果才能对上调用编号。"""
        msg: dict[str, Any] = {"role": "assistant", "content": self.content or None}
        if self.tool_calls:
            msg["tool_calls"] = [
                {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": c.arguments}}
                for c in self.tool_calls
            ]
        return msg


@dataclass(frozen=True)
class Price:
    """美元 / 百万 token。"""

    input_cache_hit: float
    input_cache_miss: float
    output: float

    def cost(self, u: Usage) -> float:
        return (
            u.cache_hit_tokens * self.input_cache_hit
            + u.cache_miss_tokens * self.input_cache_miss
            + u.completion_tokens * self.output
        ) / 1_000_000


def parse_json_content(content: str) -> tuple[dict | None, str | None]:
    """把模型输出解析成 JSON 对象，返回 (对象, 错误说明)。

    兼容模型偶尔在 JSON 外面包一层 ```json 代码块的情况；顶层不是对象也视为失败。
    """
    text = (content or "").strip()
    if not text:
        return None, "empty content"
    if text.startswith("```"):
        text = text.strip("`")
        text = text[text.find("{") :] if "{" in text else text
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as e:
        return None, f"json decode error: {e.msg}"
    if not isinstance(obj, dict):
        return None, "top-level json is not an object"
    return obj, None


class LLMClient:
    def __init__(self, cfg: LLMSettings, cache_dir: Path | None, model: str | None = None, client: Any = None) -> None:
        """client 仅供测试注入假客户端；正常使用时根据配置创建 openai.OpenAI。"""
        self.cfg = cfg
        self.model = model or cfg.model
        if not self.model:
            raise ValueError("未配置模型名：请在 .env 中设置 LLM__MODEL")
        if client is None:
            if not cfg.api_key.get_secret_value():
                raise ValueError("未配置 API Key：请在 .env 中设置 LLM__API_KEY")
            from openai import OpenAI  # 延迟导入：只有真正调用大模型时才需要 openai 包

            client = OpenAI(
                api_key=cfg.api_key.get_secret_value(),
                base_url=cfg.base_url,
                timeout=cfg.timeout_s,
                max_retries=cfg.max_retries,
            )
        self._client = client
        self.cache_dir = None if cache_dir is None else (cache_dir if cache_dir.is_absolute() else PROJECT_ROOT / cache_dir)

    def _request(self, messages: list[dict], max_tokens: int) -> dict[str, Any]:
        """组装请求参数。思考模式下不发送 temperature（DeepSeek 不支持）。"""
        req: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "response_format": {"type": "json_object"},
            "max_tokens": max_tokens,
            "extra_body": {"thinking": {"type": "enabled" if self.cfg.thinking else "disabled"}},
        }
        if not self.cfg.thinking:
            req["temperature"] = self.cfg.temperature
        return req

    @staticmethod
    def cache_key(req: dict[str, Any]) -> str:
        return hashlib.sha256(json.dumps(req, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()

    def chat_json(self, messages: list[dict], max_tokens: int) -> LLMResult:
        """发送一次对话请求，要求 JSON 输出。网络层重试用尽后仍失败时抛出异常（由调用方决定如何记录）。"""
        req = self._request(messages, max_tokens)
        key = self.cache_key(req)
        cached = self._read_cache(key)
        if cached is not None:
            return cached

        t0 = time.perf_counter()
        resp = self._client.chat.completions.create(**req)
        latency = (time.perf_counter() - t0) * 1000
        content = resp.choices[0].message.content or ""
        u = resp.usage
        usage = Usage(
            prompt_tokens=u.prompt_tokens,
            completion_tokens=u.completion_tokens,
            cache_hit_tokens=getattr(u, "prompt_cache_hit_tokens", 0) or 0,
        )
        data, err = parse_json_content(content)
        if resp.choices[0].finish_reason == "length":
            err = (err or "") + "; output truncated by max_tokens"
        result = LLMResult(content=content, data=data, usage=usage, latency_ms=round(latency, 1), error=err)
        logger.info(
            "llm call",
            extra={
                "model": self.model, "latency_ms": result.latency_ms, "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens, "cache_hit_tokens": usage.cache_hit_tokens,
                "parse_error": err,
            },
        )  # fmt: skip
        self._write_cache(key, result)
        return result

    def chat_tools(self, messages: list[dict], tools: list[dict], max_tokens: int) -> ToolTurn:
        """工具调用模式：模型自行决定调用哪些工具（可并行多个），或者不调用、直接回复。

        与 chat_json 共用缓存目录；请求里多了 tools、没有 response_format，key 自然不同。
        缓存保证重跑时工具调用序列完全一致（包括模型生成的调用编号），Agent 的整条轨迹可复现。
        """
        req: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "tools": tools,
            "max_tokens": max_tokens,
            "extra_body": {"thinking": {"type": "enabled" if self.cfg.thinking else "disabled"}},
        }
        if not self.cfg.thinking:
            req["temperature"] = self.cfg.temperature
        key = self.cache_key(req)
        path = self._cache_path(key)
        if path is not None and path.exists():
            d = json.loads(path.read_text(encoding="utf-8"))
            return ToolTurn(d["content"], [ToolCall(**c) for c in d["tool_calls"]], d["finish_reason"],
                            Usage(**d["usage"]), d["latency_ms"], from_cache=True)  # fmt: skip

        t0 = time.perf_counter()
        resp = self._client.chat.completions.create(**req)
        latency = round((time.perf_counter() - t0) * 1000, 1)
        choice = resp.choices[0]
        calls = [
            ToolCall(id=c.id, name=c.function.name, arguments=c.function.arguments or "")
            for c in (choice.message.tool_calls or [])
        ]
        u = resp.usage
        usage = Usage(u.prompt_tokens, u.completion_tokens, getattr(u, "prompt_cache_hit_tokens", 0) or 0)
        turn = ToolTurn(choice.message.content or "", calls, choice.finish_reason, usage, latency)
        logger.info(
            "llm tool call",
            extra={
                "model": self.model, "latency_ms": latency, "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens, "cache_hit_tokens": usage.cache_hit_tokens,
                "tool_calls": [c.name for c in calls], "finish_reason": choice.finish_reason,
            },
        )  # fmt: skip
        if path is not None and (calls or turn.content):  # 空回复不缓存，下次重试
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {"content": turn.content, "tool_calls": [c.__dict__ for c in calls],
                       "finish_reason": turn.finish_reason, "usage": usage.__dict__, "latency_ms": latency}  # fmt: skip
            path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return turn

    # ---------------- 本地缓存 ----------------

    def _cache_path(self, key: str) -> Path | None:
        return None if self.cache_dir is None else self.cache_dir / key[:2] / f"{key}.json"

    def _read_cache(self, key: str) -> LLMResult | None:
        path = self._cache_path(key)
        if path is None or not path.exists():
            return None
        d = json.loads(path.read_text(encoding="utf-8"))
        data, err = parse_json_content(d["content"])
        return LLMResult(d["content"], data, Usage(**d["usage"]), d["latency_ms"], from_cache=True, error=err)

    def _write_cache(self, key: str, r: LLMResult) -> None:
        path = self._cache_path(key)
        if path is None or not r.content:  # 空内容不缓存，下次重试
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"content": r.content, "usage": r.usage.__dict__, "latency_ms": r.latency_ms}
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
