"""结构化日志：trace_id 贯穿一次请求，各阶段耗时自动记录。

只用标准库 logging + contextvars，不引入额外依赖。

核心概念：
    - trace_id：一次问答 / 一次评测运行的唯一标识，同一请求内的所有日志都带上它，
      排障时按 trace_id 过滤即可串起完整链路。
    - contextvars：trace_id 存在"上下文变量"里，而不是全局变量。
      asyncio 的每个任务有独立上下文，所以 FastAPI 并发处理多个请求时 trace_id 不会串。

用法：
    setup_logging(level="INFO", json_format=True)       # 进程启动时调用一次
    logger = logging.getLogger(__name__)

    with trace_context():                               # 进入一次请求，自动生成 trace_id
        with timed("retrieve", logger, top_k=50) as info:
            hits = ...
            info["n_hits"] = len(hits)                  # 可在块内追加字段，退出时一起输出

输出示例（JSON 模式，一行一条）：
    {"ts": "2026-09-24T09:10:00.123+08:00", "level": "INFO", "logger": "retrieval.search",
     "msg": "retrieve ok", "trace_id": "3f2a9c1e7b6d4a05", "stage": "retrieve",
     "status": "ok", "elapsed_ms": 42.17, "top_k": 50, "n_hits": 50}
"""

from __future__ import annotations

import json
import logging
import sys
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime
from typing import Any, TextIO

# 当前上下文中的 trace_id；不在任何请求中时为 None
_trace_id_var: ContextVar[str | None] = ContextVar("trace_id", default=None)

# LogRecord 自带的标准属性名。
# 通过 logger.info(..., extra={...}) 传入的自定义字段会被挂到 record 上，
# 格式化时用"record 的全部属性 - 标准属性"就能找出这些自定义字段。
_STANDARD_ATTRS = set(logging.makeLogRecord({}).__dict__) | {"message", "asctime", "trace_id"}

# 标记本模块添加的 handler，重复调用 setup_logging 时只替换它，不影响其他库的 handler
_HANDLER_NAME = "report_copilot"


# ---------------------------------------------------------------------------
# trace_id 管理
# ---------------------------------------------------------------------------


def new_trace_id() -> str:
    """生成一个新的 trace_id（16 位十六进制，足够唯一且便于肉眼比对）。"""
    return uuid.uuid4().hex[:16]


def get_trace_id() -> str | None:
    """读取当前上下文的 trace_id。"""
    return _trace_id_var.get()


@contextmanager
def trace_context(trace_id: str | None = None) -> Iterator[str]:
    """在 with 块内绑定 trace_id；退出时恢复为进入前的值。

    Args:
        trace_id: 指定的 trace_id（例如从请求头 X-Trace-Id 透传过来）；为 None 时自动生成。
    """
    tid = trace_id or new_trace_id()
    token = _trace_id_var.set(tid)
    try:
        yield tid
    finally:
        # 用 token 精确恢复，支持嵌套使用
        _trace_id_var.reset(token)


# ---------------------------------------------------------------------------
# 格式化
# ---------------------------------------------------------------------------


class _TraceIdFilter(logging.Filter):
    """在每条日志产生时，把当前上下文的 trace_id 写入 record。"""

    def filter(self, record: logging.LogRecord) -> bool:
        record.trace_id = get_trace_id()
        return True  # 只附加字段，不过滤任何日志


def _extra_fields(record: logging.LogRecord) -> dict[str, Any]:
    """取出通过 extra= 传入的自定义字段。"""
    return {k: v for k, v in record.__dict__.items() if k not in _STANDARD_ATTRS}


class JsonFormatter(logging.Formatter):
    """每条日志输出为一行 JSON，便于用 jq / ELK / Loki 等工具检索。"""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            # 带时区的 ISO 8601 时间，毫秒精度
            "ts": datetime.fromtimestamp(record.created).astimezone().isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "trace_id": getattr(record, "trace_id", None),
        }
        payload.update(_extra_fields(record))
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        # ensure_ascii=False：中文原样输出；default=str：遇到 Path、datetime 等对象时转成字符串而不报错
        return json.dumps(payload, ensure_ascii=False, default=str)


class TextFormatter(logging.Formatter):
    """人类可读格式，适合本地调试：时间 级别 [trace_id] logger: 消息 key=value ..."""

    def format(self, record: logging.LogRecord) -> str:
        ts = datetime.fromtimestamp(record.created).strftime("%H:%M:%S.%f")[:-3]
        tid = getattr(record, "trace_id", None) or "-"
        extras = " ".join(f"{k}={v}" for k, v in _extra_fields(record).items())
        line = f"{ts} {record.levelname:<7} [{tid}] {record.name}: {record.getMessage()}"
        if extras:
            line += f" | {extras}"
        if record.exc_info:
            line += "\n" + self.formatException(record.exc_info)
        return line


def setup_logging(level: str = "INFO", json_format: bool = True, stream: TextIO | None = None) -> None:
    """配置根 logger。进程启动时调用一次；重复调用会替换掉上一次的配置。

    Args:
        level: 日志级别，如 "INFO"、"DEBUG"。
        json_format: True 输出 JSON，False 输出可读文本。
        stream: 输出目标，默认 stderr（stdout 留给程序正常输出）；测试时可传入 StringIO。
    """
    root = logging.getLogger()
    # 移除之前由本函数添加的 handler，保证幂等
    for h in list(root.handlers):
        if h.get_name() == _HANDLER_NAME:
            root.removeHandler(h)

    handler = logging.StreamHandler(stream or sys.stderr)
    handler.set_name(_HANDLER_NAME)
    handler.addFilter(_TraceIdFilter())
    handler.setFormatter(JsonFormatter() if json_format else TextFormatter())

    root.addHandler(handler)
    root.setLevel(level.upper())


# ---------------------------------------------------------------------------
# 阶段计时
# ---------------------------------------------------------------------------


@contextmanager
def timed(stage: str, logger: logging.Logger, **fields: Any) -> Iterator[dict[str, Any]]:
    """记录一个处理阶段的耗时，退出 with 块时输出一条日志。

    - 正常结束：INFO 级别，status="ok"
    - 抛出异常：ERROR 级别，status="error"，附带异常类型和信息；异常会继续向上抛出，不会被吞掉

    Args:
        stage: 阶段名，如 "parse"、"retrieve"、"rerank"、"generate"。
        logger: 用于输出的 logger。
        **fields: 附加到日志中的字段。注意不要与 LogRecord 标准属性重名（如 name、msg），
            否则 logging 会抛 KeyError。

    Yields:
        一个字典，可在 with 块内继续写入字段（例如结果数量），退出时一并输出。
    """
    info: dict[str, Any] = {"stage": stage, **fields}
    status = "error"  # 先假设失败，只有 with 块正常走完才改成 ok
    # perf_counter 是单调时钟，不受系统时间调整影响，适合测量耗时
    start = time.perf_counter()
    try:
        yield info
        status = "ok"
    except BaseException as exc:
        # 用 BaseException 而不是 Exception：asyncio 的 CancelledError、KeyboardInterrupt
        # 不属于 Exception，也要记录下来，方便排查"请求被取消"之类的问题
        info["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        info["status"] = status
        info["elapsed_ms"] = round((time.perf_counter() - start) * 1000, 2)
        level = logging.INFO if status == "ok" else logging.ERROR
        logger.log(level, f"{stage} {status}", extra=info)
