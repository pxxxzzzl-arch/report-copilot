"""common/log.py 的单元测试。"""

import asyncio
import io
import json
import logging
import time
from collections.abc import Iterator

import pytest

from common.log import _HANDLER_NAME, get_trace_id, setup_logging, timed, trace_context

logger = logging.getLogger("tests.log")


def _remove_our_handler() -> None:
    root = logging.getLogger()
    for h in list(root.handlers):
        if h.get_name() == _HANDLER_NAME:
            root.removeHandler(h)


@pytest.fixture
def json_stream() -> Iterator[io.StringIO]:
    """把日志以 JSON 格式输出到内存，测试结束后移除 handler。"""
    stream = io.StringIO()
    setup_logging(level="DEBUG", json_format=True, stream=stream)
    yield stream
    _remove_our_handler()


def _records(stream: io.StringIO) -> list[dict]:
    """把输出按行解析成 dict；任何一行不是合法 JSON 都会直接报错。

    只保留本测试 logger 的记录，过滤掉 asyncio 等库自己打的 DEBUG 日志。
    """
    recs = [json.loads(line) for line in stream.getvalue().splitlines()]
    return [r for r in recs if r["logger"] == logger.name]


# ---------------- trace_id ----------------


def test_log_line_is_json_with_trace_id_and_extra(json_stream: io.StringIO) -> None:
    with trace_context("abc123") as tid:
        assert tid == "abc123"
        logger.info("hello", extra={"doc_id": "600000_2024"})
    (rec,) = _records(json_stream)
    assert rec["msg"] == "hello"
    assert rec["trace_id"] == "abc123"
    assert rec["doc_id"] == "600000_2024"
    assert rec["level"] == "INFO"
    assert rec["logger"] == "tests.log"


def test_trace_id_generated_and_restored() -> None:
    assert get_trace_id() is None
    with trace_context() as outer:
        assert len(outer) == 16
        with trace_context("inner"):
            assert get_trace_id() == "inner"
        assert get_trace_id() == outer  # 嵌套退出后恢复外层
    assert get_trace_id() is None


def test_trace_id_isolated_between_async_tasks(json_stream: io.StringIO) -> None:
    """两个并发任务交替执行，各自的日志必须带各自的 trace_id。"""

    async def handle(name: str) -> None:
        with trace_context(f"tid-{name}"):
            for _ in range(3):
                await asyncio.sleep(0)  # 主动让出控制权，制造交替执行
                logger.info(name)

    async def main() -> None:
        await asyncio.gather(handle("a"), handle("b"))

    asyncio.run(main())
    recs = _records(json_stream)
    assert len(recs) == 6
    for rec in recs:
        assert rec["trace_id"] == f"tid-{rec['msg']}"


def test_chinese_not_escaped(json_stream: io.StringIO) -> None:
    logger.info("研发投入合计")
    assert "研发投入合计" in json_stream.getvalue()


# ---------------- timed ----------------


def test_timed_ok_records_fields(json_stream: io.StringIO) -> None:
    with trace_context("t1"), timed("retrieve", logger, top_k=5) as info:
        time.sleep(0.02)
        info["n_hits"] = 3  # 块内追加的字段
    (rec,) = _records(json_stream)
    assert rec["stage"] == "retrieve"
    assert rec["status"] == "ok"
    assert rec["level"] == "INFO"
    assert rec["trace_id"] == "t1"
    assert rec["top_k"] == 5
    assert rec["n_hits"] == 3
    assert rec["elapsed_ms"] >= 20  # 至少包含 sleep 的 20ms


def test_timed_error_logged_and_reraised(json_stream: io.StringIO) -> None:
    with pytest.raises(ValueError, match="boom"):
        with timed("parse", logger):
            raise ValueError("boom")
    (rec,) = _records(json_stream)
    assert rec["status"] == "error"
    assert rec["level"] == "ERROR"
    assert rec["error"] == "ValueError: boom"
    assert "elapsed_ms" in rec


def test_timed_records_cancellation(json_stream: io.StringIO) -> None:
    """CancelledError 不是 Exception 的子类，也要被记录为 error。"""

    async def main() -> None:
        async def slow() -> None:
            with timed("generate", logger):
                await asyncio.sleep(10)

        task = asyncio.create_task(slow())
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(main())
    (rec,) = _records(json_stream)
    assert rec["status"] == "error"
    assert rec["error"].startswith("CancelledError")


# ---------------- 配置 ----------------


def test_text_format() -> None:
    stream = io.StringIO()
    setup_logging(level="INFO", json_format=False, stream=stream)
    try:
        with trace_context("txt"), timed("rerank", logger, top_n=5):
            pass
    finally:
        _remove_our_handler()
    line = stream.getvalue()
    assert "[txt]" in line
    assert "rerank ok" in line
    assert "top_n=5" in line


def test_setup_logging_idempotent() -> None:
    try:
        setup_logging(stream=io.StringIO())
        setup_logging(stream=io.StringIO())
        ours = [h for h in logging.getLogger().handlers if h.get_name() == _HANDLER_NAME]
        assert len(ours) == 1
    finally:
        _remove_our_handler()


def test_level_filters_debug() -> None:
    stream = io.StringIO()
    setup_logging(level="INFO", stream=stream)
    try:
        logger.debug("hidden")
        logger.info("shown")
    finally:
        _remove_our_handler()
    assert "hidden" not in stream.getvalue()
    assert "shown" in stream.getvalue()
