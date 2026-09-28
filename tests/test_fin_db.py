"""agent/fin_db.py 的集成测试：需要本地 Postgres（deploy/docker-compose.dev.yml），连不上时自动跳过。

每次测试在一个临时 schema 里建表，结束后删除，不影响正式的 fin_metrics 数据。
"""

import uuid
from decimal import Decimal

import pytest

from agent.fin_db import MetricRow, PgFinStore
from config import get_settings

psycopg = pytest.importorskip("psycopg")


@pytest.fixture
def store():
    dsn = get_settings().store.postgres_dsn.get_secret_value()
    try:
        admin = psycopg.connect(dsn, autocommit=True, connect_timeout=2)
    except psycopg.OperationalError:
        pytest.skip("本地 Postgres 不可用")
    schema = f"test_{uuid.uuid4().hex[:8]}"
    admin.execute(f'CREATE SCHEMA "{schema}"')  # schema 名由测试自己生成，不含外部输入
    sep = "&" if "?" in dsn else "?"
    s = PgFinStore(f"{dsn}{sep}options=-csearch_path%3D{schema}", max_size=2)
    s.init_schema()
    yield s
    s.close()
    admin.execute(f'DROP SCHEMA "{schema}" CASCADE')
    admin.close()


def _row(doc: str, ry: int, value: str) -> MetricRow:
    return MetricRow(doc, "600519", "贵州茅台", ry, 2024, "revenue", Decimal(value), "元", value, "元", 6, "营业收入")


def test_replace_and_query(store: PgFinStore) -> None:
    store.replace_doc_metrics("600519_2024", [_row("600519_2024", 2024, "170899152276.34")], "t")
    store.replace_doc_metrics("600519_2025", [_row("600519_2025", 2025, "170899152276.34")], "t")
    rows = store.get_metric("600519", 2024, "revenue")
    assert [r.report_year for r in rows] == [2025, 2024]  # 新年报在前
    assert rows[0].value == Decimal("170899152276.34")  # NUMERIC 精确保存，不丢小数

    # 重新抽取时整体替换：新结果为空，旧数据不残留
    store.replace_doc_metrics("600519_2025", [], "t")
    assert [r.report_year for r in store.get_metric("600519", 2024, "revenue")] == [2024]


def test_query_is_parameterized(store: PgFinStore) -> None:
    store.replace_doc_metrics("600519_2024", [_row("600519_2024", 2024, "1")], "t")
    # 注入式输入只会被当成普通字符串比较，查不到任何数据，也不会破坏表
    assert store.get_metric("600519' OR '1'='1", 2024, "revenue") == []
    assert len(store.get_metric("600519", 2024, "revenue")) == 1


def test_log_tool_call(store: PgFinStore) -> None:
    store.log_tool_call({"tool": "calculator", "args": {"expression": "1+1"}, "ok": True, "n_results": 0,
                         "latency_ms": 0.1, "trace_id": "abc"})  # fmt: skip
    with store.pool.connection() as conn:
        tool, args = conn.execute("SELECT tool, args FROM agent_tool_calls").fetchone()
    assert tool == "calculator" and args == {"expression": "1+1"}
