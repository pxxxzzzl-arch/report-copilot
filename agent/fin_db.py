"""PostgreSQL 存储：财务指标表 fin_metrics 与工具调用记录表 agent_tool_calls（LangGraph 的 checkpoint 表由 PostgresSaver 自己建）。

安全（AGENTS.md 第 9 条）：所有 SQL 都用参数化查询（%s 占位符，由 psycopg 负责转义），
公司、年份、指标名等来自模型或用户的值绝不拼进 SQL 字符串。

用法：
    uv run python -m agent.fin_db build     # 从 12 份年报抽取核心指标并写入 fin_metrics（可重复执行，按文档整体替换）
    uv run python -m agent.fin_db show 600519 2025
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Protocol

from config import get_settings

logger = logging.getLogger("agent.fin_db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS fin_metrics (
    id          BIGSERIAL PRIMARY KEY,
    doc_id      TEXT    NOT NULL,          -- 数据来自哪份年报，如 600519_2025
    stock_code  TEXT    NOT NULL,
    company     TEXT    NOT NULL,
    report_year INT     NOT NULL,          -- 年报所属年度
    fiscal_year INT     NOT NULL,          -- 数值所属会计年度（本年或上年）
    metric      TEXT    NOT NULL,          -- ingest.tables.METRICS 的键
    value       NUMERIC NOT NULL,          -- 换算后的数值（金额为元）
    unit        TEXT    NOT NULL,          -- 元 / 元/股 / %
    raw         TEXT    NOT NULL,          -- 原文数字串
    raw_unit    TEXT    NOT NULL,          -- 原文单位
    page        INT     NOT NULL,          -- 物理页码
    label       TEXT    NOT NULL,          -- 原文指标名
    extractor   TEXT    NOT NULL,          -- 抽取器版本
    UNIQUE (doc_id, fiscal_year, metric)
);
CREATE INDEX IF NOT EXISTS fin_metrics_lookup ON fin_metrics (stock_code, fiscal_year, metric);

CREATE TABLE IF NOT EXISTS agent_tool_calls (
    id         BIGSERIAL PRIMARY KEY,
    ts         TIMESTAMPTZ NOT NULL DEFAULT now(),
    trace_id   TEXT,
    thread_id  TEXT,
    tool       TEXT    NOT NULL,
    args       JSONB   NOT NULL,
    ok         BOOLEAN NOT NULL,
    error      TEXT,
    n_results  INT,
    latency_ms REAL
);
"""


@dataclass
class MetricRow:
    doc_id: str
    stock_code: str
    company: str
    report_year: int
    fiscal_year: int
    metric: str
    value: Decimal
    unit: str
    raw: str
    raw_unit: str
    page: int
    label: str


class FinStore(Protocol):
    """Agent 工具依赖的存储接口；测试里用内存实现替代 Postgres。"""

    def get_metric(self, stock_code: str, fiscal_year: int, metric: str) -> list[MetricRow]: ...

    def log_tool_call(self, record: dict[str, Any]) -> None: ...


class PgFinStore:
    """基于 psycopg 连接池的实现（线程安全：每次操作从池中取一个连接）。"""

    def __init__(self, dsn: str, min_size: int = 1, max_size: int = 4) -> None:
        from psycopg_pool import ConnectionPool

        self.pool = ConnectionPool(dsn, min_size=min_size, max_size=max_size, open=True)

    def close(self) -> None:
        self.pool.close()

    def init_schema(self) -> None:
        with self.pool.connection() as conn:
            conn.execute(SCHEMA)

    def replace_doc_metrics(self, doc_id: str, rows: list[MetricRow], extractor: str) -> int:
        """用新抽取的结果整体替换一份年报的指标（同一事务内先删后写）：
        抽取逻辑变严格后，旧版本抽出、新版本不再认可的数值不会残留在表里。"""
        with self.pool.connection() as conn, conn.transaction():
            conn.execute("DELETE FROM fin_metrics WHERE doc_id = %s", (doc_id,))
            return self._insert(conn, rows, extractor)

    def _insert(self, conn, rows: list[MetricRow], extractor: str) -> int:
        sql = """
        INSERT INTO fin_metrics (doc_id, stock_code, company, report_year, fiscal_year, metric, value, unit,
                                 raw, raw_unit, page, label, extractor)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (doc_id, fiscal_year, metric) DO UPDATE SET
            value = EXCLUDED.value, unit = EXCLUDED.unit, raw = EXCLUDED.raw, raw_unit = EXCLUDED.raw_unit,
            page = EXCLUDED.page, label = EXCLUDED.label, extractor = EXCLUDED.extractor
        """
        params = [
            (r.doc_id, r.stock_code, r.company, r.report_year, r.fiscal_year, r.metric, r.value, r.unit,
             r.raw, r.raw_unit, r.page, r.label, extractor)
            for r in rows
        ]  # fmt: skip
        with conn.cursor() as cur:
            cur.executemany(sql, params)
        return len(params)

    def get_metric(self, stock_code: str, fiscal_year: int, metric: str) -> list[MetricRow]:
        """查询某公司某会计年度的某个指标。可能返回多行：该年度自己的年报 + 下一年年报中的"上年数"
        （两者通常相同；发生追溯调整时不同，如比亚迪 2025 年转增股本后重述了 2024 年每股收益）。
        按年报年度从新到旧排列。"""
        sql = """
        SELECT doc_id, stock_code, company, report_year, fiscal_year, metric, value, unit, raw, raw_unit, page, label
        FROM fin_metrics WHERE stock_code = %s AND fiscal_year = %s AND metric = %s
        ORDER BY report_year DESC
        """
        with self.pool.connection() as conn:
            rows = conn.execute(sql, (stock_code, fiscal_year, metric)).fetchall()
        return [MetricRow(*r) for r in rows]

    def log_tool_call(self, record: dict[str, Any]) -> None:
        """写一条工具调用记录。审计日志写失败不影响回答，只记警告。"""
        sql = """
        INSERT INTO agent_tool_calls (trace_id, thread_id, tool, args, ok, error, n_results, latency_ms)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """
        try:
            with self.pool.connection() as conn:
                conn.execute(sql, (
                    record.get("trace_id"), record.get("thread_id"), record["tool"],
                    json.dumps(record.get("args", {}), ensure_ascii=False), record["ok"], record.get("error"),
                    record.get("n_results"), record.get("latency_ms"),
                ))  # fmt: skip
        except Exception as e:  # noqa: BLE001
            logger.warning("tool call audit failed", extra={"error": repr(e)})


def build(store: PgFinStore) -> int:
    """从全部年报抽取核心指标写入数据库，返回写入条数。"""
    from ingest.download import read_manifest
    from ingest.tables import EXTRACTOR_VERSION, extract_key_metrics

    settings = get_settings()
    manifest = read_manifest(settings.paths.manifest_path)
    store.init_schema()
    total = 0
    for doc_id, meta in sorted(manifest.items()):
        values, issues = extract_key_metrics(settings.paths.raw_dir / meta.file_name, doc_id, meta.year)
        for msg in issues:
            print("  问题：", msg)
        rows = [
            MetricRow(doc_id, meta.stock_code, meta.company, meta.year, v.fiscal_year, v.metric, v.value, v.unit,
                      v.raw, v.raw_unit, v.page, v.label)
            for v in values
        ]  # fmt: skip
        total += store.replace_doc_metrics(doc_id, rows, EXTRACTOR_VERSION)
        print(f"{doc_id}: {len(rows)} 条")
    return total


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="财务指标表")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build")
    s = sub.add_parser("show")
    s.add_argument("stock_code")
    s.add_argument("fiscal_year", type=int)
    args = p.parse_args(argv)

    store = PgFinStore(get_settings().store.postgres_dsn.get_secret_value())
    try:
        if args.cmd == "build":
            print(f"共写入 {build(store)} 条")
        else:
            from ingest.tables import METRICS

            for key in METRICS:
                for r in store.get_metric(args.stock_code, args.fiscal_year, key):
                    print(f"{key:<22}{r.value:>24} {r.unit:<5} 原文 {r.raw} {r.raw_unit}（{r.doc_id} 第{r.page}页）")
    finally:
        store.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
