"""Agent 的三个工具：search_reports（检索年报）、get_financial_metric（查财务指标表）、calculator（安全计算器）。

每个工具的结果：
    - 返回给模型的文本（content）；
    - 新登记的证据（检索到的 chunk、查到的指标都编号进证据登记表，最终答案只能引用这些编号）；
    - 一条调用记录（写入状态和 Postgres 审计表，评测时统计工具使用情况）。
计算器的结果不是证据：答案中的计算结果可以来自它，但参与计算的原始数字必须引用证据。

模型给出的参数一律不可信：JSON 可能不合法、字段可能缺失或类型不对，这里全部校验，
出错时把错误说明作为工具结果返回给模型（让它改正后重试），而不是抛异常中断整个对话。
"""

from __future__ import annotations

import ast
import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from decimal import Decimal, DivisionByZero, InvalidOperation, localcontext

from agent.fin_db import FinStore
from agent.state import Evidence
from generation.prompts import format_evidence
from ingest.chunk import Chunk
from ingest.tables import METRICS
from retrieval.query import QueryParser

SearchFn = Callable[[str, int], list[Chunk]]

# ---------------- 工具说明（OpenAI tools 格式，模型据此决定调用哪个工具、传什么参数） ----------------

TOOL_SPECS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "get_financial_metric",
            "description": (
                "从结构化财务指标表中精确查询某公司某年度的核心财务指标（数据抽取自年报'主要会计数据和财务指标'表，"
                "附原文页码和单位）。只要问题涉及以下指标，优先用这个工具，而不是从表格文本里读数。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "company": {"type": "string", "description": "公司简称或股票代码，如 贵州茅台、600519"},
                    "year": {"type": "integer", "description": "会计年度，如 2024"},
                    "metric": {
                        "type": "string",
                        "enum": list(METRICS),
                        "description": "；".join(f"{k}={v[0]}" for k, v in METRICS.items()),
                    },
                },
                "required": ["company", "year", "metric"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_reports",
            "description": (
                "在年报原文中检索，返回相关段落（带公司、年份、页码）。用于现有证据里没有的信息，"
                "或者需要另一家公司、另一个年份的资料时。明确给出 company 和 year 能大幅提高准确度。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "要查找的内容，如 研发投入合计"},
                    "company": {"type": "string", "description": "公司简称或股票代码（可选）"},
                    "year": {"type": "integer", "description": "年报年度（可选）"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": (
                "精确计算算术表达式。任何计算（增长率、差额、占比、比值、单位换算）都必须用它，不要心算。"
                "只支持数字和 + - * / ( )，数字不要带千分位逗号、百分号或单位。"
            ),
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string", "description": "如 (168838102514.79-170899152276.34)/170899152276.34*100"}},
                "required": ["expression"],
            },
        },
    },
]


# ---------------- 安全计算器 ----------------


class CalcError(ValueError):
    pass


_MAX_EXPR_LEN = 300
_MAX_NODES = 200
_BINOPS = {ast.Add: Decimal.__add__, ast.Sub: Decimal.__sub__, ast.Mult: Decimal.__mul__, ast.Div: Decimal.__truediv__}


def safe_calc(expression: str) -> Decimal:
    """计算只含数字、+ - * /、括号、正负号的表达式，用 Decimal 精确计算。

    为什么不用 eval：表达式来自模型，模型的输入又包含外部文档，eval 等于允许执行任意代码。
    这里先用 ast 解析成语法树，再逐个节点检查、只对白名单节点求值：变量名、函数调用、属性访问、
    幂运算（9**9**9 可以耗尽内存）等一律拒绝。
    """
    expr = expression.strip()
    if not expr:
        raise CalcError("表达式为空")
    if len(expr) > _MAX_EXPR_LEN:
        raise CalcError(f"表达式过长（超过 {_MAX_EXPR_LEN} 个字符）")
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise CalcError(f"表达式语法错误：{e.msg}") from None
    if sum(1 for _ in ast.walk(tree)) > _MAX_NODES:
        raise CalcError("表达式过于复杂")

    def ev(node: ast.AST) -> Decimal:
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            # 直接用源码里的数字串构造 Decimal，避免 0.1 这类数先变成二进制浮点带来的误差
            return Decimal(ast.get_source_segment(expr, node))  # type: ignore[arg-type]
        if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
            return _BINOPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            v = ev(node.operand)
            return -v if isinstance(node.op, ast.USub) else v
        if isinstance(node, ast.Tuple):
            raise CalcError("数字不要带千分位逗号")
        raise CalcError(f"不支持的内容：{type(node).__name__}（只允许数字和 + - * / 括号）")

    with localcontext() as ctx:
        ctx.prec = 28
        try:
            return ev(tree.body)
        except (DivisionByZero, InvalidOperation):
            raise CalcError("除数为 0 或结果无效") from None


def format_decimal(v: Decimal, places: int = 10) -> str:
    """保留至多 places 位小数并去掉末尾的 0，不用科学计数法：Decimal('12.3400') -> '12.34'。"""
    try:
        v = v.quantize(Decimal(1).scaleb(-places))
    except InvalidOperation:  # 数值过大、超出精度时原样输出
        return format(v, "f")
    s = format(v, "f")
    return s.rstrip("0").rstrip(".") if "." in s else s


# ---------------- 工具执行 ----------------


@dataclass
class ToolOutcome:
    content: str  # 返回给模型的文本
    new_evidence: list[Evidence] = field(default_factory=list)
    record: dict = field(default_factory=dict)  # 调用记录
    calc: dict | None = None  # 计算器调用记录


def _parse_args(raw: str) -> dict:
    try:
        args = json.loads(raw or "{}")
    except json.JSONDecodeError:
        raise ValueError("参数不是合法的 JSON") from None
    if not isinstance(args, dict):
        raise ValueError("参数必须是 JSON 对象")
    return args


def _as_int(v: object, name: str) -> int:
    if isinstance(v, bool):
        raise ValueError(f"{name} 必须是整数")
    if isinstance(v, int):
        return v
    if isinstance(v, str) and v.strip().removesuffix("年").isdigit():
        return int(v.strip().removesuffix("年"))
    raise ValueError(f"{name} 必须是整数")


class ToolBox:
    def __init__(
        self,
        search: SearchFn,
        parser: QueryParser,
        store: FinStore | None,
        company_names: dict[str, str],
        search_k: int,
    ) -> None:
        """search：检索函数（与初始检索用同一条流水线）；parser：把公司名解析成股票代码；
        company_names：股票代码 -> 公司简称（用于证据的 company 字段）。"""
        self.search = search
        self.parser = parser
        self.store = store
        self.company_names = company_names
        self.search_k = search_k

    def run(self, name: str, raw_args: str, registry: Sequence[Evidence]) -> ToolOutcome:
        """执行一次工具调用。registry 是当前的证据登记表，新证据从 len(registry)+1 开始编号。"""
        t0 = time.perf_counter()
        args: dict = {}
        try:
            args = _parse_args(raw_args)
            if name == "search_reports":
                out = self._search(args, registry)
            elif name == "get_financial_metric":
                out = self._metric(args, len(registry))
            elif name == "calculator":
                out = self._calc(args)
            else:
                raise ValueError(f"没有名为 {name} 的工具")
            ok, err = True, None
        except (ValueError, CalcError) as e:  # 参数错误：告诉模型哪里错了
            out, ok, err = ToolOutcome(content=f"工具调用失败：{e}"), False, str(e)
            if name == "calculator":
                out.calc = {"expression": str(args.get("expression", "")), "error": str(e)}
        out.record = {
            "tool": name, "args": args, "ok": ok, "error": err, "n_results": len(out.new_evidence),
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
        }  # fmt: skip
        return out

    # ---- search_reports ----

    def _search(self, args: dict, registry: Sequence[Evidence]) -> ToolOutcome:
        query = args.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query 不能为空")
        company = args.get("company")
        year = _as_int(args["year"], "year") if args.get("year") is not None else None
        # 把公司和年份写进检索文本，由检索流水线的查询理解负责过滤（与初始检索同一套逻辑）
        prefix = (company.strip() if isinstance(company, str) else "") + (f"{year}年" if year else "")
        chunks = self.search(f"{prefix} {query.strip()}".strip(), self.search_k)

        seen = {e.chunk_id: e.id for e in registry if e.chunk_id}
        dup = sorted({seen[c.chunk_id] for c in chunks if c.chunk_id in seen})
        new = []
        for c in chunks:
            if c.chunk_id in seen:
                continue
            new.append(Evidence(len(registry) + len(new) + 1, "search_reports", c.doc_id, c.company, c.year,
                                c.page_start, c.page_end, c.text, c.chunk_id))  # fmt: skip
        parts = []
        if new:
            parts.append(format_evidence(new, start=new[0].id))
        if dup:
            parts.append("另有结果已在现有证据中：" + "".join(f"[{i}]" for i in dup))
        return ToolOutcome(content="\n\n".join(parts) or "没有检索到相关内容", new_evidence=new)

    # ---- get_financial_metric ----

    def _metric(self, args: dict, n_existing: int) -> ToolOutcome:
        if self.store is None:
            raise ValueError("财务指标表不可用，请改用 search_reports")
        company, metric = args.get("company"), args.get("metric")
        if not isinstance(company, str) or not company.strip():
            raise ValueError("company 不能为空")
        if metric not in METRICS:
            raise ValueError(f"metric 必须是以下之一：{', '.join(METRICS)}")
        year = _as_int(args.get("year"), "year")
        codes = self.parser.parse(company).stock_codes
        if len(codes) != 1:
            raise ValueError(f"无法唯一识别公司：{company}（可用公司简称或股票代码）")

        rows = self.store.get_metric(codes[0], year, metric)
        if not rows:
            return ToolOutcome(content=f"财务指标表中没有 {company} {year} 年的 {METRICS[metric][0]}")
        new = []
        for r in rows:
            shown = f"{r.value:,.2f}" if r.unit == "元" else str(r.value)
            text = (
                f"{r.company}{r.report_year}年年度报告第{r.page}页「主要会计数据和财务指标」表："
                f"{r.label}，{r.fiscal_year}年为 {r.raw}（原文单位：{r.raw_unit}；即 {shown} {r.unit}）"
            )
            new.append(Evidence(n_existing + len(new) + 1, "get_financial_metric", r.doc_id,
                                self.company_names.get(r.stock_code, r.company), r.report_year, r.page, r.page, text))  # fmt: skip
        content = format_evidence(new, start=new[0].id)
        if len({r.value for r in rows}) > 1:
            content += "\n注意：不同年份的年报对该数值的披露不同（后一年年报可能做了追溯调整），回答时请说明出处。"
        return ToolOutcome(content=content, new_evidence=new)

    # ---- calculator ----

    def _calc(self, args: dict) -> ToolOutcome:
        expr = args.get("expression")
        if not isinstance(expr, str):
            raise ValueError("expression 必须是字符串")
        result = format_decimal(safe_calc(expr))
        return ToolOutcome(content=f"{expr} = {result}", calc={"expression": expr, "result": result})
