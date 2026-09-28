"""请求与响应的数据模型（Pydantic）：负责参数校验，同时生成 /docs 里的 OpenAPI 文档。"""

from __future__ import annotations

from pydantic import BaseModel, Field

# 对话编号只允许字母、数字、下划线、连字符：它会作为 checkpoint 的键写进 Postgres，也会出现在日志里
THREAD_ID_PATTERN = r"^[A-Za-z0-9_-]{1,64}$"


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, description="用户问题；最大长度由 API__MAX_QUESTION_CHARS 限制")
    thread_id: str | None = Field(default=None, pattern=THREAD_ID_PATTERN, description="对话编号；不填则新建对话")


class CitationOut(BaseModel):
    ref: int  # 引用的证据编号
    quote: str  # 模型给出的逐字摘录
    valid_ref: bool
    verified: bool  # 摘录确实出现在所引证据中；前端只应展示 verified=true 的引用
    doc_id: str | None = None
    page: int | None = None  # 定位到的物理页码；无法定位到单页时为 None，用 page_start~page_end
    page_start: int | None = None
    page_end: int | None = None


class EvidenceOut(BaseModel):
    id: int
    source: str  # retrieve / search_reports / get_financial_metric
    doc_id: str
    company: str
    year: int
    page_start: int
    page_end: int
    text: str


class ToolCallOut(BaseModel):
    tool: str
    args: dict
    ok: bool
    error: str | None = None
    n_results: int = 0
    latency_ms: float | None = None


class ChatResponse(BaseModel):
    thread_id: str
    trace_id: str | None
    question: str  # 用户原话
    standalone: str  # 结合对话历史改写后的完整问题
    answer: str
    refused: bool
    format_ok: bool  # 模型输出能否解析；为 false 时 answer 为空
    citations: list[CitationOut]
    evidence: list[EvidenceOut]
    tool_calls: list[ToolCallOut]
    steps: int  # 取证阶段调用模型的次数
    latency_ms: float


class Turn(BaseModel):
    question: str
    answer: str


class ThreadOut(BaseModel):
    thread_id: str
    history: list[Turn]


class ErrorDetail(BaseModel):
    code: str  # 机器可读的错误码，如 rate_limited
    message: str  # 给人看的说明；不包含堆栈、配置或密钥
    trace_id: str | None = None


class ErrorResponse(BaseModel):
    error: ErrorDetail
