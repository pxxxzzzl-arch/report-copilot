"""统一错误处理：所有错误都返回 {"error": {"code", "message", "trace_id"}}。

原则：
    - 对外只给错误码和简短说明，不暴露堆栈、配置、SQL、密钥等内部信息；完整异常只写进服务端日志（带 trace_id）；
    - 客户端拿着 trace_id 反馈问题，服务端按 trace_id 就能在日志里找到这次请求的全部记录。

错误码：
    invalid_request 422  参数校验失败          unauthorized 401   缺少或错误的 API Key
    not_found       404  资源不存在            rate_limited 429   超过限流（带 Retry-After 响应头）
    busy            503  并发已满              upstream_error 502 大模型接口调用失败（重试用尽后）
    internal_error  500  其他未预期的错误
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from api.schemas import ErrorDetail, ErrorResponse
from common.log import get_trace_id

logger = logging.getLogger("api.errors")


class APIError(Exception):
    def __init__(self, status: int, code: str, message: str, headers: dict[str, str] | None = None) -> None:
        super().__init__(message)
        self.status, self.code, self.message, self.headers = status, code, message, headers


def error_body(code: str, message: str) -> dict:
    return ErrorResponse(error=ErrorDetail(code=code, message=message, trace_id=get_trace_id())).model_dump()


def classify_exception(exc: BaseException) -> APIError:
    """把运行 Agent 时抛出的异常归类成对外的错误（非流式接口返回对应状态码，流式接口放进 error 事件）。"""
    if isinstance(exc, APIError):
        return exc
    try:
        import openai

        if isinstance(exc, openai.APIError):  # 网络错误、限流、5xx 等，openai SDK 已按配置重试过
            return APIError(502, "upstream_error", "大模型服务暂时不可用，请稍后重试")
    except ImportError:  # pragma: no cover - openai 是项目依赖，这里只是防御
        pass
    return APIError(500, "internal_error", "服务内部错误")


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(APIError)
    async def _api_error(_: Request, exc: APIError) -> JSONResponse:
        return JSONResponse(error_body(exc.code, exc.message), status_code=exc.status, headers=exc.headers)

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, exc: RequestValidationError) -> JSONResponse:
        # 只返回出错的字段位置和原因，不回显用户输入的内容
        parts = [f"{'.'.join(str(x) for x in e['loc'])}: {e['msg']}" for e in exc.errors()]
        return JSONResponse(error_body("invalid_request", "; ".join(parts)), status_code=422)

    @app.exception_handler(StarletteHTTPException)
    async def _http(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        code = {404: "not_found", 405: "method_not_allowed"}.get(exc.status_code, "http_error")
        return JSONResponse(error_body(code, str(exc.detail)), status_code=exc.status_code)

    @app.exception_handler(Exception)
    async def _unexpected(_: Request, exc: Exception) -> JSONResponse:
        err = classify_exception(exc)
        logger.error("unhandled error", exc_info=exc, extra={"code": err.code})
        return JSONResponse(error_body(err.code, err.message), status_code=err.status)
