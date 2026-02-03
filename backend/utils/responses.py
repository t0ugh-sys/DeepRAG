"""统一响应格式"""
from typing import Any, Optional
from fastapi.responses import JSONResponse

def success_response(data: Any = None, message: str = "success") -> JSONResponse:
    """成功响应"""
    return JSONResponse({
        "ok": True,
        "message": message,
        "data": data
    })

def error_response(
    error: Optional[str] = None,
    status_code: int = 400,
    details: Optional[dict] = None,
    message: Optional[str] = None
) -> JSONResponse:
    """错误响应"""
    resolved_error = error or message or "error"
    content = {
        "ok": False,
        "error": resolved_error
    }
    if details:
        content["details"] = details
    return JSONResponse(content, status_code=status_code)

