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

def error_response(error: str, status_code: int = 400, details: Optional[dict] = None) -> JSONResponse:
    """错误响应"""
    content = {
        "ok": False,
        "error": error
    }
    if details:
        content["details"] = details
    return JSONResponse(content, status_code=status_code)

