"""Middleware: request logging and request context."""
import time
import uuid
from contextvars import ContextVar
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from backend.utils.logger import logger

_request_var: ContextVar[Request | None] = ContextVar("request_var", default=None)
_request_id_var: ContextVar[str | None] = ContextVar("request_id_var", default=None)


def get_current_request() -> Request | None:
    return _request_var.get()

def get_current_request_id() -> str | None:
    return _request_id_var.get()


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Request logging middleware."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()
        token = _request_var.set(request)
        request_id_token = _request_id_var.set(str(uuid.uuid4()))
        request_id = _request_id_var.get()

        logger.info(f"[{request_id}] -> {request.method} {request.url.path}")

        try:
            response = await call_next(request)
            process_time = time.time() - start_time

            logger.info(
                f"[{request_id}] -> {request.method} {request.url.path} | "
                f"status={response.status_code} | time={process_time:.3f}s"
            )

            response.headers["X-Process-Time"] = str(process_time)
            if request_id:
                response.headers["X-Request-Id"] = request_id
            return response
        except Exception as e:
            process_time = time.time() - start_time
            logger.error(
                f"[{request_id}] !! {request.method} {request.url.path} | "
                f"error={str(e)} | time={process_time:.3f}s"
            )
            raise
        finally:
            _request_var.reset(token)
            _request_id_var.reset(request_id_token)
