"""Middleware: request logging and request context."""
import time
from contextvars import ContextVar
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from backend.utils.logger import logger

_request_var: ContextVar[Request | None] = ContextVar("request_var", default=None)


def get_current_request() -> Request | None:
    return _request_var.get()


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Request logging middleware."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()
        token = _request_var.set(request)

        logger.info(f"-> {request.method} {request.url.path}")

        try:
            response = await call_next(request)
            process_time = time.time() - start_time

            logger.info(
                f"-> {request.method} {request.url.path} | "
                f"status={response.status_code} | time={process_time:.3f}s"
            )

            response.headers["X-Process-Time"] = str(process_time)
            return response
        except Exception as e:
            process_time = time.time() - start_time
            logger.error(
                f"!! {request.method} {request.url.path} | "
                f"error={str(e)} | time={process_time:.3f}s"
            )
            raise
        finally:
            _request_var.reset(token)
