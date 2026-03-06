"""Middleware: request logging and request context."""
import hashlib
import time
import uuid
from contextvars import ContextVar
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from backend.utils.logger import logger

_request_var: ContextVar[Request | None] = ContextVar("request_var", default=None)
_request_id_var: ContextVar[str | None] = ContextVar("request_id_var", default=None)
_AUDIT_METHODS = {"POST", "PUT", "PATCH", "DELETE"}
_AUDIT_PATH_PREFIXES = (
    "/admin/",
    "/docs",
    "/import",
    "/namespaces",
    "/cache/clear",
    "/metrics/export",
    "/metrics/clear",
    "/documents/metadata",
    "/documents/",
    "/conversations/",
)


def get_current_request() -> Request | None:
    return _request_var.get()

def get_current_request_id() -> str | None:
    return _request_id_var.get()


def _is_audit_target(method: str, path: str) -> bool:
    if method.upper() not in _AUDIT_METHODS:
        return False
    return any(path.startswith(prefix) for prefix in _AUDIT_PATH_PREFIXES)


def _mask_api_key(api_key: str | None) -> str:
    if not api_key:
        return "none"
    digest = hashlib.sha256(api_key.encode("utf-8")).hexdigest()
    return digest[:12]


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


class AuditLoggingMiddleware(BaseHTTPMiddleware):
    """Audit logging middleware for write/admin endpoints."""

    def __init__(self, app, enabled: bool = True):  # type: ignore[override]
        super().__init__(app)
        self.enabled = enabled

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if not self.enabled:
            return await call_next(request)

        method = request.method.upper()
        path = request.url.path
        if not _is_audit_target(method, path):
            return await call_next(request)

        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        request_id = get_current_request_id() or "-"
        namespace = request.query_params.get("namespace") or "-"
        actor_key = _mask_api_key(request.headers.get("X-API-Key") or request.headers.get("x-api-key"))
        logger.info(
            "audit request_id=%s method=%s path=%s status=%s namespace=%s actor_key=%s elapsed_ms=%d",
            request_id,
            method,
            path,
            response.status_code,
            namespace,
            actor_key,
            int(process_time * 1000),
        )
        return response
