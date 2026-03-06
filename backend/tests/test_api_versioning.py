import backend.server as server
from fastapi import Response


def test_legacy_successor_exact_routes() -> None:
    assert server._legacy_successor_for_request('POST', '/ask') == '/v1/ask'
    assert server._legacy_successor_for_request('GET', '/healthz') == '/v1/healthz'
    assert server._legacy_successor_for_request('POST', '/metrics/export') == '/admin/metrics/export'


def test_legacy_successor_dynamic_routes() -> None:
    assert server._legacy_successor_for_request('POST', '/documents/a/b/tags') == '/admin/documents/{path}/tags'
    assert server._legacy_successor_for_request('DELETE', '/documents/x/tags') == '/admin/documents/{path}/tags'


def test_non_legacy_route_returns_none() -> None:
    assert server._legacy_successor_for_request('POST', '/v1/ask') is None
    assert server._legacy_successor_for_request('GET', '/documents/list') is None


def test_apply_backend_headers_with_fallback() -> None:
    class _Store:
        backend = 'faiss'
        fallback_from = 'milvus'
        fallback_reason = 'connection refused'

    class _Pipeline:
        store = _Store()

    response = Response()
    server._apply_backend_headers(response, _Pipeline(), 'default')
    assert response.headers['X-RAG-Namespace'] == 'default'
    assert response.headers['X-RAG-Vector-Backend'] == 'faiss'
    assert response.headers['X-RAG-Vector-Fallback'] == 'milvus->faiss'
