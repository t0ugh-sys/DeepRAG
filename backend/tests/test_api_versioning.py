import backend.server as server


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
