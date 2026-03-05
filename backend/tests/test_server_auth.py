import pytest
from fastapi import HTTPException

import backend.server as server


def test_require_admin_api_key_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(server.settings, "admin_api_key_required", True, raising=False)
    monkeypatch.setattr(server.settings, "admin_api_key", "adm", raising=False)

    server._require_admin_api_key("adm")
    with pytest.raises(HTTPException):
        server._require_admin_api_key("bad")


def test_require_admin_api_key_falls_back_to_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(server.settings, "admin_api_key_required", False, raising=False)
    monkeypatch.setattr(server.settings, "api_key_required", True, raising=False)
    monkeypatch.setattr(server.settings, "api_key", "k", raising=False)

    server._require_admin_api_key("k")
    with pytest.raises(HTTPException):
        server._require_admin_api_key("bad")


def test_resolve_namespace_whitelist_and_api_key_namespace(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(server.settings, "default_namespace", "default", raising=False)
    monkeypatch.setattr(server.settings, "namespace_whitelist", "a,b", raising=False)
    monkeypatch.setattr(server.settings, "api_key_namespace", "a", raising=False)

    assert server._resolve_namespace("a") == "a"
    with pytest.raises(HTTPException):
        server._resolve_namespace("b")
    with pytest.raises(HTTPException):
        server._resolve_namespace("c")
