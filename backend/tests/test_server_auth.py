import pytest
from fastapi import HTTPException

import backend.server as server


def test_require_admin_api_key_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(server.settings, "admin_api_key_required", True, raising=False)
    monkeypatch.setattr(server.settings, "admin_api_key", "adm", raising=False)
    monkeypatch.setattr(server.settings, "admin_api_keys", None, raising=False)

    server._require_admin_api_key("adm")
    with pytest.raises(HTTPException):
        server._require_admin_api_key("bad")


def test_require_admin_api_key_falls_back_to_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(server.settings, "admin_api_key_required", False, raising=False)
    monkeypatch.setattr(server.settings, "admin_api_key_fallback_to_api_key", True, raising=False)
    monkeypatch.setattr(server.settings, "api_key_required", True, raising=False)
    monkeypatch.setattr(server.settings, "api_key", "k", raising=False)
    monkeypatch.setattr(server.settings, "read_api_keys", None, raising=False)

    server._require_admin_api_key("k")
    with pytest.raises(HTTPException):
        server._require_admin_api_key("bad")


def test_require_admin_api_key_no_fallback_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(server.settings, "admin_api_key_required", False, raising=False)
    monkeypatch.setattr(server.settings, "admin_api_key_fallback_to_api_key", False, raising=False)
    monkeypatch.setattr(server.settings, "api_key_required", True, raising=False)
    monkeypatch.setattr(server.settings, "api_key", "k", raising=False)

    # no-op when admin check disabled and fallback also disabled
    server._require_admin_api_key(None)


def test_require_api_key_supports_read_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(server.settings, "api_key_required", True, raising=False)
    monkeypatch.setattr(server.settings, "api_key", None, raising=False)
    monkeypatch.setattr(server.settings, "read_api_keys", "read1, read2", raising=False)
    monkeypatch.setattr(server.settings, "admin_api_key", None, raising=False)
    monkeypatch.setattr(server.settings, "admin_api_keys", None, raising=False)

    server._require_api_key("read2")
    with pytest.raises(HTTPException):
        server._require_api_key("bad")


def test_require_api_key_accepts_admin_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(server.settings, "api_key_required", True, raising=False)
    monkeypatch.setattr(server.settings, "api_key", None, raising=False)
    monkeypatch.setattr(server.settings, "read_api_keys", None, raising=False)
    monkeypatch.setattr(server.settings, "admin_api_key", "adm", raising=False)
    monkeypatch.setattr(server.settings, "admin_api_keys", "adm2", raising=False)

    server._require_api_key("adm")
    server._require_api_key("adm2")


def test_resolve_namespace_whitelist_and_api_key_namespace(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(server.settings, "default_namespace", "default", raising=False)
    monkeypatch.setattr(server.settings, "namespace_whitelist", "a,b", raising=False)
    monkeypatch.setattr(server.settings, "api_key_namespace", "a", raising=False)

    assert server._resolve_namespace("a") == "a"
    with pytest.raises(HTTPException):
        server._resolve_namespace("b")
    with pytest.raises(HTTPException):
        server._resolve_namespace("c")
