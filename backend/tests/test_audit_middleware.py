from backend.utils.middleware import _is_audit_target, _mask_api_key


def test_is_audit_target_for_admin_write() -> None:
    assert _is_audit_target('POST', '/admin/docs')
    assert _is_audit_target('DELETE', '/documents/a/tags')
    assert _is_audit_target('POST', '/conversations/1/messages')


def test_is_audit_target_for_read_or_non_admin() -> None:
    assert not _is_audit_target('GET', '/admin/docs')
    assert not _is_audit_target('POST', '/v1/ask')


def test_mask_api_key() -> None:
    assert _mask_api_key(None) == 'none'
    masked = _mask_api_key('secret-key')
    assert len(masked) == 12
    assert masked != 'secret-key'
