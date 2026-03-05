def test_stdlib_types_not_shadowed() -> None:
    import types as std_types

    assert hasattr(std_types, "MappingProxyType")
    module_path = (getattr(std_types, "__file__", "") or "").lower()
    assert "backend" not in module_path


def test_rag_types_importable() -> None:
    from backend.rag_types import RetrievedChunk

    rec = RetrievedChunk(text="t", score=0.1, meta={"path": "p", "chunk_id": 1})
    assert rec.text == "t"
