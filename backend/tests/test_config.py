"""配置模块测试"""
import os
import pytest
from backend.config import Settings, ensure_dirs


def test_settings_defaults():
    """测试默认配置"""
    settings = Settings()
    assert settings.docs_dir == os.getenv("RAG_DOCS_DIR", "data/docs")
    assert settings.index_dir == os.getenv("RAG_INDEX_DIR", "data/index")
    assert settings.top_k == int(os.getenv("RAG_TOP_K", "8"))
    assert settings.embedding_model_name is not None


def test_settings_environment_override(monkeypatch):
    """测试环境变量覆盖"""
    monkeypatch.setenv("RAG_TOP_K", "10")
    monkeypatch.setenv("RAG_BM25_ENABLED", "false")
    
    settings = Settings()
    assert settings.top_k == 10
    assert settings.bm25_enabled is False


def test_ensure_dirs(tmp_path):
    """测试目录创建"""
    test_dir = tmp_path / "test_dir"
    ensure_dirs(str(test_dir))
    assert test_dir.exists()
    assert test_dir.is_dir()
