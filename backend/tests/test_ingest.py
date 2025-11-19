"""文档摄取模块测试"""
import pytest
from backend.ingest import split_text, read_text_file


def test_split_text_basic():
    """测试基本文本分块"""
    text = "这是第一段。\n\n这是第二段。\n\n这是第三段。"
    chunks = split_text(text, chunk_size=20, chunk_overlap=5)
    assert len(chunks) > 0
    assert all(isinstance(chunk, str) for chunk in chunks)


def test_split_text_markdown():
    """测试 Markdown 文本分块"""
    text = """# 标题1

这是内容1。

## 标题2

这是内容2。

```python
def hello():
    print("world")
```

这是更多内容。"""
    
    chunks = split_text(text, chunk_size=100, chunk_overlap=20)
    assert len(chunks) > 0
    # 验证代码块被保持在一起
    assert any("```python" in chunk for chunk in chunks)


def test_split_text_empty():
    """测试空文本"""
    chunks = split_text("")
    assert len(chunks) == 0


def test_split_text_small():
    """测试小文本"""
    text = "短文本"
    chunks = split_text(text, chunk_size=100)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_read_text_file(tmp_path):
    """测试文本文件读取"""
    test_file = tmp_path / "test.txt"
    content = "测试内容\n第二行"
    test_file.write_text(content, encoding="utf-8")
    
    result = read_text_file(str(test_file))
    assert result == content
