"""工具模块测试"""
import pytest
from backend.utils.cache import QueryCache
from backend.utils.responses import success_response, error_response


class TestQueryCache:
    """查询缓存测试"""
    
    def test_cache_set_and_get(self):
        """测试缓存设置和获取"""
        cache = QueryCache(maxsize=10)
        cache.set("test query", 5, "default", ["result1", "result2"])
        
        result = cache.get("test query", 5, "default")
        assert result == ["result1", "result2"]
    
    def test_cache_miss(self):
        """测试缓存未命中"""
        cache = QueryCache(maxsize=10)
        result = cache.get("nonexistent", 5, "default")
        assert result is None
    
    def test_cache_lru_eviction(self):
        """测试 LRU 淘汰"""
        cache = QueryCache(maxsize=2)
        cache.set("query1", 5, "default", "result1")
        cache.set("query2", 5, "default", "result2")
        cache.set("query3", 5, "default", "result3")
        
        # query1 应该被淘汰
        assert cache.get("query1", 5, "default") is None
        assert cache.get("query2", 5, "default") == "result2"
        assert cache.get("query3", 5, "default") == "result3"
    
    def test_cache_clear(self):
        """测试缓存清空"""
        cache = QueryCache(maxsize=10)
        cache.set("query1", 5, "default", "result1")
        cache.clear()
        
        assert cache.size() == 0
        assert cache.get("query1", 5, "default") is None


class TestResponses:
    """响应格式测试"""
    
    def test_success_response(self):
        """测试成功响应"""
        response = success_response(data={"key": "value"}, message="操作成功")
        assert response.status_code == 200
        
        import json
        body = json.loads(response.body)
        assert body["ok"] is True
        assert body["message"] == "操作成功"
        assert body["data"]["key"] == "value"
    
    def test_error_response(self):
        """测试错误响应"""
        response = error_response("发生错误", status_code=400)
        assert response.status_code == 400
        
        import json
        body = json.loads(response.body)
        assert body["ok"] is False
        assert body["error"] == "发生错误"
    
    def test_error_response_with_details(self):
        """测试带详情的错误响应"""
        response = error_response(
            "验证失败",
            status_code=422,
            details={"field": "email", "issue": "invalid"}
        )
        
        import json
        body = json.loads(response.body)
        assert body["ok"] is False
        assert body["details"]["field"] == "email"
