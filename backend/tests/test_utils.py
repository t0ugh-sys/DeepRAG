"""Utility module tests / 工具模块测试"""
from backend.utils.cache import QueryCache
from backend.utils.responses import success_response, error_response


class TestQueryCache:
    """Query cache behavior / 查询缓存行为"""

    def test_cache_set_and_get(self):
        """Should return cached data / 应返回缓存数据"""
        cache = QueryCache(maxsize=10)
        cache.set("test query", 5, "default", ["result1", "result2"])

        result = cache.get("test query", 5, "default")
        assert result == ["result1", "result2"]

    def test_cache_miss(self):
        """Should return None on miss / 未命中时返回 None"""
        cache = QueryCache(maxsize=10)
        result = cache.get("nonexistent", 5, "default")
        assert result is None

    def test_cache_lru_eviction(self):
        """Should evict least recently used / 应驱逐最久未使用"""
        cache = QueryCache(maxsize=2)
        cache.set("query1", 5, "default", "result1")
        cache.set("query2", 5, "default", "result2")
        cache.set("query3", 5, "default", "result3")

        # LRU eviction should remove query1 / LRU 应移除 query1
        assert cache.get("query1", 5, "default") is None
        assert cache.get("query2", 5, "default") == "result2"
        assert cache.get("query3", 5, "default") == "result3"

    def test_cache_clear(self):
        """Clear should empty cache / clear 应清空缓存"""
        cache = QueryCache(maxsize=10)
        cache.set("query1", 5, "default", "result1")
        cache.clear()

        assert cache.size() == 0
        assert cache.get("query1", 5, "default") is None


class TestResponses:
    """Response helpers / 响应工具"""

    def test_success_response(self):
        """Should return ok response / 应返回成功响应"""
        response = success_response(data={"key": "value"}, message="OK / 成功")
        assert response.status_code == 200

        import json

        body = json.loads(response.body)
        assert body["ok"] is True
        assert body["message"] == "OK / 成功"
        assert body["data"]["key"] == "value"

    def test_error_response(self):
        """Should return error response / 应返回错误响应"""
        response = error_response("Bad request / 错误请求", status_code=400)
        assert response.status_code == 400

        import json

        body = json.loads(response.body)
        assert body["ok"] is False
        assert body["error"] == "Bad request / 错误请求"

    def test_error_response_message_alias(self):
        """Should accept message alias / 应支持 message 别名"""
        response = error_response(message="error message", status_code=400)
        assert response.status_code == 400

        import json

        body = json.loads(response.body)
        assert body["ok"] is False
        assert body["error"] == "error message"

    def test_error_response_with_details(self):
        """Should include details / 应包含详情"""
        response = error_response(
            "Validation failed / 校验失败",
            status_code=422,
            details={"field": "email", "issue": "invalid"},
        )

        import json

        body = json.loads(response.body)
        assert body["ok"] is False
        assert body["details"]["field"] == "email"
