"""简单的LRU缓存实现，用于缓存查询结果"""
from functools import lru_cache
from typing import List, Tuple
import hashlib
import json


class QueryCache:
    """查询结果缓存"""
    
    def __init__(self, maxsize: int = 128):
        self.maxsize = maxsize
        self._cache = {}
        self._access_order = []
    
    def _make_key(self, query: str, top_k: int, namespace: str) -> str:
        """生成缓存键"""
        data = f"{query}|{top_k}|{namespace}"
        return hashlib.md5(data.encode()).hexdigest()
    
    def get(self, query: str, top_k: int, namespace: str):
        """获取缓存"""
        key = self._make_key(query, top_k, namespace)
        if key in self._cache:
            # 更新访问顺序
            self._access_order.remove(key)
            self._access_order.append(key)
            return self._cache[key]
        return None
    
    def set(self, query: str, top_k: int, namespace: str, value):
        """设置缓存"""
        key = self._make_key(query, top_k, namespace)
        
        # 如果缓存已满，删除最久未使用的
        if len(self._cache) >= self.maxsize and key not in self._cache:
            oldest_key = self._access_order.pop(0)
            del self._cache[oldest_key]
        
        self._cache[key] = value
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)
    
    def clear(self):
        """清空缓存"""
        self._cache.clear()
        self._access_order.clear()
    
    def size(self) -> int:
        """获取当前缓存大小"""
        return len(self._cache)


# 全局缓存实例
query_cache = QueryCache(maxsize=256)


@lru_cache(maxsize=512)
def cache_embedding(text: str, model_name: str) -> Tuple[float, ...]:
    """
    缓存文本嵌入（这只是占位符，实际嵌入在VectorStore中计算）
    注意：这个函数主要用于演示，实际使用时需要在VectorStore中集成
    """
    # 实际的嵌入计算在VectorStore中进行
    # 这里只是一个示例框架
    pass

