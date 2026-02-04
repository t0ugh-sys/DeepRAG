"""Simple LRU cache utilities for query results."""
from collections import OrderedDict
from functools import lru_cache
from threading import Lock
from typing import Tuple
import hashlib
import json


class QueryCache:
    """Thread-safe LRU cache for query results."""

    def __init__(self, maxsize: int = 128):
        self.maxsize = maxsize
        self._cache: "OrderedDict[str, object]" = OrderedDict()
        self._lock = Lock()

    def _make_key(self, query: str, top_k: int, namespace: str) -> str:
        data = f"{query}|{top_k}|{namespace}"
        return hashlib.md5(data.encode()).hexdigest()

    def get(self, query: str, top_k: int, namespace: str):
        key = self._make_key(query, top_k, namespace)
        with self._lock:
            if key not in self._cache:
                return None
            self._cache.move_to_end(key)
            return self._cache[key]

    def set(self, query: str, top_k: int, namespace: str, value):
        key = self._make_key(query, top_k, namespace)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = value
            if len(self._cache) > self.maxsize:
                self._cache.popitem(last=False)

    def clear(self):
        with self._lock:
            self._cache.clear()

    def size(self) -> int:
        with self._lock:
            return len(self._cache)


# Global cache instance
query_cache = QueryCache(maxsize=256)


@lru_cache(maxsize=512)
def cache_embedding(text: str, model_name: str) -> Tuple[float, ...]:
    """
    Placeholder embedding cache. Real embedding computation happens in VectorStore.
    """
    # Real embedding calculation happens in VectorStore.
    pass
