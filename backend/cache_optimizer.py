"""
缓存优化模块

提供智能缓存策略、缓存预热、缓存分析等功能
"""

from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict
import hashlib
import time
from dataclasses import dataclass
import json


@dataclass
class CacheEntry:
    """缓存条目"""
    key: str
    value: Any
    hits: int
    created_at: float
    last_accessed: float
    size_bytes: int
    ttl: Optional[float] = None


class CacheAnalyzer:
    """缓存分析器"""
    
    def __init__(self, cache_stats: Dict[str, Any]):
        self.stats = cache_stats
    
    def get_hit_rate(self) -> float:
        """计算缓存命中率"""
        hits = self.stats.get("hits", 0)
        misses = self.stats.get("misses", 0)
        total = hits + misses
        return hits / total if total > 0 else 0.0
    
    def get_efficiency_score(self) -> float:
        """计算缓存效率分数"""
        hit_rate = self.get_hit_rate()
        size = self.stats.get("size", 0)
        max_size = self.stats.get("max_size", 1000)
        
        # 综合命中率和空间利用率
        space_utilization = size / max_size if max_size > 0 else 0
        efficiency = (hit_rate * 0.7) + (space_utilization * 0.3)
        
        return round(efficiency, 4)
    
    def suggest_optimizations(self) -> List[str]:
        """建议优化措施"""
        suggestions = []
        hit_rate = self.get_hit_rate()
        size = self.stats.get("size", 0)
        max_size = self.stats.get("max_size", 1000)
        
        # 命中率低
        if hit_rate < 0.3:
            suggestions.append("缓存命中率较低，建议：")
            suggestions.append("  - 增加缓存大小")
            suggestions.append("  - 启用查询归一化")
            suggestions.append("  - 使用缓存预热")
        elif hit_rate < 0.5:
            suggestions.append("缓存命中率中等，建议：")
            suggestions.append("  - 分析热门查询并预热")
            suggestions.append("  - 调整 TTL 策略")
        
        # 空间利用率
        space_utilization = size / max_size if max_size > 0 else 0
        if space_utilization > 0.9:
            suggestions.append("缓存空间接近满载，建议：")
            suggestions.append("  - 增加缓存容量")
            suggestions.append("  - 启用 LRU 淘汰策略")
            suggestions.append("  - 减少 TTL 时间")
        elif space_utilization < 0.3:
            suggestions.append("缓存空间利用率低，建议：")
            suggestions.append("  - 减少缓存容量以节省内存")
            suggestions.append("  - 增加缓存的查询类型")
        
        if not suggestions:
            suggestions.append("缓存运行良好，继续保持当前配置")
        
        return suggestions


class QueryNormalizer:
    """查询归一化器"""
    
    @staticmethod
    def normalize(query: str) -> str:
        """归一化查询文本"""
        # 转小写
        normalized = query.lower().strip()
        
        # 移除多余空格
        normalized = " ".join(normalized.split())
        
        # 移除标点符号（保留中文）
        import string
        translator = str.maketrans("", "", string.punctuation)
        normalized = normalized.translate(translator)
        
        return normalized
    
    @staticmethod
    def get_cache_key(query: str, **kwargs) -> str:
        """生成缓存键"""
        normalized_query = QueryNormalizer.normalize(query)
        
        # 包含其他参数
        params = sorted(kwargs.items())
        params_str = json.dumps(params, sort_keys=True)
        
        # 生成哈希
        content = f"{normalized_query}:{params_str}"
        return hashlib.md5(content.encode()).hexdigest()
    
    @staticmethod
    def are_similar(query1: str, query2: str, threshold: float = 0.8) -> bool:
        """判断两个查询是否相似"""
        norm1 = set(QueryNormalizer.normalize(query1).split())
        norm2 = set(QueryNormalizer.normalize(query2).split())
        
        if not norm1 or not norm2:
            return False
        
        # Jaccard 相似度
        intersection = len(norm1 & norm2)
        union = len(norm1 | norm2)
        similarity = intersection / union if union > 0 else 0
        
        return similarity >= threshold


class CachePrewarmer:
    """缓存预热器"""
    
    def __init__(self, retrieval_func):
        self.retrieval_func = retrieval_func
    
    def prewarm_from_hot_queries(
        self,
        hot_queries: List[Dict[str, Any]],
        top_n: int = 20
    ) -> Dict[str, Any]:
        """从热门查询预热缓存"""
        prewarmed = []
        failed = []
        
        for item in hot_queries[:top_n]:
            query = item.get("query")
            if not query:
                continue
            
            try:
                # 执行检索并缓存
                self.retrieval_func(query)
                prewarmed.append(query)
            except Exception as e:
                failed.append({"query": query, "error": str(e)})
        
        return {
            "prewarmed_count": len(prewarmed),
            "failed_count": len(failed),
            "prewarmed_queries": prewarmed,
            "failed_queries": failed
        }
    
    def prewarm_from_patterns(
        self,
        patterns: List[str]
    ) -> Dict[str, Any]:
        """从常见模式预热缓存"""
        prewarmed = []
        
        for pattern in patterns:
            try:
                self.retrieval_func(pattern)
                prewarmed.append(pattern)
            except Exception:
                continue
        
        return {
            "prewarmed_count": len(prewarmed),
            "patterns": prewarmed
        }


class CacheStrategy:
    """缓存策略"""
    
    @staticmethod
    def should_cache(
        query: str,
        result_count: int,
        execution_time: float
    ) -> bool:
        """判断是否应该缓存"""
        # 查询太短，不缓存
        if len(query.split()) < 2:
            return False
        
        # 没有结果，不缓存
        if result_count == 0:
            return False
        
        # 执行时间太短，不需要缓存
        if execution_time < 0.1:
            return False
        
        return True
    
    @staticmethod
    def get_ttl(
        query: str,
        result_count: int,
        query_frequency: int = 1
    ) -> int:
        """计算缓存 TTL（秒）"""
        # 基础 TTL
        base_ttl = 3600  # 1 小时
        
        # 根据查询频率调整
        if query_frequency > 10:
            base_ttl *= 2  # 高频查询，延长缓存
        elif query_frequency > 5:
            base_ttl *= 1.5
        
        # 根据结果数量调整
        if result_count > 10:
            base_ttl *= 1.2  # 结果多，延长缓存
        
        return int(base_ttl)
    
    @staticmethod
    def get_priority(
        query: str,
        hits: int,
        last_accessed: float
    ) -> float:
        """计算缓存优先级（用于淘汰）"""
        # LRU + LFU 混合策略
        current_time = time.time()
        time_since_access = current_time - last_accessed
        
        # 访问频率权重
        frequency_score = hits
        
        # 时间衰减权重
        time_score = 1.0 / (1.0 + time_since_access / 3600)
        
        # 综合分数
        priority = (frequency_score * 0.6) + (time_score * 0.4)
        
        return priority


class SmartCache:
    """智能缓存"""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache: Dict[str, CacheEntry] = {}
        self.hits = 0
        self.misses = 0
        self.normalizer = QueryNormalizer()
        self.strategy = CacheStrategy()
    
    def get(self, query: str, **kwargs) -> Optional[Any]:
        """获取缓存"""
        key = self.normalizer.get_cache_key(query, **kwargs)
        
        if key in self.cache:
            entry = self.cache[key]
            
            # 检查 TTL
            if entry.ttl and (time.time() - entry.created_at) > entry.ttl:
                del self.cache[key]
                self.misses += 1
                return None
            
            # 更新访问信息
            entry.hits += 1
            entry.last_accessed = time.time()
            self.hits += 1
            
            return entry.value
        
        self.misses += 1
        return None
    
    def set(
        self,
        query: str,
        value: Any,
        result_count: int = 0,
        execution_time: float = 0.0,
        **kwargs
    ):
        """设置缓存"""
        # 判断是否应该缓存
        if not self.strategy.should_cache(query, result_count, execution_time):
            return
        
        key = self.normalizer.get_cache_key(query, **kwargs)
        
        # 计算 TTL
        ttl = self.strategy.get_ttl(query, result_count)
        
        # 计算大小
        size_bytes = len(str(value).encode())
        
        # 创建缓存条目
        entry = CacheEntry(
            key=key,
            value=value,
            hits=0,
            created_at=time.time(),
            last_accessed=time.time(),
            size_bytes=size_bytes,
            ttl=ttl
        )
        
        # 检查容量
        if len(self.cache) >= self.max_size:
            self._evict()
        
        self.cache[key] = entry
    
    def _evict(self):
        """淘汰缓存"""
        if not self.cache:
            return
        
        # 计算优先级
        priorities = {}
        for key, entry in self.cache.items():
            priorities[key] = self.strategy.get_priority(
                entry.key,
                entry.hits,
                entry.last_accessed
            )
        
        # 移除优先级最低的
        min_key = min(priorities.items(), key=lambda x: x[1])[0]
        del self.cache[min_key]
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        total_size = sum(entry.size_bytes for entry in self.cache.values())
        
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hits / (self.hits + self.misses) if (self.hits + self.misses) > 0 else 0,
            "total_size_bytes": total_size,
            "avg_size_bytes": total_size / len(self.cache) if self.cache else 0
        }
    
    def clear(self):
        """清空缓存"""
        self.cache.clear()
        self.hits = 0
        self.misses = 0
    
    def analyze(self) -> Dict[str, Any]:
        """分析缓存"""
        stats = self.get_stats()
        analyzer = CacheAnalyzer(stats)
        
        return {
            "stats": stats,
            "hit_rate": analyzer.get_hit_rate(),
            "efficiency_score": analyzer.get_efficiency_score(),
            "suggestions": analyzer.suggest_optimizations()
        }


# 全局智能缓存实例
_global_smart_cache: Optional[SmartCache] = None


def get_smart_cache() -> SmartCache:
    """获取全局智能缓存实例"""
    global _global_smart_cache
    if _global_smart_cache is None:
        _global_smart_cache = SmartCache(max_size=1000)
    return _global_smart_cache
