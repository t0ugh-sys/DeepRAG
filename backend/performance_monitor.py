"""
性能监控模块

提供请求追踪、性能统计、热门查询分析等功能
"""

import time
import json
from typing import Dict, Any, List, Optional
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import threading


@dataclass
class RequestMetrics:
    """单次请求的性能指标"""
    request_id: str
    endpoint: str
    method: str
    query: Optional[str]
    timestamp: float
    duration: float  # 总耗时（秒）
    retrieval_time: Optional[float] = None  # 检索耗时
    rerank_time: Optional[float] = None  # 重排耗时
    llm_time: Optional[float] = None  # LLM 生成耗时
    top_k: Optional[int] = None
    results_count: Optional[int] = None
    status: str = "success"  # success/error
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.request_history: deque = deque(maxlen=max_history)
        self.query_counter: Dict[str, int] = defaultdict(int)
        self.endpoint_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "count": 0,
            "total_time": 0.0,
            "avg_time": 0.0,
            "min_time": float('inf'),
            "max_time": 0.0,
            "error_count": 0
        })
        self.lock = threading.Lock()
        self.start_time = time.time()
    
    def record_request(self, metrics: RequestMetrics):
        """记录请求指标"""
        with self.lock:
            # 添加到历史记录
            self.request_history.append(metrics)
            
            # 更新查询计数
            if metrics.query:
                self.query_counter[metrics.query] += 1
            
            # 更新端点统计
            stats = self.endpoint_stats[metrics.endpoint]
            stats["count"] += 1
            stats["total_time"] += metrics.duration
            stats["avg_time"] = stats["total_time"] / stats["count"]
            stats["min_time"] = min(stats["min_time"], metrics.duration)
            stats["max_time"] = max(stats["max_time"], metrics.duration)
            
            if metrics.status == "error":
                stats["error_count"] += 1
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取总体统计信息"""
        with self.lock:
            total_requests = len(self.request_history)
            
            if total_requests == 0:
                return {
                    "total_requests": 0,
                    "uptime_seconds": time.time() - self.start_time,
                    "requests_per_minute": 0.0,
                    "avg_response_time": 0.0,
                    "error_rate": 0.0
                }
            
            # 计算平均响应时间
            total_time = sum(r.duration for r in self.request_history)
            avg_time = total_time / total_requests
            
            # 计算错误率
            error_count = sum(1 for r in self.request_history if r.status == "error")
            error_rate = error_count / total_requests
            
            # 计算请求速率
            uptime = time.time() - self.start_time
            requests_per_minute = (total_requests / uptime) * 60 if uptime > 0 else 0
            
            return {
                "total_requests": total_requests,
                "uptime_seconds": uptime,
                "requests_per_minute": round(requests_per_minute, 2),
                "avg_response_time": round(avg_time, 3),
                "min_response_time": round(min(r.duration for r in self.request_history), 3),
                "max_response_time": round(max(r.duration for r in self.request_history), 3),
                "error_count": error_count,
                "error_rate": round(error_rate, 3),
                "success_count": total_requests - error_count
            }
    
    def get_endpoint_statistics(self) -> Dict[str, Dict[str, Any]]:
        """获取各端点的统计信息"""
        with self.lock:
            result = {}
            for endpoint, stats in self.endpoint_stats.items():
                result[endpoint] = {
                    "count": stats["count"],
                    "avg_time": round(stats["avg_time"], 3),
                    "min_time": round(stats["min_time"], 3) if stats["min_time"] != float('inf') else 0,
                    "max_time": round(stats["max_time"], 3),
                    "error_count": stats["error_count"],
                    "error_rate": round(stats["error_count"] / stats["count"], 3) if stats["count"] > 0 else 0
                }
            return result
    
    def get_hot_queries(self, top_n: int = 10) -> List[Dict[str, Any]]:
        """获取热门查询"""
        with self.lock:
            sorted_queries = sorted(
                self.query_counter.items(),
                key=lambda x: x[1],
                reverse=True
            )[:top_n]
            
            return [
                {"query": query, "count": count}
                for query, count in sorted_queries
            ]
    
    def get_recent_requests(self, limit: int = 50) -> List[Dict[str, Any]]:
        """获取最近的请求记录"""
        with self.lock:
            recent = list(self.request_history)[-limit:]
            return [r.to_dict() for r in reversed(recent)]
    
    def get_performance_breakdown(self) -> Dict[str, Any]:
        """获取性能分解统计"""
        with self.lock:
            retrieval_times = [r.retrieval_time for r in self.request_history if r.retrieval_time]
            rerank_times = [r.rerank_time for r in self.request_history if r.rerank_time]
            llm_times = [r.llm_time for r in self.request_history if r.llm_time]
            
            def calc_stats(times: List[float]) -> Dict[str, float]:
                if not times:
                    return {"avg": 0.0, "min": 0.0, "max": 0.0, "count": 0}
                return {
                    "avg": round(sum(times) / len(times), 3),
                    "min": round(min(times), 3),
                    "max": round(max(times), 3),
                    "count": len(times)
                }
            
            return {
                "retrieval": calc_stats(retrieval_times),
                "rerank": calc_stats(rerank_times),
                "llm_generation": calc_stats(llm_times)
            }
    
    def get_time_series(self, interval_seconds: int = 60) -> List[Dict[str, Any]]:
        """获取时间序列数据"""
        with self.lock:
            if not self.request_history:
                return []
            
            # 按时间间隔分组
            time_buckets = defaultdict(lambda: {"count": 0, "total_time": 0.0, "errors": 0})
            
            for req in self.request_history:
                bucket_time = int(req.timestamp / interval_seconds) * interval_seconds
                bucket = time_buckets[bucket_time]
                bucket["count"] += 1
                bucket["total_time"] += req.duration
                if req.status == "error":
                    bucket["errors"] += 1
            
            # 转换为列表并排序
            result = []
            for timestamp, data in sorted(time_buckets.items()):
                result.append({
                    "timestamp": timestamp,
                    "datetime": datetime.fromtimestamp(timestamp).isoformat(),
                    "request_count": data["count"],
                    "avg_response_time": round(data["total_time"] / data["count"], 3) if data["count"] > 0 else 0,
                    "error_count": data["errors"]
                })
            
            return result
    
    def clear_history(self):
        """清空历史记录"""
        with self.lock:
            self.request_history.clear()
            self.query_counter.clear()
            self.endpoint_stats.clear()
            self.start_time = time.time()
    
    def export_metrics(self, filepath: str):
        """导出指标到文件"""
        with self.lock:
            data = {
                "exported_at": datetime.now().isoformat(),
                "statistics": self.get_statistics(),
                "endpoint_stats": self.get_endpoint_statistics(),
                "hot_queries": self.get_hot_queries(50),
                "recent_requests": self.get_recent_requests(100),
                "performance_breakdown": self.get_performance_breakdown()
            }
            
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)


class RequestTimer:
    """请求计时器上下文管理器"""
    
    def __init__(self, monitor: PerformanceMonitor, endpoint: str, method: str = "POST", query: Optional[str] = None):
        self.monitor = monitor
        self.endpoint = endpoint
        self.method = method
        self.query = query
        self.start_time = None
        self.metrics = None
        self.retrieval_start = None
        self.rerank_start = None
        self.llm_start = None
    
    def __enter__(self):
        self.start_time = time.time()
        self.metrics = RequestMetrics(
            request_id=f"{self.endpoint}_{int(self.start_time * 1000)}",
            endpoint=self.endpoint,
            method=self.method,
            query=self.query,
            timestamp=self.start_time,
            duration=0.0
        )
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.metrics.duration = time.time() - self.start_time
        
        if exc_type is not None:
            self.metrics.status = "error"
            self.metrics.error_message = str(exc_val)
        
        self.monitor.record_request(self.metrics)
        return False
    
    def start_retrieval(self):
        """开始检索计时"""
        self.retrieval_start = time.time()
    
    def end_retrieval(self):
        """结束检索计时"""
        if self.retrieval_start:
            self.metrics.retrieval_time = time.time() - self.retrieval_start
    
    def start_rerank(self):
        """开始重排计时"""
        self.rerank_start = time.time()
    
    def end_rerank(self):
        """结束重排计时"""
        if self.rerank_start:
            self.metrics.rerank_time = time.time() - self.rerank_start
    
    def start_llm(self):
        """开始 LLM 计时"""
        self.llm_start = time.time()
    
    def end_llm(self):
        """结束 LLM 计时"""
        if self.llm_start:
            self.metrics.llm_time = time.time() - self.llm_start
    
    def set_results_count(self, count: int):
        """设置结果数量"""
        self.metrics.results_count = count
    
    def set_top_k(self, top_k: int):
        """设置 top_k"""
        self.metrics.top_k = top_k


# 全局监控器实例
_global_monitor: Optional[PerformanceMonitor] = None


def get_monitor() -> PerformanceMonitor:
    """获取全局监控器实例"""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = PerformanceMonitor()
    return _global_monitor


def reset_monitor():
    """重置全局监控器"""
    global _global_monitor
    _global_monitor = PerformanceMonitor()
