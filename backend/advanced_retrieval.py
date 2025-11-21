"""
高级检索功能模块

提供过滤检索、权重调优、聚合检索等高级功能
"""

from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
from datetime import datetime
from collections import defaultdict
import re


@dataclass
class FilterConfig:
    """检索过滤配置"""
    doc_types: Optional[List[str]] = None  # 文档类型过滤 ['pdf', 'markdown', 'text']
    date_from: Optional[str] = None  # 日期范围起始 'YYYY-MM-DD'
    date_to: Optional[str] = None  # 日期范围结束
    min_score: Optional[float] = None  # 最低相似度分数
    max_results: Optional[int] = None  # 最大结果数
    paths: Optional[List[str]] = None  # 指定文档路径
    tags: Optional[List[str]] = None  # 标签过滤
    has_tables: Optional[bool] = None  # 是否包含表格
    page_range: Optional[tuple] = None  # 页码范围 (min, max)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "doc_types": self.doc_types,
            "date_from": self.date_from,
            "date_to": self.date_to,
            "min_score": self.min_score,
            "max_results": self.max_results,
            "paths": self.paths,
            "tags": self.tags,
            "has_tables": self.has_tables,
            "page_range": self.page_range
        }


@dataclass
class WeightConfig:
    """混合检索权重配置"""
    vector_weight: float = 0.7  # 向量检索权重
    bm25_weight: float = 0.3  # BM25 检索权重
    reranker_enabled: bool = True  # 是否启用重排
    mmr_lambda: float = 0.5  # MMR 多样性参数
    
    def validate(self) -> bool:
        """验证权重配置"""
        total = self.vector_weight + self.bm25_weight
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"权重之和必须为 1.0，当前为 {total}")
        if not (0 <= self.vector_weight <= 1):
            raise ValueError(f"vector_weight 必须在 [0, 1] 范围内")
        if not (0 <= self.bm25_weight <= 1):
            raise ValueError(f"bm25_weight 必须在 [0, 1] 范围内")
        if not (0 <= self.mmr_lambda <= 1):
            raise ValueError(f"mmr_lambda 必须在 [0, 1] 范围内")
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "vector_weight": self.vector_weight,
            "bm25_weight": self.bm25_weight,
            "reranker_enabled": self.reranker_enabled,
            "mmr_lambda": self.mmr_lambda
        }


class AdvancedRetriever:
    """高级检索器"""
    
    def __init__(self):
        self.default_weights = WeightConfig()
    
    def filter_results(
        self,
        results: List[Dict[str, Any]],
        filter_config: FilterConfig
    ) -> List[Dict[str, Any]]:
        """
        根据过滤配置过滤检索结果
        
        Args:
            results: 原始检索结果
            filter_config: 过滤配置
            
        Returns:
            过滤后的结果
        """
        filtered = results
        
        # 文档类型过滤
        if filter_config.doc_types:
            filtered = [
                r for r in filtered
                if r.get("meta", {}).get("doc_type") in filter_config.doc_types
            ]
        
        # 路径过滤
        if filter_config.paths:
            filtered = [
                r for r in filtered
                if any(path in r.get("meta", {}).get("path", "") for path in filter_config.paths)
            ]
        
        # 分数过滤
        if filter_config.min_score is not None:
            filtered = [
                r for r in filtered
                if r.get("score", 0) >= filter_config.min_score
            ]
        
        # 表格过滤
        if filter_config.has_tables is not None:
            filtered = [
                r for r in filtered
                if r.get("meta", {}).get("has_tables") == filter_config.has_tables
            ]
        
        # 页码范围过滤
        if filter_config.page_range:
            min_page, max_page = filter_config.page_range
            filtered = [
                r for r in filtered
                if self._in_page_range(r.get("meta", {}).get("page"), min_page, max_page)
            ]
        
        # 日期过滤
        if filter_config.date_from or filter_config.date_to:
            filtered = self._filter_by_date(filtered, filter_config.date_from, filter_config.date_to)
        
        # 标签过滤
        if filter_config.tags:
            filtered = [
                r for r in filtered
                if self._has_tags(r.get("meta", {}).get("tags", []), filter_config.tags)
            ]
        
        # 限制结果数量
        if filter_config.max_results:
            filtered = filtered[:filter_config.max_results]
        
        return filtered
    
    def _in_page_range(self, page: Optional[int], min_page: int, max_page: int) -> bool:
        """检查页码是否在范围内"""
        if page is None:
            return False
        return min_page <= page <= max_page
    
    def _filter_by_date(
        self,
        results: List[Dict[str, Any]],
        date_from: Optional[str],
        date_to: Optional[str]
    ) -> List[Dict[str, Any]]:
        """按日期过滤"""
        filtered = []
        for r in results:
            doc_date = r.get("meta", {}).get("created_at")
            if not doc_date:
                continue
            
            try:
                doc_dt = datetime.fromisoformat(doc_date)
                if date_from:
                    from_dt = datetime.fromisoformat(date_from)
                    if doc_dt < from_dt:
                        continue
                if date_to:
                    to_dt = datetime.fromisoformat(date_to)
                    if doc_dt > to_dt:
                        continue
                filtered.append(r)
            except (ValueError, TypeError):
                continue
        
        return filtered
    
    def _has_tags(self, doc_tags: List[str], required_tags: List[str]) -> bool:
        """检查是否包含所需标签"""
        return any(tag in doc_tags for tag in required_tags)
    
    def adjust_weights(
        self,
        vector_scores: List[float],
        bm25_scores: List[float],
        weight_config: WeightConfig
    ) -> List[float]:
        """
        根据权重配置调整混合检索分数
        
        Args:
            vector_scores: 向量检索分数
            bm25_scores: BM25 检索分数
            weight_config: 权重配置
            
        Returns:
            混合后的分数
        """
        weight_config.validate()
        
        # 归一化分数
        vector_scores_norm = self._normalize_scores(vector_scores)
        bm25_scores_norm = self._normalize_scores(bm25_scores)
        
        # 加权融合
        hybrid_scores = [
            weight_config.vector_weight * v + weight_config.bm25_weight * b
            for v, b in zip(vector_scores_norm, bm25_scores_norm)
        ]
        
        return hybrid_scores
    
    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """归一化分数到 [0, 1]"""
        if not scores:
            return []
        
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score == min_score:
            return [1.0] * len(scores)
        
        return [(s - min_score) / (max_score - min_score) for s in scores]
    
    def aggregate_by_document(
        self,
        results: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        按文档聚合检索结果
        
        Args:
            results: 检索结果
            
        Returns:
            按文档路径聚合的结果
        """
        aggregated = defaultdict(lambda: {
            "path": "",
            "doc_type": "",
            "chunks": [],
            "max_score": 0.0,
            "avg_score": 0.0,
            "chunk_count": 0
        })
        
        for r in results:
            path = r.get("meta", {}).get("path", "unknown")
            doc_type = r.get("meta", {}).get("doc_type", "unknown")
            score = r.get("score", 0.0)
            
            agg = aggregated[path]
            agg["path"] = path
            agg["doc_type"] = doc_type
            agg["chunks"].append({
                "chunk_id": r.get("meta", {}).get("chunk_id"),
                "text": r.get("text", ""),
                "score": score,
                "page": r.get("meta", {}).get("page")
            })
            agg["max_score"] = max(agg["max_score"], score)
            agg["chunk_count"] += 1
        
        # 计算平均分数
        for path, agg in aggregated.items():
            if agg["chunk_count"] > 0:
                agg["avg_score"] = sum(c["score"] for c in agg["chunks"]) / agg["chunk_count"]
        
        return dict(aggregated)
    
    def aggregate_by_type(
        self,
        results: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        按文档类型聚合检索结果
        
        Args:
            results: 检索结果
            
        Returns:
            按文档类型聚合的结果
        """
        aggregated = defaultdict(lambda: {
            "doc_type": "",
            "documents": set(),
            "chunks": [],
            "max_score": 0.0,
            "avg_score": 0.0,
            "chunk_count": 0
        })
        
        for r in results:
            doc_type = r.get("meta", {}).get("doc_type", "unknown")
            path = r.get("meta", {}).get("path", "")
            score = r.get("score", 0.0)
            
            agg = aggregated[doc_type]
            agg["doc_type"] = doc_type
            agg["documents"].add(path)
            agg["chunks"].append({
                "path": path,
                "text": r.get("text", ""),
                "score": score
            })
            agg["max_score"] = max(agg["max_score"], score)
            agg["chunk_count"] += 1
        
        # 计算平均分数并转换 set 为 list
        result = {}
        for doc_type, agg in aggregated.items():
            if agg["chunk_count"] > 0:
                agg["avg_score"] = sum(c["score"] for c in agg["chunks"]) / agg["chunk_count"]
            agg["documents"] = list(agg["documents"])
            agg["document_count"] = len(agg["documents"])
            result[doc_type] = agg
        
        return result
    
    def get_statistics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        获取检索结果统计信息
        
        Args:
            results: 检索结果
            
        Returns:
            统计信息
        """
        if not results:
            return {
                "total_chunks": 0,
                "unique_documents": 0,
                "avg_score": 0.0,
                "max_score": 0.0,
                "min_score": 0.0,
                "doc_type_distribution": {},
                "has_tables_count": 0
            }
        
        scores = [r.get("score", 0.0) for r in results]
        paths = set(r.get("meta", {}).get("path") for r in results)
        doc_types = [r.get("meta", {}).get("doc_type", "unknown") for r in results]
        has_tables = sum(1 for r in results if r.get("meta", {}).get("has_tables", False))
        
        # 文档类型分布
        type_dist = defaultdict(int)
        for dt in doc_types:
            type_dist[dt] += 1
        
        return {
            "total_chunks": len(results),
            "unique_documents": len(paths),
            "avg_score": sum(scores) / len(scores) if scores else 0.0,
            "max_score": max(scores) if scores else 0.0,
            "min_score": min(scores) if scores else 0.0,
            "doc_type_distribution": dict(type_dist),
            "has_tables_count": has_tables
        }


def create_retriever() -> AdvancedRetriever:
    """创建高级检索器实例"""
    return AdvancedRetriever()
