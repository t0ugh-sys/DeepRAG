"""
检索优化模块

提供自动权重调优、检索质量评估、查询优化建议等功能
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from collections import defaultdict


@dataclass
class RetrievalMetrics:
    """检索质量指标"""
    avg_score: float
    max_score: float
    min_score: float
    score_variance: float
    top_k_coverage: float  # 前 K 个结果的分数覆盖度
    score_distribution: Dict[str, int]  # 分数分布


class WeightOptimizer:
    """权重优化器"""
    
    @staticmethod
    def grid_search(
        retrieval_func,
        query: str,
        top_k: int = 10,
        vector_weights: List[float] = None,
        bm25_weights: List[float] = None
    ) -> Dict[str, Any]:
        """
        网格搜索最佳权重组合
        
        Args:
            retrieval_func: 检索函数
            query: 查询文本
            top_k: 返回结果数量
            vector_weights: 向量权重候选列表
            bm25_weights: BM25 权重候选列表
        """
        if vector_weights is None:
            vector_weights = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        
        if bm25_weights is None:
            bm25_weights = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        
        results = []
        
        for vec_w in vector_weights:
            for bm25_w in bm25_weights:
                # 权重归一化
                total = vec_w + bm25_w
                if total == 0:
                    continue
                
                normalized_vec = vec_w / total
                normalized_bm25 = bm25_w / total
                
                try:
                    # 执行检索
                    retrieved = retrieval_func(
                        query=query,
                        top_k=top_k,
                        vector_weight=normalized_vec,
                        bm25_weight=normalized_bm25
                    )
                    
                    # 计算指标
                    scores = [r.get("score", 0) for r in retrieved]
                    if scores:
                        metrics = {
                            "vector_weight": round(normalized_vec, 2),
                            "bm25_weight": round(normalized_bm25, 2),
                            "avg_score": round(np.mean(scores), 4),
                            "max_score": round(max(scores), 4),
                            "min_score": round(min(scores), 4),
                            "std_score": round(np.std(scores), 4),
                            "result_count": len(retrieved)
                        }
                        results.append(metrics)
                except Exception as e:
                    continue
        
        # 按平均分数排序
        results.sort(key=lambda x: x["avg_score"], reverse=True)
        
        return {
            "best_weights": results[0] if results else None,
            "all_results": results[:10],  # 返回前 10 个最佳配置
            "total_tested": len(results)
        }
    
    @staticmethod
    def adaptive_weights(
        query: str,
        query_type: str = "auto"
    ) -> Tuple[float, float]:
        """
        根据查询类型自适应调整权重
        
        Args:
            query: 查询文本
            query_type: 查询类型 (auto/semantic/keyword/balanced)
        
        Returns:
            (vector_weight, bm25_weight)
        """
        if query_type == "semantic":
            return (0.8, 0.2)
        elif query_type == "keyword":
            return (0.3, 0.7)
        elif query_type == "balanced":
            return (0.5, 0.5)
        
        # 自动判断
        query_lower = query.lower()
        
        # 关键词查询特征
        keyword_indicators = [
            len(query.split()) <= 3,  # 短查询
            any(char.isdigit() for char in query),  # 包含数字
            '"' in query or "'" in query,  # 包含引号
            query.isupper(),  # 全大写
        ]
        
        # 语义查询特征
        semantic_indicators = [
            len(query.split()) > 5,  # 长查询
            any(word in query_lower for word in ["什么", "如何", "为什么", "怎么", "介绍", "说明"]),
            "?" in query or "？" in query,  # 问句
        ]
        
        keyword_score = sum(keyword_indicators)
        semantic_score = sum(semantic_indicators)
        
        if keyword_score > semantic_score:
            return (0.4, 0.6)  # 偏关键词
        elif semantic_score > keyword_score:
            return (0.7, 0.3)  # 偏语义
        else:
            return (0.6, 0.4)  # 默认平衡


class RetrievalAnalyzer:
    """检索结果分析器"""
    
    @staticmethod
    def analyze_results(results: List[Dict[str, Any]]) -> RetrievalMetrics:
        """分析检索结果质量"""
        if not results:
            return RetrievalMetrics(
                avg_score=0.0,
                max_score=0.0,
                min_score=0.0,
                score_variance=0.0,
                top_k_coverage=0.0,
                score_distribution={}
            )
        
        scores = [r.get("score", 0) for r in results]
        
        # 分数分布
        distribution = defaultdict(int)
        for score in scores:
            if score >= 0.8:
                distribution["excellent"] += 1
            elif score >= 0.6:
                distribution["good"] += 1
            elif score >= 0.4:
                distribution["fair"] += 1
            else:
                distribution["poor"] += 1
        
        # Top-K 覆盖度（前 K 个结果占总分的比例）
        if len(scores) > 0:
            top_k = min(5, len(scores))
            top_k_sum = sum(sorted(scores, reverse=True)[:top_k])
            total_sum = sum(scores)
            coverage = top_k_sum / total_sum if total_sum > 0 else 0
        else:
            coverage = 0
        
        return RetrievalMetrics(
            avg_score=round(float(np.mean(scores)), 4),
            max_score=round(float(max(scores)), 4),
            min_score=round(float(min(scores)), 4),
            score_variance=round(float(np.var(scores)), 4),
            top_k_coverage=round(coverage, 4),
            score_distribution=dict(distribution)
        )
    
    @staticmethod
    def suggest_improvements(
        query: str,
        results: List[Dict[str, Any]],
        metrics: RetrievalMetrics
    ) -> List[str]:
        """建议改进措施"""
        suggestions = []
        
        # 分数过低
        if metrics.avg_score < 0.5:
            suggestions.append("平均分数较低，建议：")
            suggestions.append("  - 使用查询改写（expansion/decomposition）")
            suggestions.append("  - 增加 top_k 参数")
            suggestions.append("  - 检查文档质量和分块策略")
        
        # 分数方差大
        if metrics.score_variance > 0.1:
            suggestions.append("分数分布不均，建议：")
            suggestions.append("  - 启用 Reranker 重排")
            suggestions.append("  - 调整 MMR 多样性参数")
            suggestions.append("  - 使用 min_score 过滤低分结果")
        
        # Top-K 覆盖度低
        if metrics.top_k_coverage < 0.6:
            suggestions.append("高质量结果占比低，建议：")
            suggestions.append("  - 优化文档标签和分类")
            suggestions.append("  - 使用高级检索过滤")
            suggestions.append("  - 调整混合检索权重")
        
        # 结果数量少
        if len(results) < 5:
            suggestions.append("检索结果数量少，建议：")
            suggestions.append("  - 增加文档库内容")
            suggestions.append("  - 降低 min_score 阈值")
            suggestions.append("  - 使用更宽泛的查询")
        
        # 查询特征分析
        if len(query.split()) <= 2:
            suggestions.append("查询过短，建议：")
            suggestions.append("  - 使用查询扩展策略")
            suggestions.append("  - 提供更多上下文信息")
        
        if not suggestions:
            suggestions.append("检索质量良好，可以考虑：")
            suggestions.append("  - 微调权重以进一步优化")
            suggestions.append("  - 使用聚合查看文档级相关度")
        
        return suggestions


class QueryOptimizer:
    """查询优化器"""
    
    @staticmethod
    def expand_query(query: str, expansion_terms: List[str] = None) -> str:
        """扩展查询"""
        if expansion_terms:
            return f"{query} {' '.join(expansion_terms)}"
        
        # 简单的同义词扩展（实际应该使用 LLM）
        synonyms = {
            "部署": ["安装", "配置", "启动"],
            "错误": ["异常", "问题", "bug"],
            "优化": ["改进", "提升", "增强"],
        }
        
        expanded = query
        for word, syns in synonyms.items():
            if word in query:
                expanded += " " + " ".join(syns[:2])
        
        return expanded
    
    @staticmethod
    def suggest_rewrite_strategy(query: str) -> str:
        """建议查询改写策略"""
        query_lower = query.lower()
        
        # 复杂问题 -> 分解
        if any(word in query_lower for word in ["和", "以及", "还有", "同时"]):
            if len(query.split()) > 8:
                return "decomposition"
        
        # 概念性问题 -> HyDE
        if any(word in query_lower for word in ["什么是", "介绍", "概念", "定义"]):
            return "hyde"
        
        # 简短查询 -> 扩展
        if len(query.split()) <= 3:
            return "expansion"
        
        return "none"


class RetrievalOptimizer:
    """检索优化器（整合所有优化功能）"""
    
    def __init__(self):
        self.weight_optimizer = WeightOptimizer()
        self.analyzer = RetrievalAnalyzer()
        self.query_optimizer = QueryOptimizer()
    
    def optimize(
        self,
        query: str,
        results: List[Dict[str, Any]],
        retrieval_func = None
    ) -> Dict[str, Any]:
        """
        综合优化分析
        
        Args:
            query: 查询文本
            results: 检索结果
            retrieval_func: 检索函数（用于权重优化）
        
        Returns:
            优化建议和分析结果
        """
        # 分析当前结果
        metrics = self.analyzer.analyze_results(results)
        
        # 生成改进建议
        suggestions = self.analyzer.suggest_improvements(query, results, metrics)
        
        # 推荐权重
        recommended_weights = self.weight_optimizer.adaptive_weights(query)
        
        # 推荐改写策略
        rewrite_strategy = self.query_optimizer.suggest_rewrite_strategy(query)
        
        # 权重优化（如果提供了检索函数）
        weight_optimization = None
        if retrieval_func:
            try:
                weight_optimization = self.weight_optimizer.grid_search(
                    retrieval_func,
                    query,
                    top_k=10
                )
            except Exception as e:
                pass
        
        return {
            "query": query,
            "metrics": {
                "avg_score": metrics.avg_score,
                "max_score": metrics.max_score,
                "min_score": metrics.min_score,
                "score_variance": metrics.score_variance,
                "top_k_coverage": metrics.top_k_coverage,
                "score_distribution": metrics.score_distribution
            },
            "suggestions": suggestions,
            "recommended_weights": {
                "vector_weight": recommended_weights[0],
                "bm25_weight": recommended_weights[1]
            },
            "recommended_rewrite_strategy": rewrite_strategy,
            "weight_optimization": weight_optimization
        }


# 全局实例
_global_optimizer: Optional[RetrievalOptimizer] = None


def get_optimizer() -> RetrievalOptimizer:
    """获取全局优化器实例"""
    global _global_optimizer
    if _global_optimizer is None:
        _global_optimizer = RetrievalOptimizer()
    return _global_optimizer
