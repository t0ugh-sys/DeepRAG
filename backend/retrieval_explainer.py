"""
检索结果解释模块

提供检索结果的评分解释和可视化，帮助用户理解为什么检索到这些文档
"""

import logging
from typing import List, Dict, Any
from dataclasses import dataclass
import re

logger = logging.getLogger("rag")


@dataclass
class ScoreBreakdown:
    """评分细分"""
    vector_score: float = 0.0  # 向量相似度分数
    bm25_score: float = 0.0    # BM25 分数
    rerank_score: float = 0.0  # 重排分数
    final_score: float = 0.0   # 最终分数
    
    # 解释信息
    matched_keywords: List[str] = None  # 匹配的关键词
    semantic_similarity: str = ""        # 语义相似度描述
    explanation: str = ""                # 总体解释


@dataclass
class RetrievalExplanation:
    """检索结果解释"""
    chunk_id: int
    text: str
    score_breakdown: ScoreBreakdown
    relevance_level: str  # high/medium/low
    highlight_text: str   # 高亮显示的文本
    metadata: Dict[str, Any]


class RetrievalExplainer:
    """检索结果解释器"""
    
    def __init__(self):
        """初始化解释器"""
        logger.info("RetrievalExplainer 初始化完成")
    
    def extract_keywords(self, query: str) -> List[str]:
        """
        从查询中提取关键词
        
        Args:
            query: 用户查询
        
        Returns:
            关键词列表
        """
        # 移除标点符号
        query_clean = re.sub(r'[^\w\s]', ' ', query)
        
        # 分词（简单按空格分）
        words = query_clean.split()
        
        # 过滤停用词（简化版）
        stopwords = {'的', '了', '是', '在', '有', '和', '与', '或', '等', '及', 
                     '为', '以', '到', '对', '从', '而', '但', '也', '都', '就',
                     'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'}
        
        keywords = [w for w in words if w and w.lower() not in stopwords and len(w) > 1]
        
        return keywords
    
    def find_matched_keywords(self, query: str, text: str) -> List[str]:
        """
        找出查询中在文本中出现的关键词
        
        Args:
            query: 用户查询
            text: 文档文本
        
        Returns:
            匹配的关键词列表
        """
        keywords = self.extract_keywords(query)
        text_lower = text.lower()
        
        matched = []
        for kw in keywords:
            if kw.lower() in text_lower:
                matched.append(kw)
        
        return matched
    
    def highlight_text(self, text: str, keywords: List[str], max_length: int = 300) -> str:
        """
        高亮显示文本中的关键词
        
        Args:
            text: 原文本
            keywords: 要高亮的关键词
            max_length: 最大显示长度
        
        Returns:
            高亮后的文本（使用 **keyword** 标记）
        """
        if not keywords:
            return text[:max_length] + ("..." if len(text) > max_length else "")
        
        # 找到第一个关键词的位置
        first_match_pos = len(text)
        for kw in keywords:
            pos = text.lower().find(kw.lower())
            if pos != -1 and pos < first_match_pos:
                first_match_pos = pos
        
        # 以第一个关键词为中心截取文本
        if first_match_pos < len(text):
            start = max(0, first_match_pos - 100)
            end = min(len(text), first_match_pos + 200)
            snippet = text[start:end]
            if start > 0:
                snippet = "..." + snippet
            if end < len(text):
                snippet = snippet + "..."
        else:
            snippet = text[:max_length]
        
        # 高亮关键词
        for kw in keywords:
            # 使用正则表达式进行不区分大小写的替换
            pattern = re.compile(re.escape(kw), re.IGNORECASE)
            snippet = pattern.sub(f"**{kw}**", snippet)
        
        return snippet
    
    def get_relevance_level(self, score: float) -> str:
        """
        根据分数判断相关性等级
        
        Args:
            score: 相关性分数
        
        Returns:
            相关性等级 (high/medium/low)
        """
        if score >= 0.75:
            return "high"
        elif score >= 0.5:
            return "medium"
        else:
            return "low"
    
    def get_semantic_similarity_desc(self, score: float) -> str:
        """
        获取语义相似度描述
        
        Args:
            score: 相似度分数
        
        Returns:
            描述文本
        """
        if score >= 0.9:
            return "极高相似度"
        elif score >= 0.75:
            return "高度相似"
        elif score >= 0.6:
            return "中等相似"
        elif score >= 0.4:
            return "低度相似"
        else:
            return "弱相关"
    
    def explain_retrieval(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        include_scores: bool = True
    ) -> List[RetrievalExplanation]:
        """
        解释检索结果
        
        Args:
            query: 用户查询
            retrieved_chunks: 检索到的文档片段列表
                每个片段包含: text, score, meta
            include_scores: 是否包含详细评分
        
        Returns:
            解释结果列表
        """
        explanations = []
        
        for i, chunk in enumerate(retrieved_chunks):
            text = chunk.get("text", "")
            score = chunk.get("score", 0.0)
            meta = chunk.get("meta", {})
            
            # 提取匹配的关键词
            matched_keywords = self.find_matched_keywords(query, text)
            
            # 生成高亮文本
            highlight_text = self.highlight_text(text, matched_keywords)
            
            # 评分细分
            score_breakdown = ScoreBreakdown(
                vector_score=score,  # 简化：这里只有最终分数
                final_score=score,
                matched_keywords=matched_keywords,
                semantic_similarity=self.get_semantic_similarity_desc(score)
            )
            
            # 生成解释
            explanation_parts = []
            
            if matched_keywords:
                explanation_parts.append(
                    f"匹配了 {len(matched_keywords)} 个关键词: {', '.join(matched_keywords[:5])}"
                )
            
            explanation_parts.append(
                f"语义相似度: {score_breakdown.semantic_similarity} ({score:.2f})"
            )
            
            if meta.get("page"):
                explanation_parts.append(f"来自第 {meta['page']} 页")
            
            if meta.get("has_tables"):
                explanation_parts.append("包含表格数据")
            
            score_breakdown.explanation = " | ".join(explanation_parts)
            
            # 判断相关性等级
            relevance_level = self.get_relevance_level(score)
            
            # 创建解释对象
            explanation = RetrievalExplanation(
                chunk_id=i,
                text=text,
                score_breakdown=score_breakdown,
                relevance_level=relevance_level,
                highlight_text=highlight_text,
                metadata=meta
            )
            
            explanations.append(explanation)
        
        logger.info(f"生成了 {len(explanations)} 个检索结果解释")
        return explanations
    
    def generate_summary(self, explanations: List[RetrievalExplanation]) -> Dict[str, Any]:
        """
        生成检索结果摘要
        
        Args:
            explanations: 解释列表
        
        Returns:
            摘要信息
        """
        if not explanations:
            return {
                "total_chunks": 0,
                "relevance_distribution": {},
                "avg_score": 0.0,
                "top_keywords": []
            }
        
        # 统计相关性分布
        relevance_dist = {"high": 0, "medium": 0, "low": 0}
        for exp in explanations:
            relevance_dist[exp.relevance_level] += 1
        
        # 计算平均分数
        avg_score = sum(exp.score_breakdown.final_score for exp in explanations) / len(explanations)
        
        # 统计最常见的关键词
        all_keywords = []
        for exp in explanations:
            if exp.score_breakdown.matched_keywords:
                all_keywords.extend(exp.score_breakdown.matched_keywords)
        
        from collections import Counter
        keyword_counts = Counter(all_keywords)
        top_keywords = [kw for kw, _ in keyword_counts.most_common(10)]
        
        return {
            "total_chunks": len(explanations),
            "relevance_distribution": relevance_dist,
            "avg_score": round(avg_score, 3),
            "top_keywords": top_keywords,
            "high_relevance_count": relevance_dist["high"],
            "medium_relevance_count": relevance_dist["medium"],
            "low_relevance_count": relevance_dist["low"]
        }
    
    def format_explanation_text(self, explanation: RetrievalExplanation) -> str:
        """
        格式化解释为可读文本
        
        Args:
            explanation: 解释对象
        
        Returns:
            格式化的文本
        """
        lines = []
        lines.append(f"📄 文档片段 #{explanation.chunk_id + 1}")
        lines.append(f"📊 相关性: {explanation.relevance_level.upper()} ({explanation.score_breakdown.final_score:.2f})")
        lines.append(f"💡 {explanation.score_breakdown.explanation}")
        lines.append(f"\n📝 内容预览:\n{explanation.highlight_text}")
        
        return "\n".join(lines)


def create_explainer() -> RetrievalExplainer:
    """创建检索解释器实例"""
    return RetrievalExplainer()
