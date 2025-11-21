"""
查询意图识别模块

识别用户查询的意图类型，用于优化检索策略
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import re


@dataclass
class QueryIntent:
    """查询意图"""
    intent_type: str  # 意图类型
    confidence: float  # 置信度
    keywords: List[str]  # 关键词
    suggested_strategy: str  # 建议的检索策略
    explanation: str  # 解释


class IntentRecognizer:
    """意图识别器"""
    
    # 意图模式定义
    INTENT_PATTERNS = {
        "definition": {
            "patterns": [
                r"什么是",
                r"定义",
                r"概念",
                r"介绍.*是",
                r"解释.*是",
                r".*是什么",
            ],
            "keywords": ["定义", "概念", "介绍", "解释"],
            "strategy": "hyde",
            "explanation": "定义类问题，使用 HyDE 生成假设性文档"
        },
        "how_to": {
            "patterns": [
                r"如何",
                r"怎么",
                r"怎样",
                r"如何.*操作",
                r"怎么.*使用",
                r"步骤",
                r"方法",
            ],
            "keywords": ["如何", "怎么", "步骤", "方法", "操作"],
            "strategy": "decomposition",
            "explanation": "操作类问题，使用查询分解拆分步骤"
        },
        "comparison": {
            "patterns": [
                r"比较",
                r"对比",
                r"区别",
                r"差异",
                r".*和.*的区别",
                r".*vs.*",
                r"哪个更好",
            ],
            "keywords": ["比较", "对比", "区别", "差异"],
            "strategy": "expansion",
            "explanation": "比较类问题，使用查询扩展增加相关概念"
        },
        "troubleshooting": {
            "patterns": [
                r"错误",
                r"异常",
                r"问题",
                r"失败",
                r"报错",
                r"bug",
                r"无法",
                r"不能",
                r"为什么.*不",
            ],
            "keywords": ["错误", "异常", "问题", "失败", "bug"],
            "strategy": "expansion",
            "explanation": "故障排查类问题，使用查询扩展增加错误相关词"
        },
        "factual": {
            "patterns": [
                r"^[A-Z0-9_-]+$",  # 全大写或代码
                r"API",
                r"配置",
                r"参数",
                r"版本",
                r"端口",
                r"路径",
            ],
            "keywords": ["API", "配置", "参数", "版本"],
            "strategy": "keyword",
            "explanation": "事实类问题，使用关键词匹配"
        },
        "exploratory": {
            "patterns": [
                r"有哪些",
                r"包括",
                r"列举",
                r"所有",
                r"全部",
                r"都有什么",
            ],
            "keywords": ["有哪些", "包括", "列举", "所有"],
            "strategy": "expansion",
            "explanation": "探索类问题，使用查询扩展增加覆盖面"
        },
        "reason": {
            "patterns": [
                r"为什么",
                r"原因",
                r"为何",
                r"怎么回事",
                r"导致",
            ],
            "keywords": ["为什么", "原因", "为何"],
            "strategy": "decomposition",
            "explanation": "原因分析类问题，使用查询分解多角度分析"
        },
    }
    
    @classmethod
    def recognize(cls, query: str) -> QueryIntent:
        """识别查询意图"""
        query_lower = query.lower()
        
        # 计算每个意图的匹配分数
        scores = {}
        for intent_type, config in cls.INTENT_PATTERNS.items():
            score = 0
            matched_keywords = []
            
            # 模式匹配
            for pattern in config["patterns"]:
                if re.search(pattern, query_lower):
                    score += 2
            
            # 关键词匹配
            for keyword in config["keywords"]:
                if keyword in query_lower:
                    score += 1
                    matched_keywords.append(keyword)
            
            if score > 0:
                scores[intent_type] = {
                    "score": score,
                    "keywords": matched_keywords,
                    "strategy": config["strategy"],
                    "explanation": config["explanation"]
                }
        
        # 选择得分最高的意图
        if scores:
            best_intent = max(scores.items(), key=lambda x: x[1]["score"])
            intent_type = best_intent[0]
            intent_data = best_intent[1]
            
            # 计算置信度
            max_possible_score = len(cls.INTENT_PATTERNS[intent_type]["patterns"]) * 2 + \
                                len(cls.INTENT_PATTERNS[intent_type]["keywords"])
            confidence = min(intent_data["score"] / max_possible_score, 1.0)
            
            return QueryIntent(
                intent_type=intent_type,
                confidence=round(confidence, 2),
                keywords=intent_data["keywords"],
                suggested_strategy=intent_data["strategy"],
                explanation=intent_data["explanation"]
            )
        
        # 默认意图
        return QueryIntent(
            intent_type="general",
            confidence=0.5,
            keywords=[],
            suggested_strategy="none",
            explanation="通用查询，使用默认检索策略"
        )
    
    @classmethod
    def get_retrieval_config(cls, intent: QueryIntent) -> Dict[str, Any]:
        """根据意图获取推荐的检索配置"""
        configs = {
            "definition": {
                "vector_weight": 0.8,
                "bm25_weight": 0.2,
                "top_k": 8,
                "reranker_enabled": True,
                "rewrite_strategy": "hyde"
            },
            "how_to": {
                "vector_weight": 0.6,
                "bm25_weight": 0.4,
                "top_k": 12,
                "reranker_enabled": True,
                "rewrite_strategy": "decomposition"
            },
            "comparison": {
                "vector_weight": 0.7,
                "bm25_weight": 0.3,
                "top_k": 15,
                "reranker_enabled": True,
                "rewrite_strategy": "expansion"
            },
            "troubleshooting": {
                "vector_weight": 0.5,
                "bm25_weight": 0.5,
                "top_k": 15,
                "reranker_enabled": True,
                "rewrite_strategy": "expansion"
            },
            "factual": {
                "vector_weight": 0.3,
                "bm25_weight": 0.7,
                "top_k": 10,
                "reranker_enabled": False,
                "rewrite_strategy": "none"
            },
            "exploratory": {
                "vector_weight": 0.6,
                "bm25_weight": 0.4,
                "top_k": 20,
                "reranker_enabled": True,
                "rewrite_strategy": "expansion"
            },
            "reason": {
                "vector_weight": 0.7,
                "bm25_weight": 0.3,
                "top_k": 12,
                "reranker_enabled": True,
                "rewrite_strategy": "decomposition"
            },
            "general": {
                "vector_weight": 0.6,
                "bm25_weight": 0.4,
                "top_k": 10,
                "reranker_enabled": False,
                "rewrite_strategy": "none"
            }
        }
        
        return configs.get(intent.intent_type, configs["general"])


class QueryAnalyzer:
    """查询分析器"""
    
    @staticmethod
    def analyze(query: str) -> Dict[str, Any]:
        """全面分析查询"""
        # 基本特征
        word_count = len(query.split())
        char_count = len(query)
        has_question_mark = "?" in query or "？" in query
        has_numbers = any(char.isdigit() for char in query)
        has_english = any(char.isalpha() and ord(char) < 128 for char in query)
        
        # 意图识别
        intent = IntentRecognizer.recognize(query)
        
        # 复杂度评估
        complexity = "simple"
        if word_count > 15:
            complexity = "complex"
        elif word_count > 8:
            complexity = "medium"
        
        # 查询类型
        query_type = "semantic"
        if word_count <= 3 or has_numbers:
            query_type = "keyword"
        elif word_count > 10:
            query_type = "semantic"
        else:
            query_type = "balanced"
        
        return {
            "query": query,
            "features": {
                "word_count": word_count,
                "char_count": char_count,
                "has_question_mark": has_question_mark,
                "has_numbers": has_numbers,
                "has_english": has_english,
                "complexity": complexity,
                "query_type": query_type
            },
            "intent": {
                "type": intent.intent_type,
                "confidence": intent.confidence,
                "keywords": intent.keywords,
                "explanation": intent.explanation
            },
            "recommended_config": IntentRecognizer.get_retrieval_config(intent),
            "suggestions": _generate_suggestions(query, intent, complexity)
        }


def _generate_suggestions(query: str, intent: QueryIntent, complexity: str) -> List[str]:
    """生成优化建议"""
    suggestions = []
    
    # 基于意图的建议
    if intent.intent_type == "definition":
        suggestions.append("使用 HyDE 策略生成假设性文档以提高召回率")
    elif intent.intent_type == "how_to":
        suggestions.append("使用查询分解将操作步骤拆分为多个子查询")
    elif intent.intent_type == "troubleshooting":
        suggestions.append("使用查询扩展添加错误相关的同义词")
        suggestions.append("考虑搜索日志和错误文档")
    elif intent.intent_type == "factual":
        suggestions.append("提高 BM25 权重以增强关键词匹配")
        suggestions.append("使用精确路径或标签过滤")
    
    # 基于复杂度的建议
    if complexity == "simple":
        suggestions.append("查询较短，建议使用查询扩展增加相关词")
    elif complexity == "complex":
        suggestions.append("查询较复杂，建议使用查询分解简化")
        suggestions.append("增加 top_k 以获取更多候选结果")
    
    # 通用建议
    if intent.confidence < 0.6:
        suggestions.append("意图识别置信度较低，建议明确查询目的")
    
    return suggestions


# 全局实例
_global_analyzer: Optional[QueryAnalyzer] = None


def get_query_analyzer() -> QueryAnalyzer:
    """获取全局查询分析器实例"""
    global _global_analyzer
    if _global_analyzer is None:
        _global_analyzer = QueryAnalyzer()
    return _global_analyzer
