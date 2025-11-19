"""
查询改写模块

提供多种查询优化策略：
1. 查询扩展 - 生成同义词和相关词
2. 查询分解 - 将复杂查询拆分为子查询
3. HyDE - 假设性文档嵌入
"""

import logging
from typing import List, Dict, Any
from openai import OpenAI

logger = logging.getLogger("rag")


class QueryRewriter:
    """查询改写器，使用 LLM 优化用户查询"""
    
    def __init__(self, api_key: str, base_url: str, model: str = "gpt-3.5-turbo"):
        """
        初始化查询改写器
        
        Args:
            api_key: OpenAI API Key
            base_url: API Base URL
            model: 使用的模型
        """
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        logger.info(f"QueryRewriter 初始化完成，模型: {model}")
    
    def expand_query(self, query: str, language: str = "zh") -> List[str]:
        """
        查询扩展：生成查询的多个变体
        
        Args:
            query: 原始查询
            language: 语言 (zh/en)
        
        Returns:
            扩展后的查询列表（包含原查询）
        """
        prompt = f"""你是一个查询优化助手。请将用户的查询扩展为3-5个语义相似但表达不同的查询变体。

要求：
1. 保持原意不变
2. 使用不同的表达方式
3. 添加同义词和相关词
4. 每行一个查询
5. 不要添加编号或其他标记

用户查询：{query}

扩展查询："""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=300
            )
            
            expanded = response.choices[0].message.content.strip()
            queries = [q.strip() for q in expanded.split('\n') if q.strip()]
            
            # 确保原查询在列表中
            if query not in queries:
                queries.insert(0, query)
            
            logger.info(f"查询扩展: {query} -> {len(queries)} 个变体")
            return queries[:5]  # 最多返回5个
            
        except Exception as e:
            logger.error(f"查询扩展失败: {e}")
            return [query]  # 失败时返回原查询
    
    def decompose_query(self, query: str) -> List[str]:
        """
        查询分解：将复杂查询拆分为多个子查询
        
        适用于包含多个问题或需要多步推理的查询
        
        Args:
            query: 原始查询
        
        Returns:
            子查询列表
        """
        prompt = f"""你是一个查询分析助手。请分析用户的查询，如果它包含多个问题或需要多步推理，请将其分解为2-4个简单的子查询。

要求：
1. 每个子查询应该独立且明确
2. 按逻辑顺序排列
3. 每行一个子查询
4. 如果查询已经足够简单，只返回原查询
5. 不要添加编号或其他标记

用户查询：{query}

子查询："""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=300
            )
            
            decomposed = response.choices[0].message.content.strip()
            sub_queries = [q.strip() for q in decomposed.split('\n') if q.strip()]
            
            logger.info(f"查询分解: {query} -> {len(sub_queries)} 个子查询")
            return sub_queries if sub_queries else [query]
            
        except Exception as e:
            logger.error(f"查询分解失败: {e}")
            return [query]
    
    def generate_hypothetical_document(self, query: str) -> str:
        """
        HyDE (Hypothetical Document Embeddings)
        
        生成一个假设性的文档片段来回答查询，
        然后使用这个假设文档进行检索（而不是原查询）
        
        Args:
            query: 原始查询
        
        Returns:
            假设性文档内容
        """
        prompt = f"""你是一个知识库专家。请根据用户的问题，生成一个假设性的、详细的答案段落（150-200字）。

要求：
1. 答案应该专业、准确
2. 包含相关的技术术语和概念
3. 使用陈述句，不要使用疑问句
4. 不要说"我认为"、"可能"等不确定的词
5. 直接给出答案内容，不要有前缀

用户问题：{query}

假设答案："""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
                max_tokens=400
            )
            
            hypothetical_doc = response.choices[0].message.content.strip()
            logger.info(f"HyDE 生成: {query} -> {len(hypothetical_doc)} 字符")
            return hypothetical_doc
            
        except Exception as e:
            logger.error(f"HyDE 生成失败: {e}")
            return query  # 失败时返回原查询
    
    def rewrite_for_retrieval(self, query: str, strategy: str = "expand") -> List[str]:
        """
        根据策略改写查询
        
        Args:
            query: 原始查询
            strategy: 改写策略
                - "expand": 查询扩展（默认）
                - "decompose": 查询分解
                - "hyde": 假设性文档
                - "multi": 组合策略（扩展+分解）
        
        Returns:
            改写后的查询列表
        """
        if strategy == "expand":
            return self.expand_query(query)
        
        elif strategy == "decompose":
            return self.decompose_query(query)
        
        elif strategy == "hyde":
            hyde_doc = self.generate_hypothetical_document(query)
            return [hyde_doc]
        
        elif strategy == "multi":
            # 组合策略：先分解，再对每个子查询扩展
            sub_queries = self.decompose_query(query)
            all_queries = []
            for sq in sub_queries[:2]:  # 只对前2个子查询扩展
                expanded = self.expand_query(sq)
                all_queries.extend(expanded[:2])  # 每个子查询取2个变体
            return list(set(all_queries))[:5]  # 去重并限制数量
        
        else:
            logger.warning(f"未知的改写策略: {strategy}，使用原查询")
            return [query]
    
    def analyze_query(self, query: str) -> Dict[str, Any]:
        """
        分析查询特征，推荐最佳改写策略
        
        Args:
            query: 原始查询
        
        Returns:
            分析结果，包含推荐策略
        """
        prompt = f"""分析以下查询的特征，并以JSON格式返回分析结果。

查询：{query}

请分析：
1. complexity: 复杂度 (simple/medium/complex)
2. type: 查询类型 (factual/comparison/how-to/definition/multi-part)
3. recommended_strategy: 推荐策略 (expand/decompose/hyde/multi)
4. reason: 推荐理由

只返回JSON，不要其他内容：
```json
{{
  "complexity": "...",
  "type": "...",
  "recommended_strategy": "...",
  "reason": "..."
}}
```"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=200
            )
            
            import json
            result_text = response.choices[0].message.content.strip()
            # 提取JSON部分
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0].strip()
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0].strip()
            
            analysis = json.loads(result_text)
            logger.info(f"查询分析: {query} -> {analysis['recommended_strategy']}")
            return analysis
            
        except Exception as e:
            logger.error(f"查询分析失败: {e}")
            return {
                "complexity": "simple",
                "type": "factual",
                "recommended_strategy": "expand",
                "reason": "分析失败，使用默认策略"
            }


def create_query_rewriter(api_key: str, base_url: str, model: str = "gpt-3.5-turbo") -> QueryRewriter:
    """
    创建查询改写器实例
    
    Args:
        api_key: OpenAI API Key
        base_url: API Base URL
        model: 使用的模型
    
    Returns:
        QueryRewriter 实例
    """
    return QueryRewriter(api_key=api_key, base_url=base_url, model=model)
