"""
RAG 系统评估框架

提供自动化测试、质量评估、基准测试等功能
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import time
import json
from pathlib import Path
import numpy as np


@dataclass
class EvaluationResult:
    """评估结果"""
    metric_name: str
    score: float
    details: Dict[str, Any]


@dataclass
class TestCase:
    """测试用例"""
    question: str
    expected_answer: Optional[str] = None
    expected_keywords: Optional[List[str]] = None
    expected_doc_paths: Optional[List[str]] = None
    category: str = "general"


class RetrievalMetrics:
    """检索指标计算"""
    
    @staticmethod
    def precision_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
        """Precision@K"""
        if not retrieved or not relevant:
            return 0.0
        
        retrieved_k = retrieved[:k]
        relevant_set = set(relevant)
        
        hits = sum(1 for doc in retrieved_k if doc in relevant_set)
        return hits / k if k > 0 else 0.0
    
    @staticmethod
    def recall_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
        """Recall@K"""
        if not retrieved or not relevant:
            return 0.0
        
        retrieved_k = retrieved[:k]
        relevant_set = set(relevant)
        
        hits = sum(1 for doc in retrieved_k if doc in relevant_set)
        return hits / len(relevant_set) if relevant_set else 0.0
    
    @staticmethod
    def mrr(retrieved_list: List[List[str]], relevant_list: List[List[str]]) -> float:
        """Mean Reciprocal Rank"""
        reciprocal_ranks = []
        
        for retrieved, relevant in zip(retrieved_list, relevant_list):
            relevant_set = set(relevant)
            rank = 0
            
            for i, doc in enumerate(retrieved, 1):
                if doc in relevant_set:
                    rank = i
                    break
            
            reciprocal_ranks.append(1.0 / rank if rank > 0 else 0.0)
        
        return np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0
    
    @staticmethod
    def ndcg_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
        """Normalized Discounted Cumulative Gain@K"""
        if not retrieved or not relevant:
            return 0.0
        
        retrieved_k = retrieved[:k]
        relevant_set = set(relevant)
        
        # DCG
        dcg = 0.0
        for i, doc in enumerate(retrieved_k, 1):
            if doc in relevant_set:
                dcg += 1.0 / np.log2(i + 1)
        
        # IDCG
        idcg = sum(1.0 / np.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
        
        return dcg / idcg if idcg > 0 else 0.0
    
    @staticmethod
    def hit_rate(retrieved_list: List[List[str]], relevant_list: List[List[str]]) -> float:
        """Hit Rate (至少命中一个相关文档的比例)"""
        hits = 0
        
        for retrieved, relevant in zip(retrieved_list, relevant_list):
            relevant_set = set(relevant)
            if any(doc in relevant_set for doc in retrieved):
                hits += 1
        
        return hits / len(retrieved_list) if retrieved_list else 0.0


class AnswerQualityMetrics:
    """答案质量指标"""
    
    @staticmethod
    def keyword_coverage(answer: str, keywords: List[str]) -> float:
        """关键词覆盖率"""
        if not keywords:
            return 1.0
        
        answer_lower = answer.lower()
        covered = sum(1 for kw in keywords if kw.lower() in answer_lower)
        
        return covered / len(keywords)
    
    @staticmethod
    def answer_length_score(answer: str, min_length: int = 50, max_length: int = 500) -> float:
        """答案长度评分"""
        length = len(answer)
        
        if length < min_length:
            return length / min_length
        elif length > max_length:
            return max(0.5, 1.0 - (length - max_length) / max_length)
        else:
            return 1.0
    
    @staticmethod
    def has_citation(answer: str) -> bool:
        """是否包含引用"""
        citation_patterns = ["来源", "参考", "引用", "文档", "根据"]
        return any(pattern in answer for pattern in citation_patterns)


class PerformanceMetrics:
    """性能指标"""
    
    @staticmethod
    def measure_latency(func, *args, **kwargs) -> Tuple[Any, float]:
        """测量延迟"""
        start = time.time()
        result = func(*args, **kwargs)
        latency = time.time() - start
        return result, latency
    
    @staticmethod
    def calculate_throughput(total_requests: int, total_time: float) -> float:
        """计算吞吐量（请求/秒）"""
        return total_requests / total_time if total_time > 0 else 0.0


class RAGEvaluator:
    """RAG 系统评估器"""
    
    def __init__(self, rag_pipeline):
        self.pipeline = rag_pipeline
        self.retrieval_metrics = RetrievalMetrics()
        self.answer_metrics = AnswerQualityMetrics()
        self.performance_metrics = PerformanceMetrics()
    
    def evaluate_retrieval(
        self,
        test_cases: List[TestCase],
        top_k: int = 10
    ) -> Dict[str, Any]:
        """评估检索质量"""
        all_retrieved = []
        all_relevant = []
        latencies = []
        
        for test_case in test_cases:
            # 执行检索
            results, latency = self.performance_metrics.measure_latency(
                self.pipeline.vector_store.search,
                test_case.question,
                top_k=top_k
            )
            
            latencies.append(latency)
            
            # 提取文档路径
            retrieved_docs = [r.meta.get("path", "") for r in results]
            relevant_docs = test_case.expected_doc_paths or []
            
            all_retrieved.append(retrieved_docs)
            all_relevant.append(relevant_docs)
        
        # 计算指标
        precision_scores = [
            self.retrieval_metrics.precision_at_k(ret, rel, top_k)
            for ret, rel in zip(all_retrieved, all_relevant)
        ]
        
        recall_scores = [
            self.retrieval_metrics.recall_at_k(ret, rel, top_k)
            for ret, rel in zip(all_retrieved, all_relevant)
        ]
        
        ndcg_scores = [
            self.retrieval_metrics.ndcg_at_k(ret, rel, top_k)
            for ret, rel in zip(all_retrieved, all_relevant)
        ]
        
        return {
            "precision_at_k": round(np.mean(precision_scores), 4),
            "recall_at_k": round(np.mean(recall_scores), 4),
            "ndcg_at_k": round(np.mean(ndcg_scores), 4),
            "mrr": round(self.retrieval_metrics.mrr(all_retrieved, all_relevant), 4),
            "hit_rate": round(self.retrieval_metrics.hit_rate(all_retrieved, all_relevant), 4),
            "avg_latency": round(np.mean(latencies), 4),
            "p95_latency": round(np.percentile(latencies, 95), 4),
            "total_queries": len(test_cases)
        }
    
    def evaluate_answer_quality(
        self,
        test_cases: List[TestCase]
    ) -> Dict[str, Any]:
        """评估答案质量"""
        keyword_scores = []
        length_scores = []
        citation_count = 0
        latencies = []
        
        for test_case in test_cases:
            # 生成答案
            answer, latency = self.performance_metrics.measure_latency(
                self.pipeline.ask,
                test_case.question
            )
            
            latencies.append(latency)
            
            # 关键词覆盖
            if test_case.expected_keywords:
                kw_score = self.answer_metrics.keyword_coverage(
                    answer,
                    test_case.expected_keywords
                )
                keyword_scores.append(kw_score)
            
            # 长度评分
            length_score = self.answer_metrics.answer_length_score(answer)
            length_scores.append(length_score)
            
            # 引用检查
            if self.answer_metrics.has_citation(answer):
                citation_count += 1
        
        return {
            "avg_keyword_coverage": round(np.mean(keyword_scores), 4) if keyword_scores else None,
            "avg_length_score": round(np.mean(length_scores), 4),
            "citation_rate": round(citation_count / len(test_cases), 4),
            "avg_latency": round(np.mean(latencies), 4),
            "p95_latency": round(np.percentile(latencies, 95), 4),
            "total_queries": len(test_cases)
        }
    
    def run_benchmark(
        self,
        test_cases: List[TestCase],
        top_k: int = 10
    ) -> Dict[str, Any]:
        """运行完整基准测试"""
        print(f"开始评估 {len(test_cases)} 个测试用例...")
        
        # 检索评估
        print("评估检索质量...")
        retrieval_results = self.evaluate_retrieval(test_cases, top_k)
        
        # 答案质量评估
        print("评估答案质量...")
        answer_results = self.evaluate_answer_quality(test_cases)
        
        # 按类别统计
        category_stats = {}
        for test_case in test_cases:
            category = test_case.category
            category_stats[category] = category_stats.get(category, 0) + 1
        
        return {
            "retrieval_metrics": retrieval_results,
            "answer_quality_metrics": answer_results,
            "category_distribution": category_stats,
            "total_test_cases": len(test_cases),
            "timestamp": time.time()
        }


class TestCaseGenerator:
    """测试用例生成器"""
    
    @staticmethod
    def load_from_file(filepath: str) -> List[TestCase]:
        """从文件加载测试用例"""
        test_cases = []
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for item in data:
            test_cases.append(TestCase(
                question=item["question"],
                expected_answer=item.get("expected_answer"),
                expected_keywords=item.get("expected_keywords"),
                expected_doc_paths=item.get("expected_doc_paths"),
                category=item.get("category", "general")
            ))
        
        return test_cases
    
    @staticmethod
    def save_to_file(test_cases: List[TestCase], filepath: str):
        """保存测试用例到文件"""
        data = []
        
        for tc in test_cases:
            data.append({
                "question": tc.question,
                "expected_answer": tc.expected_answer,
                "expected_keywords": tc.expected_keywords,
                "expected_doc_paths": tc.expected_doc_paths,
                "category": tc.category
            })
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    @staticmethod
    def generate_basic_tests() -> List[TestCase]:
        """生成基础测试用例"""
        return [
            TestCase(
                question="什么是 RAG",
                expected_keywords=["检索", "增强", "生成"],
                category="definition"
            ),
            TestCase(
                question="如何部署系统",
                expected_keywords=["docker", "部署", "配置"],
                category="how_to"
            ),
            TestCase(
                question="API 配置参数",
                expected_keywords=["API", "配置", "参数"],
                category="factual"
            ),
        ]


class BenchmarkReport:
    """基准测试报告"""
    
    @staticmethod
    def generate_report(results: Dict[str, Any]) -> str:
        """生成报告"""
        report = []
        report.append("=" * 60)
        report.append("RAG 系统评估报告")
        report.append("=" * 60)
        report.append("")
        
        # 检索指标
        report.append("## 检索质量指标")
        report.append("-" * 60)
        retrieval = results["retrieval_metrics"]
        report.append(f"Precision@K:  {retrieval['precision_at_k']:.4f}")
        report.append(f"Recall@K:     {retrieval['recall_at_k']:.4f}")
        report.append(f"NDCG@K:       {retrieval['ndcg_at_k']:.4f}")
        report.append(f"MRR:          {retrieval['mrr']:.4f}")
        report.append(f"Hit Rate:     {retrieval['hit_rate']:.4f}")
        report.append(f"平均延迟:     {retrieval['avg_latency']:.4f}s")
        report.append(f"P95 延迟:     {retrieval['p95_latency']:.4f}s")
        report.append("")
        
        # 答案质量
        report.append("## 答案质量指标")
        report.append("-" * 60)
        answer = results["answer_quality_metrics"]
        if answer.get("avg_keyword_coverage"):
            report.append(f"关键词覆盖:   {answer['avg_keyword_coverage']:.4f}")
        report.append(f"长度评分:     {answer['avg_length_score']:.4f}")
        report.append(f"引用率:       {answer['citation_rate']:.4f}")
        report.append(f"平均延迟:     {answer['avg_latency']:.4f}s")
        report.append(f"P95 延迟:     {answer['p95_latency']:.4f}s")
        report.append("")
        
        # 类别分布
        report.append("## 测试用例分布")
        report.append("-" * 60)
        for category, count in results["category_distribution"].items():
            report.append(f"{category}: {count}")
        report.append("")
        
        report.append(f"总测试用例数: {results['total_test_cases']}")
        report.append("=" * 60)
        
        return "\n".join(report)
    
    @staticmethod
    def save_report(results: Dict[str, Any], filepath: str):
        """保存报告"""
        report = BenchmarkReport.generate_report(results)
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(report)
        
        # 同时保存 JSON
        json_path = filepath.replace('.txt', '.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)


# 全局评估器实例
_global_evaluator: Optional[RAGEvaluator] = None


def get_evaluator(rag_pipeline) -> RAGEvaluator:
    """获取全局评估器实例"""
    global _global_evaluator
    if _global_evaluator is None:
        _global_evaluator = RAGEvaluator(rag_pipeline)
    return _global_evaluator
