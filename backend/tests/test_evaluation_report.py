from backend.evaluation import BenchmarkReport


def test_generate_report_smoke() -> None:
    results = {
        "retrieval_metrics": {
            "precision_at_k": 0.1,
            "recall_at_k": 0.2,
            "ndcg_at_k": 0.3,
            "mrr": 0.4,
            "hit_rate": 0.5,
            "avg_latency": 0.01,
            "p95_latency": 0.02,
            "total_queries": 1,
        },
        "answer_quality_metrics": {
            "avg_keyword_coverage": None,
            "avg_length_score": 0.9,
            "citation_rate": 0.0,
            "avg_latency": 0.01,
            "p95_latency": 0.02,
            "total_queries": 1,
        },
        "category_distribution": {"general": 1},
        "total_test_cases": 1,
        "timestamp": 0.0,
    }

    report = BenchmarkReport.generate_report(results)
    assert "RAG 系统评估报告" in report
    assert "Precision@K" in report
