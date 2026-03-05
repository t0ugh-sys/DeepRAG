from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict

from backend.config import Settings
from backend.evaluation import BenchmarkReport, TestCaseGenerator
from backend.rag import RAGPipeline


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DeepRAG evaluation CLI")
    parser.add_argument("--cases", type=str, required=True, help="Path to JSON test cases file")
    parser.add_argument("--namespace", type=str, default=None, help="Namespace (default: from env RAG_NAMESPACE)")
    parser.add_argument("--top_k", type=int, default=10, help="Top-K for retrieval evaluation")
    parser.add_argument(
        "--mode",
        type=str,
        default="benchmark",
        choices=["retrieval", "answer", "benchmark"],
        help="Evaluation mode",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="data/evaluation/report.txt",
        help="Output report path (.txt). JSON will be saved next to it.",
    )
    return parser.parse_args()


def _safe_write_json(path: str, data: Dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def main() -> None:
    args = _parse_args()
    settings = Settings()
    namespace = args.namespace or settings.default_namespace
    pipeline = RAGPipeline(settings, namespace)

    cases = TestCaseGenerator.load_from_file(args.cases)
    evaluator = __import__("backend.evaluation", fromlist=["get_evaluator"]).get_evaluator(pipeline)

    results: Dict[str, Any]
    if args.mode == "retrieval":
        results = {"retrieval_metrics": evaluator.evaluate_retrieval(cases, top_k=args.top_k)}
    elif args.mode == "answer":
        results = {"answer_quality_metrics": evaluator.evaluate_answer_quality(cases)}
    else:
        results = evaluator.run_benchmark(cases, top_k=args.top_k)

    out_path = str(args.out)
    BenchmarkReport.save_report(results, out_path)

    # Also dump raw json to explicit path for easy diffing.
    json_out = out_path[:-4] + ".json" if out_path.lower().endswith(".txt") else out_path + ".json"
    _safe_write_json(json_out, results)

    print(f"OK: saved report to {out_path}")
    print(f"OK: saved json to {json_out}")


if __name__ == "__main__":
    # Avoid accidental proxy influence; align with pipeline httpx trust_env=False usage.
    os.environ.setdefault("HTTP_PROXY", "")
    os.environ.setdefault("HTTPS_PROXY", "")
    main()

