from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from backend.config import Settings
from backend.evaluation import RAGEvaluator, TestCaseGenerator
from backend.rag import RAGPipeline


@dataclass(frozen=True)
class MatrixCase:
    name: str
    bm25_enabled: bool
    vec_weight: float
    bm25_weight: float
    mmr_lambda: float
    reranker_enabled: bool
    reranker_top_n: int


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='DeepRAG matrix evaluation CLI')
    parser.add_argument('--cases', type=str, required=True, help='Path to JSON test cases file')
    parser.add_argument('--matrix', type=str, required=True, help='Path to matrix config JSON file')
    parser.add_argument('--namespace', type=str, default=None, help='Namespace (default: from env RAG_NAMESPACE)')
    parser.add_argument('--top_k', type=int, default=10, help='Top-K for retrieval evaluation')
    parser.add_argument(
        '--mode',
        type=str,
        default='retrieval',
        choices=['retrieval', 'benchmark'],
        help='retrieval: only retrieval metrics; benchmark: retrieval + answer quality',
    )
    parser.add_argument(
        '--out',
        type=str,
        default='data/evaluation/matrix_report.json',
        help='Output JSON report path',
    )
    return parser.parse_args()


def _normalize_case(raw: dict[str, Any], idx: int) -> MatrixCase:
    name = str(raw.get('name') or f'case_{idx + 1}')
    bm25_enabled = bool(raw.get('bm25_enabled', True))
    vec_weight = float(raw.get('vec_weight', 0.6))
    bm25_weight = float(raw.get('bm25_weight', 0.4))
    mmr_lambda = float(raw.get('mmr_lambda', 0.7))
    reranker_enabled = bool(raw.get('reranker_enabled', False))
    reranker_top_n = int(raw.get('reranker_top_n', 4))

    if not bm25_enabled:
        vec_weight = 1.0
        bm25_weight = 0.0
    else:
        total = vec_weight + bm25_weight
        if total <= 0:
            vec_weight = 0.6
            bm25_weight = 0.4
        elif abs(total - 1.0) > 1e-9:
            vec_weight = vec_weight / total
            bm25_weight = bm25_weight / total

    mmr_lambda = min(max(mmr_lambda, 0.0), 1.0)
    reranker_top_n = max(1, reranker_top_n)
    return MatrixCase(
        name=name,
        bm25_enabled=bm25_enabled,
        vec_weight=vec_weight,
        bm25_weight=bm25_weight,
        mmr_lambda=mmr_lambda,
        reranker_enabled=reranker_enabled,
        reranker_top_n=reranker_top_n,
    )


def _load_matrix(path: str) -> list[MatrixCase]:
    with Path(path).open('r', encoding='utf-8') as fp:
        raw = json.load(fp)
    if not isinstance(raw, list) or not raw:
        raise ValueError('matrix JSON must be a non-empty list')
    return [_normalize_case(item, idx) for idx, item in enumerate(raw)]


def _run_case(case: MatrixCase, namespace: str, top_k: int, mode: str, cases_path: str) -> dict[str, Any]:
    settings = Settings()
    settings.bm25_enabled = case.bm25_enabled
    settings.vec_weight = case.vec_weight
    settings.bm25_weight = case.bm25_weight
    settings.mmr_lambda = case.mmr_lambda
    settings.reranker_enabled = case.reranker_enabled
    settings.reranker_top_n = case.reranker_top_n

    pipeline = RAGPipeline(settings, namespace)
    evaluator = RAGEvaluator(pipeline)
    test_cases = TestCaseGenerator.load_from_file(cases_path)

    if mode == 'benchmark':
        metrics = evaluator.run_benchmark(test_cases, top_k=top_k)
    else:
        metrics = {'retrieval_metrics': evaluator.evaluate_retrieval(test_cases, top_k=top_k)}

    result = {
        'config': asdict(case),
        'metrics': metrics,
    }
    retrieval_metrics = metrics.get('retrieval_metrics', {})
    result['summary'] = {
        'mrr': retrieval_metrics.get('mrr'),
        'ndcg_at_k': retrieval_metrics.get('ndcg_at_k'),
        'hit_rate': retrieval_metrics.get('hit_rate'),
        'avg_latency': retrieval_metrics.get('avg_latency'),
    }
    return result


def _save_json(path: str, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as fp:
        json.dump(payload, fp, indent=2, ensure_ascii=False)


def main() -> None:
    args = _parse_args()
    settings = Settings()
    namespace = args.namespace or settings.default_namespace or 'default'
    matrix = _load_matrix(args.matrix)

    results: list[dict[str, Any]] = []
    for case in matrix:
        print(f'[RUN] {case.name}')
        results.append(_run_case(case, namespace=namespace, top_k=args.top_k, mode=args.mode, cases_path=args.cases))

    ranked = sorted(results, key=lambda x: (x.get('summary', {}).get('mrr') or 0.0), reverse=True)
    payload = {
        'mode': args.mode,
        'namespace': namespace,
        'top_k': args.top_k,
        'matrix_size': len(matrix),
        'ranked_results': ranked,
    }
    _save_json(args.out, payload)

    print(f'OK: saved matrix report to {args.out}')
    if ranked:
        best = ranked[0]
        print(
            'BEST:',
            best.get('config', {}).get('name'),
            f"mrr={best.get('summary', {}).get('mrr')}",
            f"ndcg={best.get('summary', {}).get('ndcg_at_k')}",
        )


if __name__ == '__main__':
    os.environ.setdefault('HTTP_PROXY', '')
    os.environ.setdefault('HTTPS_PROXY', '')
    main()
