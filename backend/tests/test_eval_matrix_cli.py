import json
from pathlib import Path

from backend.eval_matrix_cli import _load_matrix, _normalize_case


def test_normalize_case_rebalance_weights() -> None:
    case = _normalize_case(
        {
            'name': 'rebalance',
            'bm25_enabled': True,
            'vec_weight': 6,
            'bm25_weight': 4,
        },
        idx=0,
    )
    assert round(case.vec_weight, 2) == 0.6
    assert round(case.bm25_weight, 2) == 0.4


def test_normalize_case_disable_bm25() -> None:
    case = _normalize_case({'name': 'vec_only', 'bm25_enabled': False}, idx=0)
    assert case.vec_weight == 1.0
    assert case.bm25_weight == 0.0


def test_load_matrix_from_file() -> None:
    path = Path('backend/data/test_matrix_tmp.json')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([{'name': 'balanced', 'bm25_enabled': True, 'vec_weight': 0.6, 'bm25_weight': 0.4}], ensure_ascii=False),
        encoding='utf-8',
    )
    try:
        matrix = _load_matrix(str(path))
        assert len(matrix) == 1
        assert matrix[0].name == 'balanced'
    finally:
        if path.exists():
            path.unlink()
