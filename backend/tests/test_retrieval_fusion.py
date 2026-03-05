from __future__ import annotations

from types import SimpleNamespace

from backend.rag import VectorStore
from backend.rag_types import RetrievedChunk


class DummyBM25:
    def __init__(self, scores: list[float]):
        self._scores = scores

    def get_scores(self, tokens):  # noqa: ANN001
        return self._scores


def _make_settings():
    # Minimal settings-like object: only fields used by _fuse_with_bm25.
    return SimpleNamespace(
        bm25_enabled=True,
        bm25_weight=0.5,
        vec_weight=0.5,
        score_threshold=0.0,
        mmr_lambda=0.7,
    )


def test_fuse_with_bm25_uses_correct_corpus_index():
    store = VectorStore.__new__(VectorStore)
    store.settings = _make_settings()
    store.texts = ["doc0", "doc1", "doc2"]
    store.metas = [
        {"path": "p0", "chunk_id": 0},
        {"path": "p1", "chunk_id": 0},
        {"path": "p2", "chunk_id": 0},
    ]
    store._bm25 = DummyBM25([10.0, 0.0, 5.0])
    store._tokenize = lambda s: ["q"]  # noqa: E731
    store.backend = "faiss"
    store.faiss_index = None

    # Build the meta index used for BM25 lookups.
    store._meta_key_to_idx = store._build_meta_key_index()

    # Vector result is doc1 (idx=1), but BM25 score at idx=0 is highest.
    vec_results = [RetrievedChunk(text="doc1", score=0.9, meta={"path": "p1", "chunk_id": 0})]

    fused = store._fuse_with_bm25("q", vec_results, top_k=10, query_vec=None)

    # Ensure doc1 doesn't incorrectly receive BM25 score from idx=0.
    doc1 = next(r for r in fused if r.meta["path"] == "p1")
    # vec_norm=(0.9+1)/2=0.95; bm_norm for idx=1 is 0.0 => fused = 0.475
    assert abs(doc1.score - 0.475) < 1e-6


def test_fuse_with_bm25_includes_bm25_only_candidates():
    store = VectorStore.__new__(VectorStore)
    store.settings = _make_settings()
    store.texts = ["doc0", "doc1", "doc2"]
    store.metas = [
        {"path": "p0", "chunk_id": 0},
        {"path": "p1", "chunk_id": 0},
        {"path": "p2", "chunk_id": 0},
    ]
    store._bm25 = DummyBM25([0.0, 3.0, 1.0])
    store._tokenize = lambda s: ["q"]  # noqa: E731
    store.backend = "faiss"
    store.faiss_index = None
    store._meta_key_to_idx = store._build_meta_key_index()

    fused = store._fuse_with_bm25("q", [], top_k=10, query_vec=None)

    assert fused
    assert fused[0].meta["path"] == "p1"
