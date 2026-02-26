from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

from backend.rag import RAGPipeline
from backend.types import RetrievedChunk


class DummyStore:
    def __init__(self):
        self.namespace = "default"
        self.backend = "faiss"
        self._corpus_revision = 0
        self.last_top_k: int | None = None

    def search(self, query: str, top_k: int):  # noqa: ANN001
        self.last_top_k = top_k
        return [
            RetrievedChunk(text=f"t{i}", score=0.1, meta={"path": f"p{i}", "chunk_id": i})
            for i in range(top_k)
        ]


class DummyReranker:
    def __init__(self):
        self.last_pairs_len: int | None = None

    def compute_score(self, pairs):  # noqa: ANN001
        self.last_pairs_len = len(pairs)
        # Highest score for the last item.
        return [float(i) for i in range(len(pairs))]


def _make_settings():
    # Minimal settings-like object used by RAGPipeline.ask().
    return SimpleNamespace(
        top_k=5,
        bm25_enabled=True,
        bm25_weight=0.4,
        vec_weight=0.6,
        mmr_lambda=0.7,
        score_threshold=0.0,
        embedding_model_name="dummy",
        reranker_enabled=True,
        reranker_top_n=4,
        strict_mode=True,
        llm_model="dummy",
    )


def test_ask_rerank_retrieves_larger_candidate_pool_and_keeps_context_window():
    pipeline = RAGPipeline.__new__(RAGPipeline)
    pipeline.settings = _make_settings()
    pipeline.store = DummyStore()
    pipeline.reranker = DummyReranker()

    class _DummyClient:
        class chat:  # noqa: N801
            class completions:  # noqa: N801
                @staticmethod
                def create(**kwargs):  # noqa: ANN001
                    @dataclass
                    class _Msg:
                        content: str

                    @dataclass
                    class _Choice:
                        message: _Msg

                    return SimpleNamespace(choices=[_Choice(message=_Msg(content="ok"))])

    pipeline._get_client_for_model = lambda model: _DummyClient()  # noqa: E731

    answer, recs = pipeline.ask("q", top_k=5, rerank_enabled=True, rerank_top_n=4, model="dummy")
    assert answer == "ok"
    # Candidate pool should be expanded (min 32).
    assert pipeline.store.last_top_k == 32
    assert pipeline.reranker.last_pairs_len == 32
    # Context window keeps top_k (max(top_k, top_n)).
    assert len(recs) == 5


def test_ask_without_rerank_uses_top_k():
    pipeline = RAGPipeline.__new__(RAGPipeline)
    pipeline.settings = _make_settings()
    pipeline.store = DummyStore()
    pipeline.reranker = None

    class _DummyClient:
        class chat:  # noqa: N801
            class completions:  # noqa: N801
                @staticmethod
                def create(**kwargs):  # noqa: ANN001
                    @dataclass
                    class _Msg:
                        content: str

                    @dataclass
                    class _Choice:
                        message: _Msg

                    return SimpleNamespace(choices=[_Choice(message=_Msg(content="ok"))])

    pipeline._get_client_for_model = lambda model: _DummyClient()  # noqa: E731

    _, recs = pipeline.ask("q", top_k=5, rerank_enabled=False, rerank_top_n=4, model="dummy")
    assert pipeline.store.last_top_k == 5
    assert len(recs) == 5

