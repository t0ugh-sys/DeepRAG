import json

import backend.server as server
from backend.rag import RetrievedChunk


class DummyStore:
    def __init__(self, recs):
        self.recs = recs

    def search(self, question, top_k):
        return self.recs


class DummyReranker:
    def __init__(self, scores):
        self.scores = scores

    def compute_score(self, pairs):
        return self.scores


class DummyPipeline:
    def __init__(self, recs, reranker):
        self.store = DummyStore(recs)
        self.reranker = reranker


def _setup(monkeypatch, pipeline):
    server.pipeline = object()
    monkeypatch.setattr(server, "_get_pipeline", lambda ns: pipeline)
    monkeypatch.setattr(server, "_resolve_namespace", lambda ns: "default")
    monkeypatch.setattr(server, "_require_api_key", lambda x_api_key=None: None)


def test_advanced_search_rerank(monkeypatch):
    recs = [
        RetrievedChunk(text="a", score=0.1, meta={"path": "p1", "chunk_id": 1}),
        RetrievedChunk(text="b", score=0.2, meta={"path": "p2", "chunk_id": 2}),
    ]
    pipeline = DummyPipeline(recs, DummyReranker([0.1, 0.9]))
    _setup(monkeypatch, pipeline)

    req = server.AdvancedSearchRequest(question="q", top_k=2, reranker_enabled=True)
    response = server.advanced_search(req)
    body = json.loads(response.body)
    results = body["data"]["results"]

    assert results[0]["text"] == "b"
    assert results[0]["score"] == 0.9


def test_advanced_search_dedupe_and_filters(monkeypatch):
    recs = [
        RetrievedChunk(text="dup", score=0.1, meta={"path": "Docs/A.pdf", "chunk_id": 1, "tags": ["TagA"]}),
        RetrievedChunk(text="dup", score=0.2, meta={"path": "docs/a.pdf", "chunk_id": 1, "tags": ["taga"]}),
        RetrievedChunk(text="keep", score=0.3, meta={"path": "docs/b.pdf", "chunk_id": 2, "tags": ["tagb"]}),
    ]
    pipeline = DummyPipeline(recs, DummyReranker([0.1, 0.2, 0.3]))
    _setup(monkeypatch, pipeline)

    req = server.AdvancedSearchRequest(
        question="q",
        top_k=5,
        reranker_enabled=False,
        tags=["TAGA"],
        paths=["docs/a.pdf"]
    )
    response = server.advanced_search(req)
    body = json.loads(response.body)
    results = body["data"]["results"]

    assert len(results) == 1
    assert results[0]["text"] == "dup"
