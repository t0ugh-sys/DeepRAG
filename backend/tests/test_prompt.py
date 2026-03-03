from backend.rag import build_prompt
from backend.types import RetrievedChunk


def test_build_prompt_does_not_show_scores_by_default():
    recs = [RetrievedChunk(text="ctx", score=0.99, meta={"path": "p", "chunk_id": 1})]
    prompt = build_prompt("q", recs, strict_mode=True, show_scores=False)
    assert "score=" not in prompt


def test_build_prompt_can_show_scores_when_enabled():
    recs = [RetrievedChunk(text="ctx", score=0.42, meta={"path": "p", "chunk_id": 1})]
    prompt = build_prompt("q", recs, strict_mode=True, show_scores=True)
    assert "score=" in prompt

