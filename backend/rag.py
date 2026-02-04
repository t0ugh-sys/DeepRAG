from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from openai import OpenAI
from pymilvus import Collection, connections
from rank_bm25 import BM25Okapi  # type: ignore
from sentence_transformers import SentenceTransformer

from backend.config import Settings
from backend.ingest import split_text
from backend.utils.cache import query_cache

try:
    from FlagEmbedding import FlagReranker  # type: ignore
except Exception:  # pragma: no cover
    FlagReranker = None  # type: ignore


@dataclass
class RetrievedChunk:
    text: str
    score: float
    meta: Dict[str, Any]


class VectorStore:
    """Vector store wrapper for Milvus/FAISS / 向量存储封装。"""

    def __init__(self, meta_path: str, embedding_model: str, settings: Settings, namespace: str | None = None) -> None:
        if not os.path.exists(meta_path):
            raise FileNotFoundError(
                "meta.jsonl not found. Run ingest first / 未找到 meta.jsonl，请先执行 ingest"
            )

        self.settings = settings
        self.namespace = namespace or settings.default_namespace
        self.texts: List[str] = []
        self.metas: List[Dict[str, Any]] = []
        self.meta_path = meta_path

        with open(meta_path, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                self.texts.append(rec["text"])
                self.metas.append(rec["meta"])

        self.embedder = SentenceTransformer(embedding_model)
        self._bm25_tokenized = [self._tokenize(t) for t in self.texts]
        self._bm25 = BM25Okapi(self._bm25_tokenized) if self._bm25_tokenized else None

        self.collection = None
        self.faiss_index = None
        self.backend = "faiss"

        collection_name = f"{settings.milvus_collection}_{self.namespace}"
        if settings.vector_backend in ("auto", "milvus"):
            try:
                connections.connect(
                    alias="default",
                    host=settings.milvus_host,
                    port=settings.milvus_port,
                    user=settings.milvus_user,
                    password=settings.milvus_password,
                    secure=settings.milvus_secure,
                )
                self.collection = Collection(collection_name)
                self.backend = "milvus"
            except Exception as exc:
                logging.getLogger("rag").debug("Milvus unavailable, fallback to FAISS / Milvus 不可用，回退 FAISS: %s", exc)
                self.collection = None
                self.backend = "faiss"

        if self.collection is None:
            faiss_path = os.path.join(os.path.dirname(meta_path), "faiss.index")
            self.faiss_path = faiss_path
            if not os.path.exists(faiss_path):
                raise FileNotFoundError(
                    "faiss.index not found. Run ingest first / 未找到 faiss.index，请先执行 ingest"
                )
            try:
                import faiss  # type: ignore
            except Exception as exc:  # pragma: no cover
                raise RuntimeError(
                    "FAISS not installed. Install faiss-cpu in your env / 未安装 FAISS，请在环境中安装 faiss-cpu"
                ) from exc
            self.faiss_index = faiss.read_index(faiss_path)

    def _expand_query(self, query: str) -> str:
        """Expand query by keywords / 关键词扩展。"""
        try:
            import jieba.analyse

            keywords = jieba.analyse.extract_tags(query, topK=5, withWeight=False)
            return f"{query} {' '.join(keywords)}".strip()
        except Exception:
            return query

    def search(self, query: str, top_k: int = 4) -> List[RetrievedChunk]:
        expanded_query = self._expand_query(query)
        vec = self.embedder.encode([expanded_query], normalize_embeddings=True)
        vec = np.array(vec).astype("float32")[0]

        results: List[RetrievedChunk] = []
        if self.backend == "milvus" and self.collection is not None:
            search_params = {"metric_type": "IP", "params": {"nprobe": 16}}
            res = self.collection.search(
                data=[vec],
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                output_fields=["path", "chunk_id", "text"],
            )
            hits = res[0]
            for hit in hits:
                meta = {"path": hit.entity.get("path"), "chunk_id": int(hit.entity.get("chunk_id"))}
                results.append(RetrievedChunk(text=hit.entity.get("text"), score=float(hit.distance), meta=meta))
            return _apply_score_threshold(_dedupe_results(results), self.settings.score_threshold)

        assert self.faiss_index is not None
        import faiss  # type: ignore

        scores, indices = self.faiss_index.search(vec.reshape(1, -1), top_k)
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append(RetrievedChunk(text=self.texts[idx], score=float(score), meta=self.metas[idx]))

        return self._fuse_with_bm25(query, _dedupe_results(results), top_k)

    def _tokenize(self, s: str) -> list[str]:
        try:
            import jieba

            words = list(jieba.cut_for_search(s.lower()))
            return [w.strip() for w in words if w.strip()]
        except Exception:
            import re

            return [w for w in re.split(r"\W+", s.lower()) if w]

    def _fuse_with_bm25(self, query: str, vec_results: List[RetrievedChunk], top_k: int) -> List[RetrievedChunk]:
        settings = self.settings
        recs = list(vec_results)
        if settings.bm25_enabled and self._bm25 is not None:
            tokens = self._tokenize(query)
            bm25_scores = self._bm25.get_scores(tokens)
            bm25_max = max(bm25_scores) if len(bm25_scores) else 1.0

            for i, r in enumerate(recs):
                bm = (bm25_scores[i] / (bm25_max or 1.0)) if i < len(bm25_scores) else 0.0
                r.score = settings.vec_weight * r.score + settings.bm25_weight * float(bm)

            if settings.score_threshold > 0:
                recs = [r for r in recs if r.score >= settings.score_threshold]
            recs = sorted(recs, key=lambda x: x.score, reverse=True)

        if len(recs) > top_k:
            recs = self._mmr(query, recs, top_k, lambda_weight=settings.mmr_lambda)
        return recs

    def _mmr(self, query: str, recs: List[RetrievedChunk], k: int, lambda_weight: float = 0.5) -> List[RetrievedChunk]:
        query_vec = self.embedder.encode([query], normalize_embeddings=True)
        query_vec = np.array(query_vec).astype("float32")[0]
        cand_vecs = self.embedder.encode([r.text for r in recs], normalize_embeddings=True)
        cand_vecs = np.array(cand_vecs).astype("float32")

        selected: list[int] = []
        remaining = set(range(len(recs)))

        def sim(a: np.ndarray, b: np.ndarray) -> float:
            return float(np.dot(a, b))

        while remaining and len(selected) < k:
            if not selected:
                idx = max(remaining, key=lambda i: sim(query_vec, cand_vecs[i]))
                selected.append(idx)
                remaining.remove(idx)
                continue
            best_idx = None
            best_score = -1e9
            for i in list(remaining):
                relevance = sim(query_vec, cand_vecs[i])
                diversity = max(sim(cand_vecs[i], cand_vecs[j]) for j in selected)
                mmr_score = lambda_weight * relevance - (1 - lambda_weight) * diversity
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = i
            if best_idx is None:
                break
            selected.append(best_idx)
            remaining.remove(best_idx)

        return [recs[i] for i in selected]

    def add_document(self, path: str, text: str) -> int:
        if self.backend == "milvus" and self.collection is not None:
            chunks = split_text(text)
            if not chunks:
                return 0
            embeddings = self.embedder.encode(chunks, normalize_embeddings=True)
            embeddings = np.array(embeddings).astype("float32")
            paths = [path] * len(chunks)
            chunk_ids = list(range(len(chunks)))
            self.collection.insert([paths, chunk_ids, chunks, embeddings])
            self.collection.flush()
            return len(chunks)

        if self.backend == "faiss" and self.faiss_index is not None:
            chunks = split_text(text)
            if not chunks:
                return 0
            import faiss  # type: ignore

            embeddings = self.embedder.encode(chunks, normalize_embeddings=True)
            vecs = np.array(embeddings).astype("float32")
            self.faiss_index.add(vecs)
            faiss.write_index(self.faiss_index, getattr(self, "faiss_path", os.path.join(os.path.dirname(self.meta_path), "faiss.index")))
            with open(self.meta_path, "a", encoding="utf-8") as f:
                for idx, chunk in enumerate(chunks):
                    meta = {"path": path, "chunk_id": idx, "chunk_size": len(chunk)}
                    rec = {"meta": meta, "text": chunk}
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    self.metas.append(meta)
                    self.texts.append(chunk)
            return len(chunks)

        raise RuntimeError("Vector backend not initialized / 向量后端未初始化")

    def delete_document(self, path: str) -> int:
        if self.backend == "milvus" and self.collection is not None:
            escaped = path.replace("'", "\\'")
            expr = "path == '" + escaped + "'"
            res = self.collection.delete(expr)
            self.collection.flush()
            try:
                return int(getattr(res, "delete_count", 0))
            except Exception:
                return 0

        if self.backend == "faiss" and self.faiss_index is not None:
            remain_texts: List[str] = []
            remain_metas: List[Dict[str, Any]] = []
            removed = 0
            for t, m in zip(self.texts, self.metas):
                if str(m.get("path")) == path:
                    removed += 1
                    continue
                remain_texts.append(t)
                remain_metas.append(m)

            with open(self.meta_path, "w", encoding="utf-8") as f:
                for m, t in zip(remain_metas, remain_texts):
                    f.write(json.dumps({"meta": m, "text": t}, ensure_ascii=False) + "\n")

            import faiss  # type: ignore

            if remain_texts:
                vecs = self.embedder.encode(remain_texts, normalize_embeddings=True)
                vecs = np.array(vecs).astype("float32")
                index = faiss.IndexFlatIP(vecs.shape[1])
                index.add(vecs)
                faiss.write_index(index, getattr(self, "faiss_path", os.path.join(os.path.dirname(self.meta_path), "faiss.index")))
                self.faiss_index = index
            else:
                dim = int(self.embedder.get_sentence_embedding_dimension())
                index = faiss.IndexFlatIP(dim)
                faiss.write_index(index, getattr(self, "faiss_path", os.path.join(os.path.dirname(self.meta_path), "faiss.index")))
                self.faiss_index = index

            self.texts = remain_texts
            self.metas = remain_metas
            return removed

        raise RuntimeError("Vector backend not initialized / 向量后端未初始化")

    def list_paths(self, limit: int = 1000) -> List[str]:
        if self.backend == "milvus" and self.collection is not None:
            try:
                recs = self.collection.query(expr="", output_fields=["path"], limit=limit * 5)
            except Exception:
                recs = []
            paths = []
            seen = set()
            for r in recs:
                p = r.get("path")
                if p and p not in seen:
                    seen.add(p)
                    paths.append(p)
                if len(paths) >= limit:
                    break
            return paths

        seen = set()
        paths: List[str] = []
        for m in self.metas:
            p = m.get("path")
            if p and p not in seen:
                seen.add(p)
                paths.append(p)
            if len(paths) >= limit:
                break
        return paths

    def list_paths_with_stats(self, limit: int = 1000) -> List[dict]:
        """Return path stats / 返回路径统计。"""
        from collections import Counter
        from datetime import datetime

        if self.backend == "milvus" and self.collection is not None:
            try:
                recs = self.collection.query(expr="", output_fields=["path", "chunk_id"], limit=limit * 100)
            except Exception:
                recs = []
            path_chunks = Counter(r.get("path") for r in recs if r.get("path"))
        else:
            path_chunks = Counter(m.get("path") for m in self.metas if m.get("path"))

        result = []
        for path, chunk_count in list(path_chunks.items())[:limit]:
            result.append({
                "path": path,
                "chunk_count": chunk_count,
                "last_updated": datetime.now().isoformat(),
            })
        return result


def build_prompt(question: str, contexts: List[RetrievedChunk], strict_mode: bool = True, custom_system_prompt: str | None = None) -> str:
    """Build RAG prompt / 构建检索增强提示。"""
    has_valid_context = len(contexts) > 0 and any(c.score > 0.1 for c in contexts)

    context_blocks: List[str] = []
    for i, c in enumerate(contexts, start=1):
        path = c.meta.get("path", "")
        filename = path.split("/")[-1] if "/" in path else path.split("\\")[-1] if "\\" in path else path
        context_blocks.append(f"[{i}] {filename} (score={c.score:.2f})\n{c.text}")
    context_text = "\n\n".join(context_blocks)

    if not has_valid_context and strict_mode:
        return (
            "You are a helpful assistant. Use ONLY the provided context. "
            "If the context does not contain the answer, reply: 'Not enough information / 信息不足'.\n\n"
            f"Question / 问题: {question}\n"
        )

    if custom_system_prompt:
        tpl = custom_system_prompt
        prompt = tpl.replace("{context}", context_text).replace("{question}", question)
        if "{context}" not in tpl:
            prompt = f"{prompt}\n\nContext / 资料:\n{context_text}"
        if "{question}" not in tpl:
            prompt = f"{prompt}\n\nQuestion / 问题: {question}"
        return prompt

    if strict_mode:
        system_instruction = (
            "You are a RAG assistant. Answer strictly based on the context. "
            "If the answer is not in the context, say: 'Not enough information / 信息不足'.\n"
            "Cite the relevant context snippets when possible.\n"
            "你是检索增强助手，只能基于资料回答；资料不足请直接说明。\n"
        )
    else:
        system_instruction = (
            "You are a RAG assistant. Prefer the context, but you may use general knowledge.\n"
            "你是检索增强助手，优先使用资料，不足时可补充常识。\n"
        )

    prompt = (
        f"{system_instruction}\n"
        f"{'=' * 60}\n"
        f"Context / 资料 ({len(contexts)}):\n{context_text}\n"
        f"{'=' * 60}\n"
        f"Question / 问题: {question}\n"
    )
    return prompt


class RAGPipeline:
    """RAG pipeline / 检索增强生成管线。"""

    def __init__(self, settings: Settings, namespace: str | None = None) -> None:
        self.settings = settings
        meta_path = os.path.join(settings.index_dir, "meta.jsonl")
        self.store = VectorStore(meta_path, settings.embedding_model_name, settings, namespace)

        client_kwargs: Dict[str, Any] = {}
        if settings.openai_api_key:
            client_kwargs["api_key"] = settings.openai_api_key
        if settings.openai_base_url:
            client_kwargs["base_url"] = settings.openai_base_url
        self.client = OpenAI(**client_kwargs)

        self.qwen_client = None
        if settings.qwen_api_key:
            qwen_kwargs: Dict[str, Any] = {
                "api_key": settings.qwen_api_key,
                "base_url": settings.qwen_base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1",
            }
            self.qwen_client = OpenAI(**qwen_kwargs)

        self.reranker = None
        if settings.reranker_enabled and FlagReranker is not None:
            try:
                self.reranker = FlagReranker(settings.reranker_model_name, use_fp16=True)
            except Exception:
                self.reranker = None

        self.query_rewriter = None
        if settings.openai_api_key and settings.openai_base_url:
            try:
                from backend.query_rewriter import QueryRewriter

                self.query_rewriter = QueryRewriter(
                    api_key=settings.openai_api_key,
                    base_url=settings.openai_base_url,
                    model=settings.llm_model,
                )
            except Exception as exc:
                logging.getLogger("rag").warning("Query rewriter disabled / 改写器不可用: %s", exc)

    def _get_client_for_model(self, model: str) -> OpenAI:
        if model and model.startswith("qwen"):
            if self.qwen_client is None:
                raise ValueError("Qwen client not configured / 未配置 QWEN_API_KEY")
            return self.qwen_client
        return self.client

    def ask(
        self,
        question: str,
        top_k: int | None = None,
        rerank_enabled: bool | None = None,
        rerank_top_n: int | None = None,
        model: str | None = None,
    ) -> Tuple[str, List[RetrievedChunk]]:
        k = top_k or self.settings.top_k

        namespace = getattr(self.store, "namespace", "default")
        cached_result = query_cache.get(question, k, namespace)
        if cached_result is not None:
            recs = cached_result
        else:
            recs = self.store.search(question, k)
            query_cache.set(question, k, namespace, recs)

        use_rr = (self.reranker is not None) and (
            self.settings.reranker_enabled if rerank_enabled is None else rerank_enabled
        )
        top_n = rerank_top_n or self.settings.reranker_top_n
        if use_rr:
            pairs = [[question, r.text] for r in recs]
            scores = self.reranker.compute_score(pairs)
            for r, s in zip(recs, scores):
                r.score = float(s)
            recs = sorted(recs, key=lambda x: x.score, reverse=True)[:top_n]

        prompt = build_prompt(question, recs, strict_mode=self.settings.strict_mode)
        target_model = model or self.settings.llm_model
        client = self._get_client_for_model(target_model)
        resp = client.chat.completions.create(
            model=target_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        answer = resp.choices[0].message.content or ""
        return answer, recs

    def ask_stream(
        self,
        question: str,
        top_k: int | None = None,
        rerank_enabled: bool | None = None,
        rerank_top_n: int | None = None,
        model: str | None = None,
        system_prompt: str | None = None,
        web_enabled: bool | None = None,
        web_top_k: int | None = None,
    ):
        k = top_k or self.settings.top_k
        recs = self.store.search(question, k)

        web_snippets: List[str] = []
        if web_enabled:
            try:
                import re
                import requests

                n = web_top_k or 3
                q = requests.utils.quote(question)
                url = f"https://duckduckgo.com/html/?q={q}"
                html = requests.get(url, timeout=5).text
                results = re.findall(
                    r'<a rel="nofollow" class="result__a" href="(.*?)".*?</a>.*?<a.*?class="result__snippet".*?>(.*?)</a>',
                    html,
                    flags=re.S,
                )
                for link, snippet in results[:n]:
                    text = re.sub('<.*?>', '', snippet)
                    web_snippets.append(f"[Web] {text}\nURL: {link}")
            except Exception:
                pass

        use_rr = (self.reranker is not None) and (
            self.settings.reranker_enabled if rerank_enabled is None else rerank_enabled
        )
        top_n = rerank_top_n or self.settings.reranker_top_n
        if use_rr:
            pairs = [[question, r.text] for r in recs]
            scores = self.reranker.compute_score(pairs)
            for r, s in zip(recs, scores):
                r.score = float(s)
            recs = sorted(recs, key=lambda x: x.score, reverse=True)[:top_n]

        if web_snippets:
            from dataclasses import dataclass

            @dataclass
            class _Tmp:
                text: str
                score: float
                meta: dict

            for w in web_snippets:
                recs.append(_Tmp(text=w, score=1.0, meta={"path": "web"}))

        prompt = build_prompt(question, recs, strict_mode=self.settings.strict_mode, custom_system_prompt=system_prompt)
        target_model = model or self.settings.llm_model
        client = self._get_client_for_model(target_model)

        def _gen():  # noqa: ANN202
            stream = client.chat.completions.create(
                model=target_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                stream=True,
            )
            for chunk in stream:
                delta = getattr(getattr(chunk.choices[0], "delta", None), "content", None)
                if delta:
                    yield delta

        return _gen(), recs

    def ask_with_query_rewriting(
        self,
        question: str,
        strategy: str = "expand",
        top_k: int | None = None,
        rerank_enabled: bool | None = None,
        rerank_top_n: int | None = None,
        model: str | None = None,
    ) -> Tuple[str, List[RetrievedChunk], Dict[str, Any]]:
        if not self.query_rewriter:
            answer, recs = self.ask(question, top_k, rerank_enabled, rerank_top_n, model)
            return answer, recs, {
                "original_query": question,
                "rewritten_queries": [question],
                "strategy": "none",
                "note": "Query rewriter disabled / 改写器不可用",
            }

        rewritten_queries = self.query_rewriter.rewrite_for_retrieval(question, strategy)
        k = top_k or self.settings.top_k
        all_recs: List[RetrievedChunk] = []
        seen_texts = set()

        for query in rewritten_queries:
            recs = self.store.search(query, k)
            for rec in recs:
                if rec.text not in seen_texts:
                    seen_texts.add(rec.text)
                    all_recs.append(rec)

        use_rr = (self.reranker is not None) and (
            self.settings.reranker_enabled if rerank_enabled is None else rerank_enabled
        )
        top_n = rerank_top_n or self.settings.reranker_top_n

        if use_rr and all_recs:
            pairs = [[question, r.text] for r in all_recs]
            scores = self.reranker.compute_score(pairs)
            for r, s in zip(all_recs, scores):
                r.score = float(s)
            all_recs = sorted(all_recs, key=lambda x: x.score, reverse=True)[:top_n]
        else:
            all_recs = sorted(all_recs, key=lambda x: x.score, reverse=True)[:top_n]

        prompt = build_prompt(question, all_recs, strict_mode=self.settings.strict_mode)
        target_model = model or self.settings.llm_model
        client = self._get_client_for_model(target_model)
        resp = client.chat.completions.create(
            model=target_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        answer = resp.choices[0].message.content or ""

        metadata = {
            "original_query": question,
            "rewritten_queries": rewritten_queries,
            "strategy": strategy,
            "total_retrieved": len(all_recs),
            "unique_documents": len(seen_texts),
        }
        return answer, all_recs, metadata

    def analyze_query(self, question: str) -> Dict[str, Any]:
        if not self.query_rewriter:
            return {
                "error": "Query rewriter disabled / 改写器不可用",
                "recommended_strategy": "none",
            }
        return self.query_rewriter.analyze_query(question)

    def add_document(self, path: str, text: str) -> int:
        return self.store.add_document(path, text)

    def delete_document(self, path: str) -> int:
        return self.store.delete_document(path)

    def list_paths(self, limit: int = 1000) -> List[str]:
        return self.store.list_paths(limit)

    def list_paths_with_stats(self, limit: int = 1000) -> List[dict]:
        return self.store.list_paths_with_stats(limit)


def _dedupe_results(recs: List[RetrievedChunk]) -> List[RetrievedChunk]:
    seen = set()
    out: List[RetrievedChunk] = []
    for r in recs:
        key = (r.meta.get("path"), r.meta.get("chunk_id"), r.text[:200])
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def _apply_score_threshold(recs: List[RetrievedChunk], threshold: float) -> List[RetrievedChunk]:
    if threshold <= 0:
        return recs
    return [r for r in recs if r.score >= threshold]
