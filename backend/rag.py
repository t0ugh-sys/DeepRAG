from __future__ import annotations

import json
import hashlib
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from openai import OpenAI
import httpx
from pymilvus import Collection, connections
from rank_bm25 import BM25Okapi  # type: ignore
from sentence_transformers import SentenceTransformer

from backend.config import Settings
from backend.ingest import split_text
from backend.types import RetrievedChunk
from backend.utils.cache import query_cache
from backend.utils.middleware import get_current_request_id

try:
    from FlagEmbedding import FlagReranker  # type: ignore
except Exception:  # pragma: no cover
    FlagReranker = None  # type: ignore


def _trace_enabled(settings: Settings) -> bool:
    return bool(getattr(settings, "trace_enabled", False))


def _trace_event(settings: Settings, event: str, fields: Dict[str, Any]) -> None:
    if not _trace_enabled(settings):
        return
    req_id = get_current_request_id() or "-"
    try:
        payload = json.dumps(fields, ensure_ascii=True, separators=(",", ":"), default=str)
    except Exception:
        payload = "{}"
    logging.getLogger("rag").info("trace event=%s request_id=%s %s", event, req_id, payload)


class VectorStore:
    """Vector store wrapper for Milvus/FAISS / 向量存储封装。"""

    def __init__(self, meta_path: str, embedding_model: str, settings: Settings, namespace: str | None = None) -> None:
        self.settings = settings
        self.namespace = namespace or settings.default_namespace
        self.texts: List[str] = []
        self.metas: List[Dict[str, Any]] = []
        self.meta_path = meta_path
        os.makedirs(os.path.dirname(meta_path), exist_ok=True)
        self._meta_loaded = False
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8-sig") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    self.texts.append(rec.get("text", ""))
                    self.metas.append(rec.get("meta", {}))
            self._meta_loaded = True

        self.embedder = SentenceTransformer(embedding_model)
        self._bm25_tokenized = [self._tokenize(t) for t in self.texts]
        self._bm25 = BM25Okapi(self._bm25_tokenized) if self._bm25_tokenized else None
        self._meta_key_to_idx = self._build_meta_key_index()
        self._corpus_revision = 0
        self._bm25_complete_corpus = True

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
            try:
                import faiss  # type: ignore
            except Exception as exc:  # pragma: no cover
                raise RuntimeError(
                    "FAISS not installed. Install faiss-cpu in your env / 未安装 FAISS，请在环境中安装 faiss-cpu"
                ) from exc
            if os.path.exists(faiss_path):
                self.faiss_index = faiss.read_index(faiss_path)
            else:
                dim = int(self.embedder.get_sentence_embedding_dimension())
                self.faiss_index = faiss.IndexFlatIP(dim)

        # In Milvus mode, BM25 relies on local meta.jsonl corpus. If Milvus contains more chunks than local corpus,
        # hybrid fusion will become misleading. Default to disabling BM25 fusion in that case.
        if self.backend == "milvus" and settings.bm25_enabled and settings.bm25_require_complete_corpus:
            try:
                milvus_count = int(getattr(self.collection, "num_entities", 0) or 0) if self.collection is not None else 0
                local_count = int(len(self.texts))
                if milvus_count and local_count:
                    # Allow a small drift (e.g. partial deletes), but reject large mismatches.
                    drift = abs(milvus_count - local_count)
                    if drift > max(10, int(milvus_count * 0.05)):
                        self._bm25_complete_corpus = False
                elif milvus_count and not local_count:
                    self._bm25_complete_corpus = False
                elif not milvus_count and local_count:
                    # Unable to verify corpus parity; treat as incomplete to avoid misleading hybrid fusion.
                    self._bm25_complete_corpus = False
            except Exception:
                # If we can't determine corpus completeness, fall back to a safe default.
                self._bm25_complete_corpus = False

    @staticmethod
    def _normalize_path(path: str) -> str:
        return path.replace("\\", "/").strip().lower()

    def _meta_key(self, meta: Dict[str, Any]) -> tuple[str, int] | None:
        path = meta.get("path")
        chunk_id = meta.get("chunk_id")
        if path is None or chunk_id is None:
            return None
        try:
            return (self._normalize_path(str(path)), int(chunk_id))
        except Exception:
            return None

    def _build_meta_key_index(self) -> Dict[tuple[str, int], int]:
        out: Dict[tuple[str, int], int] = {}
        for idx, meta in enumerate(self.metas):
            key = self._meta_key(meta)
            if key is None:
                continue
            # Keep the first occurrence; duplicates can happen after partial writes or inconsistent meta.
            out.setdefault(key, idx)
        return out

    @staticmethod
    def _normalize_vec_score(score: float) -> float:
        # Embeddings are normalized in this project, so inner-product ~= cosine similarity in [-1, 1].
        # Map to [0, 1] for stable fusion with BM25 scores.
        v = (float(score) + 1.0) / 2.0
        if v < 0.0:
            return 0.0
        if v > 1.0:
            return 1.0
        return v

    def _rebuild_bm25(self) -> None:
        self._bm25_tokenized = [self._tokenize(t) for t in self.texts]
        self._bm25 = BM25Okapi(self._bm25_tokenized) if self._bm25_tokenized else None
        self._meta_key_to_idx = self._build_meta_key_index()
        self._corpus_revision += 1

    def _bm25_score_for_doc(self, doc_index: int, query_tokens: list[str]) -> float:
        """
        Compute BM25 score for a single document to avoid full-corpus scoring on large corpora.
        """
        if self._bm25 is None:
            return 0.0
        try:
            doc_freqs = self._bm25.doc_freqs[doc_index]
            doc_len = self._bm25.doc_len[doc_index]
            avgdl = float(self._bm25.avgdl) if self._bm25.avgdl else 1.0
            k1 = float(getattr(self._bm25, "k1", 1.5))
            b = float(getattr(self._bm25, "b", 0.75))
            score = 0.0
            for tok in query_tokens:
                freq = doc_freqs.get(tok)
                if not freq:
                    continue
                idf = float(self._bm25.idf.get(tok, 0.0))
                denom = float(freq) + k1 * (1 - b + b * float(doc_len) / avgdl)
                score += idf * (float(freq) * (k1 + 1)) / (denom or 1.0)
            return float(score)
        except Exception:
            return 0.0

    def _namespace_prefix(self) -> str:
        if not getattr(self.settings, "enforce_namespace_path_prefix", False):
            return ""
        ns = (self.namespace or "").strip()
        if not ns or ns == "default":
            return ""
        return f"{ns}/"

    def _is_in_namespace(self, path: str | None) -> bool:
        prefix = self._namespace_prefix()
        if not prefix:
            return True
        if not path:
            return False
        return str(path).replace("\\", "/").startswith(prefix)

    def _expand_query(self, query: str) -> str:
        """Expand query by keywords / 关键词扩展。"""
        try:
            import jieba.analyse

            keywords = jieba.analyse.extract_tags(query, topK=5, withWeight=False)
            return f"{query} {' '.join(keywords)}".strip()
        except Exception:
            return query

    def search(self, query: str, top_k: int = 4) -> List[RetrievedChunk]:
        if not self.texts and self.collection is None and self.faiss_index is None:
            return []
        def _vector_candidates(qvec: np.ndarray, k: int) -> List[RetrievedChunk]:
            out: List[RetrievedChunk] = []
            if self.backend == "milvus" and self.collection is not None:
                search_params = {"metric_type": "IP", "params": {"nprobe": 16}}
                res = self.collection.search(
                    data=[qvec],
                    anns_field="embedding",
                    param=search_params,
                    limit=k,
                    output_fields=["path", "chunk_id", "text"],
                )
                hits = res[0]
                for hit in hits:
                    meta = {"path": hit.entity.get("path"), "chunk_id": int(hit.entity.get("chunk_id"))}
                    out.append(RetrievedChunk(text=hit.entity.get("text"), score=float(hit.distance), meta=meta))
                return out

            if self.faiss_index is None:
                return []
            import faiss  # type: ignore  # noqa: F401
            scores, indices = self.faiss_index.search(qvec.reshape(1, -1), k)
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1:
                    continue
                out.append(RetrievedChunk(text=self.texts[idx], score=float(score), meta=self.metas[idx]))
            return out

        vec0 = self.embedder.encode([query], normalize_embeddings=True)
        vec0 = np.array(vec0).astype("float32")[0]

        expanded_query = query
        if self.settings.query_expand_enabled:
            expanded_query = self._expand_query(query)
        vec1: np.ndarray | None = None
        if expanded_query and expanded_query != query:
            v = self.embedder.encode([expanded_query], normalize_embeddings=True)
            vec1 = np.array(v).astype("float32")[0]

        results: List[RetrievedChunk] = []
        mult = int(self.settings.candidate_k_mult or 4)
        k_min = int(self.settings.candidate_k_min or 20)
        k_max = int(self.settings.candidate_k_max or 200)
        candidate_k = max(top_k * mult, top_k + 10, k_min)
        candidate_k = min(candidate_k, k_max)

        # When using a shared local corpus with namespace prefix filtering, scale candidate_k by corpus ratio
        # to reduce both "too few after filter" and "overly huge pool" issues.
        if self._namespace_prefix() and self.texts:
            try:
                prefix = self._namespace_prefix()
                in_ns = sum(1 for m in self.metas if str(m.get("path", "")).replace("\\", "/").startswith(prefix))
                ratio = float(in_ns) / float(len(self.metas) or 1)
                ratio = max(ratio, 0.05)  # guard for tiny namespaces
                scaled = int(candidate_k / ratio)
                candidate_k = min(max(candidate_k, scaled), k_max)
            except Exception:
                pass
        results.extend(_vector_candidates(vec0, candidate_k))
        if vec1 is not None:
            results.extend(_vector_candidates(vec1, candidate_k))

        # Namespace isolation for shared FAISS index: filter by path prefix when enabled.
        if self._namespace_prefix():
            results = [r for r in results if self._is_in_namespace(r.meta.get("path"))]

        return self._fuse_with_bm25(query, _dedupe_results(results), top_k, query_vec=vec0)

    def _tokenize(self, s: str) -> list[str]:
        try:
            import jieba

            words = list(jieba.cut_for_search(s.lower()))
            return [w.strip() for w in words if w.strip()]
        except Exception:
            import re

            return [w for w in re.split(r"\W+", s.lower()) if w]

    def _fuse_with_bm25(
        self,
        query: str,
        vec_results: List[RetrievedChunk],
        top_k: int,
        query_vec: np.ndarray | None = None,
    ) -> List[RetrievedChunk]:
        settings = self.settings
        # Build candidate pool = vector candidates + BM25 top candidates, then fuse scores.
        vec_by_idx: Dict[int, RetrievedChunk] = {}
        synthetic_idx = -1
        for r in vec_results:
            key = self._meta_key(r.meta)
            if key is None:
                vec_by_idx[synthetic_idx] = r
                synthetic_idx -= 1
                continue
            idx = self._meta_key_to_idx.get(key)
            if idx is None:
                vec_by_idx[synthetic_idx] = r
                synthetic_idx -= 1
                continue
            # Keep the best vector score for the same corpus idx.
            prev = vec_by_idx.get(idx)
            if prev is None or r.score > prev.score:
                vec_by_idx[idx] = r

        bm25_scores: list[float] | None = None
        bm25_score_by_idx: Dict[int, float] = {}
        bm25_top: list[int] = []
        bm25_enabled = bool(settings.bm25_enabled)
        if self.backend == "milvus" and settings.bm25_require_complete_corpus and not self._bm25_complete_corpus:
            bm25_enabled = False
        if bm25_enabled and self._bm25 is not None and self.texts:
            tokens = self._tokenize(query)
            # Full-corpus BM25 scoring is O(N * |q|). For large corpora, score only candidates.
            if len(self.texts) <= int(settings.bm25_full_scan_max_docs or 0):
                bm25_scores = [float(s) for s in self._bm25.get_scores(tokens)]
                if bm25_scores:
                    bm25_k = min(len(bm25_scores), max(top_k * 4, 20))
                    arr = np.asarray(bm25_scores, dtype="float32")
                    idxs = np.argpartition(-arr, bm25_k - 1)[:bm25_k]
                    idxs = idxs[np.argsort(-arr[idxs])]
                    bm25_top = []
                    for i in idxs.tolist():
                        ii = int(i)
                        if 0 <= ii < len(self.metas) and self._is_in_namespace(self.metas[ii].get("path")):
                            bm25_top.append(ii)
            else:
                candidate_indices = [i for i in candidates.keys() if i >= 0]
                for idx in candidate_indices:
                    if 0 <= idx < len(self.texts) and self._is_in_namespace(self.metas[idx].get("path")):
                        bm25_score_by_idx[idx] = self._bm25_score_for_doc(idx, tokens)

        candidates: Dict[int, RetrievedChunk] = {}
        for idx, r in vec_by_idx.items():
            candidates[idx] = RetrievedChunk(text=r.text, score=r.score, meta=dict(r.meta))
        for idx in bm25_top:
            if idx in candidates:
                continue
            try:
                candidates[idx] = RetrievedChunk(text=self.texts[idx], score=0.0, meta=dict(self.metas[idx]))
            except Exception:
                continue

        if not candidates:
            return []

        bm25_max = 1.0
        if bm25_scores:
            try:
                bm25_max = max(bm25_scores) or 1.0
            except Exception:
                bm25_max = 1.0
        elif bm25_score_by_idx:
            bm25_max = max(bm25_score_by_idx.values()) or 1.0

        fused: List[RetrievedChunk] = []
        for idx, r in candidates.items():
            vec_norm = self._normalize_vec_score(r.score) if idx in vec_by_idx else 0.0
            bm_norm = 0.0
            if bm25_scores and 0 <= idx < len(bm25_scores):
                bm_norm = float(bm25_scores[idx] / (bm25_max or 1.0))
                if bm_norm < 0.0:
                    bm_norm = 0.0
                if bm_norm > 1.0:
                    bm_norm = 1.0
            elif idx in bm25_score_by_idx:
                bm_norm = float(bm25_score_by_idx[idx] / (bm25_max or 1.0))
                if bm_norm < 0.0:
                    bm_norm = 0.0
                if bm_norm > 1.0:
                    bm_norm = 1.0
            r.meta["score_vec"] = vec_norm
            r.meta["score_bm25"] = bm_norm
            r.score = settings.vec_weight * vec_norm + settings.bm25_weight * bm_norm
            r.meta["score_fused"] = r.score
            fused.append(r)

        fused = [r for r in fused if r.text.strip()]
        if settings.score_threshold > 0:
            fused = [r for r in fused if r.score >= settings.score_threshold]
        fused = sorted(fused, key=lambda x: x.score, reverse=True)

        if len(fused) <= top_k:
            return fused

        # MMR is expensive; limit pool size for diversification.
        pool_size = min(len(fused), max(top_k * 4, 30))
        pool = fused[:pool_size]
        return self._mmr(query, pool, top_k, lambda_weight=settings.mmr_lambda, query_vec=query_vec)

    def _mmr(
        self,
        query: str,
        recs: List[RetrievedChunk],
        k: int,
        lambda_weight: float = 0.5,
        query_vec: np.ndarray | None = None,
    ) -> List[RetrievedChunk]:
        local_query_vec = query_vec
        if local_query_vec is None:
            local_query_vec = self.embedder.encode([query], normalize_embeddings=True)
            local_query_vec = np.array(local_query_vec).astype("float32")[0]

        cand_vecs: np.ndarray | None = None
        if self.backend == "faiss" and getattr(self, "faiss_index", None) is not None and hasattr(self.faiss_index, "reconstruct"):
            idxs: list[int] = []
            ok = True
            for r in recs:
                key = self._meta_key(r.meta)
                if key is None:
                    ok = False
                    break
                idx = self._meta_key_to_idx.get(key)
                if idx is None:
                    ok = False
                    break
                idxs.append(int(idx))
            if ok and idxs:
                try:
                    cand_vecs = np.vstack([self.faiss_index.reconstruct(i) for i in idxs]).astype("float32")
                except Exception:
                    cand_vecs = None

        if cand_vecs is None:
            cand_vecs = self.embedder.encode([r.text for r in recs], normalize_embeddings=True)
            cand_vecs = np.array(cand_vecs).astype("float32")

        selected: list[int] = []
        remaining = set(range(len(recs)))

        def sim(a: np.ndarray, b: np.ndarray) -> float:
            return float(np.dot(a, b))

        while remaining and len(selected) < k:
            if not selected:
                idx = max(remaining, key=lambda i: sim(local_query_vec, cand_vecs[i]))
                selected.append(idx)
                remaining.remove(idx)
                continue
            best_idx = None
            best_score = -1e9
            for i in list(remaining):
                relevance = sim(local_query_vec, cand_vecs[i])
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

            # Keep local meta/BM25 in sync for hybrid retrieval & debugging.
            with open(self.meta_path, "a", encoding="utf-8") as f:
                for idx, chunk in enumerate(chunks):
                    meta = {"path": path, "chunk_id": idx, "chunk_size": len(chunk)}
                    f.write(json.dumps({"meta": meta, "text": chunk}, ensure_ascii=False) + "\n")
                    self.metas.append(meta)
                    self.texts.append(chunk)
            self._rebuild_bm25()
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
            self._rebuild_bm25()
            return len(chunks)

        raise RuntimeError("Vector backend not initialized / 向量后端未初始化")

    def delete_document(self, path: str) -> int:
        if self.backend == "milvus" and self.collection is not None:
            escaped = path.replace("'", "\\'")
            expr = "path == '" + escaped + "'"
            res = self.collection.delete(expr)
            self.collection.flush()

            # Best-effort local cleanup for hybrid retrieval.
            try:
                removed = 0
                remain_texts: List[str] = []
                remain_metas: List[Dict[str, Any]] = []
                for t, m in zip(self.texts, self.metas):
                    if self._normalize_path(str(m.get("path", ""))) == self._normalize_path(path):
                        removed += 1
                        continue
                    remain_texts.append(t)
                    remain_metas.append(m)

                with open(self.meta_path, "w", encoding="utf-8") as f:
                    for m, t in zip(remain_metas, remain_texts):
                        f.write(json.dumps({"meta": m, "text": t}, ensure_ascii=False) + "\n")
                self.texts = remain_texts
                self.metas = remain_metas
                self._rebuild_bm25()
                if removed:
                    return removed
            except Exception:
                pass
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
            self._rebuild_bm25()
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
            if not self._is_in_namespace(p):
                continue
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
            path_chunks = Counter(m.get("path") for m in self.metas if m.get("path") and self._is_in_namespace(m.get("path")))

        result = []
        for path, chunk_count in list(path_chunks.items())[:limit]:
            result.append({
                "path": path,
                "chunk_count": chunk_count,
                "last_updated": datetime.now().isoformat(),
            })
        return result

    def get_chunks_by_path(self, path: str, limit: int = 2000) -> List[RetrievedChunk]:
        """
        Fetch chunks by exact path match.

        This is used by non-RAG features (e.g. knowledge graph build) that need
        document text without relying on semantic search.
        """
        if not path:
            return []

        if self.backend == "milvus" and self.collection is not None:
            try:
                escaped = str(path).replace("\\", "\\\\").replace('"', '\\"')
                recs = self.collection.query(
                    expr=f'path == "{escaped}"',
                    output_fields=["path", "chunk_id", "text"],
                    limit=limit,
                )
                out: List[RetrievedChunk] = []
                for r in recs:
                    out.append(
                        RetrievedChunk(
                            text=str(r.get("text") or ""),
                            score=0.0,
                            meta={
                                "path": r.get("path"),
                                "chunk_id": r.get("chunk_id"),
                            },
                        )
                    )
                out.sort(key=lambda x: int(x.meta.get("chunk_id") or 0))
                return out
            except Exception:
                return []

        out = []
        for t, m in zip(self.texts, self.metas):
            if str(m.get("path")) != str(path):
                continue
            if not self._is_in_namespace(m.get("path")):
                continue
            out.append(RetrievedChunk(text=t, score=0.0, meta=dict(m)))
            if len(out) >= limit:
                break
        out.sort(key=lambda x: int(x.meta.get("chunk_id") or 0))
        return out


def build_prompt(
    question: str,
    contexts: List[RetrievedChunk],
    strict_mode: bool = True,
    custom_system_prompt: str | None = None,
    show_scores: bool = False,
) -> str:
    """Build RAG prompt / 构建检索增强提示。"""
    # Do not depend on numeric scores here (rerank/fusion score scales differ).
    has_valid_context = len(contexts) > 0

    context_blocks: List[str] = []
    for i, c in enumerate(contexts, start=1):
        path = c.meta.get("path", "")
        filename = path.split("/")[-1] if "/" in path else path.split("\\")[-1] if "\\" in path else path
        if show_scores:
            context_blocks.append(f"[{i}] {filename} (score={c.score:.2f})\n{c.text}")
        else:
            # Default: do not expose numeric scores to the model.
            context_blocks.append(f"[{i}] {filename}\n{c.text}")
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
        # Backward-compatibility: older server routes still reference `pipeline.vector_store`.
        self.vector_store = self.store

        self.client = None
        if settings.openai_api_key:
            client_kwargs: Dict[str, Any] = {"api_key": settings.openai_api_key}
            if settings.openai_base_url:
                client_kwargs["base_url"] = settings.openai_base_url
            client_kwargs["http_client"] = httpx.Client(trust_env=False)
            self.client = OpenAI(**client_kwargs)

        self.qwen_client = None
        if settings.qwen_api_key:
            qwen_kwargs: Dict[str, Any] = {
                "api_key": settings.qwen_api_key,
                "base_url": settings.qwen_base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1",
            }
            qwen_kwargs["http_client"] = httpx.Client(trust_env=False)
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
        if self.client is None:
            raise ValueError("OpenAI client not configured / 未配置 OPENAI_API_KEY")
        return self.client

    def ask(
        self,
        question: str,
        top_k: int | None = None,
        rerank_enabled: bool | None = None,
        rerank_top_n: int | None = None,
        model: str | None = None,
    ) -> Tuple[str, List[RetrievedChunk]]:
        t0 = time.perf_counter()
        k = top_k or self.settings.top_k

        use_rr = (self.reranker is not None) and (
            self.settings.reranker_enabled if rerank_enabled is None else rerank_enabled
        )
        top_n = rerank_top_n or self.settings.reranker_top_n

        # When rerank is enabled, retrieve a larger candidate pool then rerank.
        candidate_k = k
        if use_rr:
            candidate_k = max(k * 4, top_n * 4, 32, k, top_n)
            candidate_k = min(candidate_k, 200)

        namespace = getattr(self.store, "namespace", "default")
        q_hash = hashlib.sha1(question.encode("utf-8", errors="ignore")).hexdigest()[:10]
        _trace_event(
            self.settings,
            "ask_start",
            {
                "ns": namespace,
                "backend": str(getattr(self.store, "backend", "unknown")),
                "q_hash": q_hash,
                "q_len": len(question),
                "top_k": int(k),
                "candidate_k": int(candidate_k),
                "rerank": bool(use_rr),
                "rerank_top_n": int(top_n),
                "bm25_enabled": bool(self.settings.bm25_enabled),
                "bm25_complete_corpus": bool(getattr(self.store, "_bm25_complete_corpus", True)),
                "query_expand_enabled": bool(self.settings.query_expand_enabled),
            },
        )
        retrieval_options = {
            "rev": getattr(self.store, "_corpus_revision", 0),
            "bm25_enabled": bool(self.settings.bm25_enabled),
            "bm25_require_complete_corpus": bool(self.settings.bm25_require_complete_corpus),
            "bm25_full_scan_max_docs": int(self.settings.bm25_full_scan_max_docs or 0),
            "bm25_weight": float(self.settings.bm25_weight),
            "vec_weight": float(self.settings.vec_weight),
            "mmr_lambda": float(self.settings.mmr_lambda),
            "score_threshold": float(self.settings.score_threshold),
            "embedding_model": str(self.settings.embedding_model_name),
            "vector_backend": str(getattr(self.store, "backend", "unknown")),
            "query_expand_enabled": bool(self.settings.query_expand_enabled),
            "candidate_k_mult": int(self.settings.candidate_k_mult or 0),
            "candidate_k_min": int(self.settings.candidate_k_min or 0),
            "candidate_k_max": int(self.settings.candidate_k_max or 0),
        }
        cached_result = query_cache.get(question, candidate_k, namespace, retrieval_options)
        if cached_result is not None:
            # Clone to avoid mutating cached objects (rerank, score updates).
            recs = [RetrievedChunk(text=r.text, score=r.score, meta=dict(r.meta)) for r in cached_result]
            _trace_event(self.settings, "retrieve_done", {"q_hash": q_hash, "cache_hit": True, "n": len(recs)})
        else:
            t_search0 = time.perf_counter()
            recs = self.store.search(question, candidate_k)
            _trace_event(
                self.settings,
                "retrieve_done",
                {"q_hash": q_hash, "cache_hit": False, "ms": int((time.perf_counter() - t_search0) * 1000), "n": len(recs)},
            )
            query_cache.set(
                question,
                candidate_k,
                namespace,
                [RetrievedChunk(text=r.text, score=r.score, meta=dict(r.meta)) for r in recs],
                retrieval_options,
            )

        if use_rr:
            t_rr0 = time.perf_counter()
            pairs = [[question, r.text] for r in recs]
            scores = self.reranker.compute_score(pairs)
            for r, s in zip(recs, scores):
                r.meta["score_rerank"] = float(s)
            recs = sorted(recs, key=lambda x: float(x.meta.get("score_rerank", 0.0)), reverse=True)
            context_k = max(k, top_n)
            recs = recs[:context_k]
            _trace_event(self.settings, "rerank_done", {"q_hash": q_hash, "ms": int((time.perf_counter() - t_rr0) * 1000), "n": len(recs)})
        else:
            recs = recs[:k]

        t_prompt0 = time.perf_counter()
        prompt = build_prompt(
            question,
            recs,
            strict_mode=self.settings.strict_mode,
            show_scores=bool(self.settings.prompt_show_scores),
        )
        _trace_event(self.settings, "prompt_done", {"q_hash": q_hash, "ms": int((time.perf_counter() - t_prompt0) * 1000), "chars": len(prompt)})
        target_model = model or self.settings.llm_model
        client = self._get_client_for_model(target_model)
        t_llm0 = time.perf_counter()
        resp = client.chat.completions.create(
            model=target_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        answer = resp.choices[0].message.content or ""
        _trace_event(
            self.settings,
            "ask_done",
            {
                "q_hash": q_hash,
                "ms_total": int((time.perf_counter() - t0) * 1000),
                "ms_llm": int((time.perf_counter() - t_llm0) * 1000),
                "sources": len(recs),
                "answer_chars": len(answer),
            },
        )
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
        t0 = time.perf_counter()
        k = top_k or self.settings.top_k

        use_rr = (self.reranker is not None) and (
            self.settings.reranker_enabled if rerank_enabled is None else rerank_enabled
        )
        top_n = rerank_top_n or self.settings.reranker_top_n

        candidate_k = k
        if use_rr:
            candidate_k = max(k * 4, top_n * 4, 32, k, top_n)
            candidate_k = min(candidate_k, 200)

        namespace = getattr(self.store, "namespace", "default")
        q_hash = hashlib.sha1(question.encode("utf-8", errors="ignore")).hexdigest()[:10]
        _trace_event(
            self.settings,
            "ask_stream_start",
            {
                "ns": namespace,
                "backend": str(getattr(self.store, "backend", "unknown")),
                "q_hash": q_hash,
                "q_len": len(question),
                "top_k": int(k),
                "candidate_k": int(candidate_k),
                "rerank": bool(use_rr),
                "rerank_top_n": int(top_n),
                "bm25_enabled": bool(self.settings.bm25_enabled),
                "bm25_complete_corpus": bool(getattr(self.store, "_bm25_complete_corpus", True)),
                "query_expand_enabled": bool(self.settings.query_expand_enabled),
                "web_enabled": bool(web_enabled),
            },
        )
        retrieval_options = {
            "rev": getattr(self.store, "_corpus_revision", 0),
            "bm25_enabled": bool(self.settings.bm25_enabled),
            "bm25_require_complete_corpus": bool(self.settings.bm25_require_complete_corpus),
            "bm25_full_scan_max_docs": int(self.settings.bm25_full_scan_max_docs or 0),
            "bm25_weight": float(self.settings.bm25_weight),
            "vec_weight": float(self.settings.vec_weight),
            "mmr_lambda": float(self.settings.mmr_lambda),
            "score_threshold": float(self.settings.score_threshold),
            "embedding_model": str(self.settings.embedding_model_name),
            "vector_backend": str(getattr(self.store, "backend", "unknown")),
            "query_expand_enabled": bool(self.settings.query_expand_enabled),
            "candidate_k_mult": int(self.settings.candidate_k_mult or 0),
            "candidate_k_min": int(self.settings.candidate_k_min or 0),
            "candidate_k_max": int(self.settings.candidate_k_max or 0),
        }
        cached_result = query_cache.get(question, candidate_k, namespace, retrieval_options)
        if cached_result is not None:
            recs = [RetrievedChunk(text=r.text, score=r.score, meta=dict(r.meta)) for r in cached_result]
            _trace_event(self.settings, "retrieve_done", {"q_hash": q_hash, "cache_hit": True, "n": len(recs)})
        else:
            t_search0 = time.perf_counter()
            recs = self.store.search(question, candidate_k)
            _trace_event(
                self.settings,
                "retrieve_done",
                {"q_hash": q_hash, "cache_hit": False, "ms": int((time.perf_counter() - t_search0) * 1000), "n": len(recs)},
            )
            query_cache.set(
                question,
                candidate_k,
                namespace,
                [RetrievedChunk(text=r.text, score=r.score, meta=dict(r.meta)) for r in recs],
                retrieval_options,
            )

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

        if use_rr:
            t_rr0 = time.perf_counter()
            pairs = [[question, r.text] for r in recs]
            scores = self.reranker.compute_score(pairs)
            for r, s in zip(recs, scores):
                r.meta["score_rerank"] = float(s)
            recs = sorted(recs, key=lambda x: float(x.meta.get("score_rerank", 0.0)), reverse=True)
            context_k = max(k, top_n)
            recs = recs[:context_k]
            _trace_event(self.settings, "rerank_done", {"q_hash": q_hash, "ms": int((time.perf_counter() - t_rr0) * 1000), "n": len(recs)})
        else:
            recs = recs[:k]

        t_prompt0 = time.perf_counter()
        prompt = build_prompt(
            question,
            recs,
            strict_mode=self.settings.strict_mode,
            custom_system_prompt=system_prompt,
            show_scores=bool(self.settings.prompt_show_scores),
        )
        _trace_event(self.settings, "prompt_done", {"q_hash": q_hash, "ms": int((time.perf_counter() - t_prompt0) * 1000), "chars": len(prompt)})
        if web_snippets:
            web_block = "\n\n".join(web_snippets)
            prompt = (
                f"{prompt}\n\n"
                f"{'=' * 60}\n"
                "Web Context (untrusted) / 联网片段（不保证准确）：\n"
                f"{web_block}\n"
            )
        target_model = model or self.settings.llm_model
        client = self._get_client_for_model(target_model)

        def _gen():  # noqa: ANN202
            t_llm0 = time.perf_counter()
            first_token_emitted = False
            stream = client.chat.completions.create(
                model=target_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                stream=True,
            )
            for chunk in stream:
                delta = getattr(getattr(chunk.choices[0], "delta", None), "content", None)
                if delta:
                    if not first_token_emitted:
                        first_token_emitted = True
                        _trace_event(
                            self.settings,
                            "ask_stream_first_token",
                            {"q_hash": q_hash, "ms_to_first_token": int((time.perf_counter() - t_llm0) * 1000)},
                        )
                    yield delta
            _trace_event(
                self.settings,
                "ask_stream_done",
                {
                    "q_hash": q_hash,
                    "ms_total": int((time.perf_counter() - t0) * 1000),
                    "ms_stream": int((time.perf_counter() - t_llm0) * 1000),
                    "sources": len(recs),
                },
            )

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
        seen = set()

        for query in rewritten_queries:
            recs = self.store.search(query, k)
            for rec in recs:
                key = (rec.meta.get("path"), rec.meta.get("chunk_id"), rec.text[:200])
                if key in seen:
                    continue
                seen.add(key)
                all_recs.append(rec)

        use_rr = (self.reranker is not None) and (
            self.settings.reranker_enabled if rerank_enabled is None else rerank_enabled
        )
        top_n = rerank_top_n or self.settings.reranker_top_n

        if use_rr and all_recs:
            pairs = [[question, r.text] for r in all_recs]
            scores = self.reranker.compute_score(pairs)
            for r, s in zip(all_recs, scores):
                r.meta["score_rerank"] = float(s)
            all_recs = sorted(all_recs, key=lambda x: float(x.meta.get("score_rerank", 0.0)), reverse=True)[:top_n]
        else:
            all_recs = sorted(all_recs, key=lambda x: x.score, reverse=True)[:top_n]

        prompt = build_prompt(
            question,
            all_recs,
            strict_mode=self.settings.strict_mode,
            show_scores=bool(self.settings.prompt_show_scores),
        )
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
            "unique_chunks": len(seen),
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
