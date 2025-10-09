import json
import os
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from pymilvus import connections, Collection
from ingest import split_text
from config import Settings

try:
    from FlagEmbedding import FlagReranker  # type: ignore
except Exception:  # pragma: no cover
    FlagReranker = None  # type: ignore

from config import Settings


@dataclass
class RetrievedChunk:
    text: str
    score: float
    meta: Dict[str, Any]


class VectorStore:
    def __init__(self, meta_path: str, embedding_model: str, settings: Settings, namespace: str | None = None) -> None:
        if not os.path.exists(meta_path):
            raise FileNotFoundError("未找到 meta.jsonl，请先运行 ingest 构建索引")
        self.texts: List[str] = []
        self.metas: List[Dict[str, Any]] = []
        self.meta_path = meta_path
        with open(meta_path, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                self.texts.append(rec["text"]) 
                self.metas.append(rec["meta"]) 
        self.embedder = SentenceTransformer(embedding_model)

        # 尝试使用 Milvus，否则回退到 FAISS
        self.collection = None
        self.faiss_index = None
        self.backend = "faiss"
        collection_name = f"{settings.milvus_collection}_{(namespace or settings.default_namespace)}"
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
            except Exception:
                self.collection = None
                self.backend = "faiss"
        if self.collection is None:
            faiss_path = os.path.join(os.path.dirname(meta_path), "faiss.index")
            self.faiss_path = faiss_path
            if not os.path.exists(faiss_path):
                raise FileNotFoundError("未找到 Milvus 集合且缺少 faiss.index，请先运行 ingest 构建索引")
            try:
                import faiss  # type: ignore
            except Exception as exc:  # pragma: no cover
                raise RuntimeError("需要 FAISS 以读取本地回退索引，请使用 conda 安装 faiss-cpu: conda install -n rag-env -c conda-forge faiss-cpu") from exc
            self.faiss_index = faiss.read_index(faiss_path)

    def search(self, query: str, top_k: int = 4) -> List[RetrievedChunk]:
        vec = self.embedder.encode([query], normalize_embeddings=True)
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
            return results
        # faiss 回退
        assert self.faiss_index is not None
        # 动态导入 faiss，避免未安装时报错
        import faiss  # type: ignore
        scores, indices = self.faiss_index.search(vec.reshape(1, -1), top_k)
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append(RetrievedChunk(text=self.texts[idx], score=float(score), meta=self.metas[idx]))
        return results

    def add_document(self, path: str, text: str) -> int:
        # 支持两种后端的在线新增
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

        # FAISS 本地模式：动态追加并写回索引与 meta
        if self.backend == "faiss" and self.faiss_index is not None:
            chunks = split_text(text)
            if not chunks:
                return 0
            import faiss  # type: ignore
            embeddings = self.embedder.encode(chunks, normalize_embeddings=True)
            vecs = np.array(embeddings).astype("float32")
            self.faiss_index.add(vecs)
            # 写回索引文件
            faiss.write_index(self.faiss_index, getattr(self, "faiss_path", os.path.join(os.path.dirname(self.meta_path), "faiss.index")))
            # 追加 meta.jsonl 与内存映射
            with open(self.meta_path, "a", encoding="utf-8") as f:
                for idx, chunk in enumerate(chunks):
                    meta = {"path": path, "chunk_id": idx, "chunk_size": len(chunk)}
                    rec = {"meta": meta, "text": chunk}
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    self.metas.append(meta)
                    self.texts.append(chunk)
            return len(chunks)

        raise RuntimeError("当前向量后端未就绪，无法新增文档")
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

    def delete_document(self, path: str) -> int:
        if self.backend != "milvus" or self.collection is None:
            raise RuntimeError("FAISS 模式下暂不支持在线删除文档，请通过 ingest 重建索引")
        escaped = path.replace("'", "\\'")
        expr = "path == '" + escaped + "'"
        res = self.collection.delete(expr)
        self.collection.flush()
        cnt = 0
        try:
            cnt = int(getattr(res, "delete_count", 0))
        except Exception:
            cnt = 0
        return cnt

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
        # 从本地 meta 去重
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


def build_prompt(question: str, contexts: List[RetrievedChunk]) -> str:
    context_blocks = []
    for i, c in enumerate(contexts, start=1):
        path = c.meta.get("path", "")
        context_blocks.append(f"[片段{i}] 来源: {path}\n{c.text}")
    context_text = "\n\n".join(context_blocks)
    prompt = (
        "你是一个严谨的中文知识助手。\n"
        "请使用下列检索到的资料片段来回答问题，严格基于资料作答。\n"
        "若资料无法回答，请明确说明“资料未覆盖”，不要编造。\n\n"
        f"资料片段：\n{context_text}\n\n"
        f"问题：{question}\n\n"
        "请给出简洁、条理清晰的回答。"
    )
    return prompt


class RAGPipeline:
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
        self.reranker = None
        if settings.reranker_enabled and FlagReranker is not None:
            try:
                self.reranker = FlagReranker(settings.reranker_model_name, use_fp16=True)
            except Exception:
                self.reranker = None

    def ask(self, question: str, top_k: int | None = None, rerank_enabled: bool | None = None, rerank_top_n: int | None = None, model: str | None = None) -> Tuple[str, List[RetrievedChunk]]:
        k = top_k or self.settings.top_k
        recs = self.store.search(question, k)
        # 可选重排
        use_rr = (self.reranker is not None) and (self.settings.reranker_enabled if rerank_enabled is None else rerank_enabled)
        top_n = rerank_top_n or self.settings.reranker_top_n
        if use_rr:
            pairs = [[question, r.text] for r in recs]
            scores = self.reranker.compute_score(pairs)
            for r, s in zip(recs, scores):
                r.score = float(s)
            recs = sorted(recs, key=lambda x: x.score, reverse=True)[: top_n]
        prompt = build_prompt(question, recs)
        resp = self.client.chat.completions.create(
            model=(model or self.settings.llm_model),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        answer = resp.choices[0].message.content or ""
        return answer, recs

    def ask_stream(self, question: str, top_k: int | None = None, rerank_enabled: bool | None = None, rerank_top_n: int | None = None, model: str | None = None):  # noqa: ANN001
        """返回(生成器, 检索片段)。生成器逐块产出模型文本。"""
        k = top_k or self.settings.top_k
        recs = self.store.search(question, k)
        use_rr = (self.reranker is not None) and (self.settings.reranker_enabled if rerank_enabled is None else rerank_enabled)
        top_n = rerank_top_n or self.settings.reranker_top_n
        if use_rr:
            pairs = [[question, r.text] for r in recs]
            scores = self.reranker.compute_score(pairs)
            for r, s in zip(recs, scores):
                r.score = float(s)
            recs = sorted(recs, key=lambda x: x.score, reverse=True)[: top_n]
        prompt = build_prompt(question, recs)

        def _gen():  # noqa: ANN202
            stream = self.client.chat.completions.create(
                model=(model or self.settings.llm_model),
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                stream=True,
            )
            for chunk in stream:
                delta = getattr(getattr(chunk.choices[0], "delta", None), "content", None)
                if delta:
                    yield delta

        return _gen(), recs

    def add_document(self, path: str, text: str) -> int:
        return self.store.add_document(path, text)

    def delete_document(self, path: str) -> int:
        return self.store.delete_document(path)

    def list_paths(self, limit: int = 1000) -> List[str]:
        return self.store.list_paths(limit)


