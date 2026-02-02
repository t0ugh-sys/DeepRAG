import json
import os
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from pymilvus import connections, Collection
from backend.ingest import split_text
from backend.config import Settings
from backend.utils.cache import query_cache
from rank_bm25 import BM25Okapi  # type: ignore

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
    """
    向量存储抽象层，支持 Milvus �?FAISS 双后�?
    
    自动尝试连接 Milvus，失败时回退到本�?FAISS 索引
    """
    
    def __init__(self, meta_path: str, embedding_model: str, settings: Settings, namespace: str | None = None) -> None:
        """
        初始化向量存�?
        
        Args:
            meta_path: 元数据文件路�?(meta.jsonl)
            embedding_model: 向量化模型名�?
            settings: 配置对象
            namespace: 命名空间
        """
        if not os.path.exists(meta_path):
            raise FileNotFoundError("未找�?meta.jsonl，请先运�?ingest 构建索引")
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
        # BM25 语料（按词分�?
        self._bm25_tokenized = [self._tokenize(t) for t in self.texts]
        self._bm25 = BM25Okapi(self._bm25_tokenized) if self._bm25_tokenized else None

        # 尝试使用 Milvus，否则回退�?FAISS
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
            except Exception as e:
                # Milvus 连接失败，回退�?FAISS
                import logging
                logging.getLogger("rag").debug(f"Milvus 连接失败，回退�?FAISS: {e}")
                self.collection = None
                self.backend = "faiss"
        if self.collection is None:
            faiss_path = os.path.join(os.path.dirname(meta_path), "faiss.index")
            self.faiss_path = faiss_path
            if not os.path.exists(faiss_path):
                raise FileNotFoundError("未找�?Milvus 集合且缺�?faiss.index，请先运�?ingest 构建索引")
            try:
                import faiss  # type: ignore
            except Exception as exc:  # pragma: no cover
                raise RuntimeError("需�?FAISS 以读取本地回退索引，请使用 conda 安装 faiss-cpu: conda install -n rag-env -c conda-forge faiss-cpu") from exc
            self.faiss_index = faiss.read_index(faiss_path)

    def _expand_query(self, query: str) -> str:
        """查询扩展：提取关键词，生成多个查询变�?""
        try:
            import jieba.analyse
            # 提取关键词（TF-IDF�?
            keywords = jieba.analyse.extract_tags(query, topK=5, withWeight=False)
            # 将关键词组合回原查询
            expanded = query + " " + " ".join(keywords)
            return expanded
        except Exception:
            return query
    
    def search(self, query: str, top_k: int = 4) -> List[RetrievedChunk]:
        # 查询扩展
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
        # faiss 回退
        assert self.faiss_index is not None
        # 动态导�?faiss，避免未安装时报�?
        import faiss  # type: ignore
        scores, indices = self.faiss_index.search(vec.reshape(1, -1), top_k)
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append(RetrievedChunk(text=self.texts[idx], score=float(score), meta=self.metas[idx]))

        # 可�?BM25 融合
        return self._fuse_with_bm25(query, _dedupe_results(results), top_k)

    # --- 辅助：BM25 + MMR 融合 ---
    def _tokenize(self, s: str) -> list[str]:
        """中文友好的分词：结合 jieba 分词和字符级分词"""
        import re
        try:
            import jieba
            # 使用 jieba 进行中文分词
            words = list(jieba.cut_for_search(s.lower()))  # 搜索引擎模式，更细粒�?
            # 过滤掉空白和单字符标�?
            return [w.strip() for w in words if w.strip() and not re.match(r'^[\W_]+$', w)]
        except ImportError:
            # 如果没有 jieba，回退到简单分�?
            return [w for w in re.split(r"\W+", s.lower()) if w]

    def _fuse_with_bm25(self, query: str, vec_results: List[RetrievedChunk], top_k: int) -> List[RetrievedChunk]:
        settings = Settings()
        recs = list(vec_results)
        if settings.bm25_enabled and self._bm25 is not None:
            tokens = self._tokenize(query)
            bm25_scores = self._bm25.get_scores(tokens)
            # 归一化分�?
            import math
            def norm(x: float) -> float:
                return 0.0 if math.isnan(x) else float(x)
            bm25_max = max(bm25_scores) if len(bm25_scores) else 1.0
            fused: dict[int, float] = {}
            for i, r in enumerate(recs):
                bm = (bm25_scores[i] / (bm25_max or 1.0)) if i < len(bm25_scores) else 0.0
                fused[i] = settings.vec_weight * r.score + settings.bm25_weight * norm(bm)
                r.score = fused[i]
            # 得分阈值过�?
            if settings.score_threshold > 0:
                recs = [r for r in recs if r.score >= settings.score_threshold]
            recs = sorted(recs, key=lambda x: x.score, reverse=True)

        # MMR 多样性采�?
        if len(recs) > top_k:
            recs = self._mmr(query, recs, top_k, lambda_weight=settings.mmr_lambda)
        return recs

    def _mmr(self, query: str, recs: List[RetrievedChunk], k: int, lambda_weight: float = 0.5) -> List[RetrievedChunk]:
        # 使用嵌入空间相似度近似去冗余
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
                # 先选与 query 最相似
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
            selected.append(best_idx)  # type: ignore[arg-type]
            remaining.remove(best_idx)  # type: ignore[arg-type]
        return [recs[i] for i in selected]

    def add_document(self, path: str, text: str) -> int:
        # 支持两种后端的在线新�?
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

        # FAISS 本地模式：动态追加并写回索引�?meta
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
            # 追加 meta.jsonl 与内存映�?
            with open(self.meta_path, "a", encoding="utf-8") as f:
                for idx, chunk in enumerate(chunks):
                    meta = {"path": path, "chunk_id": idx, "chunk_size": len(chunk)}
                    rec = {"meta": meta, "text": chunk}
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    self.metas.append(meta)
                    self.texts.append(chunk)
            return len(chunks)

        raise RuntimeError("当前向量后端未就绪，无法新增文档")

    def delete_document(self, path: str) -> int:
        # Milvus 在线删除
        if self.backend == "milvus" and self.collection is not None:
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

        # FAISS 本地删除：过�?meta.jsonl，并重建索引
        if self.backend == "faiss" and self.faiss_index is not None:
            # 过滤内存中的文本与元数据
            remain_texts: List[str] = []
            remain_metas: List[Dict[str, Any]] = []
            removed = 0
            for t, m in zip(self.texts, self.metas):
                if str(m.get("path")) == path:
                    removed += 1
                    continue
                remain_texts.append(t)
                remain_metas.append(m)

            # 重写 meta.jsonl
            with open(self.meta_path, "w", encoding="utf-8") as f:
                for m, t in zip(remain_metas, remain_texts):
                    f.write(json.dumps({"meta": m, "text": t}, ensure_ascii=False) + "\n")

            # 重新编码剩余文本并重�?faiss 索引
            if remain_texts:
                import faiss  # type: ignore
                vecs = self.embedder.encode(remain_texts, normalize_embeddings=True)
                vecs = np.array(vecs).astype("float32")
                index = faiss.IndexFlatIP(vecs.shape[1])
                index.add(vecs)
                faiss.write_index(index, getattr(self, "faiss_path", os.path.join(os.path.dirname(self.meta_path), "faiss.index")))
                self.faiss_index = index
            else:
                # 空库：重建一个空索引
                import faiss  # type: ignore
                dim = int(self.embedder.get_sentence_embedding_dimension()) if hasattr(self.embedder, 'get_sentence_embedding_dimension') else len(self.embedder.encode(["dim"], normalize_embeddings=True)[0])
                index = faiss.IndexFlatIP(dim)
                faiss.write_index(index, getattr(self, "faiss_path", os.path.join(os.path.dirname(self.meta_path), "faiss.index")))
                self.faiss_index = index

            # 更新内存
            self.texts = remain_texts
            self.metas = remain_metas
            return removed

        raise RuntimeError("当前向量后端未就绪，无法删除文档")

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
        # 从本�?meta 去重
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
        """返回文档路径及其统计信息（分片数、最后更新时间等�?""
        from collections import Counter
        from datetime import datetime
        
        if self.backend == "milvus" and self.collection is not None:
            try:
                recs = self.collection.query(expr="", output_fields=["path", "chunk_id"], limit=limit * 100)
            except Exception:
                recs = []
            
            # 统计每个路径的分片数
            path_chunks = Counter(r.get("path") for r in recs if r.get("path"))
            result = []
            for path, chunk_count in list(path_chunks.items())[:limit]:
                result.append({
                    "path": path,
                    "chunk_count": chunk_count,
                    "last_updated": datetime.now().isoformat()  # Milvus 暂不支持时间�?
                })
            return result
        
        # FAISS 后端：从本地 meta 统计
        path_chunks = Counter(m.get("path") for m in self.metas if m.get("path"))
        result = []
        for path, chunk_count in list(path_chunks.items())[:limit]:
            result.append({
                "path": path,
                "chunk_count": chunk_count,
                "last_updated": datetime.now().isoformat()
            })
        return result


def build_prompt(question: str, contexts: List[RetrievedChunk], strict_mode: bool = True, custom_system_prompt: str | None = None) -> str:
    """
    构建 RAG 提示�?
    
    Args:
        question: 用户问题
        contexts: 检索到的上下文片段
        strict_mode: 严格模式。True=仅基于知识库回答；False=允许模型自由发挥
        custom_system_prompt: 自定义系统提示词，会替换{context}和{question}占位�?
    """
    # 检查是否有有效的上下文（分数阈值或为空�?
    has_valid_context = len(contexts) > 0 and any(c.score > 0.1 for c in contexts)
    
    if not has_valid_context and strict_mode:
        # 严格模式：没有命中知识库时，明确告知用户
        prompt = (
            "你是一个严谨的知识库检索助手。\n"
            f"用户问题：{question}\n\n"
            "检索结果：未在知识库中找到相关信息。\n\n"
            "请礼貌地告诉用户：\n"
            "1. 当前知识库中没有找到与该问题相关的资料\n"
            "2. 建议用户补充相关文档到知识库，或换个方式提问\n"
            "3. 不要编造或猜测答案"
        )
        return prompt
    
    # 有上下文或非严格模式：正常构建提示词
    context_blocks = []
    for i, c in enumerate(contexts, start=1):
        path = c.meta.get("path", "")
        # 提取文件名而非完整路径，更简�?
        filename = path.split('/')[-1] if '/' in path else path.split('\\')[-1] if '\\' in path else path
        score = f"相关�? {c.score:.2f}"
        context_blocks.append(f"[文档{i}: {filename}]\n{c.text}")
    context_text = "\n\n".join(context_blocks)
    
    # 如果有自定义提示词，使用它并替换占位符；若缺少占位符则自动补�?
    if custom_system_prompt:
        tpl = custom_system_prompt
        includes_context = "{context}" in tpl
        includes_question = "{question}" in tpl
        prompt = tpl.replace("{context}", context_text).replace("{question}", question)

        # 兜底：确保问题与上下文一定出现在最终提示中
        if not includes_question or not includes_context:
            prompt = (
                f"{prompt}\n\n"
                f"{'='*60}\n"
                f"用户问题：{question}\n\n"
                f"检索到的知识库文档（共 {len(contexts)} 个片段）：\n\n{context_text}\n"
            )
        return prompt
    
    # 否则使用默认提示�?
    if strict_mode:
        system_instruction = (
            "你是一个专业的知识库检索助手。\n\n"
            "**核心规则**：\n"
            "1. 仔细阅读下列所有文档片段，全面理解其内容\n"
            "2. 从文档中寻找与问题相关的**所有信�?*，包括直接和间接相关的内容\n"
            "3. 综合多个文档片段的信息进行回答\n"
            "4. 如果文档中确实没有答案，明确告知用户\n"
            "5. 回答要详细、具体，尽可能引用原文\n\n"
            "**输出格式要求**：\n"
            "1. 使用规范的中文标点符号（，。；！？）\n"
            "2. 合理分段，每段讲一个主题\n"
            "3. 使用标题、列表等 Markdown 格式提高可读性\n"
            "4. 数字和英文前后加空格（例如：YOLOv8 的结构）\n"
            "5. 避免句子过长，适当断句\n\n"
            "**注意**：即使某个文档片段看起来相关度不高，也要仔细检查是否包含有用信息�?
        )
    else:
        system_instruction = (
            "你是一个智能助手。\n\n"
            "**任务**：\n"
            "1. 优先使用下列文档片段中的信息\n"
            "2. 如果文档信息不足，可以结合你的知识进行补充\n"
            "3. 明确标注哪些来自文档，哪些是你的补充\n"
        )
    
    prompt = (
        f"{system_instruction}\n"
        f"{'='*60}\n"
        f"检索到的知识库文档（共 {len(contexts)} 个片段）：\n\n"
        f"{context_text}\n"
        f"{'='*60}\n\n"
        f"用户问题：{question}\n\n"
        "请基于以上文档，给出详细、准确的回答�?
    )
    return prompt


class RAGPipeline:
    """
    RAG (检索增强生�? 主流程管�?
    
    整合向量检索、BM25、Reranker、查询改写和 LLM 生成，提供完整的问答能力
    """
    
    def __init__(self, settings: Settings, namespace: str | None = None) -> None:
        """
        初始�?RAG Pipeline
        
        Args:
            settings: 配置对象
            namespace: 命名空间，用于多租户隔离
        """
        self.settings = settings
        meta_path = os.path.join(settings.index_dir, "meta.jsonl")
        self.store = VectorStore(meta_path, settings.embedding_model_name, settings, namespace)

        # 默认使用 DeepSeek (OpenAI) 配置
        client_kwargs: Dict[str, Any] = {}
        if settings.openai_api_key:
            client_kwargs["api_key"] = settings.openai_api_key
        if settings.openai_base_url:
            client_kwargs["base_url"] = settings.openai_base_url
        self.client = OpenAI(**client_kwargs)
        
        # �?Qwen 创建单独的客户端
        self.qwen_client = None
        if settings.qwen_api_key:
            qwen_kwargs: Dict[str, Any] = {
                "api_key": settings.qwen_api_key,
                "base_url": settings.qwen_base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1"
            }
            self.qwen_client = OpenAI(**qwen_kwargs)
        
        self.reranker = None
        if settings.reranker_enabled and FlagReranker is not None:
            try:
                self.reranker = FlagReranker(settings.reranker_model_name, use_fp16=True)
            except Exception:
                self.reranker = None
        
        # 初始化查询改写器（可选）
        self.query_rewriter = None
        if settings.openai_api_key and settings.openai_base_url:
            try:
                from backend.query_rewriter import QueryRewriter
                self.query_rewriter = QueryRewriter(
                    api_key=settings.openai_api_key,
                    base_url=settings.openai_base_url,
                    model=settings.llm_model
                )
            except Exception as e:
                import logging
                logging.getLogger("rag").warning(f"查询改写器初始化失败: {e}")
    
    def _get_client_for_model(self, model: str) -> OpenAI:
        """根据模型名称选择对应的客户端"""
        if model and model.startswith("qwen"):
            if self.qwen_client is None:
                raise ValueError(f"Qwen 模型 '{model}' 需要配�?QWEN_API_KEY")
            return self.qwen_client
        return self.client

    def ask(self, question: str, top_k: int | None = None, rerank_enabled: bool | None = None, rerank_top_n: int | None = None, model: str | None = None) -> Tuple[str, List[RetrievedChunk]]:
        k = top_k or self.settings.top_k
        
        # 尝试从缓存获取检索结�?
        namespace = getattr(self.store, 'namespace', 'default')
        cached_result = query_cache.get(question, k, namespace)
        if cached_result is not None:
            recs = cached_result
        else:
            recs = self.store.search(question, k)
            # 缓存检索结�?
            query_cache.set(question, k, namespace, recs)
        # 可选重�?
        use_rr = (self.reranker is not None) and (self.settings.reranker_enabled if rerank_enabled is None else rerank_enabled)
        top_n = rerank_top_n or self.settings.reranker_top_n
        if use_rr:
            pairs = [[question, r.text] for r in recs]
            scores = self.reranker.compute_score(pairs)
            for r, s in zip(recs, scores):
                r.score = float(s)
            recs = sorted(recs, key=lambda x: x.score, reverse=True)[: top_n]
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

    def ask_stream(self, question: str, top_k: int | None = None, rerank_enabled: bool | None = None, rerank_top_n: int | None = None, model: str | None = None, system_prompt: str | None = None, web_enabled: bool | None = None, web_top_k: int | None = None):  # noqa: ANN001
        """返回(生成�? 检索片�?。生成器逐块产出模型文本�?""
        k = top_k or self.settings.top_k
        recs = self.store.search(question, k)

        # 可选：联网搜索补充实时信息（简单实现：调用 DuckDuckGo html API�?
        web_snippets: list[str] = []
        if web_enabled:
            try:
                import requests
                import re
                n = web_top_k or 3
                q = requests.utils.quote(question)
                url = f"https://duckduckgo.com/html/?q={q}"
                html = requests.get(url, timeout=5).text
                # 极简抓取搜索结果摘要
                results = re.findall(r'<a rel="nofollow" class="result__a" href="(.*?)".*?</a>.*?<a.*?class="result__snippet".*?>(.*?)</a>', html, flags=re.S)
                for link, snippet in results[:n]:
                    text = re.sub('<.*?>', '', snippet)
                    web_snippets.append(f"[Web] {text}\nURL: {link}")
            except Exception:
                pass
        use_rr = (self.reranker is not None) and (self.settings.reranker_enabled if rerank_enabled is None else rerank_enabled)
        top_n = rerank_top_n or self.settings.reranker_top_n
        if use_rr:
            pairs = [[question, r.text] for r in recs]
            scores = self.reranker.compute_score(pairs)
            for r, s in zip(recs, scores):
                r.score = float(s)
            recs = sorted(recs, key=lambda x: x.score, reverse=True)[: top_n]
        # �?web 片段拼接到上下文尾部
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
        model: str | None = None
    ) -> Tuple[str, List[RetrievedChunk], Dict[str, Any]]:
        """
        使用查询改写增强检索效�?
        
        Args:
            question: 原始查询
            strategy: 改写策略 (expand/decompose/hyde/multi)
            top_k: 每个查询返回的文档数
            rerank_enabled: 是否启用重排
            rerank_top_n: 重排后保留的文档�?
            model: LLM 模型
        
        Returns:
            (答案, 检索片段列�? 元数�?
            元数据包�? original_query, rewritten_queries, strategy
        """
        if not self.query_rewriter:
            # 查询改写器未初始化，回退到普通检�?
            answer, recs = self.ask(question, top_k, rerank_enabled, rerank_top_n, model)
            return answer, recs, {
                "original_query": question,
                "rewritten_queries": [question],
                "strategy": "none",
                "note": "查询改写器未启用"
            }
        
        # 1. 改写查询
        rewritten_queries = self.query_rewriter.rewrite_for_retrieval(question, strategy)
        
        # 2. 对每个改写后的查询进行检�?
        k = top_k or self.settings.top_k
        all_recs: List[RetrievedChunk] = []
        seen_texts = set()
        
        for query in rewritten_queries:
            recs = self.store.search(query, k)
            # 去重：避免相同文档片段重复出�?
            for rec in recs:
                if rec.text not in seen_texts:
                    seen_texts.add(rec.text)
                    all_recs.append(rec)
        
        # 3. 对合并后的结果进行重排（可选）
        use_rr = (self.reranker is not None) and (self.settings.reranker_enabled if rerank_enabled is None else rerank_enabled)
        top_n = rerank_top_n or self.settings.reranker_top_n
        
        if use_rr and all_recs:
            # 使用原始查询进行重排
            pairs = [[question, r.text] for r in all_recs]
            scores = self.reranker.compute_score(pairs)
            for r, s in zip(all_recs, scores):
                r.score = float(s)
            all_recs = sorted(all_recs, key=lambda x: x.score, reverse=True)[:top_n]
        else:
            # 按原始分数排序并限制数量
            all_recs = sorted(all_recs, key=lambda x: x.score, reverse=True)[:top_n]
        
        # 4. 生成答案
        prompt = build_prompt(question, all_recs, strict_mode=self.settings.strict_mode)
        target_model = model or self.settings.llm_model
        client = self._get_client_for_model(target_model)
        resp = client.chat.completions.create(
            model=target_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        answer = resp.choices[0].message.content or ""
        
        # 5. 返回结果和元数据
        metadata = {
            "original_query": question,
            "rewritten_queries": rewritten_queries,
            "strategy": strategy,
            "total_retrieved": len(all_recs),
            "unique_documents": len(seen_texts)
        }
        
        return answer, all_recs, metadata
    
    def analyze_query(self, question: str) -> Dict[str, Any]:
        """
        分析查询特征并推荐最佳改写策�?
        
        Args:
            question: 用户查询
        
        Returns:
            分析结果字典
        """
        if not self.query_rewriter:
            return {
                "error": "查询改写器未启用",
                "recommended_strategy": "none"
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

