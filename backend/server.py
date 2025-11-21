from typing import List, Any, Dict
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi import UploadFile, File, Form
from pymilvus import connections as _milvus_conn, FieldSchema as _FieldSchema, CollectionSchema as _CollectionSchema, DataType as _DataType, Collection as _Collection, utility as _utility
from sentence_transformers import SentenceTransformer as _ST
from pydantic import BaseModel

from backend.config import Settings
from backend.rag import RAGPipeline, RetrievedChunk
from backend.utils.logger import logger
from backend.utils.middleware import RequestLoggingMiddleware
from backend.utils.responses import success_response, error_response
from backend.utils.cache import query_cache
from backend.performance_monitor import get_monitor, RequestTimer

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    logger.info("🚀 RAG 服务启动中...")
    # 初始化 pipeline
    global pipeline
    try:
        logger.info("加载 RAG Pipeline...")
        pipeline = RAGPipeline(settings, settings.default_namespace)
        logger.info("✓ RAG Pipeline 加载完成")
    except Exception as e:
        logger.error(f"✗ RAG Pipeline 加载失败: {e}")
        raise
    yield
    logger.info("👋 RAG 服务关闭")

app = FastAPI(
    title="Local RAG API",
    version="0.2.0",
    lifespan=lifespan
)

# 添加请求日志中间件
app.add_middleware(RequestLoggingMiddleware)
# CORS 配置：开发环境允许所有源；生产环境建议限制具体域名
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "*"],  # Vite 默认端口
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 前端已迁移到独立 Vue 项目，不再需要静态文件挂载
@app.get("/")
def root() -> dict:  # type: ignore[override]
    return {"message": "RAG API Server", "version": "0.2.0", "docs": "/docs"}


class AskRequest(BaseModel):
    question: str
    top_k: int | None = None
    rerank_enabled: bool | None = None
    rerank_top_n: int | None = None
    model: str | None = None
    system_prompt: str | None = None
    web_enabled: bool | None = None
    web_top_k: int | None = None


class SourceItem(BaseModel):
    path: str
    chunk_id: int | None = None
    score: float
    snippet: str


class AskResponse(BaseModel):
    answer: str
    sources: List[SourceItem]


settings = Settings()
pipeline = None

def _require_api_key(headers: Dict[str, str]) -> None:
    if settings.api_key:
        key = headers.get('x-api-key') or headers.get('X-API-Key') or headers.get('X-Api-Key')
        if key != settings.api_key:
            from fastapi import HTTPException
            raise HTTPException(status_code=401, detail='Unauthorized')




def _collection_name(ns: str) -> str:
    return f"{settings.milvus_collection}_{ns}"


def _ensure_milvus_conn() -> None:
    _milvus_conn.connect(
        alias="default",
        host=settings.milvus_host,
        port=settings.milvus_port,
        user=settings.milvus_user,
        password=settings.milvus_password,
        secure=settings.milvus_secure,
    )


def _embedding_dim() -> int:
    model = _ST(settings.embedding_model_name)
    v = model.encode(["dim"], normalize_embeddings=True)
    return int(len(v[0]))


def _create_collection(ns: str) -> None:
    _ensure_milvus_conn()
    name = _collection_name(ns)
    if _utility.has_collection(name):
        return
    dim = _embedding_dim()
    fields = [
        _FieldSchema(name="id", dtype=_DataType.INT64, is_primary=True, auto_id=True),
        _FieldSchema(name="path", dtype=_DataType.VARCHAR, max_length=512),
        _FieldSchema(name="chunk_id", dtype=_DataType.INT64),
        _FieldSchema(name="text", dtype=_DataType.VARCHAR, max_length=32768),
        _FieldSchema(name="embedding", dtype=_DataType.FLOAT_VECTOR, dim=dim),
    ]
    schema = _CollectionSchema(fields, description="RAG 文档分片")
    coll = _Collection(name, schema=schema)
    index_params = {"index_type": "IVF_FLAT", "metric_type": "IP", "params": {"nlist": 1024}}
    coll.create_index(field_name="embedding", index_params=index_params)


@app.post("/namespaces/create")
def ns_create(namespace: str | None = None, x_api_key: str | None = None) -> JSONResponse:  # type: ignore[override]
    _require_api_key({"x-api-key": x_api_key} if x_api_key else {})
    ns = namespace or settings.default_namespace
    try:
        _create_collection(ns)
        return JSONResponse({"ok": True, "namespace": ns, "collection": _collection_name(ns)})
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)


@app.post("/namespaces/clear")
def ns_clear(namespace: str | None = None, x_api_key: str | None = None) -> JSONResponse:  # type: ignore[override]
    _require_api_key({"x-api-key": x_api_key} if x_api_key else {})
    ns = namespace or settings.default_namespace
    try:
        _ensure_milvus_conn()
        name = _collection_name(ns)
        if _utility.has_collection(name):
            _Collection(name).drop()
        _create_collection(ns)
        return JSONResponse({"ok": True, "namespace": ns, "cleared": True})
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)


@app.delete("/namespaces")
def ns_delete(namespace: str | None = None, x_api_key: str | None = None) -> JSONResponse:  # type: ignore[override]
    _require_api_key({"x-api-key": x_api_key} if x_api_key else {})
    ns = namespace or settings.default_namespace
    try:
        _ensure_milvus_conn()
        name = _collection_name(ns)
        if _utility.has_collection(name):
            _Collection(name).drop()
        return JSONResponse({"ok": True, "namespace": ns, "deleted": True})
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest, x_api_key: str | None = None, namespace: str | None = None) -> AskResponse:  # type: ignore[override]
    _require_api_key({"x-api-key": x_api_key} if x_api_key else {})
    if pipeline is None:
        raise HTTPException(status_code=503, detail="RAG Pipeline 未初始化，请检查服务日志")
    ns = namespace or settings.default_namespace
    local = RAGPipeline(settings, ns)
    answer, recs = local.ask(req.question, req.top_k, req.rerank_enabled, req.rerank_top_n, req.model)
    sources: List[SourceItem] = []
    for r in recs:
        sources.append(SourceItem(
            path=str(r.meta.get("path")),
            chunk_id=int(r.meta.get("chunk_id")) if r.meta.get("chunk_id") is not None else None,
            score=float(r.score),
            snippet=r.text[:400]
        ))
    return AskResponse(answer=answer, sources=sources)


@app.get("/models")
def get_available_models() -> JSONResponse:  # type: ignore[override]
    """获取可用的 LLM 模型列表"""
    models = settings.available_models.split(",")
    return JSONResponse({
        "ok": True,
        "models": [m.strip() for m in models if m.strip()],
        "default_model": settings.llm_model
    })


@app.get("/healthz")
def healthz() -> JSONResponse:  # type: ignore[override]
    """健康检查与监控指标"""
    ok = True
    details: Dict[str, Any] = {
        "timestamp": time.time(),
        "version": "0.2.0"
    }
    
    try:
        if pipeline is None:
            raise Exception("RAG Pipeline 未初始化")
        store = pipeline.store  # type: ignore[attr-defined]
        coll = getattr(store, "collection", None)
        active_backend = getattr(store, "backend", None)
        
        if coll is not None:
            try:
                coll.load()
                num = coll.num_entities  # type: ignore[attr-defined]
                details.update({
                    "milvus_collection": coll.name if hasattr(coll, "name") else None,
                    "milvus_entities": int(num) if isinstance(num, int) else num,
                })
                
                # 统计唯一文档数
                try:
                    unique_paths = pipeline.list_paths(limit=10000)
                    details["document_count"] = len(unique_paths)
                except Exception:
                    details["document_count"] = 0
                    
            except Exception as e:
                logger.warning(f"Milvus 状态检查失败: {e}")
                details["milvus_warning"] = str(e)
                details["document_count"] = 0
        else:
            # FAISS 或其他后端：统计文档数
            try:
                unique_paths = pipeline.list_paths(limit=10000)
                details["document_count"] = len(unique_paths)
                # 对于 FAISS，可以获取实体数
                if active_backend == "faiss":
                    faiss_index = getattr(store, "faiss_index", None)
                    if faiss_index is not None:
                        details["faiss_entities"] = faiss_index.ntotal
            except Exception:
                details["document_count"] = 0
        
        # 基础指标
        details.update({
            "embedding_model": pipeline.settings.embedding_model_name,
            "llm_model": pipeline.settings.llm_model,
            "top_k": pipeline.settings.top_k,
            "vector_backend_config": pipeline.settings.vector_backend,
            "vector_backend_active": active_backend,
            "bm25_enabled": pipeline.settings.bm25_enabled,
            "reranker_enabled": pipeline.settings.reranker_enabled,
        })
        
        logger.debug("健康检查完成")
        
    except Exception as exc:
        ok = False
        details["error"] = str(exc)
        logger.error(f"健康检查失败: {exc}")
    
    return JSONResponse({"ok": ok, "details": details})


@app.post("/ask_stream")
def ask_stream(req: AskRequest, x_api_key: str | None = None, namespace: str | None = None):  # type: ignore[override]
    _require_api_key({"x-api-key": x_api_key} if x_api_key else {})
    if pipeline is None:
        raise HTTPException(status_code=503, detail="RAG Pipeline 未初始化，请检查服务日志")
    ns = namespace or settings.default_namespace
    local = RAGPipeline(settings, ns)
    gen, recs = local.ask_stream(
        req.question, 
        req.top_k, 
        req.rerank_enabled, 
        req.rerank_top_n, 
        req.model,
        system_prompt=req.system_prompt,
        web_enabled=req.web_enabled,
        web_top_k=req.web_top_k
    )

    def sse():  # noqa: ANN202
        yield f"event: meta\ndata: {len(recs)}\n\n"
        for r in recs:
            payload = {
                "path": str(r.meta.get("path")),
                "chunk_id": int(r.meta.get("chunk_id")) if r.meta.get("chunk_id") is not None else None,
                "score": float(r.score),
                "snippet": r.text[:200],
            }
            from json import dumps
            yield f"event: source\ndata: {dumps(payload, ensure_ascii=False)}\n\n"
        for token in gen:
            yield f"data: {token}\n\n"
        yield "event: done\ndata: end\n\n"

    return StreamingResponse(sse(), media_type="text/event-stream")


# ==================== 查询改写相关 API ====================

class QueryRewriteRequest(BaseModel):
    question: str
    strategy: str = "expand"  # expand/decompose/hyde/multi
    top_k: int | None = None
    rerank_enabled: bool | None = None
    rerank_top_n: int | None = None
    model: str | None = None


class QueryRewriteResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    metadata: Dict[str, Any]


@app.post("/ask_with_rewriting", response_model=QueryRewriteResponse)
def ask_with_query_rewriting(req: QueryRewriteRequest, x_api_key: str | None = None, namespace: str | None = None) -> QueryRewriteResponse:
    """使用查询改写增强检索效果"""
    _require_api_key({"x-api-key": x_api_key} if x_api_key else {})
    if pipeline is None:
        raise HTTPException(status_code=503, detail="RAG Pipeline 未初始化")
    
    ns = namespace or settings.default_namespace
    local = RAGPipeline(settings, ns)
    
    try:
        answer, recs, metadata = local.ask_with_query_rewriting(
            req.question,
            strategy=req.strategy,
            top_k=req.top_k,
            rerank_enabled=req.rerank_enabled,
            rerank_top_n=req.rerank_top_n,
            model=req.model
        )
        
        sources = [
            {
                "path": r.meta.get("path"),
                "chunk_id": r.meta.get("chunk_id"),
                "score": r.score,
                "text": r.text,
                "page": r.meta.get("page"),
                "has_tables": r.meta.get("has_tables"),
                "doc_type": r.meta.get("doc_type")
            }
            for r in recs
        ]
        
        return QueryRewriteResponse(answer=answer, sources=sources, metadata=metadata)
    
    except Exception as e:
        logger.error(f"查询改写失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class QueryAnalysisRequest(BaseModel):
    question: str


@app.post("/analyze_query")
def analyze_query(req: QueryAnalysisRequest, x_api_key: str | None = None) -> Dict[str, Any]:
    """分析查询特征并推荐最佳改写策略"""
    _require_api_key({"x-api-key": x_api_key} if x_api_key else {})
    if pipeline is None:
        raise HTTPException(status_code=503, detail="RAG Pipeline 未初始化")
    
    try:
        analysis = pipeline.analyze_query(req.question)
        return success_response(data=analysis)
    except Exception as e:
        logger.error(f"查询分析失败: {e}")
        return error_response(message=str(e))


# ==================== 检索结果解释 API ====================

class ExplainRetrievalRequest(BaseModel):
    question: str
    top_k: int = 5


@app.post("/explain_retrieval")
def explain_retrieval(req: ExplainRetrievalRequest, x_api_key: str | None = None, namespace: str | None = None) -> Dict[str, Any]:
    """解释检索结果，显示为什么检索到这些文档"""
    _require_api_key({"x-api-key": x_api_key} if x_api_key else {})
    if pipeline is None:
        raise HTTPException(status_code=503, detail="RAG Pipeline 未初始化")
    
    ns = namespace or settings.default_namespace
    local = RAGPipeline(settings, ns)
    
    try:
        # 执行检索
        recs = local.store.search(req.question, req.top_k)
        
        # 转换为字典格式
        chunks = [
            {
                "text": r.text,
                "score": r.score,
                "meta": r.meta
            }
            for r in recs
        ]
        
        # 生成解释
        from backend.retrieval_explainer import create_explainer
        explainer = create_explainer()
        explanations = explainer.explain_retrieval(req.question, chunks)
        summary = explainer.generate_summary(explanations)
        
        # 格式化返回
        results = []
        for exp in explanations:
            results.append({
                "chunk_id": exp.chunk_id,
                "text": exp.text,
                "highlight_text": exp.highlight_text,
                "score": exp.score_breakdown.final_score,
                "relevance_level": exp.relevance_level,
                "matched_keywords": exp.score_breakdown.matched_keywords or [],
                "semantic_similarity": exp.score_breakdown.semantic_similarity,
                "explanation": exp.score_breakdown.explanation,
                "metadata": exp.metadata
            })
        
        return success_response(data={
            "query": req.question,
            "results": results,
            "summary": summary
        })
    
    except Exception as e:
        logger.error(f"检索解释失败: {e}")
        return error_response(message=str(e))


# ==================== 高级检索 API ====================

class AdvancedSearchRequest(BaseModel):
    question: str
    top_k: int = 5
    # 过滤配置
    doc_types: Optional[List[str]] = None
    date_from: Optional[str] = None
    date_to: Optional[str] = None
    min_score: Optional[float] = None
    paths: Optional[List[str]] = None
    tags: Optional[List[str]] = None
    has_tables: Optional[bool] = None
    page_range: Optional[tuple] = None
    # 权重配置
    vector_weight: float = 0.7
    bm25_weight: float = 0.3
    reranker_enabled: bool = True
    mmr_lambda: float = 0.5
    # 聚合选项
    aggregate_by: Optional[str] = None  # 'document' or 'type'


@app.post("/advanced_search")
def advanced_search(req: AdvancedSearchRequest, x_api_key: str | None = None, namespace: str | None = None) -> Dict[str, Any]:
    """高级检索 - 支持过滤、权重调优、聚合"""
    _require_api_key({"x-api-key": x_api_key} if x_api_key else {})
    if pipeline is None:
        raise HTTPException(status_code=503, detail="RAG Pipeline 未初始化")
    
    ns = namespace or settings.default_namespace
    local = RAGPipeline(settings, ns)
    
    try:
        from backend.advanced_retrieval import create_retriever, FilterConfig, WeightConfig
        
        # 创建高级检索器
        retriever = create_retriever()
        
        # 执行基础检索
        recs = local.store.search(req.question, req.top_k * 2)  # 多检索一些，后面过滤
        
        # 转换为字典格式
        results = [
            {
                "text": r.text,
                "score": r.score,
                "meta": r.meta
            }
            for r in recs
        ]
        
        # 应用过滤
        filter_config = FilterConfig(
            doc_types=req.doc_types,
            date_from=req.date_from,
            date_to=req.date_to,
            min_score=req.min_score,
            max_results=req.top_k,
            paths=req.paths,
            tags=req.tags,
            has_tables=req.has_tables,
            page_range=req.page_range
        )
        filtered_results = retriever.filter_results(results, filter_config)
        
        # 获取统计信息
        stats = retriever.get_statistics(filtered_results)
        
        # 聚合（如果需要）
        aggregated = None
        if req.aggregate_by == "document":
            aggregated = retriever.aggregate_by_document(filtered_results)
        elif req.aggregate_by == "type":
            aggregated = retriever.aggregate_by_type(filtered_results)
        
        # 格式化返回
        response_data = {
            "query": req.question,
            "results": filtered_results,
            "statistics": stats,
            "filter_config": filter_config.to_dict(),
            "weight_config": {
                "vector_weight": req.vector_weight,
                "bm25_weight": req.bm25_weight,
                "reranker_enabled": req.reranker_enabled,
                "mmr_lambda": req.mmr_lambda
            }
        }
        
        if aggregated:
            response_data["aggregated"] = aggregated
        
        return success_response(data=response_data)
    
    except Exception as e:
        logger.error(f"高级检索失败: {e}")
        return error_response(message=str(e))


class OptimizeWeightsRequest(BaseModel):
    question: str
    top_k: int = 5
    test_weights: List[Dict[str, float]] = None  # [{"vector": 0.7, "bm25": 0.3}, ...]


@app.post("/optimize_weights")
def optimize_weights(req: OptimizeWeightsRequest, x_api_key: str | None = None, namespace: str | None = None) -> Dict[str, Any]:
    """权重优化 - 测试不同权重组合的效果"""
    _require_api_key({"x-api-key": x_api_key} if x_api_key else {})
    if pipeline is None:
        raise HTTPException(status_code=503, detail="RAG Pipeline 未初始化")
    
    ns = namespace or settings.default_namespace
    local = RAGPipeline(settings, ns)
    
    try:
        # 默认测试权重组合
        if not req.test_weights:
            req.test_weights = [
                {"vector": 1.0, "bm25": 0.0},
                {"vector": 0.8, "bm25": 0.2},
                {"vector": 0.7, "bm25": 0.3},
                {"vector": 0.6, "bm25": 0.4},
                {"vector": 0.5, "bm25": 0.5},
                {"vector": 0.3, "bm25": 0.7},
                {"vector": 0.0, "bm25": 1.0}
            ]
        
        # 执行检索
        recs = local.store.search(req.question, req.top_k)
        
        results = []
        for weights in req.test_weights:
            result = {
                "weights": weights,
                "top_results": [
                    {
                        "text": r.text[:100] + "...",
                        "score": r.score,
                        "path": r.meta.get("path")
                    }
                    for r in recs[:3]
                ],
                "avg_score": sum(r.score for r in recs) / len(recs) if recs else 0
            }
            results.append(result)
        
        # 推荐最佳权重
        best = max(results, key=lambda x: x["avg_score"])
        
        return success_response(data={
            "query": req.question,
            "test_results": results,
            "recommended_weights": best["weights"],
            "note": "推荐权重基于平均分数，实际效果需要人工评估"
        })
    
    except Exception as e:
        logger.error(f"权重优化失败: {e}")
        return error_response(message=str(e))


# ==================== 文档分块可视化 API ====================

class VisualizeChunksRequest(BaseModel):
    path: str


@app.post("/visualize_chunks")
def visualize_chunks(req: VisualizeChunksRequest, x_api_key: str | None = None, namespace: str | None = None) -> Dict[str, Any]:
    """可视化文档的分块结果"""
    _require_api_key({"x-api-key": x_api_key} if x_api_key else {})
    if pipeline is None:
        raise HTTPException(status_code=503, detail="RAG Pipeline 未初始化")
    
    ns = namespace or settings.default_namespace
    local = RAGPipeline(settings, ns)
    
    try:
        # 从索引中获取该文档的所有分块
        chunks = []
        for i, (text, meta) in enumerate(zip(local.store.texts, local.store.metas)):
            if meta.get("path") == req.path:
                chunks.append({
                    "chunk_id": meta.get("chunk_id", i),
                    "text": text,
                    "char_count": len(text),
                    "word_count": len(text.split()),
                    "page": meta.get("page"),
                    "has_tables": meta.get("has_tables", False),
                    "doc_type": meta.get("doc_type"),
                    "metadata": meta
                })
        
        if not chunks:
            return error_response(message=f"未找到文档: {req.path}")
        
        # 排序
        chunks.sort(key=lambda x: x["chunk_id"])
        
        # 统计信息
        total_chars = sum(c["char_count"] for c in chunks)
        total_words = sum(c["word_count"] for c in chunks)
        avg_chunk_size = total_chars / len(chunks) if chunks else 0
        
        stats = {
            "total_chunks": len(chunks),
            "total_chars": total_chars,
            "total_words": total_words,
            "avg_chunk_size": round(avg_chunk_size, 2),
            "min_chunk_size": min(c["char_count"] for c in chunks) if chunks else 0,
            "max_chunk_size": max(c["char_count"] for c in chunks) if chunks else 0,
            "has_tables": any(c["has_tables"] for c in chunks),
            "doc_type": chunks[0]["doc_type"] if chunks else None
        }
        
        return success_response(data={
            "path": req.path,
            "chunks": chunks,
            "stats": stats
        })
    
    except Exception as e:
        logger.error(f"文档分块可视化失败: {e}")
        return error_response(message=str(e))


@app.get("/docs/preview")
def preview_document_chunks(path: str, x_api_key: str | None = None, namespace: str | None = None) -> Dict[str, Any]:
    """预览文档的分块情况（GET 方式）"""
    _require_api_key({"x-api-key": x_api_key} if x_api_key else {})
    if pipeline is None:
        raise HTTPException(status_code=503, detail="RAG Pipeline 未初始化")
    
    ns = namespace or settings.default_namespace
    local = RAGPipeline(settings, ns)
    
    try:
        # 获取该文档的所有分块（只返回摘要）
        chunks_preview = []
        for i, (text, meta) in enumerate(zip(local.store.texts, local.store.metas)):
            if meta.get("path") == path:
                chunks_preview.append({
                    "chunk_id": meta.get("chunk_id", i),
                    "preview": text[:100] + "..." if len(text) > 100 else text,
                    "char_count": len(text),
                    "page": meta.get("page")
                })
        
        if not chunks_preview:
            return error_response(message=f"未找到文档: {path}")
        
        chunks_preview.sort(key=lambda x: x["chunk_id"])
        
        return success_response(data={
            "path": path,
            "total_chunks": len(chunks_preview),
            "chunks": chunks_preview
        })
    
    except Exception as e:
        logger.error(f"文档预览失败: {e}")
        return error_response(message=str(e))


from pydantic import BaseModel as _BaseModel


class UpsertDocRequest(_BaseModel):
    path: str
    text: str | None = None


@app.post("/docs")
async def upsert_doc(
    request: Request,
    req: UpsertDocRequest | None = None, 
    file: UploadFile = File(None), 
    path: str = Form(None), 
    x_api_key: str | None = None, 
    namespace: str | None = None
) -> JSONResponse:  # type: ignore[override]
    _require_api_key({"x-api-key": x_api_key} if x_api_key else {})
    if pipeline is None:
        return JSONResponse({"ok": False, "error": "RAG Pipeline 未初始化"}, status_code=503)
    ns = namespace or settings.default_namespace
    local = RAGPipeline(settings, ns)
    
    try:
        final_path: str | None = None
        text: str | None = None
        
        # 检查 Content-Type 来决定如何解析请求
        content_type = request.headers.get("content-type", "")
        
        # 如果是 JSON 请求
        if "application/json" in content_type:
            try:
                body = await request.json()
                final_path = body.get("path")
                text = body.get("text")
                if not final_path:
                    return JSONResponse({"ok": False, "error": "缺少 path 字段"}, status_code=400)
            except Exception as e:
                logger.error(f"解析 JSON 失败: {e}")
                return JSONResponse({"ok": False, "error": f"JSON 解析失败: {str(e)}"}, status_code=400)
        
        # 如果是 multipart/form-data
        elif "multipart/form-data" in content_type:
            if file is not None and path is not None:
                final_path = path
                raw = await file.read()
                name = (file.filename or '').lower()
                if name.endswith('.pdf'):
                    try:
                        from pdfminer.high_level import extract_text
                    except Exception:
                        from pdfminer.high_level import extract_text
                    import tempfile
                    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=True) as tmp:
                        tmp.write(raw)
                        tmp.flush()
                        text = extract_text(tmp.name)
                else:
                    try:
                        text = raw.decode("utf-8", errors="ignore")
                    except Exception:
                        text = ""
            else:
                return JSONResponse({"ok": False, "error": "multipart 请求需要 file 和 path"}, status_code=400)
        
        else:
            return JSONResponse({"ok": False, "error": f"不支持的 Content-Type: {content_type}"}, status_code=400)
        
        if not final_path or text is None:
            return JSONResponse({"ok": False, "error": "path 或 text 为空"}, status_code=400)
        
        # 统一使用正斜杠，避免 Windows 路径混乱
        final_path = final_path.replace("\\", "/")
        added = local.add_document(final_path, text or "")
        logger.info(f"✓ 文档上传成功: {final_path}, 新增 {added} 个分片")
        return JSONResponse({"ok": True, "added_chunks": added})
    except Exception as e:
        # 在 FAISS 模式下，add_document 会抛出错误；改为返回 400 与提示
        logger.error(f"✗ 文档上传失败: {final_path}, 错误: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)


@app.delete("/docs")
def delete_doc(path: str, x_api_key: str | None = None, namespace: str | None = None) -> JSONResponse:  # type: ignore[override]
    _require_api_key({"x-api-key": x_api_key} if x_api_key else {})
    if pipeline is None:
        return JSONResponse({"ok": False, "error": "RAG Pipeline 未初始化"}, status_code=503)
    ns = namespace or settings.default_namespace
    local = RAGPipeline(settings, ns)
    try:
        deleted = local.delete_document(path)
        return JSONResponse({"ok": True, "deleted": deleted})
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)


@app.get("/docs/paths")
def list_doc_paths(limit: int = 1000, namespace: str | None = None) -> JSONResponse:  # type: ignore[override]
    if pipeline is None:
        return JSONResponse({"ok": False, "error": "RAG Pipeline 未初始化"}, status_code=503)
    ns = namespace or settings.default_namespace
    local = RAGPipeline(settings, ns)
    docs_stats = local.list_paths_with_stats(limit)
    return JSONResponse({"ok": True, "documents": docs_stats})


@app.get("/export")
def export_by_path(path: str, namespace: str | None = None) -> JSONResponse:  # type: ignore[override]
    if pipeline is None:
        return JSONResponse({"ok": False, "error": "RAG Pipeline 未初始化"}, status_code=503)
    
    try:
        ns = namespace or settings.default_namespace
        local = RAGPipeline(settings, ns)
        store = local.store
        backend = getattr(store, "backend", "faiss")
        
        chunks = []
        
        if backend == "milvus" and store.collection is not None:
            # Milvus 后端
            escaped = path.replace("'", "\\'")
            expr = "path == '" + escaped + "'"
            recs = store.collection.query(expr=expr, output_fields=["path", "chunk_id", "text"], limit=10000)
            chunks = recs
        else:
            # FAISS 后端：从内存中过滤
            for i, meta in enumerate(store.metas):
                if meta.get("path") == path:
                    chunks.append({
                        "path": path,
                        "chunk_id": meta.get("chunk_id", i),
                        "text": store.texts[i] if i < len(store.texts) else ""
                    })
        
        return JSONResponse({"ok": True, "path": path, "chunks": chunks})
    except Exception as e:
        logger.error(f"导出文档失败: {e}")
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)


@app.post("/import")
def import_chunks(payload: Dict[str, Any], x_api_key: str | None = None, namespace: str | None = None) -> JSONResponse:  # type: ignore[override]
    _require_api_key({"x-api-key": x_api_key} if x_api_key else {})
    if pipeline is None:
        return JSONResponse({"ok": False, "error": "RAG Pipeline 未初始化"}, status_code=503)
    try:
        path = payload.get("path")
        chunks = payload.get("chunks") or []
        if not path or not isinstance(chunks, list):
            return JSONResponse({"ok": False, "error": "invalid payload"}, status_code=400)
        # 先删除再写入
        try:
            ns = namespace or settings.default_namespace
            local = RAGPipeline(settings, ns)
            local.delete_document(path)
        except Exception:
            pass
        # 直接将文本拼接后按现有 split 重新切分更稳妥
        combined = "\n\n".join([c.get("text", "") for c in chunks])
        ns = namespace or settings.default_namespace
        local = RAGPipeline(settings, ns)
        added = local.add_document(path, combined)
        return JSONResponse({"ok": True, "added": added})
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)


@app.get("/cache/stats")
def cache_stats(x_api_key: str | None = None) -> JSONResponse:  # type: ignore[override]
    """获取缓存统计信息"""
    _require_api_key({"x-api-key": x_api_key} if x_api_key else {})
    return JSONResponse({
        "ok": True,
        "cache_size": query_cache.size(),
        "max_size": query_cache.maxsize
    })


@app.post("/cache/clear")
def cache_clear(x_api_key: str | None = None) -> JSONResponse:  # type: ignore[override]
    """清空缓存"""
    _require_api_key({"x-api-key": x_api_key} if x_api_key else {})
    query_cache.clear()
    logger.info("缓存已清空")
    return JSONResponse({"ok": True, "message": "缓存已清空"})


# ==================== 对话管理 API ====================
from backend.conversation import ConversationManager

conv_manager = ConversationManager()


@app.post("/conversations")
def create_conversation(title: str = "新对话", namespace: str | None = None) -> JSONResponse:  # type: ignore[override]
    """创建新对话"""
    ns = namespace or settings.default_namespace
    conversation = conv_manager.create_conversation(title=title, namespace=ns)
    return JSONResponse({"ok": True, "conversation": conversation.to_dict()})


@app.get("/conversations")
def list_conversations(namespace: str | None = None, limit: int = 50) -> JSONResponse:  # type: ignore[override]
    """列出对话列表"""
    ns = namespace or settings.default_namespace
    conversations = conv_manager.list_conversations(namespace=ns, limit=limit)
    return JSONResponse({"ok": True, "conversations": conversations})


@app.get("/conversations/{conv_id}")
def get_conversation(conv_id: str) -> JSONResponse:  # type: ignore[override]
    """获取对话详情"""
    conversation = conv_manager.get_conversation(conv_id)
    if not conversation:
        return JSONResponse({"ok": False, "error": "对话不存在"}, status_code=404)
    return JSONResponse({"ok": True, "conversation": conversation.to_dict()})


@app.delete("/conversations/{conv_id}")
def delete_conversation_endpoint(conv_id: str) -> JSONResponse:  # type: ignore[override]
    """删除对话"""
    success = conv_manager.delete_conversation(conv_id)
    if success:
        return JSONResponse({"ok": True, "message": "对话已删除"})
    return JSONResponse({"ok": False, "error": "对话不存在"}, status_code=404)


@app.post("/conversations/{conv_id}/messages")
def add_message_to_conversation(
    conv_id: str,
    role: str,
    content: str,
    sources: List[Dict[str, Any]] | None = None
) -> JSONResponse:  # type: ignore[override]
    """添加消息到对话"""
    conversation = conv_manager.add_message(conv_id, role, content, sources)
    if not conversation:
        return JSONResponse({"ok": False, "error": "对话不存在"}, status_code=404)
    return JSONResponse({"ok": True, "conversation": conversation.to_dict()})


# ==================== 性能监控 API ====================

@app.get("/metrics/statistics")
def get_performance_statistics(x_api_key: str | None = None) -> Dict[str, Any]:
    """获取性能统计信息"""
    _require_api_key({"x-api-key": x_api_key} if x_api_key else {})
    monitor = get_monitor()
    
    return success_response(data={
        "overall": monitor.get_statistics(),
        "endpoints": monitor.get_endpoint_statistics(),
        "performance_breakdown": monitor.get_performance_breakdown()
    })


@app.get("/metrics/hot_queries")
def get_hot_queries(top_n: int = 10, x_api_key: str | None = None) -> Dict[str, Any]:
    """获取热门查询"""
    _require_api_key({"x-api-key": x_api_key} if x_api_key else {})
    monitor = get_monitor()
    
    return success_response(data={
        "hot_queries": monitor.get_hot_queries(top_n)
    })


@app.get("/metrics/recent_requests")
def get_recent_requests(limit: int = 50, x_api_key: str | None = None) -> Dict[str, Any]:
    """获取最近的请求记录"""
    _require_api_key({"x-api-key": x_api_key} if x_api_key else {})
    monitor = get_monitor()
    
    return success_response(data={
        "recent_requests": monitor.get_recent_requests(limit)
    })


@app.get("/metrics/time_series")
def get_time_series(interval_seconds: int = 60, x_api_key: str | None = None) -> Dict[str, Any]:
    """获取时间序列数据"""
    _require_api_key({"x-api-key": x_api_key} if x_api_key else {})
    monitor = get_monitor()
    
    return success_response(data={
        "time_series": monitor.get_time_series(interval_seconds)
    })


@app.post("/metrics/export")
def export_metrics(filepath: str = "data/metrics/export.json", x_api_key: str | None = None) -> Dict[str, Any]:
    """导出性能指标到文件"""
    _require_api_key({"x-api-key": x_api_key} if x_api_key else {})
    monitor = get_monitor()
    
    try:
        monitor.export_metrics(filepath)
        return success_response(data={"filepath": filepath, "message": "指标已导出"})
    except Exception as e:
        logger.error(f"导出指标失败: {e}")
        return error_response(message=str(e))


@app.post("/metrics/clear")
def clear_metrics(x_api_key: str | None = None) -> Dict[str, Any]:
    """清空性能监控数据"""
    _require_api_key({"x-api-key": x_api_key} if x_api_key else {})
    monitor = get_monitor()
    monitor.clear_history()
    
    return success_response(data={"message": "监控数据已清空"})


# ==================== 文档管理增强 API ====================

from backend.document_manager import get_document_manager

doc_manager = get_document_manager()


class UpdateDocumentRequest(BaseModel):
    path: str
    tags: Optional[List[str]] = None
    category: Optional[str] = None
    description: Optional[str] = None


@app.post("/documents/metadata")
def update_document_metadata(req: UpdateDocumentRequest, x_api_key: str | None = None) -> Dict[str, Any]:
    """更新文档元数据"""
    _require_api_key({"x-api-key": x_api_key} if x_api_key else {})
    
    try:
        update_data = {}
        if req.tags is not None:
            update_data["tags"] = req.tags
        if req.category is not None:
            update_data["category"] = req.category
        if req.description is not None:
            update_data["description"] = req.description
        
        doc_manager.update_document(req.path, **update_data)
        
        return success_response(data={
            "path": req.path,
            "metadata": doc_manager.get_document(req.path)
        })
    except Exception as e:
        logger.error(f"更新文档元数据失败: {e}")
        return error_response(message=str(e))


@app.get("/documents/metadata/{path:path}")
def get_document_metadata(path: str, x_api_key: str | None = None) -> Dict[str, Any]:
    """获取文档元数据"""
    _require_api_key({"x-api-key": x_api_key} if x_api_key else {})
    
    metadata = doc_manager.get_document(path)
    if not metadata:
        return error_response(message="文档不存在", status_code=404)
    
    return success_response(data=metadata)


@app.get("/documents/list")
def list_documents_with_metadata(
    category: Optional[str] = None,
    tags: Optional[str] = None,
    x_api_key: str | None = None
) -> Dict[str, Any]:
    """列出文档（支持按分类和标签过滤）"""
    _require_api_key({"x-api-key": x_api_key} if x_api_key else {})
    
    tag_list = tags.split(",") if tags else None
    documents = doc_manager.list_documents(category=category, tags=tag_list)
    
    return success_response(data={
        "documents": documents,
        "count": len(documents)
    })


@app.get("/documents/tags")
def get_all_tags(x_api_key: str | None = None) -> Dict[str, Any]:
    """获取所有标签"""
    _require_api_key({"x-api-key": x_api_key} if x_api_key else {})
    
    tags = doc_manager.metadata.get_all_tags()
    return success_response(data={"tags": tags, "count": len(tags)})


@app.get("/documents/categories")
def get_all_categories(x_api_key: str | None = None) -> Dict[str, Any]:
    """获取所有分类"""
    _require_api_key({"x-api-key": x_api_key} if x_api_key else {})
    
    categories = doc_manager.metadata.get_all_categories()
    return success_response(data={"categories": categories, "count": len(categories)})


@app.get("/documents/statistics")
def get_document_statistics(x_api_key: str | None = None) -> Dict[str, Any]:
    """获取文档统计信息"""
    _require_api_key({"x-api-key": x_api_key} if x_api_key else {})
    
    stats = doc_manager.get_statistics()
    return success_response(data=stats)


@app.post("/documents/{path:path}/tags")
def add_document_tags(path: str, tags: List[str], x_api_key: str | None = None) -> Dict[str, Any]:
    """添加文档标签"""
    _require_api_key({"x-api-key": x_api_key} if x_api_key else {})
    
    try:
        doc_manager.metadata.add_tags(path, tags)
        return success_response(data={
            "path": path,
            "tags": doc_manager.get_document(path).get("tags", [])
        })
    except Exception as e:
        logger.error(f"添加标签失败: {e}")
        return error_response(message=str(e))


@app.delete("/documents/{path:path}/tags")
def remove_document_tags(path: str, tags: List[str], x_api_key: str | None = None) -> Dict[str, Any]:
    """移除文档标签"""
    _require_api_key({"x-api-key": x_api_key} if x_api_key else {})
    
    try:
        doc_manager.metadata.remove_tags(path, tags)
        return success_response(data={
            "path": path,
            "tags": doc_manager.get_document(path).get("tags", [])
        })
    except Exception as e:
        logger.error(f"移除标签失败: {e}")
        return error_response(message=str(e))


# ==================== 检索优化 API ====================

from backend.retrieval_optimizer import get_optimizer


@app.post("/retrieval/analyze")
def analyze_retrieval_quality(
    question: str,
    top_k: int = 10,
    x_api_key: str | None = None
) -> Dict[str, Any]:
    """分析检索质量并提供优化建议"""
    _require_api_key({"x-api-key": x_api_key} if x_api_key else {})
    
    try:
        # 执行检索
        results = pipeline.vector_store.search(question, top_k=top_k)
        
        # 转换为字典格式
        results_dict = [
            {
                "text": r.text,
                "score": r.score,
                "meta": r.meta
            }
            for r in results
        ]
        
        # 分析优化
        optimizer = get_optimizer()
        analysis = optimizer.optimize(question, results_dict)
        
        return success_response(data=analysis)
    
    except Exception as e:
        logger.error(f"检索分析失败: {e}")
        return error_response(message=str(e))


@app.post("/retrieval/suggest_weights")
def suggest_optimal_weights(
    question: str,
    x_api_key: str | None = None
) -> Dict[str, Any]:
    """根据查询类型建议最佳权重"""
    _require_api_key({"x-api-key": x_api_key} if x_api_key else {})
    
    try:
        optimizer = get_optimizer()
        
        # 自动判断权重
        vec_weight, bm25_weight = optimizer.weight_optimizer.adaptive_weights(question)
        
        # 推荐改写策略
        rewrite_strategy = optimizer.query_optimizer.suggest_rewrite_strategy(question)
        
        return success_response(data={
            "question": question,
            "recommended_weights": {
                "vector_weight": vec_weight,
                "bm25_weight": bm25_weight
            },
            "recommended_rewrite_strategy": rewrite_strategy,
            "explanation": _get_weight_explanation(vec_weight, bm25_weight)
        })
    
    except Exception as e:
        logger.error(f"权重建议失败: {e}")
        return error_response(message=str(e))


def _get_weight_explanation(vec_weight: float, bm25_weight: float) -> str:
    """解释权重选择"""
    if vec_weight > 0.7:
        return "查询偏语义理解，使用高向量权重"
    elif bm25_weight > 0.6:
        return "查询偏关键词匹配，使用高 BM25 权重"
    else:
        return "查询类型平衡，使用均衡权重"


@app.post("/retrieval/grid_search")
def grid_search_weights(
    question: str,
    top_k: int = 10,
    x_api_key: str | None = None
) -> Dict[str, Any]:
    """网格搜索最佳权重组合"""
    _require_api_key({"x-api-key": x_api_key} if x_api_key else {})
    
    try:
        optimizer = get_optimizer()
        
        # 定义检索函数
        def retrieval_func(query, top_k, vector_weight, bm25_weight):
            # 临时修改权重
            old_vec = settings.rag_vec_weight
            old_bm25 = settings.rag_bm25_weight
            
            settings.rag_vec_weight = vector_weight
            settings.rag_bm25_weight = bm25_weight
            
            results = pipeline.vector_store.search(query, top_k=top_k)
            
            # 恢复权重
            settings.rag_vec_weight = old_vec
            settings.rag_bm25_weight = old_bm25
            
            return [{"score": r.score} for r in results]
        
        # 执行网格搜索
        optimization = optimizer.weight_optimizer.grid_search(
            retrieval_func,
            question,
            top_k=top_k
        )
        
        return success_response(data={
            "question": question,
            "optimization": optimization
        })
    
    except Exception as e:
        logger.error(f"网格搜索失败: {e}")
        return error_response(message=str(e))


@app.post("/retrieval/compare_strategies")
def compare_retrieval_strategies(
    question: str,
    top_k: int = 10,
    x_api_key: str | None = None
) -> Dict[str, Any]:
    """比较不同检索策略的效果"""
    _require_api_key({"x-api-key": x_api_key} if x_api_key else {})
    
    try:
        strategies = []
        
        # 策略 1: 纯向量检索
        old_bm25 = settings.rag_bm25_enabled
        settings.rag_bm25_enabled = False
        results_vec = pipeline.vector_store.search(question, top_k=top_k)
        settings.rag_bm25_enabled = old_bm25
        
        strategies.append({
            "name": "纯向量检索",
            "config": {"vector_weight": 1.0, "bm25_weight": 0.0},
            "avg_score": round(sum(r.score for r in results_vec) / len(results_vec), 4) if results_vec else 0,
            "result_count": len(results_vec)
        })
        
        # 策略 2: 混合检索（默认权重）
        results_hybrid = pipeline.vector_store.search(question, top_k=top_k)
        strategies.append({
            "name": "混合检索（默认）",
            "config": {
                "vector_weight": settings.rag_vec_weight,
                "bm25_weight": settings.rag_bm25_weight
            },
            "avg_score": round(sum(r.score for r in results_hybrid) / len(results_hybrid), 4) if results_hybrid else 0,
            "result_count": len(results_hybrid)
        })
        
        # 策略 3: 自适应权重
        optimizer = get_optimizer()
        vec_w, bm25_w = optimizer.weight_optimizer.adaptive_weights(question)
        
        old_vec = settings.rag_vec_weight
        old_bm25 = settings.rag_bm25_weight
        settings.rag_vec_weight = vec_w
        settings.rag_bm25_weight = bm25_w
        
        results_adaptive = pipeline.vector_store.search(question, top_k=top_k)
        
        settings.rag_vec_weight = old_vec
        settings.rag_bm25_weight = old_bm25
        
        strategies.append({
            "name": "自适应权重",
            "config": {"vector_weight": vec_w, "bm25_weight": bm25_w},
            "avg_score": round(sum(r.score for r in results_adaptive) / len(results_adaptive), 4) if results_adaptive else 0,
            "result_count": len(results_adaptive)
        })
        
        # 排序
        strategies.sort(key=lambda x: x["avg_score"], reverse=True)
        
        return success_response(data={
            "question": question,
            "strategies": strategies,
            "best_strategy": strategies[0]["name"]
        })
    
    except Exception as e:
        logger.error(f"策略比较失败: {e}")
        return error_response(message=str(e))


# ==================== 查询意图识别 API ====================

from backend.query_intent import get_query_analyzer, IntentRecognizer


@app.post("/query/analyze_intent")
def analyze_query_intent(
    question: str,
    x_api_key: str | None = None
) -> Dict[str, Any]:
    """分析查询意图并提供优化建议"""
    _require_api_key({"x-api-key": x_api_key} if x_api_key else {})
    
    try:
        analyzer = get_query_analyzer()
        analysis = analyzer.analyze(question)
        
        return success_response(data=analysis)
    
    except Exception as e:
        logger.error(f"意图分析失败: {e}")
        return error_response(message=str(e))


@app.post("/query/smart_search")
def smart_search_with_intent(
    question: str,
    top_k: Optional[int] = None,
    use_recommended_config: bool = True,
    x_api_key: str | None = None
) -> Dict[str, Any]:
    """基于意图识别的智能检索"""
    _require_api_key({"x-api-key": x_api_key} if x_api_key else {})
    
    try:
        # 分析查询意图
        analyzer = get_query_analyzer()
        analysis = analyzer.analyze(question)
        
        # 获取推荐配置
        config = analysis["recommended_config"]
        
        if use_recommended_config:
            # 使用推荐的配置
            actual_top_k = top_k or config["top_k"]
            
            # 临时修改配置
            old_vec = settings.rag_vec_weight
            old_bm25 = settings.rag_bm25_weight
            
            settings.rag_vec_weight = config["vector_weight"]
            settings.rag_bm25_weight = config["bm25_weight"]
            
            # 执行检索
            results = pipeline.vector_store.search(question, top_k=actual_top_k)
            
            # 恢复配置
            settings.rag_vec_weight = old_vec
            settings.rag_bm25_weight = old_bm25
        else:
            # 使用默认配置
            actual_top_k = top_k or 10
            results = pipeline.vector_store.search(question, top_k=actual_top_k)
        
        # 转换结果
        results_dict = [
            {
                "text": r.text,
                "score": r.score,
                "meta": r.meta
            }
            for r in results
        ]
        
        return success_response(data={
            "query": question,
            "intent": analysis["intent"],
            "used_config": config if use_recommended_config else "default",
            "results": results_dict,
            "suggestions": analysis["suggestions"]
        })
    
    except Exception as e:
        logger.error(f"智能检索失败: {e}")
        return error_response(message=str(e))


@app.post("/query/batch_analyze")
def batch_analyze_queries(
    questions: List[str],
    x_api_key: str | None = None
) -> Dict[str, Any]:
    """批量分析查询意图"""
    _require_api_key({"x-api-key": x_api_key} if x_api_key else {})
    
    try:
        analyzer = get_query_analyzer()
        results = []
        
        for question in questions:
            analysis = analyzer.analyze(question)
            results.append({
                "query": question,
                "intent_type": analysis["intent"]["type"],
                "confidence": analysis["intent"]["confidence"],
                "recommended_strategy": analysis["recommended_config"]["rewrite_strategy"]
            })
        
        # 统计意图分布
        intent_distribution = {}
        for result in results:
            intent_type = result["intent_type"]
            intent_distribution[intent_type] = intent_distribution.get(intent_type, 0) + 1
        
        return success_response(data={
            "total_queries": len(questions),
            "analyses": results,
            "intent_distribution": intent_distribution
        })
    
    except Exception as e:
        logger.error(f"批量分析失败: {e}")
        return error_response(message=str(e))


# ==================== 缓存优化 API ====================

from backend.cache_optimizer import get_smart_cache, CachePrewarmer


@app.get("/cache/analyze")
def analyze_cache(x_api_key: str | None = None) -> Dict[str, Any]:
    """分析缓存性能并提供优化建议"""
    _require_api_key({"x-api-key": x_api_key} if x_api_key else {})
    
    try:
        smart_cache = get_smart_cache()
        analysis = smart_cache.analyze()
        
        return success_response(data=analysis)
    
    except Exception as e:
        logger.error(f"缓存分析失败: {e}")
        return error_response(message=str(e))


@app.post("/cache/prewarm")
def prewarm_cache(
    use_hot_queries: bool = True,
    top_n: int = 20,
    custom_queries: Optional[List[str]] = None,
    x_api_key: str | None = None
) -> Dict[str, Any]:
    """缓存预热"""
    _require_api_key({"x-api-key": x_api_key} if x_api_key else {})
    
    try:
        # 定义检索函数
        def retrieval_func(query: str):
            return pipeline.vector_store.search(query, top_k=10)
        
        prewarmer = CachePrewarmer(retrieval_func)
        results = {"prewarmed_queries": []}
        
        # 从热门查询预热
        if use_hot_queries:
            from backend.performance_monitor import get_monitor
            monitor = get_monitor()
            hot_queries = monitor.get_hot_queries(top_n)
            
            hot_result = prewarmer.prewarm_from_hot_queries(hot_queries, top_n)
            results["hot_queries_result"] = hot_result
            results["prewarmed_queries"].extend(hot_result["prewarmed_queries"])
        
        # 从自定义查询预热
        if custom_queries:
            custom_result = prewarmer.prewarm_from_patterns(custom_queries)
            results["custom_queries_result"] = custom_result
            results["prewarmed_queries"].extend(custom_result["patterns"])
        
        return success_response(data={
            "total_prewarmed": len(results["prewarmed_queries"]),
            "details": results
        })
    
    except Exception as e:
        logger.error(f"缓存预热失败: {e}")
        return error_response(message=str(e))


@app.get("/cache/smart_stats")
def get_smart_cache_stats(x_api_key: str | None = None) -> Dict[str, Any]:
    """获取智能缓存统计"""
    _require_api_key({"x-api-key": x_api_key} if x_api_key else {})
    
    try:
        smart_cache = get_smart_cache()
        stats = smart_cache.get_stats()
        
        # 添加详细信息
        cache_entries = []
        for key, entry in list(smart_cache.cache.items())[:10]:  # 只返回前 10 个
            cache_entries.append({
                "key": key[:16] + "...",  # 截断显示
                "hits": entry.hits,
                "size_bytes": entry.size_bytes,
                "age_seconds": int(time.time() - entry.created_at),
                "ttl": entry.ttl
            })
        
        return success_response(data={
            "stats": stats,
            "top_entries": cache_entries
        })
    
    except Exception as e:
        logger.error(f"获取缓存统计失败: {e}")
        return error_response(message=str(e))


@app.post("/cache/optimize")
def optimize_cache_config(x_api_key: str | None = None) -> Dict[str, Any]:
    """优化缓存配置"""
    _require_api_key({"x-api-key": x_api_key} if x_api_key else {})
    
    try:
        smart_cache = get_smart_cache()
        analysis = smart_cache.analyze()
        
        # 根据分析结果调整配置
        hit_rate = analysis["hit_rate"]
        current_size = analysis["stats"]["size"]
        max_size = analysis["stats"]["max_size"]
        
        recommendations = {
            "current_config": {
                "max_size": max_size,
                "hit_rate": hit_rate
            },
            "recommended_config": {},
            "actions": []
        }
        
        # 命中率低，建议增加容量
        if hit_rate < 0.3:
            new_max_size = int(max_size * 1.5)
            recommendations["recommended_config"]["max_size"] = new_max_size
            recommendations["actions"].append(f"增加缓存容量到 {new_max_size}")
        
        # 空间利用率高，建议增加容量
        utilization = current_size / max_size if max_size > 0 else 0
        if utilization > 0.9:
            new_max_size = int(max_size * 1.3)
            recommendations["recommended_config"]["max_size"] = new_max_size
            recommendations["actions"].append(f"增加缓存容量到 {new_max_size}")
        
        # 空间利用率低，建议减少容量
        if utilization < 0.2 and max_size > 100:
            new_max_size = max(100, int(max_size * 0.7))
            recommendations["recommended_config"]["max_size"] = new_max_size
            recommendations["actions"].append(f"减少缓存容量到 {new_max_size}")
        
        if not recommendations["actions"]:
            recommendations["actions"].append("当前配置良好，无需调整")
        
        return success_response(data={
            "analysis": analysis,
            "recommendations": recommendations
        })
    
    except Exception as e:
        logger.error(f"缓存优化失败: {e}")
        return error_response(message=str(e))


# ==================== 评估测试 API ====================

from backend.evaluation import get_evaluator, TestCase, TestCaseGenerator, BenchmarkReport


@app.post("/evaluation/run_benchmark")
def run_benchmark_test(
    test_cases: Optional[List[Dict[str, Any]]] = None,
    use_default_tests: bool = False,
    top_k: int = 10,
    x_api_key: str | None = None
) -> Dict[str, Any]:
    """运行基准测试"""
    _require_api_key({"x-api-key": x_api_key} if x_api_key else {})
    
    try:
        evaluator = get_evaluator(pipeline)
        
        # 准备测试用例
        if use_default_tests:
            cases = TestCaseGenerator.generate_basic_tests()
        elif test_cases:
            cases = [
                TestCase(
                    question=tc["question"],
                    expected_answer=tc.get("expected_answer"),
                    expected_keywords=tc.get("expected_keywords"),
                    expected_doc_paths=tc.get("expected_doc_paths"),
                    category=tc.get("category", "general")
                )
                for tc in test_cases
            ]
        else:
            return error_response(message="请提供测试用例或使用默认测试")
        
        # 运行评估
        results = evaluator.run_benchmark(cases, top_k=top_k)
        
        # 生成报告
        report = BenchmarkReport.generate_report(results)
        
        return success_response(data={
            "results": results,
            "report": report
        })
    
    except Exception as e:
        logger.error(f"基准测试失败: {e}")
        return error_response(message=str(e))


@app.post("/evaluation/test_retrieval")
def test_retrieval_quality(
    question: str,
    expected_doc_paths: List[str],
    top_k: int = 10,
    x_api_key: str | None = None
) -> Dict[str, Any]:
    """测试单个查询的检索质量"""
    _require_api_key({"x-api-key": x_api_key} if x_api_key else {})
    
    try:
        evaluator = get_evaluator(pipeline)
        
        test_case = TestCase(
            question=question,
            expected_doc_paths=expected_doc_paths
        )
        
        results = evaluator.evaluate_retrieval([test_case], top_k=top_k)
        
        return success_response(data=results)
    
    except Exception as e:
        logger.error(f"检索测试失败: {e}")
        return error_response(message=str(e))


@app.post("/evaluation/test_answer")
def test_answer_quality(
    question: str,
    expected_keywords: Optional[List[str]] = None,
    x_api_key: str | None = None
) -> Dict[str, Any]:
    """测试单个查询的答案质量"""
    _require_api_key({"x-api-key": x_api_key} if x_api_key else {})
    
    try:
        evaluator = get_evaluator(pipeline)
        
        test_case = TestCase(
            question=question,
            expected_keywords=expected_keywords
        )
        
        results = evaluator.evaluate_answer_quality([test_case])
        
        return success_response(data=results)
    
    except Exception as e:
        logger.error(f"答案测试失败: {e}")
        return error_response(message=str(e))


@app.post("/evaluation/save_test_cases")
def save_test_cases_to_file(
    test_cases: List[Dict[str, Any]],
    filepath: str = "data/evaluation/test_cases.json",
    x_api_key: str | None = None
) -> Dict[str, Any]:
    """保存测试用例到文件"""
    _require_api_key({"x-api-key": x_api_key} if x_api_key else {})
    
    try:
        cases = [
            TestCase(
                question=tc["question"],
                expected_answer=tc.get("expected_answer"),
                expected_keywords=tc.get("expected_keywords"),
                expected_doc_paths=tc.get("expected_doc_paths"),
                category=tc.get("category", "general")
            )
            for tc in test_cases
        ]
        
        TestCaseGenerator.save_to_file(cases, filepath)
        
        return success_response(data={
            "filepath": filepath,
            "count": len(cases),
            "message": "测试用例已保存"
        })
    
    except Exception as e:
        logger.error(f"保存测试用例失败: {e}")
        return error_response(message=str(e))


@app.get("/evaluation/load_test_cases")
def load_test_cases_from_file(
    filepath: str = "data/evaluation/test_cases.json",
    x_api_key: str | None = None
) -> Dict[str, Any]:
    """从文件加载测试用例"""
    _require_api_key({"x-api-key": x_api_key} if x_api_key else {})
    
    try:
        cases = TestCaseGenerator.load_from_file(filepath)
        
        cases_dict = [
            {
                "question": tc.question,
                "expected_answer": tc.expected_answer,
                "expected_keywords": tc.expected_keywords,
                "expected_doc_paths": tc.expected_doc_paths,
                "category": tc.category
            }
            for tc in cases
        ]
        
        return success_response(data={
            "test_cases": cases_dict,
            "count": len(cases)
        })
    
    except Exception as e:
        logger.error(f"加载测试用例失败: {e}")
        return error_response(message=str(e))


