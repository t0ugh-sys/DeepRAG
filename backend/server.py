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
    gen, recs = local.ask_stream(req.question, req.top_k, req.rerank_enabled, req.rerank_top_n, req.model)

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


