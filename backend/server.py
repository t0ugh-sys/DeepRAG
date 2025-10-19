from typing import List, Any, Dict
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi import UploadFile, File, Form
from pymilvus import connections as _milvus_conn, FieldSchema as _FieldSchema, CollectionSchema as _CollectionSchema, DataType as _DataType, Collection as _Collection, utility as _utility
from sentence_transformers import SentenceTransformer as _ST
from pydantic import BaseModel

from config import Settings
from rag import RAGPipeline, RetrievedChunk
from utils.logger import logger
from utils.middleware import RequestLoggingMiddleware
from utils.responses import success_response, error_response

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    logger.info("🚀 RAG 服务启动中...")
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

# Frontend static site
app.mount("/web", StaticFiles(directory="web", html=True), name="web")

@app.get("/")
def root_redirect() -> RedirectResponse:  # type: ignore[override]
    return RedirectResponse(url="/web/")


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


@app.on_event("startup")
def _load_pipeline() -> None:
    global pipeline
    try:
        logger.info("加载 RAG Pipeline...")
        pipeline = RAGPipeline(settings, settings.default_namespace)
        logger.info("✓ RAG Pipeline 加载完成")
    except Exception as e:
        logger.error(f"✗ RAG Pipeline 加载失败: {e}")
        raise


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
    assert pipeline is not None
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


@app.get("/healthz")
def healthz() -> JSONResponse:  # type: ignore[override]
    """健康检查与监控指标"""
    ok = True
    details: Dict[str, Any] = {
        "timestamp": time.time(),
        "version": "0.2.0"
    }
    
    try:
        assert pipeline is not None
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
            except Exception as e:
                logger.warning(f"Milvus 状态检查失败: {e}")
                details["milvus_warning"] = str(e)
        
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
    assert pipeline is not None
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
def upsert_doc(req: UpsertDocRequest | None = None, file: UploadFile = File(None), path: str = Form(None), x_api_key: str | None = None, namespace: str | None = None) -> JSONResponse:  # type: ignore[override]
    _require_api_key({"x-api-key": x_api_key} if x_api_key else {})
    assert pipeline is not None
    ns = namespace or settings.default_namespace
    local = RAGPipeline(settings, ns)
    try:
        final_path: str | None = None
        text: str | None = None
        if req is not None and req.path:
            final_path = req.path
            text = req.text
        elif file is not None and path is not None:
            final_path = path
            raw = file.file.read()
            name = (file.filename or '').lower()
            if name.endswith('.pdf'):
                try:
                    from pdfminer_high_level import extract_text  # type: ignore
                except Exception:
                    from pdfminer.high_level import extract_text  # fallback
                # 临时存盘再解析，避免流处理复杂度
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
            return JSONResponse({"ok": False, "error": "need JSON{path,text} or multipart file+path"}, status_code=400)
        added = local.add_document(final_path, text or "")
        return JSONResponse({"ok": True, "added_chunks": added})
    except Exception as e:
        # 在 FAISS 模式下，add_document 会抛出错误；改为返回 400 与提示
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)


@app.delete("/docs")
def delete_doc(path: str, x_api_key: str | None = None, namespace: str | None = None) -> JSONResponse:  # type: ignore[override]
    _require_api_key({"x-api-key": x_api_key} if x_api_key else {})
    assert pipeline is not None
    ns = namespace or settings.default_namespace
    local = RAGPipeline(settings, ns)
    try:
        deleted = local.delete_document(path)
        return JSONResponse({"ok": True, "deleted": deleted})
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)


@app.get("/docs/paths")
def list_doc_paths(limit: int = 1000, namespace: str | None = None) -> JSONResponse:  # type: ignore[override]
    assert pipeline is not None
    ns = namespace or settings.default_namespace
    local = RAGPipeline(settings, ns)
    return JSONResponse({"ok": True, "paths": local.list_paths(limit)})


@app.get("/export")
def export_by_path(path: str, namespace: str | None = None) -> JSONResponse:  # type: ignore[override]
    assert pipeline is not None
    # 简易导出：按 path 直接查询文本与 chunk_id
    try:
        ns = namespace or settings.default_namespace
        local = RAGPipeline(settings, ns)
        coll = local.store.collection  # type: ignore[attr-defined]
        escaped = path.replace("'", "\\'")
        expr = "path == '" + escaped + "'"
        recs = coll.query(expr=expr, output_fields=["path", "chunk_id", "text"], limit=10000)
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)
    return JSONResponse({"ok": True, "path": path, "chunks": recs})


@app.post("/import")
def import_chunks(payload: Dict[str, Any], x_api_key: str | None = None, namespace: str | None = None) -> JSONResponse:  # type: ignore[override]
    _require_api_key({"x-api-key": x_api_key} if x_api_key else {})
    assert pipeline is not None
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


