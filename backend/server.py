from typing import List, Any, Dict, Optional
import time
from contextlib import asynccontextmanager
from threading import Lock

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
from backend.utils.middleware import RequestLoggingMiddleware, get_current_request
from backend.utils.responses import success_response, error_response
from backend.utils.cache import query_cache
from backend.performance_monitor import get_monitor, RequestTimer

settings = Settings()
pipeline: RAGPipeline | None = None
pipelines: Dict[str, RAGPipeline] = {}
_pipeline_lock = Lock()
cors_origins = [o.strip() for o in settings.cors_allow_origins.split(",") if o.strip()]
cors_methods = [m.strip() for m in settings.cors_allow_methods.split(",") if m.strip()]
cors_headers = [h.strip() for h in settings.cors_allow_headers.split(",") if h.strip()]

def _get_pipeline(ns: str) -> RAGPipeline:
    with _pipeline_lock:
        if ns in pipelines:
            return pipelines[ns]
        local = RAGPipeline(settings, ns)
        pipelines[ns] = local
        return local

@asynccontextmanager
async def lifespan(app: FastAPI):
    """搴旂敤鐢熷懡鍛ㄦ湡绠＄悊"""
    logger.info("馃殌 RAG 鏈嶅姟鍚姩涓?..")
    # 鍒濆鍖?pipeline
    global pipeline
    global pipelines
    try:
        logger.info("鍔犺浇 RAG Pipeline...")
        pipeline = _get_pipeline(settings.default_namespace)
        logger.info("鉁?RAG Pipeline 鍔犺浇瀹屾垚")
    except Exception as e:
        logger.error(f"鉁?RAG Pipeline 鍔犺浇澶辫触: {e}")
        raise
    yield
    logger.info("馃憢 RAG 鏈嶅姟鍏抽棴")

app = FastAPI(
    title="Local RAG API",
    version="0.2.0",
    lifespan=lifespan
)

# 娣诲姞璇锋眰鏃ュ織涓棿浠?
app.add_middleware(RequestLoggingMiddleware)
# CORS 閰嶇疆锛氬紑鍙戠幆澧冨厑璁告墍鏈夋簮锛涚敓浜х幆澧冨缓璁檺鍒跺叿浣撳煙鍚?
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,  # Vite dev
    allow_credentials=settings.cors_allow_credentials,
    allow_methods=cors_methods,
    allow_headers=cors_headers,
)

# 鍓嶇宸茶縼绉诲埌鐙珛 Vue 椤圭洰锛屼笉鍐嶉渶瑕侀潤鎬佹枃浠舵寕杞?
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


def _require_api_key(x_api_key: str | None = None) -> None:
    if not settings.api_key_required:
        return
    if not settings.api_key:
        raise HTTPException(status_code=500, detail="API key required but not configured")
    key = x_api_key
    if not key:
        request = get_current_request()
        if request is not None:
            key = request.headers.get("X-API-Key") or request.headers.get("x-api-key")
    if key != settings.api_key:
        raise HTTPException(status_code=401, detail='Unauthorized')

def _resolve_namespace(namespace: str | None) -> str:
    ns = namespace or settings.default_namespace
    if settings.namespace_whitelist:
        allowed = [n.strip() for n in settings.namespace_whitelist.split(",") if n.strip()]
        if allowed and ns not in allowed:
            raise HTTPException(status_code=403, detail="Namespace not allowed")
    if settings.api_key_namespace:
        if ns != settings.api_key_namespace:
            raise HTTPException(status_code=403, detail="Namespace not allowed for this API key")
    return ns



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
    schema = _CollectionSchema(fields, description="RAG 鏂囨。鍒嗙墖")
    coll = _Collection(name, schema=schema)
    index_params = {"index_type": "IVF_FLAT", "metric_type": "IP", "params": {"nlist": 1024}}
    coll.create_index(field_name="embedding", index_params=index_params)


@app.post("/namespaces/create")
def ns_create(namespace: str | None = None, x_api_key: str | None = None) -> JSONResponse:  # type: ignore[override]
    _require_api_key(x_api_key)
    ns = _resolve_namespace(namespace)
    try:
        _create_collection(ns)
        return JSONResponse({"ok": True, "namespace": ns, "collection": _collection_name(ns)})
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)


@app.post("/namespaces/clear")
def ns_clear(namespace: str | None = None, x_api_key: str | None = None) -> JSONResponse:  # type: ignore[override]
    _require_api_key(x_api_key)
    ns = _resolve_namespace(namespace)
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
    _require_api_key(x_api_key)
    ns = _resolve_namespace(namespace)
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
    _require_api_key(x_api_key)
    if pipeline is None:
        raise HTTPException(status_code=503, detail="RAG Pipeline 鏈垵濮嬪寲锛岃妫€鏌ユ湇鍔℃棩蹇?)
    ns = _resolve_namespace(namespace)
    local = _get_pipeline(ns)
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
    """鑾峰彇鍙敤鐨?LLM 妯″瀷鍒楄〃"""
    models = settings.available_models.split(",")
    return JSONResponse({
        "ok": True,
        "models": [m.strip() for m in models if m.strip()],
        "default_model": settings.llm_model
    })


@app.get("/healthz")
def healthz() -> JSONResponse:  # type: ignore[override]
    """鍋ュ悍妫€鏌ヤ笌鐩戞帶鎸囨爣"""
    ok = True
    details: Dict[str, Any] = {
        "timestamp": time.time(),
        "version": "0.2.0"
    }
    
    try:
        if pipeline is None:
            raise Exception("RAG Pipeline 鏈垵濮嬪寲")
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
                
                # 缁熻鍞竴鏂囨。鏁?
                try:
                    unique_paths = pipeline.list_paths(limit=10000)
                    details["document_count"] = len(unique_paths)
                except Exception:
                    details["document_count"] = 0
                    
            except Exception as e:
                logger.warning(f"Milvus 鐘舵€佹鏌ュけ璐? {e}")
                details["milvus_warning"] = str(e)
                details["document_count"] = 0
        else:
            # FAISS 鎴栧叾浠栧悗绔細缁熻鏂囨。鏁?
            try:
                unique_paths = pipeline.list_paths(limit=10000)
                details["document_count"] = len(unique_paths)
                # 瀵逛簬 FAISS锛屽彲浠ヨ幏鍙栧疄浣撴暟
                if active_backend == "faiss":
                    faiss_index = getattr(store, "faiss_index", None)
                    if faiss_index is not None:
                        details["faiss_entities"] = faiss_index.ntotal
            except Exception:
                details["document_count"] = 0
        
        # 鍩虹鎸囨爣
        details.update({
            "embedding_model": pipeline.settings.embedding_model_name,
            "llm_model": pipeline.settings.llm_model,
            "top_k": pipeline.settings.top_k,
            "vector_backend_config": pipeline.settings.vector_backend,
            "vector_backend_active": active_backend,
            "bm25_enabled": pipeline.settings.bm25_enabled,
            "reranker_enabled": pipeline.settings.reranker_enabled,
        })
        
        logger.debug("鍋ュ悍妫€鏌ュ畬鎴?)
        
    except Exception as exc:
        ok = False
        details["error"] = str(exc)
        logger.error(f"鍋ュ悍妫€鏌ュけ璐? {exc}")
    
    return JSONResponse({"ok": ok, "details": details})


@app.post("/ask_stream")
def ask_stream(req: AskRequest, x_api_key: str | None = None, namespace: str | None = None):  # type: ignore[override]
    _require_api_key(x_api_key)
    if pipeline is None:
        raise HTTPException(status_code=503, detail="RAG Pipeline 鏈垵濮嬪寲锛岃妫€鏌ユ湇鍔℃棩蹇?)
    ns = _resolve_namespace(namespace)
    local = _get_pipeline(ns)
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


# ==================== 鏌ヨ鏀瑰啓鐩稿叧 API ====================

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
    """浣跨敤鏌ヨ鏀瑰啓澧炲己妫€绱㈡晥鏋?""
    _require_api_key(x_api_key)
    if pipeline is None:
        raise HTTPException(status_code=503, detail="RAG Pipeline 鏈垵濮嬪寲")
    
    ns = _resolve_namespace(namespace)
    local = _get_pipeline(ns)
    
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
        logger.error(f"鏌ヨ鏀瑰啓澶辫触: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class QueryAnalysisRequest(BaseModel):
    question: str


@app.post("/analyze_query")
def analyze_query(req: QueryAnalysisRequest, x_api_key: str | None = None) -> Dict[str, Any]:
    """鍒嗘瀽鏌ヨ鐗瑰緛骞舵帹鑽愭渶浣虫敼鍐欑瓥鐣?""
    _require_api_key(x_api_key)
    if pipeline is None:
        raise HTTPException(status_code=503, detail="RAG Pipeline 鏈垵濮嬪寲")
    
    try:
        analysis = pipeline.analyze_query(req.question)
        return success_response(data=analysis)
    except Exception as e:
        logger.error(f"鏌ヨ鍒嗘瀽澶辫触: {e}")
        return error_response(message=str(e))


# ==================== 妫€绱㈢粨鏋滆В閲?API ====================

class ExplainRetrievalRequest(BaseModel):
    question: str
    top_k: int = 5


@app.post("/explain_retrieval")
def explain_retrieval(req: ExplainRetrievalRequest, x_api_key: str | None = None, namespace: str | None = None) -> Dict[str, Any]:
    """瑙ｉ噴妫€绱㈢粨鏋滐紝鏄剧ず涓轰粈涔堟绱㈠埌杩欎簺鏂囨。"""
    _require_api_key(x_api_key)
    if pipeline is None:
        raise HTTPException(status_code=503, detail="RAG Pipeline 鏈垵濮嬪寲")
    
    ns = _resolve_namespace(namespace)
    local = _get_pipeline(ns)
    
    try:
        # 鎵ц妫€绱?
        recs = local.store.search(req.question, req.top_k)
        
        # 杞崲涓哄瓧鍏告牸寮?
        chunks = [
            {
                "text": r.text,
                "score": r.score,
                "meta": r.meta
            }
            for r in recs
        ]
        
        # 鐢熸垚瑙ｉ噴
        from backend.retrieval_explainer import create_explainer
        explainer = create_explainer()
        explanations = explainer.explain_retrieval(req.question, chunks)
        summary = explainer.generate_summary(explanations)
        
        # 鏍煎紡鍖栬繑鍥?
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
        logger.error(f"妫€绱㈣В閲婂け璐? {e}")
        return error_response(message=str(e))


# ==================== 楂樼骇妫€绱?API ====================

class AdvancedSearchRequest(BaseModel):
    question: str
    top_k: int = 5
    # 杩囨护閰嶇疆
    doc_types: Optional[List[str]] = None
    date_from: Optional[str] = None
    date_to: Optional[str] = None
    min_score: Optional[float] = None
    paths: Optional[List[str]] = None
    tags: Optional[List[str]] = None
    has_tables: Optional[bool] = None
    page_range: Optional[tuple] = None
    # 鏉冮噸閰嶇疆
    vector_weight: float = 0.7
    bm25_weight: float = 0.3
    reranker_enabled: bool = True
    mmr_lambda: float = 0.5
    # 鑱氬悎閫夐」
    aggregate_by: Optional[str] = None  # 'document' or 'type'


@app.post("/advanced_search")
def advanced_search(req: AdvancedSearchRequest, x_api_key: str | None = None, namespace: str | None = None) -> Dict[str, Any]:
    """楂樼骇妫€绱?- 鏀寔杩囨护銆佹潈閲嶈皟浼樸€佽仛鍚?""
    _require_api_key(x_api_key)
    if pipeline is None:
        raise HTTPException(status_code=503, detail="RAG Pipeline 鏈垵濮嬪寲")
    
    ns = _resolve_namespace(namespace)
    local = _get_pipeline(ns)
    
    try:
        from backend.advanced_retrieval import create_retriever, FilterConfig, WeightConfig
        
        # 鍒涘缓楂樼骇妫€绱㈠櫒
        retriever = create_retriever()
        
        # 鎵ц鍩虹妫€绱?
        recs = local.store.search(req.question, req.top_k * 2)  # 澶氭绱竴浜涳紝鍚庨潰杩囨护
        
        # 杞崲涓哄瓧鍏告牸寮?
        results = [
            {
                "text": r.text,
                "score": r.score,
                "meta": r.meta
            }
            for r in recs
        ]
        
        # 搴旂敤杩囨护
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
        
        # 鑾峰彇缁熻淇℃伅
        stats = retriever.get_statistics(filtered_results)
        
        # 鑱氬悎锛堝鏋滈渶瑕侊級
        aggregated = None
        if req.aggregate_by == "document":
            aggregated = retriever.aggregate_by_document(filtered_results)
        elif req.aggregate_by == "type":
            aggregated = retriever.aggregate_by_type(filtered_results)
        
        # 鏍煎紡鍖栬繑鍥?
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
        logger.error(f"楂樼骇妫€绱㈠け璐? {e}")
        return error_response(message=str(e))


class OptimizeWeightsRequest(BaseModel):
    question: str
    top_k: int = 5
    test_weights: List[Dict[str, float]] = None  # [{"vector": 0.7, "bm25": 0.3}, ...]


@app.post("/optimize_weights")
def optimize_weights(req: OptimizeWeightsRequest, x_api_key: str | None = None, namespace: str | None = None) -> Dict[str, Any]:
    """鏉冮噸浼樺寲 - 娴嬭瘯涓嶅悓鏉冮噸缁勫悎鐨勬晥鏋?""
    _require_api_key(x_api_key)
    if pipeline is None:
        raise HTTPException(status_code=503, detail="RAG Pipeline 鏈垵濮嬪寲")
    
    ns = _resolve_namespace(namespace)
    local = _get_pipeline(ns)
    
    try:
        # 榛樿娴嬭瘯鏉冮噸缁勫悎
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
        
        # 鎵ц妫€绱?
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
        
        # 鎺ㄨ崘鏈€浣虫潈閲?
        best = max(results, key=lambda x: x["avg_score"])
        
        return success_response(data={
            "query": req.question,
            "test_results": results,
            "recommended_weights": best["weights"],
            "note": "鎺ㄨ崘鏉冮噸鍩轰簬骞冲潎鍒嗘暟锛屽疄闄呮晥鏋滈渶瑕佷汉宸ヨ瘎浼?
        })
    
    except Exception as e:
        logger.error(f"鏉冮噸浼樺寲澶辫触: {e}")
        return error_response(message=str(e))


# ==================== 鏂囨。鍒嗗潡鍙鍖?API ====================

class VisualizeChunksRequest(BaseModel):
    path: str


@app.post("/visualize_chunks")
def visualize_chunks(req: VisualizeChunksRequest, x_api_key: str | None = None, namespace: str | None = None) -> Dict[str, Any]:
    """鍙鍖栨枃妗ｇ殑鍒嗗潡缁撴灉"""
    _require_api_key(x_api_key)
    if pipeline is None:
        raise HTTPException(status_code=503, detail="RAG Pipeline 鏈垵濮嬪寲")
    
    ns = _resolve_namespace(namespace)
    local = _get_pipeline(ns)
    
    try:
        # 浠庣储寮曚腑鑾峰彇璇ユ枃妗ｇ殑鎵€鏈夊垎鍧?
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
            return error_response(message=f"鏈壘鍒版枃妗? {req.path}")
        
        # 鎺掑簭
        chunks.sort(key=lambda x: x["chunk_id"])
        
        # 缁熻淇℃伅
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
        logger.error(f"鏂囨。鍒嗗潡鍙鍖栧け璐? {e}")
        return error_response(message=str(e))


@app.get("/docs/preview")
def preview_document_chunks(path: str, x_api_key: str | None = None, namespace: str | None = None) -> Dict[str, Any]:
    """棰勮鏂囨。鐨勫垎鍧楁儏鍐碉紙GET 鏂瑰紡锛?""
    _require_api_key(x_api_key)
    if pipeline is None:
        raise HTTPException(status_code=503, detail="RAG Pipeline 鏈垵濮嬪寲")
    
    ns = _resolve_namespace(namespace)
    local = _get_pipeline(ns)
    
    try:
        # 鑾峰彇璇ユ枃妗ｇ殑鎵€鏈夊垎鍧楋紙鍙繑鍥炴憳瑕侊級
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
            return error_response(message=f"鏈壘鍒版枃妗? {path}")
        
        chunks_preview.sort(key=lambda x: x["chunk_id"])
        
        return success_response(data={
            "path": path,
            "total_chunks": len(chunks_preview),
            "chunks": chunks_preview
        })
    
    except Exception as e:
        logger.error(f"鏂囨。棰勮澶辫触: {e}")
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
    _require_api_key(x_api_key)
    if pipeline is None:
        return JSONResponse({"ok": False, "error": "RAG Pipeline 鏈垵濮嬪寲"}, status_code=503)
    ns = _resolve_namespace(namespace)
    local = _get_pipeline(ns)
    
    try:
        final_path: str | None = None
        text: str | None = None
        
        # 妫€鏌?Content-Type 鏉ュ喅瀹氬浣曡В鏋愯姹?
        content_type = request.headers.get("content-type", "")
        
        # 濡傛灉鏄?JSON 璇锋眰
        if "application/json" in content_type:
            try:
                body = await request.json()
                final_path = body.get("path")
                text = body.get("text")
                if not final_path:
                    return JSONResponse({"ok": False, "error": "缂哄皯 path 瀛楁"}, status_code=400)
            except Exception as e:
                logger.error(f"瑙ｆ瀽 JSON 澶辫触: {e}")
                return JSONResponse({"ok": False, "error": f"JSON 瑙ｆ瀽澶辫触: {str(e)}"}, status_code=400)
        
        # 濡傛灉鏄?multipart/form-data
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
                return JSONResponse({"ok": False, "error": "multipart 璇锋眰闇€瑕?file 鍜?path"}, status_code=400)
        
        else:
            return JSONResponse({"ok": False, "error": f"涓嶆敮鎸佺殑 Content-Type: {content_type}"}, status_code=400)
        
        if not final_path or text is None:
            return JSONResponse({"ok": False, "error": "path 鎴?text 涓虹┖"}, status_code=400)
        
        # 缁熶竴浣跨敤姝ｆ枩鏉狅紝閬垮厤 Windows 璺緞娣蜂贡
        final_path = final_path.replace("\\", "/")
        added = local.add_document(final_path, text or "")
        logger.info(f"鉁?鏂囨。涓婁紶鎴愬姛: {final_path}, 鏂板 {added} 涓垎鐗?)
        return JSONResponse({"ok": True, "added_chunks": added})
    except Exception as e:
        # 鍦?FAISS 妯″紡涓嬶紝add_document 浼氭姏鍑洪敊璇紱鏀逛负杩斿洖 400 涓庢彁绀?
        logger.error(f"鉁?鏂囨。涓婁紶澶辫触: {final_path}, 閿欒: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)


@app.delete("/docs")
def delete_doc(path: str, x_api_key: str | None = None, namespace: str | None = None) -> JSONResponse:  # type: ignore[override]
    _require_api_key(x_api_key)
    if pipeline is None:
        return JSONResponse({"ok": False, "error": "RAG Pipeline 鏈垵濮嬪寲"}, status_code=503)
    ns = _resolve_namespace(namespace)
    local = _get_pipeline(ns)
    try:
        deleted = local.delete_document(path)
        return JSONResponse({"ok": True, "deleted": deleted})
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)


@app.get("/docs/paths")
def list_doc_paths(limit: int = 1000, namespace: str | None = None) -> JSONResponse:  # type: ignore[override]
    if pipeline is None:
        return JSONResponse({"ok": False, "error": "RAG Pipeline 鏈垵濮嬪寲"}, status_code=503)
    ns = _resolve_namespace(namespace)
    local = _get_pipeline(ns)
    docs_stats = local.list_paths_with_stats(limit)
    return JSONResponse({"ok": True, "documents": docs_stats})


@app.get("/export")
def export_by_path(path: str, namespace: str | None = None) -> JSONResponse:  # type: ignore[override]
    if pipeline is None:
        return JSONResponse({"ok": False, "error": "RAG Pipeline 鏈垵濮嬪寲"}, status_code=503)
    
    try:
        ns = _resolve_namespace(namespace)
        local = _get_pipeline(ns)
        store = local.store
        backend = getattr(store, "backend", "faiss")
        
        chunks = []
        
        if backend == "milvus" and store.collection is not None:
            # Milvus 鍚庣
            escaped = path.replace("'", "\\'")
            expr = "path == '" + escaped + "'"
            recs = store.collection.query(expr=expr, output_fields=["path", "chunk_id", "text"], limit=10000)
            chunks = recs
        else:
            # FAISS 鍚庣锛氫粠鍐呭瓨涓繃婊?
            for i, meta in enumerate(store.metas):
                if meta.get("path") == path:
                    chunks.append({
                        "path": path,
                        "chunk_id": meta.get("chunk_id", i),
                        "text": store.texts[i] if i < len(store.texts) else ""
                    })
        
        return JSONResponse({"ok": True, "path": path, "chunks": chunks})
    except Exception as e:
        logger.error(f"瀵煎嚭鏂囨。澶辫触: {e}")
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)


@app.post("/import")
def import_chunks(payload: Dict[str, Any], x_api_key: str | None = None, namespace: str | None = None) -> JSONResponse:  # type: ignore[override]
    _require_api_key(x_api_key)
    if pipeline is None:
        return JSONResponse({"ok": False, "error": "RAG Pipeline 鏈垵濮嬪寲"}, status_code=503)
    try:
        path = payload.get("path")
        chunks = payload.get("chunks") or []
        if not path or not isinstance(chunks, list):
            return JSONResponse({"ok": False, "error": "invalid payload"}, status_code=400)
        # 鍏堝垹闄ゅ啀鍐欏叆
        try:
            ns = _resolve_namespace(namespace)
            local = _get_pipeline(ns)
            local.delete_document(path)
        except Exception:
            pass
        # 鐩存帴灏嗘枃鏈嫾鎺ュ悗鎸夌幇鏈?split 閲嶆柊鍒囧垎鏇寸ǔ濡?
        combined = "\n\n".join([c.get("text", "") for c in chunks])
        ns = _resolve_namespace(namespace)
        local = _get_pipeline(ns)
        added = local.add_document(path, combined)
        return JSONResponse({"ok": True, "added": added})
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)


@app.get("/cache/stats")
def cache_stats(x_api_key: str | None = None) -> JSONResponse:  # type: ignore[override]
    """鑾峰彇缂撳瓨缁熻淇℃伅"""
    _require_api_key(x_api_key)
    return JSONResponse({
        "ok": True,
        "cache_size": query_cache.size(),
        "max_size": query_cache.maxsize
    })


@app.post("/cache/clear")
def cache_clear(x_api_key: str | None = None) -> JSONResponse:  # type: ignore[override]
    """娓呯┖缂撳瓨"""
    _require_api_key(x_api_key)
    query_cache.clear()
    logger.info("缂撳瓨宸叉竻绌?)
    return JSONResponse({"ok": True, "message": "缂撳瓨宸叉竻绌?})


# ==================== 瀵硅瘽绠＄悊 API ====================
from backend.conversation import ConversationManager

conv_manager = ConversationManager()


@app.post("/conversations")
def create_conversation(title: str = "鏂板璇?, namespace: str | None = None) -> JSONResponse:  # type: ignore[override]
    """鍒涘缓鏂板璇?""
    ns = _resolve_namespace(namespace)
    conversation = conv_manager.create_conversation(title=title, namespace=ns)
    return JSONResponse({"ok": True, "conversation": conversation.to_dict()})


@app.get("/conversations")
def list_conversations(namespace: str | None = None, limit: int = 50, query: str | None = None) -> JSONResponse:  # type: ignore[override]
    """鍒楀嚭瀵硅瘽鍒楄〃"""
    ns = _resolve_namespace(namespace)
    conversations = conv_manager.list_conversations(namespace=ns, limit=limit, query=query)
    return JSONResponse({"ok": True, "conversations": conversations})


@app.get("/conversations/{conv_id}")
def get_conversation(conv_id: str) -> JSONResponse:  # type: ignore[override]
    """鑾峰彇瀵硅瘽璇︽儏"""
    conversation = conv_manager.get_conversation(conv_id)
    if not conversation:
        return JSONResponse({"ok": False, "error": "瀵硅瘽涓嶅瓨鍦?}, status_code=404)
    return JSONResponse({"ok": True, "conversation": conversation.to_dict()})


@app.delete("/conversations/{conv_id}")
def delete_conversation_endpoint(conv_id: str) -> JSONResponse:  # type: ignore[override]
    """鍒犻櫎瀵硅瘽"""
    success = conv_manager.delete_conversation(conv_id)
    if success:
        return JSONResponse({"ok": True, "message": "瀵硅瘽宸插垹闄?})
    return JSONResponse({"ok": False, "error": "瀵硅瘽涓嶅瓨鍦?}, status_code=404)


@app.post("/conversations/{conv_id}/messages")
def add_message_to_conversation(
    conv_id: str,
    role: str,
    content: str,
    sources: List[Dict[str, Any]] | None = None
) -> JSONResponse:  # type: ignore[override]
    """娣诲姞娑堟伅鍒板璇?""
    conversation = conv_manager.add_message(conv_id, role, content, sources)
    if not conversation:
        return JSONResponse({"ok": False, "error": "瀵硅瘽涓嶅瓨鍦?}, status_code=404)
    return JSONResponse({"ok": True, "conversation": conversation.to_dict()})


# ==================== 鎬ц兘鐩戞帶 API ====================

@app.get("/metrics/statistics")
def get_performance_statistics(x_api_key: str | None = None) -> Dict[str, Any]:
    """鑾峰彇鎬ц兘缁熻淇℃伅"""
    _require_api_key(x_api_key)
    monitor = get_monitor()
    
    return success_response(data={
        "overall": monitor.get_statistics(),
        "endpoints": monitor.get_endpoint_statistics(),
        "performance_breakdown": monitor.get_performance_breakdown()
    })


@app.get("/metrics/hot_queries")
def get_hot_queries(top_n: int = 10, x_api_key: str | None = None) -> Dict[str, Any]:
    """鑾峰彇鐑棬鏌ヨ"""
    _require_api_key(x_api_key)
    monitor = get_monitor()
    
    return success_response(data={
        "hot_queries": monitor.get_hot_queries(top_n)
    })


@app.get("/metrics/recent_requests")
def get_recent_requests(limit: int = 50, x_api_key: str | None = None) -> Dict[str, Any]:
    """鑾峰彇鏈€杩戠殑璇锋眰璁板綍"""
    _require_api_key(x_api_key)
    monitor = get_monitor()
    
    return success_response(data={
        "recent_requests": monitor.get_recent_requests(limit)
    })


@app.get("/metrics/time_series")
def get_time_series(interval_seconds: int = 60, x_api_key: str | None = None) -> Dict[str, Any]:
    """鑾峰彇鏃堕棿搴忓垪鏁版嵁"""
    _require_api_key(x_api_key)
    monitor = get_monitor()
    
    return success_response(data={
        "time_series": monitor.get_time_series(interval_seconds)
    })


@app.post("/metrics/export")
def export_metrics(filepath: str = "data/metrics/export.json", x_api_key: str | None = None) -> Dict[str, Any]:
    """瀵煎嚭鎬ц兘鎸囨爣鍒版枃浠?""
    _require_api_key(x_api_key)
    monitor = get_monitor()
    
    try:
        monitor.export_metrics(filepath)
        return success_response(data={"filepath": filepath, "message": "鎸囨爣宸插鍑?})
    except Exception as e:
        logger.error(f"瀵煎嚭鎸囨爣澶辫触: {e}")
        return error_response(message=str(e))


@app.post("/metrics/clear")
def clear_metrics(x_api_key: str | None = None) -> Dict[str, Any]:
    """娓呯┖鎬ц兘鐩戞帶鏁版嵁"""
    _require_api_key(x_api_key)
    monitor = get_monitor()
    monitor.clear_history()
    
    return success_response(data={"message": "鐩戞帶鏁版嵁宸叉竻绌?})


# ==================== 鏂囨。绠＄悊澧炲己 API ====================

from backend.document_manager import get_document_manager

doc_manager = get_document_manager()


class UpdateDocumentRequest(BaseModel):
    path: str
    tags: Optional[List[str]] = None
    category: Optional[str] = None
    description: Optional[str] = None


@app.post("/documents/metadata")
def update_document_metadata(req: UpdateDocumentRequest, x_api_key: str | None = None) -> Dict[str, Any]:
    """鏇存柊鏂囨。鍏冩暟鎹?""
    _require_api_key(x_api_key)
    
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
        logger.error(f"鏇存柊鏂囨。鍏冩暟鎹け璐? {e}")
        return error_response(message=str(e))


@app.get("/documents/metadata/{path:path}")
def get_document_metadata(path: str, x_api_key: str | None = None) -> Dict[str, Any]:
    """鑾峰彇鏂囨。鍏冩暟鎹?""
    _require_api_key(x_api_key)
    
    metadata = doc_manager.get_document(path)
    if not metadata:
        return error_response(message="鏂囨。涓嶅瓨鍦?, status_code=404)
    
    return success_response(data=metadata)


@app.get("/documents/list")
def list_documents_with_metadata(
    category: Optional[str] = None,
    tags: Optional[str] = None,
    x_api_key: str | None = None
) -> Dict[str, Any]:
    """鍒楀嚭鏂囨。锛堟敮鎸佹寜鍒嗙被鍜屾爣绛捐繃婊わ級"""
    _require_api_key(x_api_key)
    
    tag_list = tags.split(",") if tags else None
    documents = doc_manager.list_documents(category=category, tags=tag_list)
    
    return success_response(data={
        "documents": documents,
        "count": len(documents)
    })


@app.get("/documents/tags")
def get_all_tags(x_api_key: str | None = None) -> Dict[str, Any]:
    """鑾峰彇鎵€鏈夋爣绛?""
    _require_api_key(x_api_key)
    
    tags = doc_manager.metadata.get_all_tags()
    return success_response(data={"tags": tags, "count": len(tags)})


@app.get("/documents/categories")
def get_all_categories(x_api_key: str | None = None) -> Dict[str, Any]:
    """鑾峰彇鎵€鏈夊垎绫?""
    _require_api_key(x_api_key)
    
    categories = doc_manager.metadata.get_all_categories()
    return success_response(data={"categories": categories, "count": len(categories)})


@app.get("/documents/statistics")
def get_document_statistics(x_api_key: str | None = None) -> Dict[str, Any]:
    """鑾峰彇鏂囨。缁熻淇℃伅"""
    _require_api_key(x_api_key)
    
    stats = doc_manager.get_statistics()
    return success_response(data=stats)


@app.post("/documents/{path:path}/tags")
def add_document_tags(path: str, tags: List[str], x_api_key: str | None = None) -> Dict[str, Any]:
    """娣诲姞鏂囨。鏍囩"""
    _require_api_key(x_api_key)
    
    try:
        doc_manager.metadata.add_tags(path, tags)
        return success_response(data={
            "path": path,
            "tags": doc_manager.get_document(path).get("tags", [])
        })
    except Exception as e:
        logger.error(f"娣诲姞鏍囩澶辫触: {e}")
        return error_response(message=str(e))


@app.delete("/documents/{path:path}/tags")
def remove_document_tags(path: str, tags: List[str], x_api_key: str | None = None) -> Dict[str, Any]:
    """绉婚櫎鏂囨。鏍囩"""
    _require_api_key(x_api_key)
    
    try:
        doc_manager.metadata.remove_tags(path, tags)
        return success_response(data={
            "path": path,
            "tags": doc_manager.get_document(path).get("tags", [])
        })
    except Exception as e:
        logger.error(f"绉婚櫎鏍囩澶辫触: {e}")
        return error_response(message=str(e))


# ==================== 妫€绱紭鍖?API ====================

from backend.retrieval_optimizer import get_optimizer


@app.post("/retrieval/analyze")
def analyze_retrieval_quality(
    question: str,
    top_k: int = 10,
    x_api_key: str | None = None
) -> Dict[str, Any]:
    """鍒嗘瀽妫€绱㈣川閲忓苟鎻愪緵浼樺寲寤鸿"""
    _require_api_key(x_api_key)
    
    try:
        # 鎵ц妫€绱?
        results = pipeline.vector_store.search(question, top_k=top_k)
        
        # 杞崲涓哄瓧鍏告牸寮?
        results_dict = [
            {
                "text": r.text,
                "score": r.score,
                "meta": r.meta
            }
            for r in results
        ]
        
        # 鍒嗘瀽浼樺寲
        optimizer = get_optimizer()
        analysis = optimizer.optimize(question, results_dict)
        
        return success_response(data=analysis)
    
    except Exception as e:
        logger.error(f"妫€绱㈠垎鏋愬け璐? {e}")
        return error_response(message=str(e))


@app.post("/retrieval/suggest_weights")
def suggest_optimal_weights(
    question: str,
    x_api_key: str | None = None
) -> Dict[str, Any]:
    """鏍规嵁鏌ヨ绫诲瀷寤鸿鏈€浣虫潈閲?""
    _require_api_key(x_api_key)
    
    try:
        optimizer = get_optimizer()
        
        # 鑷姩鍒ゆ柇鏉冮噸
        vec_weight, bm25_weight = optimizer.weight_optimizer.adaptive_weights(question)
        
        # 鎺ㄨ崘鏀瑰啓绛栫暐
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
        logger.error(f"鏉冮噸寤鸿澶辫触: {e}")
        return error_response(message=str(e))


def _get_weight_explanation(vec_weight: float, bm25_weight: float) -> str:
    """瑙ｉ噴鏉冮噸閫夋嫨"""
    if vec_weight > 0.7:
        return "鏌ヨ鍋忚涔夌悊瑙ｏ紝浣跨敤楂樺悜閲忔潈閲?
    elif bm25_weight > 0.6:
        return "鏌ヨ鍋忓叧閿瘝鍖归厤锛屼娇鐢ㄩ珮 BM25 鏉冮噸"
    else:
        return "鏌ヨ绫诲瀷骞宠　锛屼娇鐢ㄥ潎琛℃潈閲?


@app.post("/retrieval/grid_search")
def grid_search_weights(
    question: str,
    top_k: int = 10,
    x_api_key: str | None = None
) -> Dict[str, Any]:
    """缃戞牸鎼滅储鏈€浣虫潈閲嶇粍鍚?""
    _require_api_key(x_api_key)
    
    try:
        optimizer = get_optimizer()
        
        # 瀹氫箟妫€绱㈠嚱鏁?
        def retrieval_func(query, top_k, vector_weight, bm25_weight):
            # 涓存椂淇敼鏉冮噸
            old_vec = settings.rag_vec_weight
            old_bm25 = settings.rag_bm25_weight
            
            settings.rag_vec_weight = vector_weight
            settings.rag_bm25_weight = bm25_weight
            
            results = pipeline.vector_store.search(query, top_k=top_k)
            
            # 鎭㈠鏉冮噸
            settings.rag_vec_weight = old_vec
            settings.rag_bm25_weight = old_bm25
            
            return [{"score": r.score} for r in results]
        
        # 鎵ц缃戞牸鎼滅储
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
        logger.error(f"缃戞牸鎼滅储澶辫触: {e}")
        return error_response(message=str(e))


@app.post("/retrieval/compare_strategies")
def compare_retrieval_strategies(
    question: str,
    top_k: int = 10,
    x_api_key: str | None = None
) -> Dict[str, Any]:
    """姣旇緝涓嶅悓妫€绱㈢瓥鐣ョ殑鏁堟灉"""
    _require_api_key(x_api_key)
    
    try:
        strategies = []
        
        # 绛栫暐 1: 绾悜閲忔绱?
        old_bm25 = settings.rag_bm25_enabled
        settings.rag_bm25_enabled = False
        results_vec = pipeline.vector_store.search(question, top_k=top_k)
        settings.rag_bm25_enabled = old_bm25
        
        strategies.append({
            "name": "绾悜閲忔绱?,
            "config": {"vector_weight": 1.0, "bm25_weight": 0.0},
            "avg_score": round(sum(r.score for r in results_vec) / len(results_vec), 4) if results_vec else 0,
            "result_count": len(results_vec)
        })
        
        # 绛栫暐 2: 娣峰悎妫€绱紙榛樿鏉冮噸锛?
        results_hybrid = pipeline.vector_store.search(question, top_k=top_k)
        strategies.append({
            "name": "娣峰悎妫€绱紙榛樿锛?,
            "config": {
                "vector_weight": settings.rag_vec_weight,
                "bm25_weight": settings.rag_bm25_weight
            },
            "avg_score": round(sum(r.score for r in results_hybrid) / len(results_hybrid), 4) if results_hybrid else 0,
            "result_count": len(results_hybrid)
        })
        
        # 绛栫暐 3: 鑷€傚簲鏉冮噸
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
            "name": "鑷€傚簲鏉冮噸",
            "config": {"vector_weight": vec_w, "bm25_weight": bm25_w},
            "avg_score": round(sum(r.score for r in results_adaptive) / len(results_adaptive), 4) if results_adaptive else 0,
            "result_count": len(results_adaptive)
        })
        
        # 鎺掑簭
        strategies.sort(key=lambda x: x["avg_score"], reverse=True)
        
        return success_response(data={
            "question": question,
            "strategies": strategies,
            "best_strategy": strategies[0]["name"]
        })
    
    except Exception as e:
        logger.error(f"绛栫暐姣旇緝澶辫触: {e}")
        return error_response(message=str(e))


# ==================== 鏌ヨ鎰忓浘璇嗗埆 API ====================

from backend.query_intent import get_query_analyzer, IntentRecognizer


@app.post("/query/analyze_intent")
def analyze_query_intent(
    question: str,
    x_api_key: str | None = None
) -> Dict[str, Any]:
    """鍒嗘瀽鏌ヨ鎰忓浘骞舵彁渚涗紭鍖栧缓璁?""
    _require_api_key(x_api_key)
    
    try:
        analyzer = get_query_analyzer()
        analysis = analyzer.analyze(question)
        
        return success_response(data=analysis)
    
    except Exception as e:
        logger.error(f"鎰忓浘鍒嗘瀽澶辫触: {e}")
        return error_response(message=str(e))


@app.post("/query/smart_search")
def smart_search_with_intent(
    question: str,
    top_k: Optional[int] = None,
    use_recommended_config: bool = True,
    x_api_key: str | None = None
) -> Dict[str, Any]:
    """鍩轰簬鎰忓浘璇嗗埆鐨勬櫤鑳芥绱?""
    _require_api_key(x_api_key)
    
    try:
        # 鍒嗘瀽鏌ヨ鎰忓浘
        analyzer = get_query_analyzer()
        analysis = analyzer.analyze(question)
        
        # 鑾峰彇鎺ㄨ崘閰嶇疆
        config = analysis["recommended_config"]
        
        if use_recommended_config:
            # 浣跨敤鎺ㄨ崘鐨勯厤缃?
            actual_top_k = top_k or config["top_k"]
            
            # 涓存椂淇敼閰嶇疆
            old_vec = settings.rag_vec_weight
            old_bm25 = settings.rag_bm25_weight
            
            settings.rag_vec_weight = config["vector_weight"]
            settings.rag_bm25_weight = config["bm25_weight"]
            
            # 鎵ц妫€绱?
            results = pipeline.vector_store.search(question, top_k=actual_top_k)
            
            # 鎭㈠閰嶇疆
            settings.rag_vec_weight = old_vec
            settings.rag_bm25_weight = old_bm25
        else:
            # 浣跨敤榛樿閰嶇疆
            actual_top_k = top_k or 10
            results = pipeline.vector_store.search(question, top_k=actual_top_k)
        
        # 杞崲缁撴灉
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
        logger.error(f"鏅鸿兘妫€绱㈠け璐? {e}")
        return error_response(message=str(e))


@app.post("/query/batch_analyze")
def batch_analyze_queries(
    questions: List[str],
    x_api_key: str | None = None
) -> Dict[str, Any]:
    """鎵归噺鍒嗘瀽鏌ヨ鎰忓浘"""
    _require_api_key(x_api_key)
    
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
        
        # 缁熻鎰忓浘鍒嗗竷
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
        logger.error(f"鎵归噺鍒嗘瀽澶辫触: {e}")
        return error_response(message=str(e))


# ==================== 缂撳瓨浼樺寲 API ====================

from backend.cache_optimizer import get_smart_cache, CachePrewarmer


@app.get("/cache/analyze")
def analyze_cache(x_api_key: str | None = None) -> Dict[str, Any]:
    """鍒嗘瀽缂撳瓨鎬ц兘骞舵彁渚涗紭鍖栧缓璁?""
    _require_api_key(x_api_key)
    
    try:
        smart_cache = get_smart_cache()
        analysis = smart_cache.analyze()
        
        return success_response(data=analysis)
    
    except Exception as e:
        logger.error(f"缂撳瓨鍒嗘瀽澶辫触: {e}")
        return error_response(message=str(e))


@app.post("/cache/prewarm")
def prewarm_cache(
    use_hot_queries: bool = True,
    top_n: int = 20,
    custom_queries: Optional[List[str]] = None,
    x_api_key: str | None = None
) -> Dict[str, Any]:
    """缂撳瓨棰勭儹"""
    _require_api_key(x_api_key)
    
    try:
        # 瀹氫箟妫€绱㈠嚱鏁?
        def retrieval_func(query: str):
            return pipeline.vector_store.search(query, top_k=10)
        
        prewarmer = CachePrewarmer(retrieval_func)
        results = {"prewarmed_queries": []}
        
        # 浠庣儹闂ㄦ煡璇㈤鐑?
        if use_hot_queries:
            from backend.performance_monitor import get_monitor
            monitor = get_monitor()
            hot_queries = monitor.get_hot_queries(top_n)
            
            hot_result = prewarmer.prewarm_from_hot_queries(hot_queries, top_n)
            results["hot_queries_result"] = hot_result
            results["prewarmed_queries"].extend(hot_result["prewarmed_queries"])
        
        # 浠庤嚜瀹氫箟鏌ヨ棰勭儹
        if custom_queries:
            custom_result = prewarmer.prewarm_from_patterns(custom_queries)
            results["custom_queries_result"] = custom_result
            results["prewarmed_queries"].extend(custom_result["patterns"])
        
        return success_response(data={
            "total_prewarmed": len(results["prewarmed_queries"]),
            "details": results
        })
    
    except Exception as e:
        logger.error(f"缂撳瓨棰勭儹澶辫触: {e}")
        return error_response(message=str(e))


@app.get("/cache/smart_stats")
def get_smart_cache_stats(x_api_key: str | None = None) -> Dict[str, Any]:
    """鑾峰彇鏅鸿兘缂撳瓨缁熻"""
    _require_api_key(x_api_key)
    
    try:
        smart_cache = get_smart_cache()
        stats = smart_cache.get_stats()
        
        # 娣诲姞璇︾粏淇℃伅
        cache_entries = []
        for key, entry in list(smart_cache.cache.items())[:10]:  # 鍙繑鍥炲墠 10 涓?
            cache_entries.append({
                "key": key[:16] + "...",  # 鎴柇鏄剧ず
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
        logger.error(f"鑾峰彇缂撳瓨缁熻澶辫触: {e}")
        return error_response(message=str(e))


@app.post("/cache/optimize")
def optimize_cache_config(x_api_key: str | None = None) -> Dict[str, Any]:
    """浼樺寲缂撳瓨閰嶇疆"""
    _require_api_key(x_api_key)
    
    try:
        smart_cache = get_smart_cache()
        analysis = smart_cache.analyze()
        
        # 鏍规嵁鍒嗘瀽缁撴灉璋冩暣閰嶇疆
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
        
        # 鍛戒腑鐜囦綆锛屽缓璁鍔犲閲?
        if hit_rate < 0.3:
            new_max_size = int(max_size * 1.5)
            recommendations["recommended_config"]["max_size"] = new_max_size
            recommendations["actions"].append(f"澧炲姞缂撳瓨瀹归噺鍒?{new_max_size}")
        
        # 绌洪棿鍒╃敤鐜囬珮锛屽缓璁鍔犲閲?
        utilization = current_size / max_size if max_size > 0 else 0
        if utilization > 0.9:
            new_max_size = int(max_size * 1.3)
            recommendations["recommended_config"]["max_size"] = new_max_size
            recommendations["actions"].append(f"澧炲姞缂撳瓨瀹归噺鍒?{new_max_size}")
        
        # 绌洪棿鍒╃敤鐜囦綆锛屽缓璁噺灏戝閲?
        if utilization < 0.2 and max_size > 100:
            new_max_size = max(100, int(max_size * 0.7))
            recommendations["recommended_config"]["max_size"] = new_max_size
            recommendations["actions"].append(f"鍑忓皯缂撳瓨瀹归噺鍒?{new_max_size}")
        
        if not recommendations["actions"]:
            recommendations["actions"].append("褰撳墠閰嶇疆鑹ソ锛屾棤闇€璋冩暣")
        
        return success_response(data={
            "analysis": analysis,
            "recommendations": recommendations
        })
    
    except Exception as e:
        logger.error(f"缂撳瓨浼樺寲澶辫触: {e}")
        return error_response(message=str(e))


# ==================== 璇勪及娴嬭瘯 API ====================

from backend.evaluation import get_evaluator, TestCase, TestCaseGenerator, BenchmarkReport


@app.post("/evaluation/run_benchmark")
def run_benchmark_test(
    test_cases: Optional[List[Dict[str, Any]]] = None,
    use_default_tests: bool = False,
    top_k: int = 10,
    x_api_key: str | None = None
) -> Dict[str, Any]:
    """杩愯鍩哄噯娴嬭瘯"""
    _require_api_key(x_api_key)
    
    try:
        evaluator = get_evaluator(pipeline)
        
        # 鍑嗗娴嬭瘯鐢ㄤ緥
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
            return error_response(message="璇锋彁渚涙祴璇曠敤渚嬫垨浣跨敤榛樿娴嬭瘯")
        
        # 杩愯璇勪及
        results = evaluator.run_benchmark(cases, top_k=top_k)
        
        # 鐢熸垚鎶ュ憡
        report = BenchmarkReport.generate_report(results)
        
        return success_response(data={
            "results": results,
            "report": report
        })
    
    except Exception as e:
        logger.error(f"鍩哄噯娴嬭瘯澶辫触: {e}")
        return error_response(message=str(e))


@app.post("/evaluation/test_retrieval")
def test_retrieval_quality(
    question: str,
    expected_doc_paths: List[str],
    top_k: int = 10,
    x_api_key: str | None = None
) -> Dict[str, Any]:
    """娴嬭瘯鍗曚釜鏌ヨ鐨勬绱㈣川閲?""
    _require_api_key(x_api_key)
    
    try:
        evaluator = get_evaluator(pipeline)
        
        test_case = TestCase(
            question=question,
            expected_doc_paths=expected_doc_paths
        )
        
        results = evaluator.evaluate_retrieval([test_case], top_k=top_k)
        
        return success_response(data=results)
    
    except Exception as e:
        logger.error(f"妫€绱㈡祴璇曞け璐? {e}")
        return error_response(message=str(e))


@app.post("/evaluation/test_answer")
def test_answer_quality(
    question: str,
    expected_keywords: Optional[List[str]] = None,
    x_api_key: str | None = None
) -> Dict[str, Any]:
    """娴嬭瘯鍗曚釜鏌ヨ鐨勭瓟妗堣川閲?""
    _require_api_key(x_api_key)
    
    try:
        evaluator = get_evaluator(pipeline)
        
        test_case = TestCase(
            question=question,
            expected_keywords=expected_keywords
        )
        
        results = evaluator.evaluate_answer_quality([test_case])
        
        return success_response(data=results)
    
    except Exception as e:
        logger.error(f"绛旀娴嬭瘯澶辫触: {e}")
        return error_response(message=str(e))


@app.post("/evaluation/save_test_cases")
def save_test_cases_to_file(
    test_cases: List[Dict[str, Any]],
    filepath: str = "data/evaluation/test_cases.json",
    x_api_key: str | None = None
) -> Dict[str, Any]:
    """淇濆瓨娴嬭瘯鐢ㄤ緥鍒版枃浠?""
    _require_api_key(x_api_key)
    
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
            "message": "娴嬭瘯鐢ㄤ緥宸蹭繚瀛?
        })
    
    except Exception as e:
        logger.error(f"淇濆瓨娴嬭瘯鐢ㄤ緥澶辫触: {e}")
        return error_response(message=str(e))


@app.get("/evaluation/load_test_cases")
def load_test_cases_from_file(
    filepath: str = "data/evaluation/test_cases.json",
    x_api_key: str | None = None
) -> Dict[str, Any]:
    """浠庢枃浠跺姞杞芥祴璇曠敤渚?""
    _require_api_key(x_api_key)
    
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
        logger.error(f"鍔犺浇娴嬭瘯鐢ㄤ緥澶辫触: {e}")
        return error_response(message=str(e))


# ==================== 鐭ヨ瘑鍥捐氨 API ====================

from backend.knowledge_graph import (
    get_knowledge_graph, set_knowledge_graph,
    KnowledgeGraphBuilder, GraphEnhancedRetriever
)


@app.post("/kg/build")
def build_knowledge_graph(
    doc_paths: Optional[List[str]] = None,
    x_api_key: str | None = None
) -> Dict[str, Any]:
    """鏋勫缓鐭ヨ瘑鍥捐氨"""
    _require_api_key(x_api_key)
    
    try:
        # 鑾峰彇鏂囨。
        if doc_paths:
            # 浠庢寚瀹氳矾寰勫姞杞?
            documents = []
            for path in doc_paths:
                # 浠庡悜閲忓瓨鍌ㄤ腑鑾峰彇鏂囨。
                results = pipeline.vector_store.search(path, top_k=100)
                for r in results:
                    if r.meta.get("path") == path:
                        documents.append({
                            "text": r.text,
                            "path": r.meta.get("path", "")
                        })
        else:
            # 浣跨敤鎵€鏈夋枃妗?
            documents = []
            # 杩欓噷绠€鍖栧鐞嗭紝瀹為檯搴旇浠庡悜閲忓瓨鍌ㄤ腑鑾峰彇鎵€鏈夋枃妗?
            logger.info("浣跨敤鍚戦噺瀛樺偍涓殑鎵€鏈夋枃妗ｆ瀯寤虹煡璇嗗浘璋?)
        
        # 鏋勫缓鍥捐氨
        builder = KnowledgeGraphBuilder()
        kg = builder.build_from_documents(documents)
        
        # 璁剧疆鍏ㄥ眬鍥捐氨
        set_knowledge_graph(kg)
        
        # 鑾峰彇缁熻淇℃伅
        stats = kg.get_statistics()
        
        return success_response(data={
            "message": "鐭ヨ瘑鍥捐氨鏋勫缓瀹屾垚",
            "statistics": stats
        })
    
    except Exception as e:
        logger.error(f"鏋勫缓鐭ヨ瘑鍥捐氨澶辫触: {e}")
        return error_response(message=str(e))


@app.get("/kg/statistics")
def get_kg_statistics(x_api_key: str | None = None) -> Dict[str, Any]:
    """鑾峰彇鐭ヨ瘑鍥捐氨缁熻淇℃伅"""
    _require_api_key(x_api_key)
    
    try:
        kg = get_knowledge_graph()
        stats = kg.get_statistics()
        
        return success_response(data=stats)
    
    except Exception as e:
        logger.error(f"鑾峰彇缁熻淇℃伅澶辫触: {e}")
        return error_response(message=str(e))


@app.get("/kg/entity/{entity_name}")
def get_entity_info(
    entity_name: str,
    x_api_key: str | None = None
) -> Dict[str, Any]:
    """鑾峰彇瀹炰綋淇℃伅"""
    _require_api_key(x_api_key)
    
    try:
        kg = get_knowledge_graph()
        entity = kg.get_entity(entity_name)
        
        if not entity:
            return error_response(message="瀹炰綋涓嶅瓨鍦?, status_code=404)
        
        return success_response(data={
            "name": entity.name,
            "type": entity.type,
            "mentions": entity.mentions[:10],  # 鏈€澶氳繑鍥?10 涓?
            "doc_paths": entity.doc_paths,
            "doc_count": len(entity.doc_paths),
            "attributes": entity.attributes
        })
    
    except Exception as e:
        logger.error(f"鑾峰彇瀹炰綋淇℃伅澶辫触: {e}")
        return error_response(message=str(e))


@app.get("/kg/subgraph/{entity_name}")
def get_entity_subgraph(
    entity_name: str,
    depth: int = 2,
    x_api_key: str | None = None
) -> Dict[str, Any]:
    """鑾峰彇瀹炰綋瀛愬浘"""
    _require_api_key(x_api_key)
    
    try:
        kg = get_knowledge_graph()
        subgraph = kg.get_subgraph(entity_name, depth)
        
        return success_response(data=subgraph)
    
    except Exception as e:
        logger.error(f"鑾峰彇瀛愬浘澶辫触: {e}")
        return error_response(message=str(e))


@app.get("/kg/search")
def search_entities(
    query: str,
    entity_type: Optional[str] = None,
    limit: int = 10,
    x_api_key: str | None = None
) -> Dict[str, Any]:
    """鎼滅储瀹炰綋"""
    _require_api_key(x_api_key)
    
    try:
        kg = get_knowledge_graph()
        entities = kg.search_entities(query, entity_type)
        
        results = [
            {
                "name": e.name,
                "type": e.type,
                "doc_count": len(e.doc_paths)
            }
            for e in entities[:limit]
        ]
        
        return success_response(data={
            "query": query,
            "entity_type": entity_type,
            "results": results,
            "count": len(results)
        })
    
    except Exception as e:
        logger.error(f"鎼滅储瀹炰綋澶辫触: {e}")
        return error_response(message=str(e))


@app.get("/kg/path")
def find_entity_path(
    start: str,
    end: str,
    max_depth: int = 3,
    x_api_key: str | None = None
) -> Dict[str, Any]:
    """鏌ユ壘瀹炰綋璺緞"""
    _require_api_key(x_api_key)
    
    try:
        kg = get_knowledge_graph()
        paths = kg.find_path(start, end, max_depth)
        
        return success_response(data={
            "start": start,
            "end": end,
            "paths": paths[:5],  # 鏈€澶氳繑鍥?5 鏉¤矾寰?
            "path_count": len(paths)
        })
    
    except Exception as e:
        logger.error(f"鏌ユ壘璺緞澶辫触: {e}")
        return error_response(message=str(e))


@app.post("/kg/enhanced_search")
def graph_enhanced_search(
    question: str,
    top_k: int = 10,
    use_graph_expansion: bool = True,
    x_api_key: str | None = None
) -> Dict[str, Any]:
    """鍥捐氨澧炲己妫€绱?""
    _require_api_key(x_api_key)
    
    try:
        kg = get_knowledge_graph()
        retriever = GraphEnhancedRetriever(kg)
        
        # 鏌ヨ鎵╁睍
        expanded_queries = [question]
        if use_graph_expansion:
            expanded_queries = retriever.expand_query_with_graph(question)
        
        # 鑾峰彇鐩稿叧瀹炰綋
        related_entities = retriever.get_related_entities(question)
        
        # 鎵ц妫€绱?
        all_results = []
        for query in expanded_queries[:3]:  # 鏈€澶氫娇鐢?3 涓墿灞曟煡璇?
            results = pipeline.vector_store.search(query, top_k=top_k)
            all_results.extend(results)
        
        # 鍘婚噸骞舵帓搴?
        seen = set()
        unique_results = []
        for r in all_results:
            key = (r.text, r.meta.get("path"))
            if key not in seen:
                seen.add(key)
                unique_results.append({
                    "text": r.text,
                    "score": r.score,
                    "meta": r.meta
                })
        
        unique_results.sort(key=lambda x: x["score"], reverse=True)
        
        return success_response(data={
            "question": question,
            "expanded_queries": expanded_queries,
            "related_entities": related_entities,
            "results": unique_results[:top_k],
            "total_results": len(unique_results)
        })
    
    except Exception as e:
        logger.error(f"鍥捐氨澧炲己妫€绱㈠け璐? {e}")
        return error_response(message=str(e))


@app.post("/kg/export")
def export_knowledge_graph(
    filepath: str = "data/kg/knowledge_graph.json",
    x_api_key: str | None = None
) -> Dict[str, Any]:
    """瀵煎嚭鐭ヨ瘑鍥捐氨"""
    _require_api_key(x_api_key)
    
    try:
        kg = get_knowledge_graph()
        kg.export_to_json(filepath)
        
        return success_response(data={
            "filepath": filepath,
            "message": "鐭ヨ瘑鍥捐氨宸插鍑?
        })
    
    except Exception as e:
        logger.error(f"瀵煎嚭鐭ヨ瘑鍥捐氨澶辫触: {e}")
        return error_response(message=str(e))


@app.post("/kg/import")
def import_knowledge_graph(
    filepath: str = "data/kg/knowledge_graph.json",
    x_api_key: str | None = None
) -> Dict[str, Any]:
    """瀵煎叆鐭ヨ瘑鍥捐氨"""
    _require_api_key(x_api_key)
    
    try:
        kg = get_knowledge_graph()
        kg.import_from_json(filepath)
        
        stats = kg.get_statistics()
        
        return success_response(data={
            "filepath": filepath,
            "message": "鐭ヨ瘑鍥捐氨宸插鍏?,
            "statistics": stats
        })
    
    except Exception as e:
        logger.error(f"瀵煎叆鐭ヨ瘑鍥捐氨澶辫触: {e}")
        return error_response(message=str(e))










