import os
import logging
from dataclasses import dataclass
from datetime import date


logger = logging.getLogger(__name__)


# Load .env.local then .env if available / 优先加载 .env.local，再加载 .env
# This keeps local overrides without committing secrets. / 本地覆盖，避免提交敏感信息。
def _load_env_files() -> None:
    try:
        from dotenv import load_dotenv

        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        for fname in ('.env.local', '.env'):
            env_path = os.path.join(project_root, fname)
            try:
                if os.path.exists(env_path):
                    load_dotenv(dotenv_path=env_path, override=True)
                    # Avoid printing secrets to stdout; rely on standard logging.
                    if os.getenv('RAG_DEBUG_CONFIG', '').lower() in {'1', 'true', 'yes'}:
                        logger.info('Loaded env file: %s', env_path)
            except Exception as e:
                logger.warning('Failed to load env file: %s (%s)', env_path, e)
    except Exception as e:
        # python-dotenv is optional in some deployments (e.g. env is injected by container/orchestrator).
        logger.debug('python-dotenv not installed or failed to import: %s', e)


_load_env_files()


@dataclass
class Settings:
    # Data paths / 数据路径
    docs_dir: str | None = None
    index_dir: str | None = None

    # Embeddings / 向量模型
    embedding_model_name: str | None = None

    # LLM (OpenAI-compatible) / 大模型（兼容 OpenAI）
    openai_api_key: str | None = None
    openai_base_url: str | None = None
    llm_model: str | None = None

    # Qwen (optional) / 通义千问（可选）
    qwen_api_key: str | None = None
    qwen_base_url: str | None = None

    # Available models list / 可用模型列表
    available_models: str | None = None

    # Retrieval defaults / 检索默认参数
    top_k: int | None = None

    # Reranker / 重排
    reranker_enabled: bool | None = None
    reranker_model_name: str | None = None
    reranker_top_n: int | None = None

    # Milvus / 向量数据库
    milvus_host: str | None = None
    milvus_port: str | None = None
    milvus_user: str | None = None
    milvus_password: str | None = None
    milvus_secure: bool | None = None
    milvus_db: str | None = None
    milvus_collection: str | None = None

    # Namespace & API key / 命名空间与鉴权
    default_namespace: str | None = None
    namespace_whitelist: str | None = None
    api_key_namespace: str | None = None
    api_key: str | None = None
    api_key_required: bool | None = None
    admin_api_key: str | None = None
    admin_api_key_required: bool | None = None
    admin_api_key_fallback_to_api_key: bool | None = None

    # CORS
    cors_allow_origins: str | None = None
    cors_allow_credentials: bool | None = None
    cors_allow_methods: str | None = None
    cors_allow_headers: str | None = None
    env: str | None = None

    # Vector backend / 向量后端: auto | milvus | faiss
    vector_backend: str | None = None

    # Hybrid retrieval / 混合检索
    bm25_enabled: bool | None = None
    bm25_weight: float | None = None
    vec_weight: float | None = None
    score_threshold: float | None = None
    mmr_lambda: float | None = None
    bm25_full_scan_max_docs: int | None = None
    candidate_k_mult: int | None = None
    candidate_k_min: int | None = None
    candidate_k_max: int | None = None

    # Strict mode / 严格模式
    strict_mode: bool | None = None
    enforce_namespace_path_prefix: bool | None = None
    query_expand_enabled: bool | None = None
    bm25_require_complete_corpus: bool | None = None
    prompt_show_scores: bool | None = None
    trace_enabled: bool | None = None
    auto_ingest_on_startup: bool | None = None
    audit_log_enabled: bool | None = None
    disable_legacy_routes: bool | None = None
    legacy_routes_sunset_date: str | None = None

    def __post_init__(self) -> None:
        self.docs_dir = self.docs_dir or os.getenv('RAG_DOCS_DIR', 'data/docs')
        self.index_dir = self.index_dir or os.getenv('RAG_INDEX_DIR', 'data/index')
        self.embedding_model_name = self.embedding_model_name or os.getenv(
            'RAG_EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2'
        )

        self.openai_api_key = self.openai_api_key or os.getenv('OPENAI_API_KEY')
        self.openai_base_url = self.openai_base_url or os.getenv('OPENAI_BASE_URL')
        self.llm_model = self.llm_model or os.getenv('RAG_MODEL', 'deepseek-reasoner')

        self.qwen_api_key = self.qwen_api_key or os.getenv('QWEN_API_KEY')
        self.qwen_base_url = self.qwen_base_url or os.getenv(
            'QWEN_BASE_URL', 'https://dashscope.aliyuncs.com/compatible-mode/v1'
        )

        self.available_models = self.available_models or os.getenv(
            'AVAILABLE_MODELS', 'deepseek-reasoner,deepseek-chat,qwen-turbo,qwen-plus,qwen-max'
        )

        if self.top_k is None:
            self.top_k = int(os.getenv('RAG_TOP_K', '8'))

        if self.reranker_enabled is None:
            self.reranker_enabled = os.getenv('RAG_RERANKER_ENABLED', 'false').lower() in {'1', 'true', 'yes'}
        self.reranker_model_name = self.reranker_model_name or os.getenv(
            'RAG_RERANKER_MODEL', 'BAAI/bge-reranker-base'
        )
        if self.reranker_top_n is None:
            self.reranker_top_n = int(os.getenv('RAG_RERANKER_TOP_N', '4'))

        self.milvus_host = self.milvus_host or os.getenv('MILVUS_HOST', '127.0.0.1')
        self.milvus_port = self.milvus_port or os.getenv('MILVUS_PORT', '19530')
        self.milvus_user = self.milvus_user or os.getenv('MILVUS_USER')
        self.milvus_password = self.milvus_password or os.getenv('MILVUS_PASSWORD')
        if self.milvus_secure is None:
            self.milvus_secure = os.getenv('MILVUS_SECURE', 'false').lower() in {'1', 'true', 'yes'}
        self.milvus_db = self.milvus_db or os.getenv('MILVUS_DB', None)
        self.milvus_collection = self.milvus_collection or os.getenv('MILVUS_COLLECTION', 'rag_chunks')

        self.default_namespace = self.default_namespace or os.getenv('RAG_NAMESPACE', 'default')
        self.namespace_whitelist = self.namespace_whitelist or os.getenv('RAG_NAMESPACE_WHITELIST')
        self.api_key_namespace = self.api_key_namespace or os.getenv('RAG_API_KEY_NAMESPACE')
        self.api_key = self.api_key or os.getenv('RAG_API_KEY')
        self.admin_api_key = self.admin_api_key or os.getenv('RAG_ADMIN_API_KEY')

        self.cors_allow_origins = self.cors_allow_origins or os.getenv(
            'RAG_CORS_ALLOW_ORIGINS', 'http://localhost:5173,http://127.0.0.1:5173'
        )
        if self.cors_allow_credentials is None:
            self.cors_allow_credentials = os.getenv('RAG_CORS_ALLOW_CREDENTIALS', 'true').lower() in {'1', 'true', 'yes'}
        self.cors_allow_methods = self.cors_allow_methods or os.getenv('RAG_CORS_ALLOW_METHODS', '*')
        self.cors_allow_headers = self.cors_allow_headers or os.getenv('RAG_CORS_ALLOW_HEADERS', '*')
        self.env = self.env or os.getenv('RAG_ENV', 'dev')

        self.vector_backend = (self.vector_backend or os.getenv('VECTOR_BACKEND', 'auto')).lower()

        if self.bm25_enabled is None:
            self.bm25_enabled = os.getenv('RAG_BM25_ENABLED', 'true').lower() in {'1', 'true', 'yes'}
        if self.bm25_weight is None:
            self.bm25_weight = float(os.getenv('RAG_BM25_WEIGHT', '0.4'))
        if self.vec_weight is None:
            self.vec_weight = float(os.getenv('RAG_VEC_WEIGHT', '0.6'))
        if self.score_threshold is None:
            self.score_threshold = float(os.getenv('RAG_SCORE_THRESHOLD', '0.0'))
        if self.mmr_lambda is None:
            self.mmr_lambda = float(os.getenv('RAG_MMR_LAMBDA', '0.7'))
        if self.bm25_full_scan_max_docs is None:
            self.bm25_full_scan_max_docs = int(os.getenv('RAG_BM25_FULL_SCAN_MAX_DOCS', '20000'))
        if self.candidate_k_mult is None:
            self.candidate_k_mult = int(os.getenv('RAG_CANDIDATE_K_MULT', '4'))
        if self.candidate_k_min is None:
            self.candidate_k_min = int(os.getenv('RAG_CANDIDATE_K_MIN', '20'))
        if self.candidate_k_max is None:
            self.candidate_k_max = int(os.getenv('RAG_CANDIDATE_K_MAX', '200'))

        if self.strict_mode is None:
            self.strict_mode = os.getenv('RAG_STRICT_MODE', 'true').lower() in {'1', 'true', 'yes'}

        if self.query_expand_enabled is None:
            self.query_expand_enabled = os.getenv('RAG_QUERY_EXPAND_ENABLED', 'false').lower() in {'1', 'true', 'yes'}

        if self.bm25_require_complete_corpus is None:
            self.bm25_require_complete_corpus = os.getenv('RAG_BM25_REQUIRE_COMPLETE_CORPUS', 'true').lower() in {'1', 'true', 'yes'}

        if self.prompt_show_scores is None:
            self.prompt_show_scores = os.getenv('RAG_PROMPT_SHOW_SCORES', 'false').lower() in {'1', 'true', 'yes'}

        if self.trace_enabled is None:
            self.trace_enabled = os.getenv('RAG_TRACE', 'false').lower() in {'1', 'true', 'yes'}

        if self.auto_ingest_on_startup is None:
            # Default to false: avoid slow/blocking cold starts and accidental ingestion in production.
            self.auto_ingest_on_startup = os.getenv('RAG_AUTO_INGEST_ON_STARTUP', 'false').lower() in {'1', 'true', 'yes'}

        if self.audit_log_enabled is None:
            self.audit_log_enabled = os.getenv('RAG_AUDIT_LOG_ENABLED', 'true').lower() in {'1', 'true', 'yes'}

        if self.disable_legacy_routes is None:
            self.disable_legacy_routes = os.getenv('RAG_DISABLE_LEGACY_ROUTES', 'false').lower() in {'1', 'true', 'yes'}

        self.legacy_routes_sunset_date = self.legacy_routes_sunset_date or os.getenv(
            'RAG_LEGACY_ROUTES_SUNSET_DATE',
            date(2026, 12, 31).isoformat(),
        )

        if os.getenv('RAG_API_KEY_REQUIRED') is not None:
            self.api_key_required = os.getenv('RAG_API_KEY_REQUIRED', 'false').lower() in {'1', 'true', 'yes'}
        elif self.api_key_required is None:
            self.api_key_required = bool(self.api_key)

        if os.getenv('RAG_ADMIN_API_KEY_REQUIRED') is not None:
            self.admin_api_key_required = os.getenv('RAG_ADMIN_API_KEY_REQUIRED', 'false').lower() in {'1', 'true', 'yes'}
        elif self.admin_api_key_required is None:
            # If admin key is configured, default to requiring it for admin endpoints.
            self.admin_api_key_required = bool(self.admin_api_key)

        if self.admin_api_key_fallback_to_api_key is None:
            # Safer default: when admin key check is disabled, do NOT implicitly fallback to regular API key.
            # Can be re-enabled explicitly for backward compatibility.
            self.admin_api_key_fallback_to_api_key = os.getenv(
                'RAG_ADMIN_API_KEY_FALLBACK_TO_API_KEY', 'false'
            ).lower() in {'1', 'true', 'yes'}

        if self.enforce_namespace_path_prefix is None:
            self.enforce_namespace_path_prefix = os.getenv('RAG_ENFORCE_NAMESPACE_PATH_PREFIX', 'false').lower() in {'1', 'true', 'yes'}

        if '*' in self.cors_allow_origins and self.cors_allow_credentials:
            self.cors_allow_credentials = False

        if os.getenv('RAG_DEBUG_CONFIG', '').lower() in {'1', 'true', 'yes'}:
            logger.info('OPENAI_API_KEY present: %s', bool(self.openai_api_key))

    # Backward-compatibility aliases used by older server routes.
    @property
    def rag_vec_weight(self) -> float:
        return float(self.vec_weight or 0.0)

    @rag_vec_weight.setter
    def rag_vec_weight(self, v: float) -> None:
        self.vec_weight = float(v)

    @property
    def rag_bm25_weight(self) -> float:
        return float(self.bm25_weight or 0.0)

    @rag_bm25_weight.setter
    def rag_bm25_weight(self, v: float) -> None:
        self.bm25_weight = float(v)

    @property
    def rag_bm25_enabled(self) -> bool:
        return bool(self.bm25_enabled)

    @rag_bm25_enabled.setter
    def rag_bm25_enabled(self, v: bool) -> None:
        self.bm25_enabled = bool(v)


def ensure_dirs(path: str) -> None:
    os.makedirs(path, exist_ok=True)
