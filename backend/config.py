import os
from dataclasses import dataclass

# 尝试加载 .env �?.env.local（若存在�?
def _load_env_files():
    """在导入时立即加载环境变量"""
    try:
        from dotenv import load_dotenv
        
        # 获取项目根目录（backend 的父目录�?
        _project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # 优先加载 .env.local，其次加�?.env；覆盖已存在的进程环境变�?
        for _fname in (".env.local", ".env"):
            _env_path = os.path.join(_project_root, _fname)
            try:
                if os.path.exists(_env_path):
                    load_dotenv(dotenv_path=_env_path, override=True)
                    print(f"[OK] Loaded env file: {_env_path}")
                    # 调试：立即检查是否加载成�?
                    api_key = os.getenv("OPENAI_API_KEY")
                    if api_key:
                        print(f"  -> OPENAI_API_KEY loaded (length: {len(api_key)})")
                    else:
                        print(f"  -> OPENAI_API_KEY still empty!")
            except Exception as e:
                print(f"[ERROR] Failed to load env file: {_env_path}, {e}")
    except Exception as e:
        # 若未安装 python-dotenv，跳过，不影响运�?
        print(f"[ERROR] python-dotenv not installed: {e}")

# 立即执行加载
_load_env_files()


@dataclass
class Settings:
    # 文档与索引目�?
    docs_dir: str = os.getenv("RAG_DOCS_DIR", "data/docs")
    index_dir: str = os.getenv("RAG_INDEX_DIR", "data/index")

    # 向量化模�?
    embedding_model_name: str = os.getenv("RAG_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

    # 大模型配置（OpenAI 兼容�?
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    openai_base_url: str | None = os.getenv("OPENAI_BASE_URL")
    llm_model: str = os.getenv("RAG_MODEL", "deepseek-chat")
    
    # Qwen API 配置
    qwen_api_key: str | None = os.getenv("QWEN_API_KEY")
    qwen_base_url: str | None = os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    
    # 可用模型列表
    available_models: str = os.getenv("AVAILABLE_MODELS", "deepseek-chat,qwen-turbo,qwen-plus,qwen-max")

    # 检索参�?
    top_k: int = int(os.getenv("RAG_TOP_K", "8"))  # 增加默认检索数�?

    # Reranker 配置
    reranker_enabled: bool = os.getenv("RAG_RERANKER_ENABLED", "false").lower() in {"1", "true", "yes"}
    reranker_model_name: str = os.getenv("RAG_RERANKER_MODEL", "BAAI/bge-reranker-base")
    reranker_top_n: int = int(os.getenv("RAG_RERANKER_TOP_N", "4"))

    # Milvus 配置
    milvus_host: str = os.getenv("MILVUS_HOST", "127.0.0.1")
    milvus_port: str = os.getenv("MILVUS_PORT", "19530")
    milvus_user: str | None = os.getenv("MILVUS_USER")
    milvus_password: str | None = os.getenv("MILVUS_PASSWORD")
    milvus_secure: bool = os.getenv("MILVUS_SECURE", "false").lower() in {"1", "true", "yes"}
    milvus_db: str | None = os.getenv("MILVUS_DB", None)
    milvus_collection: str = os.getenv("MILVUS_COLLECTION", "rag_chunks")

    # 多租户命名空间与鉴权
    default_namespace: str = os.getenv("RAG_NAMESPACE", "default")
    api_key: str | None = os.getenv("RAG_API_KEY")
    api_key_required: bool = False

    # CORS configuration
    cors_allow_origins: str = os.getenv("RAG_CORS_ALLOW_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173")
    cors_allow_credentials: bool = os.getenv("RAG_CORS_ALLOW_CREDENTIALS", "true").lower() in {"1", "true", "yes"}
    cors_allow_methods: str = os.getenv("RAG_CORS_ALLOW_METHODS", "*")
    cors_allow_headers: str = os.getenv("RAG_CORS_ALLOW_HEADERS", "*")
    env: str = os.getenv("RAG_ENV", "dev")
    # 向量后端：auto | milvus | faiss
    vector_backend: str = os.getenv("VECTOR_BACKEND", "auto").lower()

    # 检索质量参�?
    bm25_enabled: bool = os.getenv("RAG_BM25_ENABLED", "true").lower() in {"1", "true", "yes"}
    bm25_weight: float = float(os.getenv("RAG_BM25_WEIGHT", "0.4"))  # 提高 BM25 权重
    vec_weight: float = float(os.getenv("RAG_VEC_WEIGHT", "0.6"))
    score_threshold: float = float(os.getenv("RAG_SCORE_THRESHOLD", "0.0"))  # 不过滤低分，增加召回
    mmr_lambda: float = float(os.getenv("RAG_MMR_LAMBDA", "0.7"))  # 降低多样性，增加相关�?
    
    # 严格模式：True=仅基于知识库回答，False=允许模型自由发挥
    strict_mode: bool = os.getenv("RAG_STRICT_MODE", "true").lower() in {"1", "true", "yes"}
    
    def __post_init__(self):
        if os.getenv("RAG_API_KEY_REQUIRED") is not None:
            self.api_key_required = os.getenv("RAG_API_KEY_REQUIRED", "false").lower() in {"1", "true", "yes"}
        else:
            self.api_key_required = bool(self.api_key)

        if "*" in self.cors_allow_origins and self.cors_allow_credentials:
            self.cors_allow_credentials = False

        # debug: show key presence
        if self.openai_api_key:
            print(f"[OK] OPENAI_API_KEY loaded (length: {len(self.openai_api_key)})")
        else:
            print("[ERROR] OPENAI_API_KEY not found in environment!")


def ensure_dirs(path: str) -> None:
    os.makedirs(path, exist_ok=True)



