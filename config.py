import os
from dataclasses import dataclass

# 尝试加载 .env 与 .env.local（若存在）
try:  # pragma: no cover
    from dotenv import load_dotenv

    # 优先加载 .env.local，其次加载 .env；不覆盖已存在的进程环境变量
    for _fname in (".env.local", ".env"):
        try:
            if os.path.exists(_fname):
                load_dotenv(dotenv_path=_fname, override=False)
        except Exception:
            pass
except Exception:
    # 若未安装 python-dotenv，跳过，不影响运行
    pass


@dataclass
class Settings:
    # 文档与索引目录
    docs_dir: str = os.getenv("RAG_DOCS_DIR", "data/docs")
    index_dir: str = os.getenv("RAG_INDEX_DIR", "data/index")

    # 向量化模型
    embedding_model_name: str = os.getenv("RAG_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

    # 大模型配置（OpenAI 兼容）
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    openai_base_url: str | None = os.getenv("OPENAI_BASE_URL")
    llm_model: str = os.getenv("RAG_MODEL", "deepseek-chat")

    # 检索参数
    top_k: int = int(os.getenv("RAG_TOP_K", "4"))

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

    # 向量后端：auto | milvus | faiss
    vector_backend: str = os.getenv("VECTOR_BACKEND", "auto").lower()

    # 检索质量参数
    bm25_enabled: bool = os.getenv("RAG_BM25_ENABLED", "true").lower() in {"1", "true", "yes"}
    bm25_weight: float = float(os.getenv("RAG_BM25_WEIGHT", "0.35"))
    vec_weight: float = float(os.getenv("RAG_VEC_WEIGHT", "0.65"))
    score_threshold: float = float(os.getenv("RAG_SCORE_THRESHOLD", "0.0"))  # 过滤过低分
    mmr_lambda: float = float(os.getenv("RAG_MMR_LAMBDA", "0.5"))  # 多样性权衡


def ensure_dirs(path: str) -> None:
    os.makedirs(path, exist_ok=True)


