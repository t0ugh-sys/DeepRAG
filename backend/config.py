import os
from dataclasses import dataclass

# 尝试加载 .env 与 .env.local（若存在）
def _load_env_files():
    """在导入时立即加载环境变量"""
    try:
        from dotenv import load_dotenv
        
        # 获取项目根目录（backend 的父目录）
        _project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # 优先加载 .env.local，其次加载 .env；覆盖已存在的进程环境变量
        for _fname in (".env.local", ".env"):
            _env_path = os.path.join(_project_root, _fname)
            try:
                if os.path.exists(_env_path):
                    load_dotenv(dotenv_path=_env_path, override=True)
                    print(f"✓ 加载环境变量文件: {_env_path}")
                    # 调试：立即检查是否加载成功
                    api_key = os.getenv("OPENAI_API_KEY")
                    if api_key:
                        print(f"  → OPENAI_API_KEY 已加载 (长度: {len(api_key)})")
                    else:
                        print(f"  → OPENAI_API_KEY 仍然为空！")
            except Exception as e:
                print(f"✗ 加载环境变量文件失败: {_env_path}, {e}")
    except Exception as e:
        # 若未安装 python-dotenv，跳过，不影响运行
        print(f"✗ python-dotenv 未安装: {e}")

# 立即执行加载
_load_env_files()


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
    
    # Qwen API 配置
    qwen_api_key: str | None = os.getenv("QWEN_API_KEY")
    qwen_base_url: str | None = os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    
    # 可用模型列表
    available_models: str = os.getenv("AVAILABLE_MODELS", "deepseek-chat,qwen-turbo,qwen-plus,qwen-max")

    # 检索参数
    top_k: int = int(os.getenv("RAG_TOP_K", "8"))  # 增加默认检索数量

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
    bm25_weight: float = float(os.getenv("RAG_BM25_WEIGHT", "0.4"))  # 提高 BM25 权重
    vec_weight: float = float(os.getenv("RAG_VEC_WEIGHT", "0.6"))
    score_threshold: float = float(os.getenv("RAG_SCORE_THRESHOLD", "0.0"))  # 不过滤低分，增加召回
    mmr_lambda: float = float(os.getenv("RAG_MMR_LAMBDA", "0.7"))  # 降低多样性，增加相关性
    
    # 严格模式：True=仅基于知识库回答，False=允许模型自由发挥
    strict_mode: bool = os.getenv("RAG_STRICT_MODE", "true").lower() in {"1", "true", "yes"}
    
    def __post_init__(self):
        # 调试：打印 API key 是否存在
        if self.openai_api_key:
            print(f"✓ OPENAI_API_KEY loaded (length: {len(self.openai_api_key)})")
        else:
            print("✗ OPENAI_API_KEY not found in environment!")


def ensure_dirs(path: str) -> None:
    os.makedirs(path, exist_ok=True)


