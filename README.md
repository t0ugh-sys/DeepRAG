## 本地 RAG 知识库 + 大模型 API 调用（FastAPI）

### 功能

- 文档入库（ingest）：支持 `.txt`、`.md`、`.pdf`，自动切分、向量化、默认建立 FAISS 索引（如配置了 Milvus 则优先写入 Milvus）
- 检索增强生成（RAG）：相似检索 + 提示词拼接
- 大模型调用：支持 OpenAI 兼容接口（可自定义 `base_url` 和 `api_key`）
- FastAPI 服务：`POST /ask` 提问，返回答案与引用来源

### 目录结构

```
.
├── requirements.txt
├── README.md
├── config.py
├── ingest.py
├── rag.py
├── server.py
└── web/
    ├── index.html
    ├── app.js
    └── styles.css
```

### 环境准备

1) 安装依赖（建议 Python 3.10+）
   
   ```bash
   pip install -r requirements.txt
   ```

2) 准备环境变量（Windows PowerShell 示例）：
   
   ```powershell
   $env:OPENAI_API_KEY = "YOUR_API_KEY"
   # 如使用 OpenAI 兼容服务（如本地服务/第三方），可设置自定义 base_url
   # $env:OPENAI_BASE_URL = "https://your.base.url/v1"
   # 指定模型名
   $env:RAG_MODEL = "gpt-4o-mini"
   ```

也可以在项目根目录新建 `.env` 或 `.env.local` 文件（不覆盖系统环境变量），例如：

```
OPENAI_API_KEY=sk-xxx
OPENAI_BASE_URL=http://localhost:11434/v1
RAG_MODEL=qwen2.5:14b-instruct
RAG_TOP_K=4
VECTOR_BACKEND=auto   # auto|milvus|faiss
MILVUS_HOST=127.0.0.1
MILVUS_PORT=19530
MILVUS_COLLECTION=rag_chunks
RAG_NAMESPACE=default
RAG_API_KEY=           # 若设置则服务端要求请求头 X-API-Key
```

3) 准备待入库文档：将文件放入 `data/docs` 目录（不存在会自动创建）。支持 `.txt`、`.md`、`.pdf`。

### 文档入库（构建索引）

```bash
python ingest.py --docs_dir data/docs --index_dir data/index --chunk_size 800 --chunk_overlap 120
```

运行完成后，会在 `data/index` 下生成 `faiss.index` 和 `meta.jsonl`。若已配置 Milvus，向量会优先写入 Milvus 集合：`{MILVUS_COLLECTION}_{RAG_NAMESPACE}`。

### 启动服务

```bash
uvicorn server:app --host 0.0.0.0 --port 8000
```

### 提问示例

```bash
curl -X POST http://127.0.0.1:8000/ask ^
  -H "Content-Type: application/json" ^
  -d "{\"question\": \"请总结知识库的要点\", \"top_k\": 4}"
```

响应字段：

- `answer`：模型回答
- `sources`：引用的文档切片与相似度分数

### 使用自定义/本地 OpenAI 兼容服务

设置环境变量：

```powershell
$env:OPENAI_API_KEY = "sk-xxx"
$env:OPENAI_BASE_URL = "http://localhost:11434/v1"  # 示例
$env:RAG_MODEL = "qwen2.5:14b-instruct"             # 示例
```

### Milvus 使用说明（可选）

1) 安装并启动 Milvus（standalone）：参考官方文档。

2) 在 `.env` 中设置：

```
VECTOR_BACKEND=milvus
MILVUS_HOST=127.0.0.1
MILVUS_PORT=19530
MILVUS_COLLECTION=rag_chunks
RAG_NAMESPACE=default
```

3) 构建索引时会自动创建/覆盖集合并写入分片；服务端检索时若 Milvus 连接失败会自动回退到本地 FAISS。

### 常见问题

- 若 `faiss` 安装失败，优先尝试 `faiss-cpu` 对应的 Python 版本；或在 Conda 环境中安装。
- PDF 解析可能较慢或对复杂排版不完美，可优先提供 `.txt`/`.md`。
- 首次运行 `sentence-transformers` 会下载模型，需联网或预下载至缓存。
