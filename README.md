# 🚀 DeepRAG - 企业级 RAG 知识库系统

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111+-green.svg)](https://fastapi.tiangolo.com/)
[![Vue](https://img.shields.io/badge/Vue-3.5+-brightgreen.svg)](https://vuejs.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

基于 FastAPI + Vue 3 的前后端分离 RAG（检索增强生成）系统，支持 Milvus/FAISS 向量存储、BM25 混合检索、Reranker 重排，以及多租户隔离。

## ✨ 核心特性

- 🔍 **混合检索**：向量检索 + BM25 融合，MMR 多样性采样
- 🎯 **智能重排**：可选 FlagEmbedding Reranker 提升相关性
- 💾 **灵活存储**：Milvus 云原生 / FAISS 本地，自动回退
- 🔐 **多租户**：命名空间隔离 + API Key 鉴权
- 📄 **文档解析**：支持 `.txt`, `.md`, `.pdf`，智能分块
- 🌐 **现代前端**：Vue 3 + Vite，组件化架构，代码高亮
- 📊 **可观测性**：结构化日志、健康检查、监控指标
- 🐳 **易部署**：Docker Compose 一键启动

## 🏗️ 架构

```
┌─────────────────┐         ┌──────────────────┐
│   Vue 3 前端    │◄───────►│  FastAPI 后端    │
│  (Vite 5173)   │  HTTP   │   (uvicorn 8000) │
└─────────────────┘         └──────────┬───────┘
                                       │
                      ┌────────────────┼────────────────┐
                      │                │                │
                 ┌────▼────┐     ┌────▼────┐     ┌────▼────┐
                 │ Milvus  │     │ OpenAI  │     │ BM25    │
                 │ /FAISS  │     │  兼容   │     │ Rerank  │
                 └─────────┘     └─────────┘     └─────────┘
```

## 🚀 快速开始

### 环境要求

- Python 3.10+
- Node.js 20+ (前端)
- Conda (推荐，用于 FAISS)

### 1. 安装依赖
   
   ```bash
# 后端（推荐使用 conda 环境）
conda create -n rag-env python=3.10 -y
conda activate rag-env
   pip install -r requirements.txt
conda install -c conda-forge faiss-cpu rank-bm25 -y

# 前端
cd frontend
npm install
```

### 2. 配置环境变量

在项目根目录创建 `.env` 文件：

```bash
# 大模型配置（OpenAI 兼容）
OPENAI_API_KEY=sk-your-key
OPENAI_BASE_URL=https://api.deepseek.com/v1  # 或其他兼容服务
RAG_MODEL=deepseek-chat

# 向量存储（auto 自动选择 Milvus/FAISS）
VECTOR_BACKEND=auto
MILVUS_HOST=127.0.0.1
MILVUS_PORT=19530

# 检索参数
RAG_TOP_K=4
RAG_BM25_ENABLED=true
RAG_RERANKER_ENABLED=false

# 多租户与鉴权
RAG_NAMESPACE=default
RAG_API_KEY=  # 可选，设置后需请求头 X-API-Key
```

### 3. 准备文档并构建索引

```bash
# 将文档放入 data/docs/ 目录
mkdir -p data/docs
echo "你的知识库内容" > data/docs/sample.txt

# 构建索引
python ingest.py --docs_dir data/docs --index_dir data/index
```

### 4. 启动服务

```bash
# 后端（终端 1）
uvicorn server:app --host 0.0.0.0 --port 8000 --reload

# 前端（终端 2）
cd frontend
npm run dev  # 访问 http://localhost:5173
```

## 📚 API 文档

启动后访问 `http://localhost:8000/docs` 查看 Swagger 交互式文档。

### 主要端点

- `POST /ask_stream` - 流式问答（SSE）
- `POST /docs` - 上传文档
- `DELETE /docs?path=xxx` - 删除文档
- `GET /docs/paths` - 列出已入库路径
- `POST /namespaces/create` - 创建命名空间
- `GET /healthz` - 健康检查

## 🐳 Docker 部署

```bash
# 构建并启动（包含 Milvus）
docker-compose up -d

# 仅后端
docker build -t rag-backend .
docker run -p 8000:8000 --env-file .env rag-backend
```

## 📖 使用示例

### Python SDK

```python
import requests

response = requests.post(
    "http://localhost:8000/ask_stream",
    json={"question": "什么是 RAG？", "top_k": 4},
    headers={"X-API-Key": "your-key"},  # 可选
    stream=True
)

for line in response.iter_lines():
    if line.startswith(b"data: "):
        print(line.decode()[6:], end="")
```

### cURL

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "介绍一下系统功能", "top_k": 4}'
```

## 🛠️ 高级配置

### BM25 混合检索

```bash
RAG_BM25_ENABLED=true
RAG_BM25_WEIGHT=0.35
RAG_VEC_WEIGHT=0.65
RAG_MMR_LAMBDA=0.5  # 多样性权衡
```

### Reranker 重排

```bash
RAG_RERANKER_ENABLED=true
RAG_RERANKER_MODEL=BAAI/bge-reranker-base
RAG_RERANKER_TOP_N=4
```

### Milvus 云端部署

```bash
VECTOR_BACKEND=milvus
MILVUS_HOST=your-milvus.cloud
MILVUS_PORT=19530
MILVUS_USER=xxx
MILVUS_PASSWORD=xxx
MILVUS_SECURE=true
```

## 🤝 贡献指南

欢迎贡献！请先 Fork 本仓库，然后：

1. 创建特性分支：`git checkout -b feature/amazing-feature`
2. 提交改动：`git commit -m 'feat: add amazing feature'`
3. 推送分支：`git push origin feature/amazing-feature`
4. 提交 Pull Request

### 分支规范

- `main` - 生产稳定版本
- `dev` - 开发集成分支
- `feature/*` - 新功能开发
- `hotfix/*` - 紧急修复

## 📝 开发路线图

- [ ] 多轮对话记忆与上下文管理
- [ ] 评测框架（nDCG、MRR、Hit@K）
- [ ] 知识图谱增强
- [ ] 多模态支持（图片、表格）
- [ ] Web UI 响应式适配
- [ ] K8s Helm Chart

## 📄 许可证

本项目采用 [MIT License](LICENSE) 开源。

## 🙏 致谢

- [FastAPI](https://fastapi.tiangolo.com/) - 现代高性能 Web 框架
- [Milvus](https://milvus.io/) - 云原生向量数据库
- [Sentence Transformers](https://www.sbert.net/) - 语义向量模型
- [Vue.js](https://vuejs.org/) - 渐进式前端框架

---

**Star ⭐ 本项目以支持开发者持续维护！**
