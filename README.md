# DeepRAG - 企业级 RAG 知识库系统

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111+-green.svg)](https://fastapi.tiangolo.com/)
[![Vue](https://img.shields.io/badge/Vue-3.5+-brightgreen.svg)](https://vuejs.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

基于 FastAPI + Vue 3 的前后端分离 RAG（检索增强生成）系统，支持 Milvus/FAISS 向量存储、BM25 混合检索、Reranker 重排，以及多租户隔离。

## 核心特性

- **混合检索**：向量检索 + BM25 融合，MMR 多样性采样
- **智能重排**：可选 FlagEmbedding Reranker 提升相关性
- **查询改写**：多种策略优化检索效果
  - 查询扩展 - 生成同义词和相关表达
  - 查询分解 - 将复杂查询拆分为子查询
  - HyDE - 假设性文档嵌入
  - 智能分析 - 自动推荐最佳策略
- **灵活存储**：Milvus 云原生 / FAISS 本地，自动回退
- **多租户**：命名空间隔离 + API Key 鉴权
- **深度文档理解**：
  - 支持 `.txt`, `.md`, `.pdf`, `.docx`, `.xlsx`
  - PDF 表格自动提取和解析
  - OCR 图片文字识别（可选）
  - 智能分块保持语义边界
- **精确引用溯源**：记录页码信息，支持精确定位
- **多轮对话管理**：上下文感知的对话历史
- **现代前端**：Vue 3 + Vite，组件化架构，代码高亮，原生暗黑模式
- **可观测性**：结构化日志、健康检查、监控指标
- **易部署**：Docker Compose 一键启动
- **模型可选**：设置里卡片式选择 DeepSeek / Qwen 等兼容模型
- **联网搜索（可选）**：可将实时搜索结果并入上下文（Serper / DuckDuckGo）

## 架构

```
┌─────────────────┐         ┌──────────────────┐
│   Vue 3 前端     │ ------- │  FastAPI 后端     │
│  (Vite 5173)    │  HTTP   │   (uvicorn 8000) │
└─────────────────┘         └──────────┬───────┘
                                       │
                      ┌────────────────┼────────────────┐
                      │                │                │
                 ┌────▼────┐     ┌────▼────┐     ┌────▼────┐
                 │ Milvus  │     │ OpenAI  │     │ BM25    │
                 │ /FAISS  │     │  兼容   │     │ Rerank  │
                 └─────────┘     └─────────┘     └─────────┘
```

## 快速开始

### 环境要求

- Python 3.10+
- Node.js 20+ (前端)
- Conda (推荐，用于 FAISS)
- Tesseract OCR (可选，用于图片文字识别)

### 1. 安装依赖

```bash
# 后端（推荐使用 conda 环境）
conda create -n rag-env python=3.10 -y
conda activate rag-env
pip install -r backend/requirements.txt

# 可选：安装 Tesseract OCR（用于图片识别）
# Windows: 下载安装 https://github.com/UB-Mannheim/tesseract/wiki
# macOS: brew install tesseract tesseract-lang
# Linux: sudo apt-get install tesseract-ocr tesseract-ocr-chi-sim

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

# 可用模型（逗号分隔），前端将用来渲染模型卡片
AVAILABLE_MODELS=deepseek-chat,qwen-turbo,qwen-plus,qwen-max

# Qwen 兼容配置（可选，若使用阿里灵积）
QWEN_API_KEY=sk-your-qwen-key
QWEN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1

# 向量存储（auto 自动选择 Milvus/FAISS）
VECTOR_BACKEND=auto
MILVUS_HOST=127.0.0.1
MILVUS_PORT=19530

# 检索参数（推荐默认）
RAG_TOP_K=8
RAG_BM25_ENABLED=true
RAG_BM25_WEIGHT=0.4
RAG_VEC_WEIGHT=0.6
RAG_MMR_LAMBDA=0.7
RAG_RERANKER_ENABLED=false

# 联网搜索（可选）
RAG_WEB_SEARCH_ENABLED=false
SERPER_API_KEY=

# 多租户与鉴权
RAG_NAMESPACE=default
RAG_NAMESPACE_WHITELIST=default  # 允许的命名空间（逗号分隔，可选）
RAG_API_KEY_NAMESPACE=default  # API key 绑定命名空间（可选）
RAG_API_KEY=  # 可选，设置后需请求头 X-API-Key
RAG_API_KEY_REQUIRED=false  # 是否强制鉴权（true/false）

# CORS 配置
RAG_CORS_ALLOW_ORIGINS=http://localhost:5173,http://127.0.0.1:5173
RAG_CORS_ALLOW_CREDENTIALS=true
RAG_CORS_ALLOW_METHODS=*
RAG_CORS_ALLOW_HEADERS=*
```

### 4. 启动服务

```bash
# 后端（终端 1，从项目根目录运行）
python -m uvicorn backend.server:app --host 0.0.0.0 --port 8000 --reload

# 前端（终端 2）
cd frontend
npm run dev  # 访问 http://localhost:5173
```

## API 文档

启动后访问 `http://localhost:8000/docs` 查看 Swagger 交互式文档。

### 主要端点

**问答相关**
- `POST /ask` - 基础问答
- `POST /ask_stream` - 流式问答（SSE）
- `POST /ask_with_rewriting` - 使用查询改写增强检索
- `POST /analyze_query` - 分析查询并推荐策略
- `POST /explain_retrieval` - 解释检索结果（评分+高亮）
- `POST /advanced_search` - 高级检索（过滤+权重+聚合）
- `POST /optimize_weights` - 权重优化测试
- `GET /models` - 返回可用大模型列表

**文档管理**
- `POST /docs` - 上传文档（支持 PDF/Word/Excel）
- `DELETE /docs?path=xxx` - 删除文档
- `GET /docs/paths` - 列出已入库路径
- `GET /docs/preview?path=xxx` - 预览文档分块
- `POST /visualize_chunks` - 可视化文档分块详情
- `GET /export?path=xxx` - 导出文档分块
- `POST /documents/metadata` - 更新文档元数据（标签/分类/描述）
- `GET /documents/metadata/{path}` - 获取文档元数据
- `GET /documents/list` - 列出文档（支持按分类/标签过滤）
- `GET /documents/tags` - 获取所有标签
- `GET /documents/categories` - 获取所有分类
- `GET /documents/statistics` - 获取文档统计信息
- `POST /documents/{path}/tags` - 添加文档标签
- `DELETE /documents/{path}/tags` - 移除文档标签

**对话管理**
- `POST /conversations` - 创建新对话
- `GET /conversations` - 列出对话列表
- `GET /conversations/{id}` - 获取对话详情
- `DELETE /conversations/{id}` - 删除对话
- `POST /conversations/{id}/messages` - 添加消息

**系统管理**
- `POST /namespaces/create` - 创建命名空间
- `GET /healthz` - 健康检查
- `GET /cache/stats` - 缓存统计
- `GET /cache/analyze` - 缓存性能分析
- `POST /cache/prewarm` - 缓存预热
- `GET /cache/smart_stats` - 智能缓存统计
- `POST /cache/optimize` - 缓存配置优化

**性能监控**
- `GET /metrics/statistics` - 性能统计信息
- `GET /metrics/hot_queries` - 热门查询排行
- `GET /metrics/recent_requests` - 最近请求记录
- `GET /metrics/time_series` - 时间序列数据
- `POST /metrics/export` - 导出性能指标
- `POST /metrics/clear` - 清空监控数据

**检索优化**
- `POST /retrieval/analyze` - 分析检索质量并提供优化建议
- `POST /retrieval/suggest_weights` - 根据查询类型建议最佳权重
- `POST /retrieval/grid_search` - 网格搜索最佳权重组合
- `POST /retrieval/compare_strategies` - 比较不同检索策略效果

**查询意图识别**
- `POST /query/analyze_intent` - 分析查询意图并提供优化建议
- `POST /query/smart_search` - 基于意图识别的智能检索
- `POST /query/batch_analyze` - 批量分析查询意图

**评估测试**
- `POST /evaluation/run_benchmark` - 运行基准测试
- `POST /evaluation/test_retrieval` - 测试检索质量
- `POST /evaluation/test_answer` - 测试答案质量
- `POST /evaluation/save_test_cases` - 保存测试用例
- `GET /evaluation/load_test_cases` - 加载测试用例

**知识图谱**
- `POST /kg/build` - 构建知识图谱
- `GET /kg/statistics` - 获取图谱统计信息
- `GET /kg/entity/{name}` - 获取实体信息
- `GET /kg/subgraph/{name}` - 获取实体子图
- `GET /kg/search` - 搜索实体
- `GET /kg/path` - 查找实体路径
- `POST /kg/enhanced_search` - 图谱增强检索
- `POST /kg/export` - 导出知识图谱
- `POST /kg/import` - 导入知识图谱

## Docker 部署

### 快速开始

```bash
# 1. 进入部署目录
cd deploy

# 2. 复制环境变量模板
cp .env.example ../.env

# 3. 编辑 .env 文件，填入你的 API Key
vim ../.env

# 4. 一键启动（推荐）
./start.sh

# 或手动启动
docker-compose up -d
```

### 服务访问

- **前端界面**: http://localhost:5173
- **后端 API**: http://localhost:8000
- **API 文档**: http://localhost:8000/docs

### 常用命令

```bash
# 查看服务状态
docker-compose ps

# 查看日志
docker-compose logs -f backend

# 停止服务
./stop.sh
# 或
docker-compose down

# 重启服务
docker-compose restart
```

详细部署文档请查看 [deploy/README.md](deploy/README.md)

## 使用示例

### Python SDK

```python
import requests

response = requests.post(
    "http://localhost:8000/ask_stream",
    json={
        "question": "什么是 RAG？",
        "top_k": 8,
        "model": "deepseek-chat",          # 可选
        "system_prompt": "请用清晰小节回答：\n{context}\n问题：{question}",
        "web_enabled": True,                 # 可选
        "web_top_k": 3
    },
    headers={"X-API-Key": "your-key"},  # 可选
    stream=True
)

for line in response.iter_lines():
    if line.startswith(b"data: "):
        print(line.decode()[6:], end="")
```

### cURL

```bash
curl -X POST http://localhost:8000/ask_stream \
  -H "Content-Type: application/json" \
  -d '{
    "question": "介绍一下系统功能",
    "top_k": 8,
    "model": "qwen-plus",
    "system_prompt": "{context}\n\n请基于以上内容回答：{question}",
    "web_enabled": True,
    "web_top_k": 3
  }'
```

## 高级配置

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

## 贡献指南

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

## 开发路线图

- [ ] 多轮对话记忆与上下文管理
- [ ] 评测框架（nDCG、MRR、Hit@K）
- [ ] 知识图谱增强
- [ ] 多模态支持（图片、表格）
- [ ] Web UI 响应式适配
- [ ] K8s Helm Chart

## 许可证

本项目采用 [MIT License](LICENSE) 开源。

## 致谢

- [FastAPI](https://fastapi.tiangolo.com/) - 现代高性能 Web 框架
- [Milvus](https://milvus.io/) - 云原生向量数据库
- [Sentence Transformers](https://www.sbert.net/) - 语义向量模型
- [Vue.js](https://vuejs.org/) - 渐进式前端框架

---

**Star 本项目以支持开发者持续维护！**

## 注意事项

- 不提交或上传以下文件：AGENTS.md。
- 如需启用鉴权：设置 RAG_API_KEY_REQUIRED=true 并配置 RAG_API_KEY，客户端需在请求头携带 X-API-Key。
- CORS 规则由 RAG_CORS_* 控制；当 RAG_CORS_ALLOW_ORIGINS 包含 * 时，将自动关闭 allow_credentials。
- 运行/调试前先复制 .env.example 为 .env 并填写必要参数。
- 避免手工修改生成目录（如 data/index）与依赖目录（如 frontend/node_modules）。
- 修改功能后请执行对应测试（后端 pytest，前端如有 npm test / npm run lint）。
