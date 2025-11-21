# 🐳 DeepRAG Docker 部署指南

本目录包含 DeepRAG 项目的完整 Docker 部署配置。

---

## 📁 目录结构

```
deploy/
├── docker-compose.yml      # Docker Compose 配置
├── Dockerfile.backend      # 后端 Dockerfile
├── Dockerfile.frontend     # 前端 Dockerfile
├── nginx.conf             # Nginx 配置
├── .env.example           # 环境变量示例
└── README.md              # 本文件
```

---

## 🚀 快速开始

### 1. 准备环境变量

```bash
# 复制环境变量模板
cp .env.example ../.env

# 编辑 .env 文件，填入你的 API Key
vim ../.env
```

**必须配置的环境变量**：
- `OPENAI_API_KEY` - DeepSeek API Key
- `OPENAI_BASE_URL` - DeepSeek API 地址

### 2. 启动所有服务

```bash
# 在 deploy 目录下执行
docker-compose up -d
```

这将启动以下服务：
- **Milvus** - 向量数据库（端口 19530）
- **etcd** - Milvus 依赖
- **MinIO** - Milvus 存储
- **Backend** - RAG 后端服务（端口 8000）
- **Frontend** - Vue 前端（端口 5173）

### 3. 验证服务

```bash
# 检查服务状态
docker-compose ps

# 查看后端日志
docker-compose logs -f backend

# 查看前端日志
docker-compose logs -f frontend
```

### 4. 访问应用

- **前端界面**: http://localhost:5173
- **后端 API**: http://localhost:8000
- **API 文档**: http://localhost:8000/docs
- **健康检查**: http://localhost:8000/healthz

---

## 📦 服务说明

### Backend 服务

**镜像**: 基于 Python 3.10
**端口**: 8000
**功能**:
- FastAPI 后端服务
- 文档处理和向量化
- RAG 检索和生成
- 多轮对话管理

**健康检查**:
```bash
curl http://localhost:8000/healthz
```

### Frontend 服务

**镜像**: 基于 Nginx + Node.js
**端口**: 5173 (映射到容器内的 80)
**功能**:
- Vue 3 前端界面
- 对话管理
- 文档上传
- 设置管理

### Milvus 服务

**镜像**: milvusdb/milvus:v2.3.3
**端口**: 19530 (gRPC), 9091 (HTTP)
**功能**:
- 向量存储和检索
- 高性能相似度搜索

---

## 🔧 常用命令

### 启动服务

```bash
# 启动所有服务
docker-compose up -d

# 启动指定服务
docker-compose up -d backend

# 查看日志
docker-compose logs -f

# 查看特定服务日志
docker-compose logs -f backend
```

### 停止服务

```bash
# 停止所有服务
docker-compose down

# 停止并删除数据卷（谨慎使用）
docker-compose down -v
```

### 重启服务

```bash
# 重启所有服务
docker-compose restart

# 重启指定服务
docker-compose restart backend
```

### 重新构建

```bash
# 重新构建所有镜像
docker-compose build

# 重新构建指定服务
docker-compose build backend

# 重新构建并启动
docker-compose up -d --build
```

### 查看状态

```bash
# 查看服务状态
docker-compose ps

# 查看资源使用
docker stats

# 进入容器
docker-compose exec backend bash
docker-compose exec frontend sh
```

---

## 📊 数据持久化

数据卷配置：

```yaml
volumes:
  etcd_data:      # etcd 数据
  minio_data:     # MinIO 对象存储
  milvus_data:    # Milvus 向量数据
```

宿主机挂载：

```yaml
volumes:
  - ../data:/app/data  # 文档和索引数据
```

**备份数据**：

```bash
# 备份数据目录
tar -czf deeprag-data-backup.tar.gz ../data

# 备份 Docker 数据卷
docker run --rm -v deeprag_milvus_data:/data -v $(pwd):/backup \
  alpine tar -czf /backup/milvus-backup.tar.gz /data
```

---

## 🔐 安全配置

### 1. API Key 鉴权

在 `.env` 中设置：

```bash
API_KEY=your_secure_api_key_here
```

### 2. Nginx 安全头

已在 `nginx.conf` 中配置：
- X-Frame-Options
- X-Content-Type-Options
- X-XSS-Protection

### 3. 网络隔离

所有服务运行在独立的 Docker 网络 `deeprag-network` 中。

---

## 🐛 故障排查

### 后端无法连接 Milvus

**症状**: 后端日志显示 "Failed to connect to Milvus"

**解决方案**:
```bash
# 检查 Milvus 是否健康
docker-compose ps milvus

# 查看 Milvus 日志
docker-compose logs milvus

# 重启 Milvus
docker-compose restart milvus
```

### 前端无法访问后端 API

**症状**: 前端显示网络错误

**解决方案**:
```bash
# 检查 nginx 配置
docker-compose exec frontend cat /etc/nginx/conf.d/default.conf

# 检查后端健康
curl http://localhost:8000/healthz

# 查看 nginx 日志
docker-compose logs frontend
```

### 容器频繁重启

**症状**: `docker-compose ps` 显示服务不断重启

**解决方案**:
```bash
# 查看详细日志
docker-compose logs --tail=100 backend

# 检查资源使用
docker stats

# 检查健康检查配置
docker inspect deeprag-backend | grep -A 10 Healthcheck
```

### 数据丢失

**症状**: 重启后文档和索引消失

**解决方案**:
```bash
# 检查数据卷
docker volume ls | grep deeprag

# 检查挂载
docker-compose exec backend ls -la /app/data

# 确保使用了持久化卷
docker-compose down  # 不要加 -v 参数
```

---

## 🔄 更新部署

### 更新代码

```bash
# 1. 拉取最新代码
cd /path/to/DeepRAG
git pull origin main

# 2. 重新构建镜像
cd deploy
docker-compose build

# 3. 重启服务
docker-compose up -d
```

### 更新依赖

```bash
# 1. 修改 requirements.txt 或 package.json

# 2. 重新构建
docker-compose build --no-cache backend
docker-compose build --no-cache frontend

# 3. 重启
docker-compose up -d
```

---

## 📈 性能优化

### 1. 资源限制

在 `docker-compose.yml` 中添加：

```yaml
services:
  backend:
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
        reservations:
          cpus: '1'
          memory: 2G
```

### 2. 缓存优化

```bash
# 使用 BuildKit 加速构建
DOCKER_BUILDKIT=1 docker-compose build
```

### 3. 网络优化

```yaml
networks:
  deeprag-network:
    driver: bridge
    driver_opts:
      com.docker.network.driver.mtu: 1500
```

---

## 🌐 生产环境部署

### 1. 使用 HTTPS

```bash
# 安装 Certbot
apt-get install certbot python3-certbot-nginx

# 获取证书
certbot --nginx -d your-domain.com

# 更新 nginx.conf 添加 SSL 配置
```

### 2. 使用域名

修改 `nginx.conf`:

```nginx
server {
    listen 80;
    server_name your-domain.com;
    # ... 其他配置
}
```

### 3. 配置反向代理

如果使用外部反向代理（如 Nginx/Traefik），移除端口映射：

```yaml
services:
  backend:
    # ports:
    #   - "8000:8000"
    expose:
      - "8000"
```

---

## 📝 环境变量说明

| 变量名 | 说明 | 默认值 | 必需 |
|--------|------|--------|------|
| `OPENAI_API_KEY` | DeepSeek API Key | - | ✅ |
| `OPENAI_BASE_URL` | API 地址 | https://api.deepseek.com | ✅ |
| `VECTOR_BACKEND` | 向量后端 | milvus | ❌ |
| `MILVUS_HOST` | Milvus 地址 | milvus | ❌ |
| `EMBEDDING_MODEL_NAME` | Embedding 模型 | BAAI/bge-small-zh-v1.5 | ❌ |
| `LLM_MODEL` | LLM 模型 | deepseek-chat | ❌ |
| `TOP_K` | 检索数量 | 5 | ❌ |
| `STRICT_MODE` | 严格模式 | true | ❌ |

---

## 📞 支持

遇到问题？

1. 查看 [故障排查](#-故障排查) 部分
2. 查看服务日志: `docker-compose logs`
3. 提交 Issue: https://github.com/t0ugh-sys/DeepRAG/issues

---

## 📄 许可证

MIT License - 详见项目根目录 LICENSE 文件
