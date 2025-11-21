#!/bin/bash

# DeepRAG 快速启动脚本

set -e

echo "🚀 DeepRAG 部署脚本"
echo "===================="

# 检查 Docker 是否安装
if ! command -v docker &> /dev/null; then
    echo "❌ 错误: Docker 未安装"
    echo "请先安装 Docker: https://docs.docker.com/get-docker/"
    exit 1
fi

# 检查 Docker Compose 是否安装
if ! command -v docker-compose &> /dev/null; then
    echo "❌ 错误: Docker Compose 未安装"
    echo "请先安装 Docker Compose: https://docs.docker.com/compose/install/"
    exit 1
fi

# 检查 .env 文件
if [ ! -f "../.env" ]; then
    echo "⚠️  警告: 未找到 .env 文件"
    echo "正在从模板创建..."
    cp .env.example ../.env
    echo "✅ 已创建 .env 文件"
    echo ""
    echo "⚠️  请编辑 ../.env 文件，填入你的 API Key"
    echo "   OPENAI_API_KEY=your_api_key_here"
    echo ""
    read -p "是否已配置 API Key? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "请先配置 API Key 后再运行此脚本"
        exit 1
    fi
fi

# 创建数据目录
echo "📁 创建数据目录..."
mkdir -p ../data/docs ../data/index ../data/conversations
echo "✅ 数据目录创建完成"

# 拉取镜像
echo ""
echo "📦 拉取 Docker 镜像..."
docker-compose pull

# 构建自定义镜像
echo ""
echo "🔨 构建应用镜像..."
docker-compose build

# 启动服务
echo ""
echo "🚀 启动服务..."
docker-compose up -d

# 等待服务启动
echo ""
echo "⏳ 等待服务启动..."
sleep 10

# 检查服务状态
echo ""
echo "📊 服务状态:"
docker-compose ps

# 健康检查
echo ""
echo "🏥 健康检查:"

# 检查后端
if curl -f http://localhost:8000/healthz &> /dev/null; then
    echo "✅ 后端服务正常"
else
    echo "❌ 后端服务异常"
fi

# 检查前端
if curl -f http://localhost:5173 &> /dev/null; then
    echo "✅ 前端服务正常"
else
    echo "⚠️  前端服务可能需要更多时间启动"
fi

# 显示访问信息
echo ""
echo "===================="
echo "✅ DeepRAG 部署完成!"
echo "===================="
echo ""
echo "📍 访问地址:"
echo "   前端界面: http://localhost:5173"
echo "   后端 API: http://localhost:8000"
echo "   API 文档: http://localhost:8000/docs"
echo ""
echo "📝 常用命令:"
echo "   查看日志: docker-compose logs -f"
echo "   停止服务: docker-compose down"
echo "   重启服务: docker-compose restart"
echo ""
echo "💡 提示: 首次启动可能需要下载模型，请耐心等待"
echo ""
