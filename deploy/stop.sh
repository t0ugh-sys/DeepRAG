#!/bin/bash

# DeepRAG 停止脚本

set -e

echo "🛑 停止 DeepRAG 服务"
echo "===================="

# 检查是否有运行的容器
if [ -z "$(docker-compose ps -q)" ]; then
    echo "ℹ️  没有运行中的服务"
    exit 0
fi

# 询问是否删除数据
echo ""
read -p "是否删除数据卷? (y/n) " -n 1 -r
echo

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "⚠️  警告: 这将删除所有向量数据和 Milvus 数据"
    read -p "确定要继续吗? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "🗑️  停止服务并删除数据卷..."
        docker-compose down -v
        echo "✅ 服务已停止，数据卷已删除"
    else
        echo "🛑 停止服务（保留数据）..."
        docker-compose down
        echo "✅ 服务已停止，数据已保留"
    fi
else
    echo "🛑 停止服务（保留数据）..."
    docker-compose down
    echo "✅ 服务已停止，数据已保留"
fi

echo ""
echo "📝 提示:"
echo "   重新启动: ./start.sh"
echo "   查看数据卷: docker volume ls | grep deeprag"
echo ""
