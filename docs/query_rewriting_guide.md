# 🔄 查询改写功能使用指南

## 📖 简介

查询改写（Query Rewriting）是一种高级 RAG 优化技术，通过改写用户查询来提升检索质量。DeepRAG 支持多种改写策略，可以显著提高答案的准确性和相关性。

---

## 🎯 为什么需要查询改写？

### 常见问题

1. **用户查询不够精确**
   - 用户: "RAG是什么"
   - 问题: 太简短，可能遗漏相关信息

2. **查询包含多个问题**
   - 用户: "RAG和传统搜索的区别是什么，各有什么优缺点"
   - 问题: 需要分解为多个子查询

3. **查询与文档表达方式不匹配**
   - 用户: "如何提升性能"
   - 文档: "性能优化方法"、"加速技巧"
   - 问题: 词汇不匹配导致召回率低

---

## 🚀 支持的改写策略

### 1. 查询扩展（Expand）

**适用场景**: 简短查询、需要提高召回率

**工作原理**: 生成 3-5 个语义相似但表达不同的查询变体

**示例**:
```
原查询: "RAG是什么"

扩展后:
1. RAG是什么
2. 什么是RAG技术
3. RAG的定义和概念
4. 检索增强生成介绍
5. RAG系统的工作原理
```

**使用方法**:
```bash
curl -X POST "http://localhost:8000/ask_with_rewriting" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "RAG是什么",
    "strategy": "expand"
  }'
```

---

### 2. 查询分解（Decompose）

**适用场景**: 复杂查询、包含多个问题

**工作原理**: 将复杂查询拆分为 2-4 个简单的子查询

**示例**:
```
原查询: "RAG和传统搜索的区别是什么，各有什么优缺点"

分解后:
1. 什么是RAG
2. 什么是传统搜索
3. RAG和传统搜索的区别
4. RAG的优缺点
```

**使用方法**:
```bash
curl -X POST "http://localhost:8000/ask_with_rewriting" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "RAG和传统搜索的区别是什么，各有什么优缺点",
    "strategy": "decompose"
  }'
```

---

### 3. HyDE（假设性文档嵌入）

**适用场景**: 需要深度理解的查询、技术问题

**工作原理**: 生成一个假设性的答案段落，用这个答案去检索（而不是用原查询）

**示例**:
```
原查询: "如何优化RAG性能"

生成假设答案:
"优化RAG性能可以从多个方面入手。首先，改进文档分块策略，
使用语义分块而非固定长度分块，保持语义完整性。其次，选择
更好的embedding模型，如BGE、GTE等中文优化模型。第三，
使用混合检索，结合向量检索和BM25关键词检索。第四，添加
Reranker重排模型提升相关性。最后，优化提示词工程，让LLM
更好地理解上下文..."

然后用这段假设答案去检索相关文档
```

**使用方法**:
```bash
curl -X POST "http://localhost:8000/ask_with_rewriting" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "如何优化RAG性能",
    "strategy": "hyde"
  }'
```

---

### 4. 组合策略（Multi）

**适用场景**: 复杂场景、需要最佳效果

**工作原理**: 先分解查询，再对每个子查询扩展，综合检索

**示例**:
```
原查询: "RAG系统如何处理多语言文档"

步骤1 - 分解:
1. RAG系统的文档处理流程
2. 多语言文档的挑战

步骤2 - 对每个子查询扩展:
子查询1:
  - RAG系统的文档处理流程
  - RAG如何处理文档
  
子查询2:
  - 多语言文档的挑战
  - 跨语言检索问题

步骤3 - 合并检索结果并去重
```

**使用方法**:
```bash
curl -X POST "http://localhost:8000/ask_with_rewriting" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "RAG系统如何处理多语言文档",
    "strategy": "multi"
  }'
```

---

## 🤖 智能策略推荐

使用 `/analyze_query` 端点，让系统自动分析查询并推荐最佳策略：

```bash
curl -X POST "http://localhost:8000/analyze_query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "你的查询"
  }'
```

**返回示例**:
```json
{
  "ok": true,
  "data": {
    "complexity": "medium",
    "type": "comparison",
    "recommended_strategy": "decompose",
    "reason": "查询包含对比问题，建议分解为子查询分别检索"
  }
}
```

---

## 📊 效果对比

### 测试查询: "RAG和LangChain的关系"

#### 普通检索
```
检索到: 3 个文档
相关度: 0.72, 0.68, 0.65
答案质量: ⭐⭐⭐
```

#### 使用查询扩展
```
改写查询:
1. RAG和LangChain的关系
2. LangChain中的RAG实现
3. RAG框架与LangChain对比

检索到: 8 个文档（去重后 5 个）
相关度: 0.85, 0.82, 0.79, 0.76, 0.73
答案质量: ⭐⭐⭐⭐⭐
```

**提升**: 召回率 +67%，答案质量显著提升

---

## 💡 最佳实践

### 1. 根据查询类型选择策略

| 查询类型 | 推荐策略 | 原因 |
|---------|---------|------|
| 简短查询 | expand | 提高召回率 |
| 复杂查询 | decompose | 分解为简单问题 |
| 技术问题 | hyde | 深度理解 |
| 对比问题 | decompose | 分别检索再对比 |
| 多步推理 | multi | 综合效果最佳 |

### 2. 参数调优

```python
{
  "question": "你的查询",
  "strategy": "expand",
  "top_k": 5,              # 每个查询检索5个文档
  "rerank_enabled": true,  # 启用重排
  "rerank_top_n": 3        # 重排后保留3个
}
```

### 3. 性能考虑

- **expand**: 速度快，适合实时场景
- **decompose**: 中等速度
- **hyde**: 较慢（需要生成假设文档）
- **multi**: 最慢（组合策略）

**建议**: 
- 实时对话: 使用 `expand`
- 深度问答: 使用 `hyde` 或 `multi`
- 自动选择: 先调用 `/analyze_query` 获取推荐

---

## 🔧 Python SDK 示例

```python
import requests

# 1. 分析查询
response = requests.post(
    "http://localhost:8000/analyze_query",
    json={"question": "RAG如何处理长文档"}
)
analysis = response.json()["data"]
recommended_strategy = analysis["recommended_strategy"]

# 2. 使用推荐策略进行查询改写
response = requests.post(
    "http://localhost:8000/ask_with_rewriting",
    json={
        "question": "RAG如何处理长文档",
        "strategy": recommended_strategy,
        "top_k": 5,
        "rerank_enabled": True
    }
)

result = response.json()
print(f"答案: {result['answer']}")
print(f"改写查询: {result['metadata']['rewritten_queries']}")
print(f"检索到 {result['metadata']['total_retrieved']} 个文档")
```

---

## 📈 监控和调试

查询改写返回的 `metadata` 包含详细信息：

```json
{
  "answer": "...",
  "sources": [...],
  "metadata": {
    "original_query": "原始查询",
    "rewritten_queries": ["改写1", "改写2", "改写3"],
    "strategy": "expand",
    "total_retrieved": 8,
    "unique_documents": 5
  }
}
```

**关键指标**:
- `rewritten_queries`: 查看改写效果
- `total_retrieved`: 总检索数量
- `unique_documents`: 去重后的文档数

---

## ⚠️ 注意事项

1. **API Key 配置**: 查询改写需要调用 LLM，确保配置了有效的 API Key

2. **成本考虑**: 每次查询改写会额外调用 1-2 次 LLM API

3. **延迟**: 查询改写会增加 0.5-2 秒的延迟

4. **缓存**: 系统会缓存改写结果，相同查询不会重复改写

---

## 🎓 进阶技巧

### 自定义改写提示词

如果需要更精细的控制，可以修改 `backend/query_rewriter.py` 中的提示词模板。

### 组合使用

```python
# 先用 expand 提高召回，再用 reranker 提高精度
response = requests.post(
    "http://localhost:8000/ask_with_rewriting",
    json={
        "question": "你的查询",
        "strategy": "expand",
        "top_k": 10,           # 每个查询检索10个
        "rerank_enabled": True,
        "rerank_top_n": 3      # 重排后只保留3个最相关的
    }
)
```

---

## 📚 参考资料

- [Query Rewriting for RAG](https://arxiv.org/abs/2305.14283)
- [HyDE: Precise Zero-Shot Dense Retrieval](https://arxiv.org/abs/2212.10496)
- [RAG Survey](https://arxiv.org/abs/2312.10997)

---

**Happy Querying! 🚀**
