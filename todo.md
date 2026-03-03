## TODO / 质量与稳定性问题清单（持续更新）

说明：本文件记录“命中/质量/安全/可维护性”方面的关键问题与改进进度。

已落地的配置项（部分）：
- `RAG_BM25_REQUIRE_COMPLETE_CORPUS`：Milvus 模式下若 BM25 语料不完整则禁用 BM25 融合（避免伪混合检索）。
- `RAG_QUERY_EXPAND_ENABLED`：是否启用 query expand（启用后为“原 query + 扩展 query 并行召回”）。
- `RAG_PROMPT_SHOW_SCORES`：是否在 prompt 中展示分数（默认不展示）。
- `RAG_BM25_FULL_SCAN_MAX_DOCS`：BM25 全库打分的最大语料阈值（超出后仅对候选集合计算 BM25）。

---

### 1) 关键问题：Milvus 模式下 BM25 语料不完整，混合检索会失真

状态：已缓解（安全降级）✅ / 仍有后续可增强 🔧

- `backend/rag.py:VectorStore.__init__` 会从本地 `meta.jsonl` 读 `texts/metas` 来做 BM25。
- 但 Milvus 检索时，`VectorStore.search()` 先从 Milvus 返回候选，然后 `_fuse_with_bm25()` 用 BM25 分数融合。
- 如果你的 Milvus 库不是“同一台机器同一个 index_dir 写出来的 meta.jsonl”，或者有历史数据只在 Milvus 里而 meta.jsonl 缺失，BM25 就只能在“本地这点文本”上算分，融合权重再合理也会把结果带偏。
- 这会直接影响命中率：你以为是“向量+BM25”，实际可能变成“向量为主 + 一个不完整 BM25 噪声项”。

已做：在 Milvus 模式下若检测到 Milvus 实体数与本地语料明显不一致，则禁用 BM25 融合（仅保留向量召回 + 融合流程中的向量项）。
后续建议：
- 要么从 Milvus 拉取语料构建 BM25（成本高）；要么明确声明 BM25 仅在本地/FAISS 模式提供强保证，并在日志/接口返回中提示“BM25 已降级/禁用”。

### 2) Query Expand 只做了关键词拼接，中文场景容易加噪声降命中

状态：已改为更安全策略 ✅ / 仍需进一步约束 🔧

- `backend/rag.py:VectorStore._expand_query()` 用 `jieba.analyse.extract_tags` 抽 5 个关键词直接拼到 query 后面。
- 常见副作用：把泛化词、停用词式关键词加进去，向量检索会更“语义发散”，反而更难命中包含关键约束（人名/编号/条款/时间）的 chunk。
- 如果要做扩展，通常需要：白名单/黑名单、保留原 query 并行检索、或对扩展 query 做约束保真（不能丢掉否定、范围、数字）。

已做：改为“原 query + 扩展 query 并行召回”后合并，且提供开关 `RAG_QUERY_EXPAND_ENABLED`（默认关闭）。
后续建议：
- 对关键词扩展做约束（数字/时间/否定/范围优先保留；黑名单过滤泛化词）；并提供日志标识扩展 query 的召回贡献占比。

### 3) rerank 打开后，分数语义变了，但提示里还展示 score，且阈值逻辑不一致

状态：已修复 ✅

- 不开 rerank 时 `score` 是融合后的 [0,1] 概念（vec_norm+bm_norm），开 rerank 后 `score` 变成 `FlagReranker` 的打分（尺度完全不同）。
- `build_prompt()` 把 `(score=xx)`直接展示给模型和用户（`backend/rag.py:build_prompt()`），这在不同模式下不可比，会误导调参/分析，也可能影响模型“过度相信某些片段”。
- 同时 `has_valid_context` 用 `c.score > 0.1` 判断“是否有有效上下文”（`backend/rag.py:build_prompt()`）。在 rerank 模式下这个阈值没意义，可能导致“明明没找到证据也不触发信息不足”或相反。

已做：
- 融合/重排分数语义拆分：`score_fused` / `score_rerank` 等写入 `meta`，不再混用同一个 `score`。
- prompt 默认不展示分数（`RAG_PROMPT_SHOW_SCORES=false`），避免模型被数值误导。
- `has_valid_context` 不再依赖 score 阈值，避免不同分数尺度下逻辑失真。

### 4) 召回候选池策略偏拍脑袋：candidate_k 在 namespace 场景激进增大，但没有分类型约束

状态：未完成 ❌

- `VectorStore.search()` 里 `candidate_k = max(top_k*4, top_k+10)`，namespace 前缀存在时会拉到至少 50，最多 500。
- 这会带来两类问题：
  - 性能：BM25 还要全量打分（见下条），长文档库会变慢。
  - 质量：候选池过大时，融合/排序的噪声更大，MMR 多样性会把“同一答案必需的连续证据”拆散，导致最终上下文不够聚焦。

后续建议：
- candidate_k 按查询类型/是否 namespace/语料规模动态调整；并对 BM25、MMR、Rerank 各阶段的“pool size”加硬上限与分桶策略。

### 5) BM25 每次查询都对全库算分，规模上去后“慢”会逼你牺牲 top_k/重排，从而损失命中

状态：已缓解 ✅ / 仍有后续可增强 🔧

- `_fuse_with_bm25()` 里 `self._bm25.get_scores(tokens)` 是对整个语料打分（`backend/rag.py:_fuse_with_bm25()`）。
- 语料一大（几万~几十万 chunk），这个会成为瓶颈；你为了延迟只能降低候选、关 rerank、关改写，最终命中/质量下降。
- 通常要做：BM25 候选集截断（倒排索引/Whoosh/Elastic），或至少做文档级索引与缓存，不要全量扫。

已做：新增 `RAG_BM25_FULL_SCAN_MAX_DOCS`，语料超过阈值时不再全库打分，改为仅对候选集合逐文档算 BM25。
后续建议：引入倒排索引（Whoosh/ES 等）或更系统的分层召回，彻底避免 BM25 全量扫描。

### 6) `ask_with_query_rewriting` 的合并逻辑用“文本去重”，会错杀/漏掉关键证据

状态：已修复 ✅

- `ask_with_query_rewriting()` 里 `seen_texts` 只按 `rec.text` 去重（`backend/rag.py:ask_with_query_rewriting()`）。
- 不足：不同来源/不同页码的 chunk 文本可能相同（尤其表格标题、页眉页脚、模板化段落），你会把它们当重复删掉，丢掉正确引用/定位信息；反之同一 chunk 轻微差异又无法合并。
- 更稳：用 `(path, chunk_id)` 或稳定 chunk hash 去重。

### 7) 文档解析的页码/字符区间追踪不够严谨，影响“引用可信度”进而影响回答质量

状态：未完成 ❌

- `backend/document_parser.py:_parse_pdf()` 的 `char_start/char_end` 是用 `"\n\n".join(text_parts)` 的累计长度算的，但 `text_parts` 中途还会插入表格 markdown（而且页码信息只在“有 page_text 时”记录）。
- 这会导致：字符区间和页码映射不稳定/不一致；后续如果要按 span 精确引用，很容易错位。
- 对“回答质量”的影响是：引用一旦不准，用户会不信答案，即使答案内容对。

后续建议：把“页码/区间”的计算与“表格注入”解耦，分开维护 offset；同时把页码信息做成 chunk 级元数据（chunk_start/end）而不是依赖 `text.find(chunk)`。

### 8) Web 搜索注入上下文的方式非常粗糙，容易污染 RAG 上下文

状态：已缓解 ✅ / 仍有后续可增强 🔧

- `ask_stream()` 里用 duckduckgo html 抓 snippet，然后给 `score=1.0` 直接 append 到 `recs`（`backend/rag.py:ask_stream()`）。
- 这会把外部片段“硬塞进最可信的证据层”，很容易把模型带偏（尤其 strict_mode 下用户以为全是知识库证据）。
- 如果追求质量，web 结果至少要：单独分区、显式标注、降低权重、并且不与内部 chunk 混排。

已做：web snippet 降权（不再给高分）并标注来源字段（`source=web`）。
后续建议：把 web 结果单独分区（例如 `Web Context` 段落），不要与内部 chunk 混排。

如果你要优先改“命中/质量”，我会建议先做这 3 个改动（投入小、收益大）：

1. 统一检索语料源：Milvus 模式下 BM25 的语料要么也来自 Milvus（可 query 出来构建 BM25），要么明确声明 BM25 仅在 FAISS 本地模式可用，并在 UI/日志里提示，避免“伪混合检索”。
2. 把 query expand 改成“并行检索 + 保留原 query”的安全策略，并提供开关与日志（展示原 query/扩展 query 各自召回贡献）。
3. 规范 score 语义：融合分数与 rerank 分数分开字段，不要混用同一个 `score`；`has_valid_context` 改成基于“召回数量/最小相关性阈值（按模式）/或是否命中文档路径”等更稳定指标。
