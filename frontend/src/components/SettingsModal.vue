<template>
  <div class="modal-overlay" @click.self="$emit('close')">
    <div class="modal-content">
      <div class="modal-header">
        <h3>⚙️ 系统设置</h3>
        <button class="close-btn" @click="$emit('close')">✕</button>
      </div>
      
      <div class="modal-body">
        <!-- 模型设置 -->
        <div class="setting-section">
          <h4 class="section-title">🤖 模型配置</h4>
          
          <div class="model-selection">
            <label class="selection-label">
              <span>选择 LLM 模型</span>
              <span class="label-desc">点击卡片选择要使用的大语言模型</span>
            </label>
            <div class="model-cards">
              <div 
                v-for="model in modelOptions" 
                :key="model.value"
                class="model-card"
                :class="{ active: settings.llmModel === model.value }"
                @click="selectModel(model.value)"
              >
                <div class="model-icon">{{ model.icon }}</div>
                <div class="model-info">
                  <div class="model-name">{{ model.name }}</div>
                  <div class="model-desc">{{ model.desc }}</div>
                </div>
                <div class="model-check" v-if="settings.llmModel === model.value">
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
                    <path d="M20 6L9 17l-5-5" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                  </svg>
                </div>
              </div>
            </div>
          </div>
          
          <div class="model-selection" style="margin-top: 24px;">
            <label class="selection-label">
              <span>选择嵌入模型</span>
              <span class="label-desc">用于文本向量化，影响检索质量</span>
            </label>
            <div class="embedding-cards">
              <div 
                v-for="model in embeddingOptions" 
                :key="model.value"
                class="embedding-card"
                :class="{ active: settings.embeddingModel === model.value }"
                @click="selectEmbedding(model.value)"
              >
                <div class="embedding-info">
                  <div class="embedding-icon">{{ model.icon }}</div>
                  <div class="embedding-details">
                    <div class="embedding-name">{{ model.name }}</div>
                    <div class="embedding-desc">{{ model.desc }}</div>
                  </div>
                </div>
                <div class="model-check" v-if="settings.embeddingModel === model.value">
                  <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
                    <path d="M20 6L9 17l-5-5" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                  </svg>
                </div>
              </div>
            </div>
          </div>
        </div>
        
        <!-- 检索设置 -->
        <div class="setting-section">
          <h4 class="section-title">🔍 检索配置</h4>
          
          <div class="setting-item">
            <label class="setting-label">
              <span>Top K</span>
              <span class="label-desc">检索返回的文档片段数量</span>
            </label>
            <div class="slider-control">
              <input 
                v-model.number="settings.topK" 
                type="range" 
                class="setting-slider"
                min="1"
                max="20"
              />
              <span class="slider-value">{{ settings.topK }}</span>
            </div>
          </div>
          
          <div class="setting-item">
            <label class="setting-label">
              <span>相似度阈值</span>
              <span class="label-desc">过滤低相关度文档（0=不过滤，越高越严格）</span>
            </label>
            <div class="slider-control">
              <input 
                v-model.number="settings.scoreThreshold" 
                type="range" 
                class="setting-slider"
                min="0"
                max="1"
                step="0.05"
              />
              <span class="slider-value">{{ settings.scoreThreshold.toFixed(2) }}</span>
            </div>
          </div>
          
          <div class="setting-item">
            <label class="setting-label">
              <span>BM25 权重</span>
              <span class="label-desc">关键词匹配的权重（提高可改善精确匹配）</span>
            </label>
            <div class="slider-control">
              <input 
                v-model.number="settings.bm25Weight" 
                type="range" 
                class="setting-slider"
                min="0"
                max="1"
                step="0.05"
              />
              <span class="slider-value">{{ settings.bm25Weight.toFixed(2) }}</span>
            </div>
          </div>
          
          <div class="setting-item">
            <label class="setting-label">
              <span>向量权重</span>
              <span class="label-desc">语义相似度的权重</span>
            </label>
            <div class="slider-control">
              <input 
                v-model.number="settings.vectorWeight" 
                type="range" 
                class="setting-slider"
                min="0"
                max="1"
                step="0.05"
              />
              <span class="slider-value">{{ settings.vectorWeight.toFixed(2) }}</span>
            </div>
          </div>
          
          <div class="setting-item">
            <label class="setting-label">
              <span>结果多样性</span>
              <span class="label-desc">MMR 多样性参数（越高结果越多样）</span>
            </label>
            <div class="slider-control">
              <input 
                v-model.number="settings.mmrLambda" 
                type="range" 
                class="setting-slider"
                min="0"
                max="1"
                step="0.05"
              />
              <span class="slider-value">{{ settings.mmrLambda.toFixed(2) }}</span>
            </div>
          </div>
          
          <div class="setting-item">
            <label class="setting-label">
              <span>严格模式</span>
              <span class="label-desc">仅基于知识库回答，不允许模型自由发挥</span>
            </label>
            <div class="toggle-switch">
              <input 
                id="strict-mode"
                v-model="settings.strictMode" 
                type="checkbox"
                class="toggle-input"
              />
              <label for="strict-mode" class="toggle-label"></label>
            </div>
          </div>
          
          <div class="setting-item">
            <label class="setting-label">
              <span>启用 BM25</span>
              <span class="label-desc">混合检索：语义 + 关键词匹配</span>
            </label>
            <div class="toggle-switch">
              <input 
                id="bm25-enabled"
                v-model="settings.bm25Enabled" 
                type="checkbox"
                class="toggle-input"
              />
              <label for="bm25-enabled" class="toggle-label"></label>
            </div>
          </div>
          
          <div class="setting-item">
            <label class="setting-label">
              <span>启用重排序</span>
              <span class="label-desc">使用 Reranker 模型提升检索精度</span>
            </label>
            <div class="toggle-switch">
              <input 
                id="rerank-enabled"
                v-model="settings.rerankEnabled" 
                type="checkbox"
                class="toggle-input"
              />
              <label for="rerank-enabled" class="toggle-label"></label>
            </div>
          </div>
          
          <div v-if="settings.rerankEnabled" class="setting-item">
            <label class="setting-label">
              <span>重排序 Top N</span>
              <span class="label-desc">重排序后保留的片段数量</span>
            </label>
            <div class="slider-control">
              <input 
                v-model.number="settings.rerankTopN" 
                type="range" 
                class="setting-slider"
                min="1"
                :max="settings.topK"
              />
              <span class="slider-value">{{ settings.rerankTopN }}</span>
            </div>
          </div>
          
          <!-- 快速预设 -->
          <div class="preset-buttons">
            <button 
              class="preset-btn" 
              :class="{ active: currentPreset === 'balanced' }"
              @click="applyPreset('balanced')"
            >
              ⚖️ 平衡模式
            </button>
            <button 
              class="preset-btn" 
              :class="{ active: currentPreset === 'recall' }"
              @click="applyPreset('recall')"
            >
              📊 高召回模式
            </button>
            <button 
              class="preset-btn" 
              :class="{ active: currentPreset === 'precision' }"
              @click="applyPreset('precision')"
            >
              🎯 高精度模式
            </button>
          </div>
        </div>
        
        <!-- 提示词设置 -->
        <div class="setting-section prompt-section">
          <div class="section-header">
            <h4 class="section-title">
              <span class="title-icon">✨</span>
              <span>提示词工坊</span>
            </h4>
            <span class="section-badge">自定义 AI 行为</span>
          </div>
          
          <div class="preset-prompts-top">
            <button 
              class="preset-prompt-card" 
              :class="{ active: currentPromptPreset === 'default' }"
              @click="applyPromptPreset('default')"
            >
              <div class="card-icon">📝</div>
              <div class="card-content">
                <div class="card-title">默认助手</div>
                <div class="card-desc">专业、准确、格式规范</div>
              </div>
            </button>
            <button 
              class="preset-prompt-card" 
              :class="{ active: currentPromptPreset === 'detailed' }"
              @click="applyPromptPreset('detailed')"
            >
              <div class="card-icon">📚</div>
              <div class="card-content">
                <div class="card-title">详细解答</div>
                <div class="card-desc">深入全面、技术文档</div>
              </div>
            </button>
            <button 
              class="preset-prompt-card" 
              :class="{ active: currentPromptPreset === 'concise' }"
              @click="applyPromptPreset('concise')"
            >
              <div class="card-icon">⚡</div>
              <div class="card-content">
                <div class="card-title">简洁模式</div>
                <div class="card-desc">快速直接、要点明确</div>
              </div>
            </button>
          </div>
          
          <div class="prompt-editor-wrapper">
            <div class="editor-header">
              <span class="editor-label">
                <span class="label-icon">🎯</span>
                系统提示词
              </span>
              <div class="editor-actions">
                <button class="action-btn" @click="formatPrompt" title="格式化">
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                    <path d="M4 7h16M4 12h16M4 17h10" stroke-width="2" stroke-linecap="round"/>
                  </svg>
                </button>
                <button class="action-btn" @click="resetPrompt" title="重置">
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                    <path d="M3 12a9 9 0 0 1 9-9 9.75 9.75 0 0 1 6.74 2.74L21 8" stroke-width="2" stroke-linecap="round"/>
                    <path d="M21 3v5h-5" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                  </svg>
                </button>
              </div>
            </div>
            <textarea 
              v-model="settings.systemPrompt" 
              class="prompt-textarea"
              rows="12"
              placeholder="编写你的系统提示词..."
              spellcheck="false"
            ></textarea>
            <div class="editor-footer">
              <div class="prompt-hints">
                <span class="hint-badge">
                  <span class="hint-icon">💡</span>
                  使用 <code>{context}</code> 插入检索内容
                </span>
                <span class="hint-badge">
                  <span class="hint-icon">💬</span>
                  使用 <code>{question}</code> 插入用户问题
                </span>
              </div>
              <div class="char-count">
                {{ settings.systemPrompt.length }} 字符
              </div>
            </div>
          </div>
        </div>
        
        <!-- 界面设置 -->
        <div class="setting-section">
          <h4 class="section-title">🎨 界面配置</h4>
          
          <div class="setting-item">
            <label class="setting-label">
              <span>深色模式</span>
              <span class="label-desc">切换明暗主题</span>
            </label>
            <div class="toggle-switch">
              <input 
                id="dark-mode"
                v-model="settings.darkMode" 
                type="checkbox"
                class="toggle-input"
                @change="toggleTheme"
              />
              <label for="dark-mode" class="toggle-label"></label>
            </div>
          </div>
          
          <div class="setting-item">
            <label class="setting-label">
              <span>流式输出速度</span>
              <span class="label-desc">打字机效果的延迟（毫秒）</span>
            </label>
            <input 
              v-model.number="settings.streamDelay" 
              type="range" 
              class="setting-range"
              min="0"
              max="50"
              step="5"
            />
            <span class="range-value">{{ settings.streamDelay }} ms</span>
          </div>
          
          <div class="setting-item">
            <label class="setting-label">
              <span>自动保存对话</span>
              <span class="label-desc">自动保存聊天历史到本地</span>
            </label>
            <div class="toggle-switch">
              <input 
                id="auto-save"
                v-model="settings.autoSave" 
                type="checkbox"
                class="toggle-input"
              />
              <label for="auto-save" class="toggle-label"></label>
            </div>
          </div>
        </div>
        
        <!-- 缓存管理 -->
        <div class="setting-section">
          <h4 class="section-title">💾 缓存管理</h4>
          
          <div class="cache-stats">
            <div class="cache-info">
              <span class="cache-label">缓存大小</span>
              <span class="cache-value">{{ cacheSize }} / {{ cacheMaxSize }}</span>
            </div>
            <button class="btn btn-secondary" @click="clearCache">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
                <polyline points="3 6 5 6 21 6" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
              </svg>
              清空缓存
            </button>
          </div>
          
          <div class="cache-info-text">
            缓存可以加速重复查询，但占用内存。如遇到检索结果不准确，可尝试清空缓存。
          </div>
        </div>
        
        <!-- 数据管理 -->
        <div class="setting-section">
          <h4 class="section-title">🗄️ 数据管理</h4>
          
          <div class="data-actions">
            <button class="btn btn-secondary" @click="exportSettings">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4M7 10l5 5 5-5M12 15V3" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
              </svg>
              导出设置
            </button>
            
            <button class="btn btn-secondary" @click="importSettings">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4M17 8l-5-5-5 5M12 3v12" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
              </svg>
              导入设置
            </button>
            
            <button class="btn btn-danger" @click="resetSettings">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
                <path d="M3 12a9 9 0 0 1 9-9 9.75 9.75 0 0 1 6.74 2.74L21 8" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                <path d="M21 3v5h-5M21 12a9 9 0 0 1-9 9 9.75 9.75 0 0 1-6.74-2.74L3 16" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                <path d="M3 21v-5h5" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
              </svg>
              恢复默认
            </button>
          </div>
        </div>
      </div>
      
      <div class="modal-footer">
        <button class="btn btn-secondary" @click="$emit('close')">取消</button>
        <button class="btn btn-primary" @click="saveSettings">保存设置</button>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, watch } from 'vue';
import api from '../api';

const emit = defineEmits(['close', 'settings-changed']);

const settings = ref({
  llmModel: 'deepseek-chat',
  embeddingModel: 'sentence-transformers/all-MiniLM-L6-v2',
  topK: 8,
  scoreThreshold: 0.0,
  bm25Weight: 0.4,
  vectorWeight: 0.6,
  mmrLambda: 0.7,
  strictMode: true,
  bm25Enabled: true,
  rerankEnabled: false,
  rerankTopN: 5,
  darkMode: false,
  streamDelay: 5,
  autoSave: true,
  systemPrompt: `你是一个专业的知识库检索助手。

**核心规则**：
1. 仔细阅读下列所有文档片段，全面理解其内容
2. 从文档中寻找与问题相关的所有信息，包括直接和间接相关的内容
3. 综合多个文档片段的信息进行回答
4. 如果文档中确实没有答案，明确告知用户
5. 回答要详细、具体，尽可能引用原文

**输出格式要求**：
1. 使用规范的中文标点符号（，。；！？）
2. 合理分段，每段讲一个主题
3. 使用标题、列表等 Markdown 格式提高可读性
4. 数字和英文前后加空格（例如：YOLOv8 的结构）
5. 避免句子过长，适当断句`
});

const currentPromptPreset = ref('default');
const availableModels = ref(['deepseek-chat', 'qwen-turbo', 'qwen-plus', 'qwen-max']);
const modelOptions = ref([
  {
    value: 'deepseek-chat',
    name: 'DeepSeek Chat',
    desc: '高性价比，推理能力强',
    icon: '🚀'
  },
  {
    value: 'qwen-turbo',
    name: 'Qwen Turbo',
    desc: '快速响应，适合日常对话',
    icon: '⚡'
  },
  {
    value: 'qwen-plus',
    name: 'Qwen Plus',
    desc: '平衡性能与成本，推荐',
    icon: '✨'
  },
  {
    value: 'qwen-max',
    name: 'Qwen Max',
    desc: '最强性能，复杂任务首选',
    icon: '🎯'
  }
]);

const embeddingOptions = ref([
  {
    value: 'sentence-transformers/all-MiniLM-L6-v2',
    name: 'MiniLM-L6',
    desc: '轻量快速，适合快速测试',
    icon: '💨'
  },
  {
    value: 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
    name: 'Multilingual-MiniLM',
    desc: '多语言支持，中英文通用',
    icon: '🌏'
  },
  {
    value: 'BAAI/bge-small-zh-v1.5',
    name: 'BGE-Small-ZH',
    desc: '中文优化，性能均衡',
    icon: '🇨🇳'
  },
  {
    value: 'BAAI/bge-base-zh-v1.5',
    name: 'BGE-Base-ZH',
    desc: '中文基础模型，推荐使用',
    icon: '⭐'
  },
  {
    value: 'BAAI/bge-large-zh-v1.5',
    name: 'BGE-Large-ZH',
    desc: '中文最强效果，质量最高',
    icon: '🏆'
  },
  {
    value: 'moka-ai/m3e-base',
    name: 'M3E-Base',
    desc: '中文开源模型，效果优秀',
    icon: '🔥'
  }
]);

const cacheSize = ref(0);
const cacheMaxSize = ref(256);
const currentPreset = ref('balanced'); // 当前选中的预设模式

// 选择 LLM 模型
function selectModel(modelValue) {
  settings.value.llmModel = modelValue;
}

// 选择嵌入模型
function selectEmbedding(modelValue) {
  settings.value.embeddingModel = modelValue;
}

// 格式化提示词
function formatPrompt() {
  // 简单的格式化：移除多余空行
  settings.value.systemPrompt = settings.value.systemPrompt
    .replace(/\n{3,}/g, '\n\n')
    .trim();
}

// 重置提示词
function resetPrompt() {
  applyPromptPreset('default');
}

// 应用提示词预设
function applyPromptPreset(preset) {
  currentPromptPreset.value = preset;
  switch (preset) {
    case 'default':
      settings.value.systemPrompt = `你是一个专业的知识库检索助手。

**核心规则**：
1. 仔细阅读下列所有文档片段，全面理解其内容
2. 从文档中寻找与问题相关的所有信息，包括直接和间接相关的内容
3. 综合多个文档片段的信息进行回答
4. 如果文档中确实没有答案，明确告知用户
5. 回答要详细、具体，尽可能引用原文

**输出格式要求**：
1. 使用规范的中文标点符号（，。；！？）
2. 合理分段，每段讲一个主题
3. 使用标题、列表等 Markdown 格式提高可读性
4. 数字和英文前后加空格（例如：YOLOv8 的结构）
5. 避免句子过长，适当断句`;
      break;
    case 'detailed':
      settings.value.systemPrompt = `你是一个详细的技术文档助手。

**回答原则**：
1. 提供深入、全面的解答，覆盖所有相关细节
2. 使用专业术语，并提供必要的解释
3. 引用原文时使用引用格式
4. 提供示例和类比帮助理解
5. 如果有多种解释，列举所有可能性

**格式要求**：
- 使用标题和子标题组织内容
- 使用编号列表展示步骤
- 使用代码块展示技术内容
- 重点内容使用粗体或斜体`;
      break;
    case 'concise':
      settings.value.systemPrompt = `你是一个简洁高效的助手。

**回答原则**：
1. 提供简明扼要的答案，直击要点
2. 避免冗余信息，只保留核心内容
3. 使用简短的段落和句子
4. 优先使用列表而非长段落
5. 如果问题简单，一句话回答即可

**格式要求**：
- 简洁的标题
- 要点列表
- 必要时提供简短示例`;
      break;
  }
}

// 应用检索预设
function applyPreset(preset) {
  currentPreset.value = preset; // 更新当前选中的预设
  
  switch (preset) {
    case 'balanced': // 平衡模式：默认推荐
      settings.value.topK = 8;
      settings.value.scoreThreshold = 0.0;
      settings.value.bm25Weight = 0.4;
      settings.value.vectorWeight = 0.6;
      settings.value.mmrLambda = 0.7;
      settings.value.bm25Enabled = true;
      break;
    case 'recall': // 高召回模式：宽松检索，增加命中率
      settings.value.topK = 12;
      settings.value.scoreThreshold = 0.0;
      settings.value.bm25Weight = 0.5;
      settings.value.vectorWeight = 0.5;
      settings.value.mmrLambda = 0.5;
      settings.value.bm25Enabled = true;
      break;
    case 'precision': // 高精度模式：严格筛选
      settings.value.topK = 5;
      settings.value.scoreThreshold = 0.3;
      settings.value.bm25Weight = 0.3;
      settings.value.vectorWeight = 0.7;
      settings.value.mmrLambda = 0.8;
      settings.value.bm25Enabled = true;
      break;
  }
}

// 加载设置
function loadSettings() {
  const saved = localStorage.getItem('app-settings');
  if (saved) {
    try {
      const parsed = JSON.parse(saved);
      // 先合并设置
      settings.value = { ...settings.value, ...parsed };
      
      // 验证 llmModel 是否在可用列表中
      if (availableModels.value.length > 0 && !availableModels.value.includes(settings.value.llmModel)) {
        console.warn(`Invalid model '${settings.value.llmModel}' in localStorage, resetting to default`);
        settings.value.llmModel = availableModels.value[0] || 'deepseek-chat';
        // 只保存到 localStorage，不触发 emit
        localStorage.setItem('app-settings', JSON.stringify(settings.value));
      }
    } catch (e) {
      console.error('加载设置失败:', e);
    }
  }
  
  // 检查主题并立即应用
  const theme = localStorage.getItem('theme');
  settings.value.darkMode = theme === 'dark';
  
  // 立即应用主题到 DOM
  document.documentElement.setAttribute('data-theme', theme || 'light');
}

// 保存设置
function saveSettings() {
  console.log('Saving settings:', settings.value);
  // 自动补齐提示词占位符
  if (settings.value.systemPrompt) {
    let p = settings.value.systemPrompt;
    if (!p.includes('{question}')) p += "\n\n用户问题：{question}";
    if (!p.includes('{context}')) p += "\n\n文档片段：\n{context}";
    settings.value.systemPrompt = p;
  }
  localStorage.setItem('app-settings', JSON.stringify(settings.value));
  
  // 同时保存主题设置
  const theme = settings.value.darkMode ? 'dark' : 'light';
  localStorage.setItem('theme', theme);
  document.documentElement.setAttribute('data-theme', theme);
  
  emit('settings-changed', settings.value);
  emit('close');
  
  // 显示保存成功提示
  console.log('Settings saved successfully!');
}

// 主题切换（通过 watch 自动触发）
function toggleTheme() {
  const theme = settings.value.darkMode ? 'dark' : 'light';
  console.log('Toggling theme to:', theme);
  localStorage.setItem('theme', theme);
  document.documentElement.setAttribute('data-theme', theme);
  console.log('Theme applied to DOM, current attribute:', document.documentElement.getAttribute('data-theme'));
}

// 监听主题变化
watch(() => settings.value.darkMode, (newValue) => {
  console.log('Dark mode changed to:', newValue);
  toggleTheme();
});

// 加载缓存统计
async function loadCacheStats() {
  try {
    const res = await api.get('/cache/stats');
    if (res.data.ok) {
      cacheSize.value = res.data.cache_size;
      cacheMaxSize.value = res.data.max_size;
    }
  } catch (e) {
    console.error('加载缓存统计失败:', e);
  }
}

// 清空缓存
async function clearCache() {
  if (!confirm('确认清空所有查询缓存？')) return;
  
  try {
    const res = await api.post('/cache/clear');
    if (res.data.ok) {
      cacheSize.value = 0;
      alert('缓存已清空');
    }
  } catch (e) {
    alert('清空缓存失败: ' + e.message);
  }
}

// 导出设置
function exportSettings() {
  const data = JSON.stringify(settings.value, null, 2);
  const blob = new Blob([data], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'deeplearning-settings.json';
  a.click();
  URL.revokeObjectURL(url);
}

// 导入设置
function importSettings() {
  const input = document.createElement('input');
  input.type = 'file';
  input.accept = '.json';
  input.onchange = (e) => {
    const file = e.target.files[0];
    if (!file) return;
    
    const reader = new FileReader();
    reader.onload = (evt) => {
      try {
        const imported = JSON.parse(evt.target.result);
        settings.value = { ...settings.value, ...imported };
        alert('设置导入成功');
      } catch (err) {
        alert('导入失败：文件格式错误');
      }
    };
    reader.readAsText(file);
  };
  input.click();
}

// 恢复默认设置
function resetSettings() {
  if (!confirm('确认恢复所有默认设置？')) return;
  
  settings.value = {
    llmModel: 'deepseek-chat',
    embeddingModel: 'sentence-transformers/all-MiniLM-L6-v2',
    topK: 8,
    scoreThreshold: 0.0,
    bm25Weight: 0.4,
    vectorWeight: 0.6,
    mmrLambda: 0.7,
    strictMode: true,
    bm25Enabled: true,
    rerankEnabled: false,
    rerankTopN: 5,
    darkMode: false,
    streamDelay: 5,
    autoSave: true
  };
  
  alert('已恢复默认设置');
}

// 加载可用模型列表
async function loadAvailableModels() {
  try {
    const res = await api.getModels();
    if (res.data.ok) {
      availableModels.value = res.data.models;
      
      // 模型配置映射
      const modelConfigMap = {
        'deepseek-chat': { name: 'DeepSeek Chat', desc: '高性价比，推理能力强', icon: '🚀' },
        'qwen-turbo': { name: 'Qwen Turbo', desc: '快速响应，适合日常对话', icon: '⚡' },
        'qwen-plus': { name: 'Qwen Plus', desc: '平衡性能与成本，推荐', icon: '✨' },
        'qwen-max': { name: 'Qwen Max', desc: '最强性能，复杂任务首选', icon: '🎯' },
        'gpt-4': { name: 'GPT-4', desc: 'OpenAI 最强模型', icon: '🤖' },
        'gpt-4o': { name: 'GPT-4o', desc: 'OpenAI 多模态模型', icon: '🌟' },
        'gpt-4o-mini': { name: 'GPT-4o Mini', desc: 'OpenAI 轻量模型', icon: '💫' },
        'gpt-3.5-turbo': { name: 'GPT-3.5 Turbo', desc: '快速且经济', icon: '💨' }
      };
      
      // 根据后端返回的模型列表更新 modelOptions
      modelOptions.value = res.data.models.map(model => ({
        value: model,
        name: modelConfigMap[model]?.name || model,
        desc: modelConfigMap[model]?.desc || '大语言模型',
        icon: modelConfigMap[model]?.icon || '🔮'
      }));
      
      // 如果当前模型不在列表中，设置为默认模型并保存
      if (!availableModels.value.includes(settings.value.llmModel)) {
        settings.value.llmModel = res.data.default_model || availableModels.value[0];
        // 立即保存到 localStorage，避免显示旧模型
        const saved = localStorage.getItem('app-settings');
        if (saved) {
          try {
            const parsed = JSON.parse(saved);
            parsed.llmModel = settings.value.llmModel;
            localStorage.setItem('app-settings', JSON.stringify(parsed));
          } catch (e) {
            console.error('更新模型设置失败:', e);
          }
        }
      }
    }
  } catch (e) {
    console.error('加载模型列表失败:', e);
  }
}

onMounted(async () => {
  // 先加载模型列表，再加载设置，确保模型验证正确
  await loadAvailableModels();
  loadSettings();
  loadCacheStats();
});
</script>

<style scoped>
.modal-overlay {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0, 0, 0, 0.5);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1000;
  backdrop-filter: blur(4px);
  animation: fadeIn 0.15s ease-out;
}

@keyframes fadeIn {
  from { opacity: 0; }
  to { opacity: 1; }
}

.modal-content {
  background: var(--bg-primary);
  border-radius: 16px;
  box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
  max-width: 700px;
  width: 90%;
  max-height: 85vh;
  display: flex;
  flex-direction: column;
  overflow: hidden;
  animation: slideUp 0.2s ease-out;
}

@keyframes slideUp {
  from {
    transform: translateY(20px);
    opacity: 0;
  }
  to {
    transform: translateY(0);
    opacity: 1;
  }
}

.modal-header {
  padding: 20px 24px;
  border-bottom: 1px solid var(--border-primary);
  display: flex;
  justify-content: space-between;
  align-items: center;
  background: var(--bg-secondary);
}

.modal-header h3 {
  font-size: 18px;
  font-weight: 600;
  color: var(--text-primary);
  margin: 0;
}

.close-btn {
  background: transparent;
  border: none;
  font-size: 24px;
  color: var(--text-secondary);
  cursor: pointer;
  transition: color 0.2s;
  padding: 0;
  width: 32px;
  height: 32px;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 6px;
}

.close-btn:hover {
  color: var(--text-primary);
  background: var(--bg-tertiary);
}

.modal-body {
  padding: 20px 24px;
  overflow-y: auto;
  flex: 1;
}

.setting-section {
  margin-bottom: 28px;
  padding-bottom: 24px;
  border-bottom: 1px solid var(--border-primary);
}

.setting-section:last-child {
  border-bottom: none;
  margin-bottom: 0;
  padding-bottom: 0;
}

.section-title {
  font-size: 15px;
  font-weight: 600;
  color: var(--text-primary);
  margin: 0 0 16px 0;
}

.setting-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 14px 0;
  gap: 16px;
}

.setting-label {
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.setting-label > span:first-child {
  font-size: 14px;
  font-weight: 500;
  color: var(--text-primary);
}

.label-desc {
  font-size: 12px;
  color: var(--text-secondary);
}

.setting-input,
.setting-select {
  width: 200px;
  padding: 8px 12px;
  border: 1.5px solid var(--border-primary);
  border-radius: 8px;
  font-size: 14px;
  transition: all 0.2s;
  background: var(--bg-primary);
  color: var(--text-primary);
}

.setting-input:focus,
.setting-select:focus {
  outline: none;
  border-color: var(--accent-primary);
  box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
}

.setting-select {
  cursor: pointer;
  appearance: none;
  background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='12' viewBox='0 0 12 12'%3E%3Cpath fill='%236b7280' d='M6 9L1 4h10z'/%3E%3C/svg%3E");
  background-repeat: no-repeat;
  background-position: right 10px center;
  padding-right: 32px;
}

/* 滑块控制 */
.slider-control {
  display: flex;
  align-items: center;
  gap: 12px;
  width: 220px;
}

.setting-slider {
  flex: 1;
  height: 4px;
  border-radius: 2px;
  background: var(--border-primary);
  outline: none;
  cursor: pointer;
  appearance: none;
}

.setting-slider::-webkit-slider-thumb {
  appearance: none;
  width: 16px;
  height: 16px;
  border-radius: 50%;
  background: var(--accent-primary);
  cursor: pointer;
  box-shadow: 0 2px 4px rgba(59, 130, 246, 0.3);
  transition: all 0.2s;
}

.setting-slider::-webkit-slider-thumb:hover {
  background: var(--accent-hover);
  transform: scale(1.1);
  box-shadow: 0 3px 6px rgba(59, 130, 246, 0.4);
}

.setting-slider::-moz-range-thumb {
  width: 16px;
  height: 16px;
  border-radius: 50%;
  background: var(--accent-primary);
  cursor: pointer;
  border: none;
  box-shadow: 0 2px 4px rgba(59, 130, 246, 0.3);
  transition: all 0.2s;
}

.setting-slider::-moz-range-thumb:hover {
  background: var(--accent-hover);
  transform: scale(1.1);
  box-shadow: 0 3px 6px rgba(59, 130, 246, 0.4);
}

.slider-value {
  font-size: 14px;
  font-weight: 600;
  color: var(--accent-primary);
  min-width: 40px;
  text-align: right;
}

/* 预设按钮 */
.preset-buttons {
  display: flex;
  gap: 8px;
  margin-top: 16px;
  padding-top: 16px;
  border-top: 1px solid var(--border-primary);
}

.preset-btn {
  flex: 1;
  padding: 10px 16px;
  background: var(--bg-secondary);
  border: 1.5px solid var(--border-primary);
  border-radius: 8px;
  font-size: 13px;
  font-weight: 500;
  color: var(--text-primary);
  cursor: pointer;
  transition: all 0.2s;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 6px;
}

.preset-btn:hover {
  background: var(--bg-primary);
  border-color: var(--accent-primary);
  color: var(--accent-primary);
  transform: translateY(-1px);
  box-shadow: 0 2px 8px rgba(59, 130, 246, 0.15);
}

.preset-btn.active {
  background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
  border-color: #3b82f6;
  color: #ffffff;
  box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
  font-weight: 600;
}

.preset-btn.active:hover {
  background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
  transform: translateY(-1px);
}

/* 提示词配置区域 */
.prompt-section {
  background: var(--bg-secondary);
  border: 2px solid var(--border-primary);
  padding: 24px;
}

.section-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 20px;
}

.section-title {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 16px;
  font-weight: 600;
  color: var(--text-primary);
  margin: 0;
}

.title-icon {
  font-size: 20px;
}

.section-badge {
  padding: 4px 12px;
  background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
  color: white;
  font-size: 11px;
  font-weight: 600;
  border-radius: 12px;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

/* 预设提示词卡片 */
.preset-prompts-top {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 12px;
  margin-bottom: 20px;
}

.preset-prompt-card {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 16px;
  background: var(--bg-primary);
  border: 2px solid var(--border-primary);
  border-radius: 12px;
  cursor: pointer;
  transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
}

.preset-prompt-card:hover {
  border-color: #3b82f6;
  transform: translateY(-2px);
  box-shadow: 0 8px 16px rgba(59, 130, 246, 0.15);
}

.preset-prompt-card.active {
  background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
  border-color: #2563eb;
  box-shadow: 0 8px 20px rgba(59, 130, 246, 0.35);
  transform: translateY(-2px);
}

.preset-prompt-card.active .card-icon {
  font-size: 28px;
  filter: drop-shadow(0 2px 4px rgba(0, 0, 0, 0.1));
}

.preset-prompt-card.active .card-title,
.preset-prompt-card.active .card-desc {
  color: white;
}

.card-icon {
  font-size: 24px;
  transition: all 0.25s;
}

.card-content {
  flex: 1;
  text-align: left;
}

.card-title {
  font-size: 14px;
  font-weight: 600;
  color: var(--text-primary);
  margin-bottom: 2px;
  transition: color 0.25s;
}

.card-desc {
  font-size: 11px;
  color: var(--text-secondary);
  transition: color 0.25s;
}

/* 编辑器容器 */
.prompt-editor-wrapper {
  background: var(--bg-primary);
  border: 2px solid var(--border-primary);
  border-radius: 12px;
  overflow: hidden;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
}

.editor-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 12px 16px;
  background: var(--bg-secondary);
  border-bottom: 1px solid var(--border-primary);
}

.editor-label {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 13px;
  font-weight: 600;
  color: var(--text-secondary);
}

.label-icon {
  font-size: 16px;
}

.editor-actions {
  display: flex;
  gap: 6px;
}

.action-btn {
  padding: 6px 8px;
  background: var(--bg-primary);
  border: 1px solid var(--border-primary);
  border-radius: 6px;
  cursor: pointer;
  color: var(--text-secondary);
  transition: all 0.2s;
  display: flex;
  align-items: center;
  justify-content: center;
}

.action-btn:hover {
  background: var(--bg-secondary);
  border-color: var(--border-secondary);
  color: var(--accent-primary);
  transform: translateY(-1px);
}

/* 文本编辑区 */
.prompt-textarea {
  width: 100%;
  padding: 16px;
  border: none;
  font-size: 13px;
  font-family: 'SF Mono', 'Monaco', 'Consolas', 'Liberation Mono', monospace;
  line-height: 1.6;
  resize: vertical;
  min-height: 280px;
  background: var(--bg-primary);
  color: var(--text-primary);
  transition: all 0.2s;
}

.prompt-textarea:focus {
  outline: none;
  background: var(--bg-secondary);
}

.prompt-textarea::placeholder {
  color: #94a3b8;
}

/* 编辑器底部 */
.editor-footer {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 10px 16px;
  background: var(--bg-secondary);
  border-top: 1px solid var(--border-primary);
  flex-wrap: wrap;
  gap: 8px;
}

.prompt-hints {
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
}

.hint-badge {
  display: flex;
  align-items: center;
  gap: 4px;
  padding: 4px 10px;
  background: var(--bg-primary);
  border: 1px solid var(--border-primary);
  border-radius: 6px;
  font-size: 11px;
  color: var(--text-secondary);
}

.hint-icon {
  font-size: 13px;
}

.hint-badge code {
  padding: 2px 6px;
  background: var(--bg-tertiary);
  border: 1px solid var(--border-secondary);
  border-radius: 4px;
  font-family: 'SF Mono', monospace;
  font-size: 11px;
  color: var(--accent-primary);
  font-weight: 600;
}

.char-count {
  font-size: 11px;
  color: var(--text-tertiary);
  font-weight: 500;
  padding: 4px 8px;
  background: var(--bg-primary);
  border: 1px solid var(--border-primary);
  border-radius: 6px;
}

/* 模型选择卡片 */
.model-selection {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

/* 嵌入模型卡片 */
.embedding-cards {
  display: flex;
  flex-direction: column;
  gap: 6px;
  max-height: 400px;
  overflow-y: auto;
  padding: 4px;
}

.embedding-card {
  position: relative;
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 10px 12px;
  background: var(--bg-secondary);
  border: 1.5px solid var(--border-primary);
  border-radius: 8px;
  cursor: pointer;
  transition: all 0.2s;
}

.embedding-card:hover {
  background: var(--bg-primary);
  border-color: var(--border-secondary);
  transform: translateX(4px);
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
}

.embedding-card.active {
  background: var(--bg-primary);
  border-color: var(--accent-primary);
  box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
}

.embedding-card.active:hover {
  background: var(--bg-secondary);
}

.embedding-info {
  display: flex;
  align-items: center;
  gap: 12px;
  flex: 1;
  min-width: 0;
}

.embedding-icon {
  font-size: 18px;
  flex-shrink: 0;
  line-height: 1;
}

.embedding-details {
  flex: 1;
  min-width: 0;
}

.embedding-name {
  font-size: 13px;
  font-weight: 600;
  color: var(--text-primary);
  margin-bottom: 2px;
}

.embedding-desc {
  font-size: 11px;
  color: var(--text-secondary);
  line-height: 1.3;
}

.selection-label {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.selection-label > span:first-child {
  font-size: 14px;
  font-weight: 500;
  color: var(--text-primary);
}

.model-cards {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 8px;
}

.model-card {
  position: relative;
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 12px;
  background: var(--bg-secondary);
  border: 1.5px solid var(--border-primary);
  border-radius: 10px;
  cursor: pointer;
  transition: all 0.2s;
}

.model-card:hover {
  background: var(--bg-primary);
  border-color: var(--border-secondary);
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
}

.model-card.active {
  background: var(--bg-primary);
  border-color: var(--accent-primary);
  box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
}

.model-card.active:hover {
  background: var(--bg-primary);
}

.model-icon {
  font-size: 22px;
  flex-shrink: 0;
  line-height: 1;
}

.model-info {
  flex: 1;
  min-width: 0;
}

.model-name {
  font-size: 13px;
  font-weight: 600;
  color: var(--text-primary);
  margin-bottom: 4px;
}

.model-desc {
  font-size: 11px;
  color: var(--text-secondary);
  line-height: 1.4;
}

.model-check {
  color: #3b82f6;
  flex-shrink: 0;
  animation: checkIn 0.3s ease-out;
}

/* 已移除前置复选框，仅保留右侧对勾 */

@keyframes checkIn {
  0% {
    transform: scale(0);
    opacity: 0;
  }
  50% {
    transform: scale(1.2);
  }
  100% {
    transform: scale(1);
    opacity: 1;
  }
}

.toggle-switch {
  position: relative;
  width: 48px;
  height: 28px;
}

.toggle-input {
  display: none;
}

.toggle-label {
  position: absolute;
  top: 0;
  left: 0;
  width: 48px;
  height: 28px;
  background: #d1d5db;
  border-radius: 14px;
  cursor: pointer;
  transition: all 0.3s;
}

.toggle-label::after {
  content: '';
  position: absolute;
  top: 3px;
  left: 3px;
  width: 22px;
  height: 22px;
  background: #ffffff;
  border-radius: 50%;
  transition: all 0.3s;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
}

.toggle-input:checked + .toggle-label {
  background: #3b82f6;
}

.toggle-input:checked + .toggle-label::after {
  left: 23px;
}

.setting-range {
  width: 160px;
  height: 6px;
  border-radius: 3px;
  background: var(--border-primary);
  outline: none;
  -webkit-appearance: none;
}

.setting-range::-webkit-slider-thumb {
  -webkit-appearance: none;
  appearance: none;
  width: 18px;
  height: 18px;
  border-radius: 50%;
  background: var(--accent-primary);
  cursor: pointer;
  transition: all 0.2s;
}

.setting-range::-webkit-slider-thumb:hover {
  background: var(--accent-hover);
  transform: scale(1.1);
}

.range-value {
  font-size: 13px;
  color: var(--text-secondary);
  font-weight: 500;
  min-width: 50px;
  text-align: right;
}

.cache-stats {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 14px 16px;
  background: var(--bg-secondary);
  border-radius: 10px;
  margin-bottom: 12px;
}

.cache-info {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.cache-label {
  font-size: 12px;
  color: var(--text-secondary);
  font-weight: 500;
}

.cache-value {
  font-size: 18px;
  font-weight: 700;
  color: var(--text-primary);
}

.cache-info-text {
  font-size: 12px;
  color: var(--text-secondary);
  line-height: 1.5;
}

.data-actions {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
}

.btn {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 10px 16px;
  border-radius: 8px;
  font-size: 14px;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.2s;
  border: none;
}

.btn-primary {
  background: #3b82f6;
  color: #ffffff;
}

.btn-primary:hover {
  background: #2563eb;
  box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
}

.btn-secondary {
  background: var(--bg-secondary);
  color: var(--text-primary);
  border: 1px solid var(--border-primary);
}

.btn-secondary:hover {
  background: var(--bg-primary);
  border-color: var(--border-secondary);
}

.btn-danger {
  background: #fee2e2;
  color: #dc2626;
  border: 1px solid #fca5a5;
}

.btn-danger:hover {
  background: #fca5a5;
  border-color: #f87171;
}

.modal-footer {
  padding: 16px 24px;
  border-top: 1px solid var(--border-primary);
  display: flex;
  justify-content: flex-end;
  gap: 8px;
  background: var(--bg-secondary);
}
</style>

