<template>
  <section class="chat">
    <div class="messages" ref="messagesEl">
      <div v-if="conversation.length === 0" class="empty-state">
        <svg width="64" height="64" viewBox="0 0 24 24" fill="none">
          <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
          <circle cx="9" cy="10" r="1" fill="currentColor"/>
          <circle cx="12" cy="10" r="1" fill="currentColor"/>
          <circle cx="15" cy="10" r="1" fill="currentColor"/>
        </svg>
        <h3>开始新的对话</h3>
        <p>输入您的问题，我会基于知识库为您提供答案</p>
      </div>
      
      <div v-for="(msg, i) in conversation" :key="i" class="message" :class="msg.role">
        <div class="avatar">
          <svg v-if="msg.role === 'user'" width="20" height="20" viewBox="0 0 24 24" fill="none">
            <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2M12 11a4 4 0 1 0 0-8 4 4 0 0 0 0 8z" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
          </svg>
          <svg v-else width="20" height="20" viewBox="0 0 24 24" fill="none">
            <rect x="3" y="3" width="18" height="18" rx="3" stroke="currentColor" stroke-width="2"/>
            <path d="M9 9h6M9 12h6M9 15h4" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/>
          </svg>
        </div>
        <div class="msg-content">
          <div class="msg-header">
            <span class="msg-role">{{ msg.role === 'user' ? '你' : getModelNameByValue(msg.model) }}</span>
            <span class="msg-time">{{ formatTime(msg.ts) }}</span>
          </div>
          <div class="msg-text" v-html="renderMsg(msg)"></div>
        </div>
      </div>
      
      <div v-if="loading && streamingContent" class="message assistant">
        <div class="avatar">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
            <rect x="3" y="3" width="18" height="18" rx="3" stroke="currentColor" stroke-width="2"/>
            <path d="M9 9h6M9 12h6M9 15h4" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/>
          </svg>
        </div>
        <div class="msg-content">
          <div class="msg-header">
            <span class="msg-role">{{ getCurrentModel().name }}</span>
            <span class="typing-indicator">
              <span></span><span></span><span></span>
            </span>
          </div>
          <div class="msg-text" v-html="renderMsg({ role: 'assistant', content: streamingContent })"></div>
        </div>
      </div>
    </div>
    
    <div class="composer">
      <div class="input-wrapper">
        <!-- 模型选择器 - 美化版 -->
        <div class="model-selector-wrapper" @click.stop>
          <button class="model-selector-btn" @click.stop="toggleModelDropdown">
            <span class="model-icon">{{ getCurrentModel().icon }}</span>
            <span class="model-name">{{ getCurrentModel().name }}</span>
            <svg class="dropdown-arrow" :class="{ open: showModelDropdown }" width="12" height="12" viewBox="0 0 24 24" fill="none">
              <path d="M6 9l6 6 6-6" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
            </svg>
          </button>
          
          <div v-if="showModelDropdown" class="model-dropdown">
            <div class="dropdown-header">Select model</div>
            <div 
              v-for="model in availableModels" 
              :key="model.value" 
              class="model-option"
              :class="{ active: selectedModel === model.value }"
              @click="selectModelAndClose(model.value)"
            >
              <span class="option-icon">{{ model.icon }}</span>
              <div class="option-content">
                <div class="option-name">{{ model.name }}</div>
                <div class="option-desc">{{ model.desc }}</div>
              </div>
              <svg v-if="selectedModel === model.value" class="check-icon" width="16" height="16" viewBox="0 0 24 24" fill="none">
                <path d="M20 6L9 17l-5-5" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
              </svg>
            </div>
          </div>
        </div>
        
        <textarea
          v-model="question" 
          @keydown.enter.exact.prevent="sendQuestion"
          placeholder="输入你的问题 (Shift+Enter 换行)" 
          :disabled="loading"
          rows="1"
          ref="textareaEl"
        />
        <button v-if="!loading" class="send-btn" @click="sendQuestion" :disabled="!question.trim()">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
            <path d="M22 2L11 13M22 2l-7 20-4-9-9-4 20-7z" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
          </svg>
        </button>
        <button v-else class="send-btn danger" @click="stopGeneration" title="Stop">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
            <rect x="6" y="6" width="12" height="12" rx="2" stroke="currentColor" stroke-width="2"/>
          </svg>
        </button>
      </div>
    </div>
  </section>
</template>

<script setup>
import { ref, nextTick, watch, onMounted, onUnmounted } from 'vue';
import { marked } from 'marked';
import api from '../api';

const props = defineProps(['conversation']);
const emit = defineEmits(['new-message']);

const question = ref('');
const loading = ref(false);
const streamingContent = ref('');
const messagesEl = ref(null);
const textareaEl = ref(null);
const selectedModel = ref('deepseek-chat');
let currentAbortController = null;
const showModelDropdown = ref(false);
const availableModels = ref([
  { value: 'deepseek-chat', name: 'DeepSeek', icon: 'DS', desc: 'Balanced quality and speed' },
  { value: 'qwen-turbo', name: 'Qwen Turbo', icon: 'QT', desc: 'Fast responses' },
  { value: 'qwen-plus', name: 'Qwen Plus', icon: 'QP', desc: 'Better reasoning' },
  { value: 'qwen-max', name: 'Qwen Max', icon: 'QM', desc: 'Highest capability' }
]);

watch(question, () => {
  if (textareaEl.value) {
    textareaEl.value.style.height = 'auto';
    textareaEl.value.style.height = Math.min(textareaEl.value.scrollHeight, 200) + 'px';
  }
});

watch(() => props.conversation, async () => {
  await nextTick();
  scrollToBottom();
}, { deep: true });

function scrollToBottom() {
  if (messagesEl.value) {
    messagesEl.value.scrollTop = messagesEl.value.scrollHeight;
  }
}

function formatTime(ts) {
  if (!ts) return '';
  const date = new Date(ts);
  return date.toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' });
}

function renderMsg(msg) {
  if (msg.role === 'assistant') {
    return marked.parse(msg.content || '');
  }
  return (msg.content || '').replace(/\n/g, '<br>');
}

function getCurrentModel() {
  return availableModels.value.find(m => m.value === selectedModel.value) || availableModels.value[0];
}

function getModelNameByValue(value) {
  const m = availableModels.value.find(m => m.value === value);
  return m ? m.name : 'Assistant';
}

function toggleModelDropdown() {
  showModelDropdown.value = !showModelDropdown.value;
}

function selectModelAndClose(modelValue) {
  selectedModel.value = modelValue;
  showModelDropdown.value = false;
  localStorage.setItem('selectedModel', modelValue);
}

// 点击外部关闭下拉菜单
let clickOutsideHandler = null;

onMounted(() => {
  clickOutsideHandler = (e) => {
    const wrapper = document.querySelector('.model-selector-wrapper');
    if (wrapper && !wrapper.contains(e.target) && showModelDropdown.value) {
      showModelDropdown.value = false;
    }
  };
  
  document.addEventListener('click', clickOutsideHandler);
});

onUnmounted(() => {
  if (clickOutsideHandler) {
    document.removeEventListener('click', clickOutsideHandler);
  }
});

async function sendQuestion() {
  if (!question.value.trim() || loading.value) return;
  const q = question.value.trim();
  question.value = '';
  
  // 重置 textarea 高度
  if (textareaEl.value) {
    textareaEl.value.style.height = 'auto';
  }
  
  emit('new-message', { role: 'user', content: q, ts: Date.now() });
  
  loading.value = true;
  streamingContent.value = '';
  // 使用选中的模型
  const model = selectedModel.value;
  
  // 从 localStorage 获取系统提示词
  const settings = JSON.parse(localStorage.getItem('app-settings') || '{}');
  const systemPrompt = settings.systemPrompt;
  const topK = Number.isFinite(settings.topK) ? settings.topK : 4;
  
  try {
    // 若上一次流还未结束，先中止
    if (currentAbortController) {
      currentAbortController.abort();
    }
    currentAbortController = new AbortController();
    const res = await api.askStream(
      q,
      model,
      topK,
      systemPrompt,
      null,
      null,
      currentAbortController.signal
    );
    if (!res.ok || !res.body) {
      const errorText = await res.text();
      throw new Error(errorText || `Request failed with status ${res.status}`);
    }
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let answer = '';
    let buffer = ''; // 缓冲区，用于处理不完整的 UTF-8 字符
    
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      
      const chunk = decoder.decode(value, { stream: true });
      buffer += chunk;
      const lines = buffer.split('\n\n');
      
      // 保留最后一个可能不完整的块
      buffer = lines.pop() || '';
      
      for (const line of lines) {
        if (!line) continue;
        let eventType = 'message';
        let dataPayload = '';
        for (const part of line.split('\n')) {
          if (part.startsWith('event: ')) {
            eventType = part.slice(7).trim();
            continue;
          }
          if (part.startsWith('data: ')) {
            dataPayload = part.slice(6);
          }
        }
        if (!dataPayload) continue;
        if (eventType === 'source' || eventType === 'meta' || eventType === 'done') {
          continue;
        }
        const token = dataPayload;

        if (token.length <= 2) {
          answer += token;
          streamingContent.value = answer;
          await nextTick();
          scrollToBottom();
        } else {
          for (const char of token) {
            answer += char;
            streamingContent.value = answer;
            await nextTick();
            scrollToBottom();
            await new Promise(resolve => setTimeout(resolve, 5));
          }
        }
      }
    }
    
    emit('new-message', { role: 'assistant', model, content: answer, ts: Date.now() });
  } catch (e) {
    console.error(e);
    if (e.name === 'AbortError') {
      // 用户主动停止，无需追加消息
    } else {
      emit('new-message', { role: 'assistant', model, content: '请求失败: ' + e.message, ts: Date.now() });
    }
  } finally {
    loading.value = false;
    streamingContent.value = '';
    currentAbortController = null;
    await nextTick();
    scrollToBottom();
  }
}

function stopGeneration() {
  if (currentAbortController) {
    currentAbortController.abort();
  }
}

// 加载可用模型列表
async function loadAvailableModels() {
  try {
    const res = await api.getModels();
    if (res.data.ok) {
      const modelConfigMap = {
        'deepseek-chat': { name: 'DeepSeek Chat', icon: 'DS', desc: 'Balanced quality and speed' },
        'qwen-turbo': { name: 'Qwen Turbo', icon: 'QT', desc: 'Fast responses' },
        'qwen-plus': { name: 'Qwen Plus', icon: 'QP', desc: 'Better reasoning' },
        'qwen-max': { name: 'Qwen Max', icon: 'QM', desc: 'Highest capability' }
      };
      
      availableModels.value = res.data.models.map(model => ({
        value: model,
        name: modelConfigMap[model]?.name || model,
        icon: modelConfigMap[model]?.icon || 'LLM',
        desc: modelConfigMap[model]?.desc || ''
      }));
      
      // 从 localStorage 恢复上次选择的模型
      const savedModel = localStorage.getItem('selectedModel');
      if (savedModel && res.data.models.includes(savedModel)) {
        selectedModel.value = savedModel;
      } else {
        selectedModel.value = res.data.default_model || res.data.models[0];
      }
    }
  } catch (e) {
    console.error('加载模型列表失败:', e);
  }
}

// 监听模型选择变化，保存到 localStorage
watch(selectedModel, (newModel) => {
  localStorage.setItem('selectedModel', newModel);
});

onMounted(() => {
  loadAvailableModels();
});
</script>

<style scoped>
.chat {
  display: flex;
  flex-direction: column;
  height: 100%;
  background: #ffffff;
  overflow: hidden; /* 防止内容溢出 */
}

.messages {
  flex: 1;
  overflow-y: auto;
  overflow-x: hidden;
  padding: 24px;
  background: var(--bg-primary);
  display: flex;
  flex-direction: column;
  align-items: center;
  min-height: 0; /* 确保 flex item 可以缩小 */
}

.messages > * {
  width: 100%;
  max-width: 900px;
}

.messages::-webkit-scrollbar {
  width: 6px;
}

.messages::-webkit-scrollbar-track {
  background: transparent;
}

.messages::-webkit-scrollbar-thumb {
  background: var(--border-secondary);
  border-radius: 3px;
}

.messages::-webkit-scrollbar-thumb:hover {
  background: var(--text-tertiary);
}

.empty-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 100%;
  color: var(--text-tertiary);
  gap: 16px;
  padding: 40px;
}

.empty-state h3 {
  font-size: 20px;
  font-weight: 600;
  color: var(--text-primary);
  margin: 0;
}

.empty-state p {
  font-size: 14px;
  color: #6b7280;
  margin: 0;
  text-align: center;
  max-width: 400px;
}

.message {
  display: flex;
  gap: 12px;
  margin-bottom: 24px;
  width: 100%;
}

.message.user {
  flex-direction: row-reverse;
  justify-content: flex-start; /* 用户消息靠右 */
}

.message.assistant {
  justify-content: flex-start; /* 助手消息靠左 */
}

.avatar {
  width: 36px;
  height: 36px;
  border-radius: 50%;
  background: var(--bg-tertiary);
  display: flex;
  align-items: center;
  justify-content: center;
  color: var(--text-secondary);
  flex-shrink: 0;
  border: 1px solid var(--border-primary);
}

.message.assistant .avatar {
  background: var(--text-primary);
  color: var(--bg-primary);
  border: none;
}

.msg-content {
  flex: 1;
  min-width: 0;
  max-width: 70%; /* 限制消息宽度，避免占满整行 */
}

.message.user .msg-content {
  display: flex;
  flex-direction: column;
  align-items: flex-end; /* 用户消息内容右对齐 */
}

.msg-header {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 8px;
}

.msg-role {
  font-size: 13px;
  font-weight: 600;
  color: var(--text-primary);
}

.msg-time {
  font-size: 12px;
  color: var(--text-tertiary);
}

.typing-indicator {
  display: flex;
  gap: 4px;
  align-items: center;
}

.typing-indicator span {
  width: 4px;
  height: 4px;
  border-radius: 50%;
  background: #9ca3af;
  animation: bounce 1.4s infinite ease-in-out;
}

.typing-indicator span:nth-child(1) {
  animation-delay: -0.32s;
}

.typing-indicator span:nth-child(2) {
  animation-delay: -0.16s;
}

@keyframes bounce {
  0%, 80%, 100% { 
    transform: scale(0);
    opacity: 0.5;
  }
  40% { 
    transform: scale(1);
    opacity: 1;
  }
}

.msg-text {
  font-size: 15px;
  line-height: 1.7;
  color: var(--text-primary);
  word-wrap: break-word;
}

/* 确保 Markdown 在深色模式下可见 */
.msg-text :where(p, ul, ol, li, h1, h2, h3, h4, h5, h6, strong, em, span) {
  color: var(--text-primary) !important;
}
.msg-text a { color: var(--accent-primary); }
.msg-text code { 
  background: var(--bg-tertiary);
  border: 1px solid var(--border-primary);
  padding: 0 4px;
  border-radius: 4px;
}
.msg-text pre { 
  background: var(--bg-tertiary);
  border: 1px solid var(--border-primary);
  color: var(--text-primary);
  padding: 12px;
  border-radius: 8px;
  overflow: auto;
}
.msg-text blockquote {
  border-left: 3px solid var(--border-secondary);
  padding-left: 10px;
  color: var(--text-secondary);
}

.message.user .msg-text {
  background: var(--bg-secondary);
  color: var(--text-primary);
  padding: 12px 16px;
  border-radius: 12px;
  border: 1px solid var(--border-primary);
  box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
}

.composer {
  flex-shrink: 0; /* 输入框不被压缩 */
  border-top: 1px solid var(--border-primary);
  padding: 16px 24px 24px;
  background: var(--bg-primary);
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 12px;
}

.composer-controls {
  width: 100%;
  max-width: 900px;
  display: flex;
  justify-content: flex-start;
  padding: 0 4px;
}

.model-selector {
  display: flex;
  align-items: center;
  gap: 8px;
}

.model-dropdown-legacy {
  padding: 6px 12px;
  background: #f3f4f6;
  border: 1px solid #e5e7eb;
  border-radius: 8px;
  font-size: 13px;
  font-weight: 500;
  color: #374151;
  cursor: pointer;
  transition: all 0.2s;
  outline: none;
}

.model-dropdown-legacy:hover {
  background: #e5e7eb;
  border-color: #d1d5db;
}

.model-dropdown-legacy:focus {
  border-color: #3b82f6;
  box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
}

.input-wrapper {
  display: flex;
  align-items: center;
  gap: 12px;
  width: 100%;
  max-width: 900px;
  background: var(--bg-primary);
  border: 1.5px solid var(--border-primary);
  border-radius: 12px;
  padding: 12px 16px;
  transition: all 0.2s;
}

.input-wrapper:focus-within {
  border-color: var(--text-primary);
  background: var(--bg-primary);
  box-shadow: 0 0 0 3px rgba(17, 24, 39, 0.05);
}

/* 模型选择器 */
.model-selector-wrapper {
  position: relative;
  flex-shrink: 0;
}

.model-selector-btn {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 6px 10px;
  background: var(--bg-secondary);
  border: 1.5px solid var(--border-primary);
  border-radius: 8px;
  cursor: pointer;
  transition: all 0.2s;
  font-size: 12px;
  font-weight: 500;
  color: var(--text-primary);
  white-space: nowrap;
}

.model-selector-btn:hover {
  background: var(--bg-tertiary);
  border-color: var(--border-secondary);
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
}

.model-icon {
  width: 20px;
  height: 20px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  font-size: 12px;
  font-weight: 600;
  line-height: 1;
}

.model-name {
  font-weight: 600;
  color: var(--text-primary);
}

.dropdown-arrow {
  color: #64748b;
  transition: transform 0.2s;
}

.dropdown-arrow.open {
  transform: rotate(180deg);
}

/* 下拉菜单 */
.model-dropdown {
  position: absolute;
  bottom: calc(100% + 8px); /* 向上展开 */
  left: 0;
  min-width: 200px;
  background: var(--bg-primary);
  border: 1px solid var(--border-primary);
  border-radius: 12px;
  box-shadow: 0 10px 30px rgba(0, 0, 0, 0.12);
  z-index: 1000;
  overflow: hidden;
  animation: dropdownSlideIn 0.2s ease-out;
}

@keyframes dropdownSlideIn {
  from {
    opacity: 0;
    transform: translateY(-8px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

.dropdown-header {
  padding: 12px 16px;
  background: var(--bg-secondary);
  border-bottom: 1px solid var(--border-primary);
  font-size: 12px;
  font-weight: 600;
  color: var(--text-secondary);
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

.model-option {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 12px 16px;
  cursor: pointer;
  transition: all 0.15s;
  border-bottom: 1px solid var(--border-primary);
}

.model-option:last-child {
  border-bottom: none;
}

.model-option:hover {
  background: var(--bg-secondary);
}

.model-option.active {
  background: var(--bg-tertiary);
  border-left: 3px solid var(--accent-primary);
}

.option-icon {
  width: 20px;
  height: 20px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  font-size: 12px;
  font-weight: 600;
  line-height: 1;
}

.option-content {
  flex: 1;
}

.option-name {
  font-size: 14px;
  font-weight: 600;
  color: var(--text-primary);
  margin-bottom: 2px;
}

.option-desc {
  font-size: 11px;
  color: var(--text-secondary);
}

.model-option.active .option-name {
  color: #1e40af;
}

.model-option.active .option-desc {
  color: #3b82f6;
}

.check-icon {
  color: var(--accent-primary);
  flex-shrink: 0;
}

textarea {
  flex: 1;
  border: none;
  background: transparent;
  color: var(--text-primary);
  font-size: 15px;
  font-family: inherit;
  resize: none;
  outline: none;
  line-height: 1.5;
  max-height: 200px;
  overflow-y: auto;
}

textarea::placeholder {
  color: var(--text-tertiary);
}

textarea::-webkit-scrollbar {
  width: 4px;
}

textarea::-webkit-scrollbar-thumb {
  background: #d1d5db;
  border-radius: 2px;
}

.send-btn {
  width: 36px;
  height: 36px;
  border-radius: 8px;
  border: none;
  background: #111827;
  color: #ffffff;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all 0.2s;
  flex-shrink: 0;
}

.send-btn:not(:disabled):hover {
  background: #1f2937;
  transform: scale(1.05);
}

.send-btn:disabled {
  background: #e5e7eb;
  color: #9ca3af;
  cursor: not-allowed;
  transform: none;
}

.send-btn.danger {
  background: var(--danger);
}

.loading-spinner {
  width: 16px;
  height: 16px;
  border: 2px solid #ffffff;
  border-top-color: transparent;
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}

/* Markdown 样式 */
.msg-text :deep(pre) {
  background: #f7f8fa;
  border: 1px solid #e5e7eb;
  border-radius: 8px;
  padding: 16px;
  overflow-x: auto;
  margin: 12px 0;
  font-size: 14px;
}

.msg-text :deep(code) {
  background: #f3f4f6;
  padding: 2px 6px;
  border-radius: 4px;
  font-family: 'Consolas', 'Monaco', 'Menlo', monospace;
  font-size: 0.9em;
  color: #374151;
  border: 1px solid #e5e7eb;
}

.msg-text :deep(pre code) {
  background: transparent;
  padding: 0;
  border: none;
  color: #1f2937;
}

.msg-text :deep(h1), 
.msg-text :deep(h2), 
.msg-text :deep(h3) {
  margin: 16px 0 8px;
  color: #111827;
  font-weight: 600;
}

.msg-text :deep(h1) { font-size: 1.5em; }
.msg-text :deep(h2) { font-size: 1.3em; }
.msg-text :deep(h3) { font-size: 1.1em; }

.msg-text :deep(p) {
  margin: 8px 0;
}

.msg-text :deep(ul), 
.msg-text :deep(ol) {
  margin: 8px 0;
  padding-left: 24px;
}

.msg-text :deep(li) {
  margin: 4px 0;
}

.msg-text :deep(a) {
  color: #2563eb;
  text-decoration: none;
  border-bottom: 1px solid rgba(37,99,235,0.3);
  transition: all 0.2s;
}

.msg-text :deep(a:hover) {
  color: #1d4ed8;
  border-bottom-color: rgba(29,78,216,0.6);
}

.msg-text :deep(strong) {
  font-weight: 600;
  color: #111827;
}

.msg-text :deep(blockquote) {
  border-left: 3px solid #e5e7eb;
  padding-left: 16px;
  margin: 12px 0;
  color: #6b7280;
}
</style>

