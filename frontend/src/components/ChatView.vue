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
            <span class="msg-role">{{ msg.role === 'user' ? '你' : 'DeepLearning' }}</span>
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
            <span class="msg-role">DeepLearning</span>
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
        <textarea
          v-model="question" 
          @keydown.enter.exact.prevent="sendQuestion"
          placeholder="输入你的问题 (Shift+Enter 换行)" 
          :disabled="loading"
          rows="1"
          ref="textareaEl"
        />
        <button class="send-btn" @click="sendQuestion" :disabled="loading || !question.trim()">
          <svg v-if="!loading" width="20" height="20" viewBox="0 0 24 24" fill="none">
            <path d="M22 2L11 13M22 2l-7 20-4-9-9-4 20-7z" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
          </svg>
          <div v-else class="loading-spinner"></div>
        </button>
      </div>
    </div>
  </section>
</template>

<script setup>
import { ref, nextTick, watch } from 'vue';
import { marked } from 'marked';
import api from '../api';

const props = defineProps(['conversation']);
const emit = defineEmits(['new-message']);

const question = ref('');
const loading = ref(false);
const streamingContent = ref('');
const messagesEl = ref(null);
const textareaEl = ref(null);

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
  const model = localStorage.getItem('model') || undefined;
  
  try {
    const res = await api.askStream(q, model, 4);
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let answer = '';
    
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      const chunk = decoder.decode(value, { stream: true });
      const lines = chunk.split('\n\n');
      for (const line of lines) {
        if (!line) continue;
        if (line.startsWith('data: ')) {
          const token = line.slice(6);
          answer += token;
          streamingContent.value = answer;
          await nextTick();
          scrollToBottom();
        }
      }
    }
    
    emit('new-message', { role: 'assistant', content: answer, ts: Date.now() });
  } catch (e) {
    console.error(e);
    emit('new-message', { role: 'assistant', content: '请求失败: ' + e.message, ts: Date.now() });
  } finally {
    loading.value = false;
    streamingContent.value = '';
    await nextTick();
    scrollToBottom();
  }
}
</script>

<style scoped>
.chat {
  display: flex;
  flex-direction: column;
  height: 100vh;
  background: #ffffff;
}

.messages {
  flex: 1;
  overflow-y: auto;
  padding: 24px;
  background: #ffffff;
  display: flex;
  flex-direction: column;
  align-items: center;
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
  background: #d1d5db;
  border-radius: 3px;
}

.messages::-webkit-scrollbar-thumb:hover {
  background: #9ca3af;
}

.empty-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 100%;
  color: #9ca3af;
  gap: 16px;
  padding: 40px;
}

.empty-state h3 {
  font-size: 20px;
  font-weight: 600;
  color: #374151;
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
}

.avatar {
  width: 36px;
  height: 36px;
  border-radius: 50%;
  background: #f3f4f6;
  display: flex;
  align-items: center;
  justify-content: center;
  color: #6b7280;
  flex-shrink: 0;
  border: 1px solid #e5e7eb;
}

.message.assistant .avatar {
  background: #111827;
  color: #ffffff;
  border: none;
}

.msg-content {
  flex: 1;
  min-width: 0;
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
  color: #374151;
}

.msg-time {
  font-size: 12px;
  color: #9ca3af;
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
  color: #374151;
  word-wrap: break-word;
}

.message.user .msg-text {
  color: #111827;
}

.composer {
  border-top: 1px solid #e5e7eb;
  padding: 16px 24px 24px;
  background: #ffffff;
  display: flex;
  justify-content: center;
}

.input-wrapper {
  display: flex;
  align-items: flex-end;
  gap: 12px;
  width: 100%;
  max-width: 900px;
  background: #f7f8fa;
  border: 1.5px solid #e5e7eb;
  border-radius: 12px;
  padding: 12px 16px;
  transition: all 0.2s;
}

.input-wrapper:focus-within {
  border-color: #111827;
  background: #ffffff;
  box-shadow: 0 0 0 3px rgba(17, 24, 39, 0.05);
}

textarea {
  flex: 1;
  border: none;
  background: transparent;
  color: #111827;
  font-size: 15px;
  font-family: inherit;
  resize: none;
  outline: none;
  line-height: 1.5;
  max-height: 200px;
  overflow-y: auto;
}

textarea::placeholder {
  color: #9ca3af;
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

