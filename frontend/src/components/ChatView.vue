<template>
  <section class="chat">
    <div class="messages" ref="messagesEl">
      <div v-for="(msg, i) in conversation" :key="i" class="message">
        <div class="avatar">{{ msg.role === 'user' ? 'U' : 'A' }}</div>
        <div class="msg" :class="msg.role" v-html="renderMsg(msg)"></div>
      </div>
    </div>
    
    <div class="typing" v-if="loading">正在生成...</div>
    
    <div class="composer">
      <input 
        v-model="question" 
        @keydown.enter="sendQuestion" 
        type="text" 
        placeholder="输入你的问题..." 
        :disabled="loading"
      />
      <button class="btn primary" @click="sendQuestion" :disabled="loading">发送</button>
    </div>
  </section>
</template>

<script setup>
import { ref, nextTick } from 'vue';
import { marked } from 'marked';
import api from '../api';

const props = defineProps(['conversation']);
const emit = defineEmits(['new-message']);

const question = ref('');
const loading = ref(false);
const messagesEl = ref(null);

function renderMsg(msg) {
  if (msg.role === 'assistant') {
    return marked.parse(msg.content || '');
  }
  return msg.content || '';
}

async function sendQuestion() {
  if (!question.value.trim() || loading.value) return;
  const q = question.value.trim();
  question.value = '';
  
  emit('new-message', { role: 'user', content: q, ts: Date.now() });
  
  loading.value = true;
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
        }
      }
    }
    
    emit('new-message', { role: 'assistant', content: answer, ts: Date.now() });
  } catch (e) {
    console.error(e);
    emit('new-message', { role: 'assistant', content: '请求失败: ' + e.message, ts: Date.now() });
  } finally {
    loading.value = false;
    await nextTick();
    if (messagesEl.value) messagesEl.value.scrollTop = messagesEl.value.scrollHeight;
  }
}
</script>

<style scoped>
.chat {
  background: #101623;
  border: 1px solid #1d2639;
  border-radius: 12px;
  padding: 16px;
  box-shadow: 0 8px 20px rgba(0,0,0,0.25);
  display: flex;
  flex-direction: column;
  min-height: calc(100vh - 220px);
}

.messages {
  flex: 1;
  overflow: auto;
  padding: 12px;
  background: #0e1421;
  border-radius: 10px;
  border: 1px solid #1d2639;
  max-height: 500px;
}

.message { display: flex; gap: 10px; margin: 10px 0; }
.avatar {
  width: 28px;
  height: 28px;
  border-radius: 50%;
  background: #27324a;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 14px;
  color: #c6d1f0;
  flex-shrink: 0;
}

.msg {
  flex: 1;
  padding: 10px 12px;
  border-radius: 10px;
  line-height: 1.7;
  max-width: 88%;
}

.msg.user {
  background: linear-gradient(180deg,#0b254a,#0b1e38);
  color: #a0cfff;
  border: 1px solid #213456;
}

.msg.assistant {
  background: #0f1524;
  color: #e6e8eb;
  border: 1px solid #27324a;
}

.typing { margin: 8px 0; font-size: 12px; opacity: 0.8; }

.composer {
  display: flex;
  gap: 8px;
  margin-top: 12px;
}

input {
  flex: 1;
  padding: 10px 12px;
  border-radius: 8px;
  border: 1px solid #27324a;
  background: #0e1421;
  color: #e6e8eb;
}

.btn {
  padding: 10px 16px;
  border-radius: 8px;
  border: 1px solid #2a3a66;
  background: #161f35;
  color: #e6e8eb;
  cursor: pointer;
}

.btn:hover { background: #1b2746; }
.btn:disabled { background: #2a3454; cursor: not-allowed; }
.btn.primary { background: #2e5bea; border-color: #2e5bea; }
.btn.primary:hover { background: #3a69ff; }
</style>

