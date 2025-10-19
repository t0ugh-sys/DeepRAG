<template>
  <div class="app-container" :class="{ light: theme === 'light' }">
    <AppHeader 
      @toggle-theme="toggleTheme"
      @export-conv="exportConv"
      @clear-conv="clearConv"
    />
    
    <div class="main-layout">
      <Sidebar />
      <ChatView 
        :conversation="conversation"
        @new-message="handleNewMessage"
      />
      <SourcesPanel :sources="sources" :paths="paths" @refresh-paths="loadPaths" />
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue';
import AppHeader from './components/AppHeader.vue';
import Sidebar from './components/Sidebar.vue';
import ChatView from './components/ChatView.vue';
import SourcesPanel from './components/SourcesPanel.vue';
import api from './api';

const theme = ref(localStorage.getItem('theme') || 'dark');
const conversation = ref(JSON.parse(localStorage.getItem('conv') || '[]'));
const sources = ref([]);
const paths = ref([]);

function toggleTheme() {
  theme.value = theme.value === 'dark' ? 'light' : 'dark';
  localStorage.setItem('theme', theme.value);
}

function exportConv() {
  const choice = window.prompt('导出格式: 输入 json 或 md', 'json');
  if (!choice) return;
  const data = choice.toLowerCase().startsWith('m') 
    ? toMarkdown(conversation.value) 
    : JSON.stringify({ items: conversation.value }, null, 2);
  const type = choice.toLowerCase().startsWith('m') ? 'text/markdown' : 'application/json';
  const ext = choice.toLowerCase().startsWith('m') ? 'md' : 'json';
  const blob = new Blob([data], { type });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `conversation.${ext}`;
  a.click();
  URL.revokeObjectURL(url);
}

function toMarkdown(conv) {
  const lines = ['# Conversation Export'];
  for (const m of conv) {
    const time = new Date(m.ts || Date.now()).toISOString();
    lines.push(`\n## ${m.role} | ${time}`, '', m.role === 'assistant' ? (m.content || '') : `> ${m.content || ''}`);
    if (m.sources && m.sources.length) {
      lines.push('\n### sources');
      for (const s of m.sources) lines.push(`- ${s.path} #${s.chunk_id} (score ${Number(s.score).toFixed(4)})`);
    }
  }
  return lines.join('\n');
}

function clearConv() {
  if (!confirm('确认清空会话？')) return;
  conversation.value = [];
  localStorage.setItem('conv', '[]');
  sources.value = [];
}

async function handleNewMessage(msg) {
  conversation.value.push(msg);
  localStorage.setItem('conv', JSON.stringify(conversation.value));
}

async function loadPaths() {
  try {
    const res = await api.listPaths();
    if (res.data.ok) paths.value = res.data.paths;
  } catch (e) {
    console.error(e);
  }
}

onMounted(() => {
  loadPaths();
});
</script>

<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }

.app-container { min-height: 100vh; background: #0b0f17; color: #e6e8eb; }
.app-container.light { background: #f6f7fb; color: #1b1f2a; }

.main-layout { 
  display: grid; 
  grid-template-columns: 260px 1fr 360px; 
  gap: 24px; 
  max-width: 1400px; 
  margin: 0 auto; 
  padding: 24px; 
}
</style>
