<template>
  <div class="app-container">
    <Sidebar 
      :conversations="conversations"
      :currentConvId="currentConvId"
      @new-conversation="newConversation"
      @select-conversation="selectConversation"
      @delete-conversation="deleteConversation"
    />
    
    <div class="main-content">
      <AppHeader 
        @toggle-theme="toggleTheme"
        @export-conv="exportConv"
        @clear-conv="clearConv"
        @import-docs="showImportModal = true"
        @manage-knowledge="showKnowledgeManager = true"
        @open-settings="showSettings = true"
      />
      
      <ChatView 
        :conversation="currentConversation"
        @new-message="handleNewMessage"
      />
    </div>
    
    <!-- 导入文档模态窗口 -->
    <ImportModal 
      v-if="showImportModal"
      @close="showImportModal = false"
      @refresh="loadPaths"
    />
    
    <!-- 知识库管理窗口 -->
    <KnowledgeManager 
      v-if="showKnowledgeManager"
      @close="showKnowledgeManager = false"
      @refresh="loadPaths"
    />
    
    <!-- 设置窗口 -->
    <SettingsModal 
      v-if="showSettings"
      @close="showSettings = false"
      @settings-changed="handleSettingsChanged"
    />
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue';
import AppHeader from './components/AppHeader.vue';
import Sidebar from './components/Sidebar.vue';
import ChatView from './components/ChatView.vue';
import ImportModal from './components/ImportModal.vue';
import KnowledgeManager from './components/KnowledgeManager.vue';
import SettingsModal from './components/SettingsModal.vue';
import api from './api';

const theme = ref(localStorage.getItem('theme') || 'light');
const conversations = ref(JSON.parse(localStorage.getItem('conversations') || '[]'));
const currentConvId = ref(localStorage.getItem('currentConvId') || null);
const sources = ref([]);
const paths = ref([]);
const showImportModal = ref(false);
const showKnowledgeManager = ref(false);
const showSettings = ref(false);

// 如果没有会话，创建一个默认会话
if (conversations.value.length === 0) {
  const newConv = {
    id: Date.now().toString(),
    title: '新对话',
    messages: [],
    createdAt: Date.now()
  };
  conversations.value.push(newConv);
  currentConvId.value = newConv.id;
  saveConversations();
}

const currentConversation = computed(() => {
  const conv = conversations.value.find(c => c.id === currentConvId.value);
  return conv ? conv.messages : [];
});

function saveConversations() {
  localStorage.setItem('conversations', JSON.stringify(conversations.value));
  localStorage.setItem('currentConvId', currentConvId.value);
}

function toggleTheme() {
  theme.value = theme.value === 'dark' ? 'light' : 'dark';
  localStorage.setItem('theme', theme.value);
  document.documentElement.setAttribute('data-theme', theme.value);
}

function newConversation() {
  const newConv = {
    id: Date.now().toString(),
    title: '新对话',
    messages: [],
    createdAt: Date.now()
  };
  conversations.value.unshift(newConv);
  currentConvId.value = newConv.id;
  saveConversations();
}

function selectConversation(id) {
  currentConvId.value = id;
  localStorage.setItem('currentConvId', id);
}

function deleteConversation(id) {
  if (!confirm('确认删除该对话？')) return;
  
  const index = conversations.value.findIndex(c => c.id === id);
  if (index === -1) return;
  
  conversations.value.splice(index, 1);
  
  // 如果删除的是当前对话，切换到第一个对话或创建新对话
  if (currentConvId.value === id) {
    if (conversations.value.length > 0) {
      currentConvId.value = conversations.value[0].id;
    } else {
      // 创建新对话
      newConversation();
    }
  }
  
  saveConversations();
}

function exportConv() {
  const conv = conversations.value.find(c => c.id === currentConvId.value);
  if (!conv) return;
  
  const choice = window.prompt('导出格式: 输入 json 或 md', 'json');
  if (!choice) return;
  const data = choice.toLowerCase().startsWith('m') 
    ? toMarkdown(conv.messages) 
    : JSON.stringify({ title: conv.title, messages: conv.messages }, null, 2);
  const type = choice.toLowerCase().startsWith('m') ? 'text/markdown' : 'application/json';
  const ext = choice.toLowerCase().startsWith('m') ? 'md' : 'json';
  const blob = new Blob([data], { type });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `conversation-${conv.id}.${ext}`;
  a.click();
  URL.revokeObjectURL(url);
}

function toMarkdown(messages) {
  const lines = ['# 对话导出'];
  for (const m of messages) {
    const time = new Date(m.ts || Date.now()).toLocaleString();
    lines.push(`\n## ${m.role === 'user' ? '用户' : '助手'} | ${time}`, '', m.content || '');
    if (m.sources && m.sources.length) {
      lines.push('\n### 来源');
      for (const s of m.sources) lines.push(`- ${s.path} #${s.chunk_id} (score ${Number(s.score).toFixed(4)})`);
    }
  }
  return lines.join('\n');
}

function clearConv() {
  if (!confirm('确认清空当前会话？')) return;
  const conv = conversations.value.find(c => c.id === currentConvId.value);
  if (conv) {
    conv.messages = [];
    conv.title = '新对话';
    saveConversations();
  }
  sources.value = [];
}

async function handleNewMessage(msg) {
  const conv = conversations.value.find(c => c.id === currentConvId.value);
  if (conv) {
    conv.messages.push(msg);
    // 更新会话标题（使用第一条用户消息）
    if (conv.title === '新对话' && msg.role === 'user') {
      conv.title = msg.content.slice(0, 30) + (msg.content.length > 30 ? '...' : '');
    }
    saveConversations();
  }
}

async function loadPaths() {
  try {
    const res = await api.listPaths();
    if (res.data.ok) paths.value = res.data.paths;
  } catch (e) {
    console.error(e);
  }
}

function handleSettingsChanged(newSettings) {
  // 处理设置变更
  console.log('设置已更新:', newSettings);
}

onMounted(() => {
  loadPaths();
  // 初始化主题
  document.documentElement.setAttribute('data-theme', theme.value);
});
</script>

<style>
.app-container { 
  display: flex;
  width: 100%;
  height: 100vh;
  background: var(--bg-secondary);
  overflow: hidden;
}

.main-content {
  flex: 1;
  display: flex;
  flex-direction: column;
  width: 100%;
  overflow: hidden;
  background: var(--bg-secondary);
}
</style>
