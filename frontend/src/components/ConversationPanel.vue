<template>
  <div class="conversation-panel">
    <div class="panel-header">
      <h3>会话列表 / Conversations</h3>
      <button @click="createNewConversation" class="new-btn" title="新建对话 / New conversation">
        ＋ 新建对话 / New conversation
      </button>
    </div>

    <div class="search-bar">
      <input v-model="searchQuery" type="text" placeholder="搜索对话... / Search" class="search-input" />
    </div>

    <div v-if="loading" class="loading">
      <div class="spinner"></div>
    </div>

    <div v-else class="conversations-list">
      <div
        v-for="conv in filteredConversations"
        :key="conv.id"
        class="conversation-item"
        :class="{ active: conv.id === activeConversationId }"
        @click="selectConversation(conv.id)"
      >
        <div class="conv-header">
          <span class="conv-title">{{ conv.title || '未命名对话 / Untitled' }}</span>
          <button
            @click.stop="deleteConversation(conv.id)"
            class="delete-btn"
            title="删除对话 / Delete"
          >
            删除 / Delete
          </button>
        </div>
        <div class="conv-meta">
          <span class="conv-time">{{ formatTime(conv.created_at) }}</span>
          <span class="conv-count">{{ conv.message_count }} 条消息 / messages</span>
        </div>
      </div>

      <div v-if="filteredConversations.length === 0" class="empty-state">
        <p>暂无会话 / No conversations</p>
        <p class="hint">点击“新建对话 / New conversation”开始</p>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, watch } from 'vue';

import api from '../api';

const emit = defineEmits(['select-conversation', 'new-conversation']);

const loading = ref(false);
const conversations = ref([]);
const searchQuery = ref('');
const activeConversationId = ref(null);

const filteredConversations = computed(() => {
  if (!searchQuery.value.trim()) return conversations.value;
  const q = searchQuery.value.toLowerCase();
  return conversations.value.filter((c) => (c.title || '').toLowerCase().includes(q));
});

watch(searchQuery, () => {
  loadConversations();
});

const loadConversations = async () => {
  loading.value = true;
  try {
    const response = await api.listConversations({
      query: searchQuery.value || undefined,
    });
    const result = response.data;
    if (result.ok) {
      conversations.value = result.conversations || [];
    }
  } catch (err) {
    console.error('加载会话失败 / Failed to load conversations:', err);
  } finally {
    loading.value = false;
  }
};

const selectConversation = (id) => {
  activeConversationId.value = id;
  emit('select-conversation', id);
};

const createNewConversation = () => {
  activeConversationId.value = null;
  emit('new-conversation');
};

const deleteConversation = async (id) => {
  if (!confirm('确定删除该会话？/ Delete this conversation?')) return;
  try {
    const response = await api.deleteConversation(id);
    const result = response.data;
    if (result.ok) {
      conversations.value = conversations.value.filter((c) => c.id !== id);
      if (activeConversationId.value === id) {
        activeConversationId.value = null;
      }
    }
  } catch (err) {
    console.error('删除会话失败 / Failed to delete conversation:', err);
    alert('删除失败 / Delete failed');
  }
};

const formatTime = (timestamp) => {
  const date = new Date(timestamp);
  const now = new Date();
  const diff = now - date;

  if (diff < 60000) {
    return '刚刚 / Just now';
  }

  if (diff < 3600000) {
    return `${Math.floor(diff / 60000)} 分钟前 / min ago`;
  }

  if (diff < 86400000) {
    return `${Math.floor(diff / 3600000)} 小时前 / hours ago`;
  }

  if (diff < 604800000) {
    return `${Math.floor(diff / 86400000)} 天前 / days ago`;
  }

  return date.toLocaleDateString('zh-CN', {
    month: 'short',
    day: 'numeric',
  });
};

onMounted(() => {
  loadConversations();
});

// Expose for parent refresh / 暴露给父组件刷新
defineExpose({
  loadConversations,
});
</script>
