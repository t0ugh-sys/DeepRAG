<template>
  <div class="conversation-panel">
    <div class="panel-header">
      <h3>💬 对话历史</h3>
      <button @click="createNewConversation" class="new-btn" title="新建对话">
        �?新对�?
      </button>
    </div>

    <div class="search-bar">
      <input v-model="searchQuery" type="text" placeholder="�����Ի�..." class="search-input" />
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
          <span class="conv-title">{{ conv.title || '新对�? }}</span>
          <button 
            @click.stop="deleteConversation(conv.id)" 
            class="delete-btn"
            title="删除对话"
          >
            🗑�?
          </button>
        </div>
        <div class="conv-meta">
          <span class="conv-time">{{ formatTime(conv.created_at) }}</span>
          <span class="conv-count">{{ conv.message_count }} 条消�?/span>
        </div>
      </div>

      <div v-if="filteredConversations.length === 0" class="empty-state">
        <p>暂无对话历史</p>
        <p class="hint">点击"新对�?开始聊�?/p>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, watch } from 'vue'

import api from '../api'
const emit = defineEmits(['select-conversation', 'new-conversation'])

const loading = ref(false)
const conversations = ref([])
const searchQuery = ref('')
const filteredConversations = computed(() => {
  if (!searchQuery.value.trim()) return conversations.value;
  const q = searchQuery.value.toLowerCase();
  return conversations.value.filter(c => (c.title || '').toLowerCase().includes(q));
});
const activeConversationId = ref(null)

watch(searchQuery, () => {
  loadConversations()
})

const loadConversations = async () => {
  loading.value = true
  try {
    const response = await api.listConversations({
      query: searchQuery.value || undefined
    })
    const result = response.data
    if (result.ok) {
      conversations.value = result.conversations || []
    }
  } catch (err) {
    console.error('���ضԻ��б�ʧ��:', err)
  } finally {
    loading.value = false
  }
}
  } catch (err) {
    console.error('加载对话列表失败:', err)
  } finally {
    loading.value = false
  }
}

const selectConversation = (id) => {
  activeConversationId.value = id
  emit('select-conversation', id)
}

const createNewConversation = () => {
  activeConversationId.value = null
  emit('new-conversation')
}

const deleteConversation = async (id) => {
  if (!confirm('ȷ��Ҫɾ������Ի���')) return
  try {
    const response = await api.deleteConversation(id)
    const result = response.data
    if (result.ok) {
      conversations.value = conversations.value.filter(c => c.id !== id)
      if (activeConversationId.value === id) {
        activeConversationId.value = null
      }
    }
  } catch (err) {
    console.error('ɾ���Ի�ʧ��:', err)
    alert('ɾ��ʧ��')
  }
}

const formatTime = (timestamp) => {
  const date = new Date(timestamp)
  const now = new Date()
  const diff = now - date
  
  // 小于1分钟
  if (diff < 60000) {
    return '刚刚'
  }
  
  // 小于1小时
  if (diff < 3600000) {
    return `${Math.floor(diff / 60000)} 分钟前`
  }
  
  // 小于1�?
  if (diff < 86400000) {
    return `${Math.floor(diff / 3600000)} 小时前`
  }
  
  // 小于7�?
  if (diff < 604800000) {
    return `${Math.floor(diff / 86400000)} 天前`
  }
  
  // 显示日期
  return date.toLocaleDateString('zh-CN', { 
    month: 'short', 
    day: 'numeric' 
  })
}

onMounted(() => {
  loadConversations()
})

// 暴露方法给父组件
defineExpose({
  loadConversations
})
</script>

<style scoped>
.conversation-panel {
  height: 100%;
  display: flex;
  flex-direction: column;
  background: var(--bg-primary);
}

.search-bar {
  padding: 0 16px 8px;
}
.search-input {
  width: 100%;
  padding: 8px 10px;
  border: 1.5px solid var(--border-color);
  border-radius: 8px;
  background: var(--bg-secondary);
  color: var(--text-primary);
  font-size: 13px;
}

.panel-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 16px;
  border-bottom: 1px solid var(--border-color);
}

.panel-header h3 {
  margin: 0;
  font-size: 16px;
}

.new-btn {
  padding: 6px 12px;
  background: var(--primary-color);
  color: white;
  border: none;
  border-radius: 6px;
  cursor: pointer;
  font-size: 13px;
  transition: all 0.2s;
}

.new-btn:hover {
  background: var(--primary-hover);
  transform: translateY(-1px);
}

.loading {
  display: flex;
  justify-content: center;
  align-items: center;
  padding: 40px;
}

.spinner {
  width: 30px;
  height: 30px;
  border: 3px solid var(--border-color);
  border-top-color: var(--primary-color);
  border-radius: 50%;
  animation: spin 1s linear infinite;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}

.conversations-list {
  flex: 1;
  overflow-y: auto;
  padding: 8px;
}

.conversation-item {
  padding: 12px;
  margin-bottom: 8px;
  background: var(--bg-secondary);
  border: 1px solid var(--border-color);
  border-radius: 8px;
  cursor: pointer;
  transition: all 0.2s;
}

.conversation-item:hover {
  background: var(--bg-hover);
  border-color: var(--primary-color);
}

.conversation-item.active {
  background: var(--primary-bg);
  border-color: var(--primary-color);
}

.conv-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 6px;
}

.conv-title {
  font-weight: 500;
  color: var(--text-primary);
  font-size: 14px;
  flex: 1;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.delete-btn {
  background: none;
  border: none;
  cursor: pointer;
  font-size: 16px;
  opacity: 0.6;
  transition: opacity 0.2s;
  padding: 4px;
}

.delete-btn:hover {
  opacity: 1;
}

.conv-meta {
  display: flex;
  justify-content: space-between;
  font-size: 12px;
  color: var(--text-secondary);
}

.conv-time {
  opacity: 0.8;
}

.conv-count {
  opacity: 0.6;
}

.empty-state {
  text-align: center;
  padding: 40px 20px;
  color: var(--text-secondary);
}

.empty-state p {
  margin: 8px 0;
}

.empty-state .hint {
  font-size: 12px;
  opacity: 0.7;
}
</style>
