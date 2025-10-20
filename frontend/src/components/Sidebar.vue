<template>
  <aside class="sidebar">
    <div class="sidebar-header">
      <div class="logo">
        <svg width="28" height="28" viewBox="0 0 24 24" fill="none">
          <rect x="3" y="3" width="18" height="18" rx="3" stroke="currentColor" stroke-width="2"/>
          <path d="M8 12h8M12 8v8" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
        </svg>
        <span class="logo-text">DeepLearning</span>
      </div>
    </div>
    
    <button class="new-chat-btn" @click="$emit('new-conversation')">
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
        <path d="M12 5v14M5 12h14" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
      </svg>
      <span>新建对话</span>
    </button>
    
    <div class="conv-list">
      <div class="conv-section-title">最近对话</div>
      <div 
        v-for="conv in conversations" 
        :key="conv.id"
        class="conv-item"
        :class="{ active: conv.id === currentConvId }"
        @click="$emit('select-conversation', conv.id)"
      >
        <svg class="conv-icon" width="16" height="16" viewBox="0 0 24 24" fill="none">
          <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
        <span class="conv-title">{{ conv.title }}</span>
        <span class="conv-time">{{ formatTime(conv.createdAt) }}</span>
        <button 
          class="delete-btn" 
          @click.stop="$emit('delete-conversation', conv.id)"
          title="删除对话"
        >
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none">
            <path d="M3 6h18M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2M10 11v6M14 11v6" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
          </svg>
        </button>
      </div>
    </div>
  </aside>
</template>

<script setup>
defineProps(['conversations', 'currentConvId']);
defineEmits(['new-conversation', 'select-conversation', 'delete-conversation']);

function formatTime(ts) {
  const date = new Date(ts);
  const now = new Date();
  const diff = now - date;
  const days = Math.floor(diff / (1000 * 60 * 60 * 24));
  
  if (days === 0) return '今天';
  if (days === 1) return '昨天';
  if (days < 7) return `${days}天前`;
  return date.toLocaleDateString('zh-CN', { month: 'numeric', day: 'numeric' });
}
</script>

<style scoped>
.sidebar {
  width: 260px;
  background: var(--bg-primary);
  border-right: 1px solid var(--border-primary);
  display: flex;
  flex-direction: column;
  height: 100vh;
  flex-shrink: 0;
}

.sidebar-header { 
  padding: 16px 20px;
  border-bottom: 1px solid var(--border-primary);
}

.logo {
  display: flex;
  align-items: center;
  gap: 10px;
  color: var(--text-primary);
}

.logo-text {
  font-size: 18px;
  font-weight: 600;
  letter-spacing: -0.02em;
}

.new-chat-btn {
  margin: 12px 16px;
  padding: 10px 16px;
  background: var(--bg-secondary);
  border: 1px solid var(--border-primary);
  border-radius: 8px;
  display: flex;
  align-items: center;
  gap: 8px;
  cursor: pointer;
  font-size: 14px;
  font-weight: 500;
  color: var(--text-primary);
  transition: all 0.2s;
  width: calc(100% - 32px);
}

.new-chat-btn:hover {
  background: var(--bg-primary);
  border-color: var(--border-secondary);
  box-shadow: 0 1px 3px rgba(0,0,0,0.08);
}

.conv-list {
  flex: 1;
  overflow-y: auto;
  padding: 8px 16px;
}

.conv-list::-webkit-scrollbar {
  width: 4px;
}

.conv-list::-webkit-scrollbar-thumb {
  background: var(--border-secondary);
  border-radius: 2px;
}

.conv-section-title {
  font-size: 12px;
  font-weight: 600;
  color: var(--text-tertiary);
  text-transform: uppercase;
  letter-spacing: 0.05em;
  padding: 8px 12px 8px;
  margin-bottom: 4px;
}

.conv-item {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 10px 12px;
  border-radius: 8px;
  cursor: pointer;
  transition: all 0.15s;
  margin-bottom: 2px;
  color: var(--text-secondary);
  position: relative;
}

.conv-item:hover {
  background: var(--bg-secondary);
  color: var(--text-primary);
}

.conv-item.active {
  background: var(--bg-tertiary);
  color: var(--text-primary);
  font-weight: 500;
}

.conv-icon {
  flex-shrink: 0;
  opacity: 0.7;
}

.conv-title {
  flex: 1;
  font-size: 14px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  line-height: 1.4;
}

.conv-time {
  font-size: 11px;
  color: var(--text-tertiary);
  flex-shrink: 0;
  transition: opacity 0.2s;
}

.conv-item.active .conv-time {
  color: var(--text-secondary);
}

.delete-btn {
  width: 24px;
  height: 24px;
  border-radius: 4px;
  border: none;
  background: transparent;
  color: var(--text-tertiary);
  cursor: pointer;
  display: none;
  align-items: center;
  justify-content: center;
  transition: all 0.15s;
  flex-shrink: 0;
  padding: 0;
}

.conv-item:hover .delete-btn {
  display: flex;
}

.conv-item:hover .conv-time {
  display: none;
}

.delete-btn:hover {
  background: rgba(220,38,38,0.1);
  color: var(--danger);
}
</style>

