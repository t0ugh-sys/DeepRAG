<template>
  <header class="app-header">
    <div class="header-actions">
      <button class="icon-btn" @click="$emit('import-docs')" title="导入文档">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
          <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4M17 8l-5-5-5 5M12 3v12" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
      </button>
      <button class="icon-btn" @click="$emit('clear-conv')" title="清空当前对话">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
          <path d="M3 6h18M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
      </button>
      <button class="icon-btn" @click="$emit('export-conv')" title="导出对话">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
          <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4M7 10l5 5 5-5M12 15V3" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
      </button>
      <div class="divider"></div>
      <div class="status-badge" :class="statusClass">
        <span class="status-dot"></span>
        {{ statusText }}
      </div>
    </div>
  </header>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue';
import api from '../api';

const status = ref('checking');
const docCount = ref(0);

const statusClass = computed(() => {
  if (status.value === 'ok') return 'status-ok';
  if (status.value === 'offline') return 'status-offline';
  return 'status-checking';
});

const statusText = computed(() => {
  if (status.value === 'ok') return `已连接 · ${docCount.value} 文档`;
  if (status.value === 'offline') return '未连接';
  return '检查中...';
});

async function checkHealth() {
  try {
    const res = await api.healthz();
    if (res.data.ok) {
      status.value = 'ok';
      const d = res.data.details;
      docCount.value = d.milvus_entities || 0;
    } else {
      status.value = 'offline';
    }
  } catch {
    status.value = 'offline';
  }
}

onMounted(() => {
  checkHealth();
  // 每30秒检查一次健康状态
  setInterval(checkHealth, 30000);
});
</script>

<style scoped>
.app-header {
  display: flex;
  align-items: center;
  justify-content: flex-end;
  padding: 12px 24px;
  border-bottom: 1px solid #f0f0f0;
  background: #ffffff;
}

.header-actions {
  display: flex;
  align-items: center;
  gap: 8px;
}

.icon-btn {
  width: 36px;
  height: 36px;
  border-radius: 8px;
  border: none;
  background: transparent;
  color: #6b7280;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all 0.15s;
}

.icon-btn:hover {
  background: #f3f4f6;
  color: #111827;
}

.divider {
  width: 1px;
  height: 20px;
  background: #e5e7eb;
  margin: 0 4px;
}

.status-badge {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 6px 12px;
  border-radius: 6px;
  font-size: 13px;
  font-weight: 500;
  transition: all 0.2s;
}

.status-dot {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  transition: all 0.2s;
}

.status-ok {
  background: #f0fdf4;
  color: #16a34a;
}

.status-ok .status-dot {
  background: #16a34a;
  box-shadow: 0 0 0 2px rgba(22, 163, 74, 0.2);
}

.status-offline {
  background: #fef2f2;
  color: #dc2626;
}

.status-offline .status-dot {
  background: #dc2626;
  box-shadow: 0 0 0 2px rgba(220, 38, 38, 0.2);
}

.status-checking {
  background: #f7f8fa;
  color: #6b7280;
}

.status-checking .status-dot {
  background: #9ca3af;
  animation: pulse 2s ease-in-out infinite;
}

@keyframes pulse {
  0%, 100% {
    opacity: 1;
  }
  50% {
    opacity: 0.5;
  }
}
</style>

