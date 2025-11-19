<template>
  <div class="chunk-visualizer">
    <div class="visualizer-header">
      <h3>📊 文档分块可视化</h3>
      <button @click="$emit('close')" class="close-btn">✕</button>
    </div>

    <div v-if="loading" class="loading">
      <div class="spinner"></div>
      <p>加载中...</p>
    </div>

    <div v-else-if="error" class="error">
      <p>{{ error }}</p>
      <button @click="loadChunks" class="retry-btn">重试</button>
    </div>

    <div v-else-if="data" class="visualizer-content">
      <!-- 统计信息 -->
      <div class="stats-panel">
        <div class="stat-card">
          <div class="stat-label">总分块数</div>
          <div class="stat-value">{{ data.stats.total_chunks }}</div>
        </div>
        <div class="stat-card">
          <div class="stat-label">总字符数</div>
          <div class="stat-value">{{ data.stats.total_chars.toLocaleString() }}</div>
        </div>
        <div class="stat-card">
          <div class="stat-label">平均分块大小</div>
          <div class="stat-value">{{ Math.round(data.stats.avg_chunk_size) }}</div>
        </div>
        <div class="stat-card">
          <div class="stat-label">文档类型</div>
          <div class="stat-value">{{ data.stats.doc_type || 'text' }}</div>
        </div>
      </div>

      <!-- 分块大小分布图 -->
      <div class="chart-panel">
        <h4>分块大小分布</h4>
        <div class="chart">
          <div 
            v-for="(chunk, index) in data.chunks" 
            :key="chunk.chunk_id"
            class="chart-bar"
            :style="{ 
              height: `${(chunk.char_count / data.stats.max_chunk_size) * 100}%`,
              backgroundColor: getChunkColor(chunk)
            }"
            :title="`分块 #${chunk.chunk_id}: ${chunk.char_count} 字符`"
          >
          </div>
        </div>
        <div class="chart-legend">
          <span class="legend-item">
            <span class="legend-color" style="background: #10b981"></span>
            普通文本
          </span>
          <span class="legend-item">
            <span class="legend-color" style="background: #f59e0b"></span>
            包含表格
          </span>
        </div>
      </div>

      <!-- 分块列表 -->
      <div class="chunks-list">
        <h4>分块详情 ({{ data.chunks.length }})</h4>
        <div class="chunks-container">
          <div 
            v-for="chunk in data.chunks" 
            :key="chunk.chunk_id"
            class="chunk-item"
            :class="{ 'has-table': chunk.has_tables }"
          >
            <div class="chunk-header">
              <span class="chunk-id">#{{ chunk.chunk_id }}</span>
              <span v-if="chunk.page" class="chunk-page">第 {{ chunk.page }} 页</span>
              <span v-if="chunk.has_tables" class="chunk-badge">📋 表格</span>
              <span class="chunk-size">{{ chunk.char_count }} 字符</span>
            </div>
            <div class="chunk-text">
              {{ chunk.text }}
            </div>
            <div class="chunk-footer">
              <span>字数: {{ chunk.word_count }}</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'

const props = defineProps({
  documentPath: {
    type: String,
    required: true
  }
})

const emit = defineEmits(['close'])

const loading = ref(true)
const error = ref(null)
const data = ref(null)

const loadChunks = async () => {
  loading.value = true
  error.value = null
  
  try {
    const response = await fetch('http://localhost:8000/visualize_chunks', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        path: props.documentPath
      })
    })
    
    const result = await response.json()
    
    if (result.ok) {
      data.value = result.data
    } else {
      error.value = result.message || '加载失败'
    }
  } catch (err) {
    error.value = '网络错误: ' + err.message
  } finally {
    loading.value = false
  }
}

const getChunkColor = (chunk) => {
  return chunk.has_tables ? '#f59e0b' : '#10b981'
}

onMounted(() => {
  loadChunks()
})
</script>

<style scoped>
.chunk-visualizer {
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
  padding: 20px;
}

.visualizer-content {
  background: var(--bg-primary);
  border-radius: 12px;
  max-width: 1200px;
  width: 100%;
  max-height: 90vh;
  overflow-y: auto;
  box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
}

.visualizer-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 20px;
  border-bottom: 1px solid var(--border-color);
  position: sticky;
  top: 0;
  background: var(--bg-primary);
  z-index: 10;
}

.visualizer-header h3 {
  margin: 0;
  font-size: 20px;
}

.close-btn {
  background: none;
  border: none;
  font-size: 24px;
  cursor: pointer;
  color: var(--text-secondary);
  padding: 0;
  width: 32px;
  height: 32px;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 4px;
  transition: all 0.2s;
}

.close-btn:hover {
  background: var(--bg-secondary);
  color: var(--text-primary);
}

.loading, .error {
  text-align: center;
  padding: 60px 20px;
}

.spinner {
  width: 40px;
  height: 40px;
  border: 4px solid var(--border-color);
  border-top-color: var(--primary-color);
  border-radius: 50%;
  animation: spin 1s linear infinite;
  margin: 0 auto 20px;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}

.error {
  color: #ef4444;
}

.retry-btn {
  margin-top: 16px;
  padding: 8px 16px;
  background: var(--primary-color);
  color: white;
  border: none;
  border-radius: 6px;
  cursor: pointer;
}

.stats-panel {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 16px;
  padding: 20px;
}

.stat-card {
  background: var(--bg-secondary);
  padding: 16px;
  border-radius: 8px;
  text-align: center;
}

.stat-label {
  font-size: 12px;
  color: var(--text-secondary);
  margin-bottom: 8px;
}

.stat-value {
  font-size: 24px;
  font-weight: 600;
  color: var(--primary-color);
}

.chart-panel {
  padding: 20px;
  border-top: 1px solid var(--border-color);
}

.chart-panel h4 {
  margin: 0 0 16px 0;
  font-size: 16px;
}

.chart {
  display: flex;
  align-items: flex-end;
  gap: 2px;
  height: 150px;
  background: var(--bg-secondary);
  padding: 10px;
  border-radius: 8px;
  overflow-x: auto;
}

.chart-bar {
  flex: 1;
  min-width: 8px;
  border-radius: 2px 2px 0 0;
  transition: all 0.2s;
  cursor: pointer;
}

.chart-bar:hover {
  opacity: 0.8;
  transform: translateY(-2px);
}

.chart-legend {
  display: flex;
  gap: 16px;
  margin-top: 12px;
  font-size: 12px;
}

.legend-item {
  display: flex;
  align-items: center;
  gap: 6px;
}

.legend-color {
  width: 16px;
  height: 16px;
  border-radius: 3px;
}

.chunks-list {
  padding: 20px;
  border-top: 1px solid var(--border-color);
}

.chunks-list h4 {
  margin: 0 0 16px 0;
  font-size: 16px;
}

.chunks-container {
  display: flex;
  flex-direction: column;
  gap: 12px;
  max-height: 500px;
  overflow-y: auto;
}

.chunk-item {
  background: var(--bg-secondary);
  border: 1px solid var(--border-color);
  border-left: 3px solid #10b981;
  border-radius: 6px;
  padding: 12px;
  transition: all 0.2s;
}

.chunk-item.has-table {
  border-left-color: #f59e0b;
}

.chunk-item:hover {
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

.chunk-header {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 8px;
  flex-wrap: wrap;
}

.chunk-id {
  font-weight: 600;
  color: var(--primary-color);
}

.chunk-page {
  font-size: 12px;
  color: var(--text-secondary);
  background: var(--bg-primary);
  padding: 2px 8px;
  border-radius: 4px;
}

.chunk-badge {
  font-size: 12px;
  background: #fef3c7;
  color: #92400e;
  padding: 2px 8px;
  border-radius: 4px;
}

.chunk-size {
  font-size: 12px;
  color: var(--text-secondary);
  margin-left: auto;
}

.chunk-text {
  font-size: 14px;
  line-height: 1.6;
  color: var(--text-primary);
  white-space: pre-wrap;
  word-break: break-word;
  max-height: 200px;
  overflow-y: auto;
  padding: 8px;
  background: var(--bg-primary);
  border-radius: 4px;
}

.chunk-footer {
  margin-top: 8px;
  font-size: 12px;
  color: var(--text-secondary);
}
</style>
