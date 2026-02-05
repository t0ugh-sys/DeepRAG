<template>
  <div class="modal-overlay" @click.self="$emit('close')">
    <div class="modal-content">
      <div class="modal-header">
        <h3>📚 知识库管理</h3>
        <button class="close-btn" @click="$emit('close')">×</button>
      </div>
      
      <div class="modal-body">
        <!-- 搜索栏 -->
        <div class="search-bar">
          <input 
            v-model="searchQuery" 
            type="text" 
            placeholder="搜索文档..." 
            class="search-input"
          />
          <button class="btn-refresh" @click="loadDocuments" title="刷新">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
              <path d="M23 4v6h-6M1 20v-6h6M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
            </svg>
          </button>
        </div>
        <div class="filter-bar">
          <select v-model="selectedCategory" class="filter-select">
            <option value="">全部分类</option>
            <option v-for="c in categories" :key="c" :value="c">{{ c }}</option>
          </select>
          <select v-model="selectedTag" class="filter-select">
            <option value="">全部标签</option>
            <option v-for="t in tags" :key="t" :value="t">{{ t }}</option>
          </select>
        </div>
        
        <!-- 统计信息 -->
        <div class="stats-bar">
          <div class="stat-item">
            <span class="stat-label">总文档数</span>
            <span class="stat-value">{{ documents.length }}</span>
          </div>
          <div class="stat-item">
            <span class="stat-label">已筛选</span>
            <span class="stat-value">{{ filteredDocuments.length }}</span>
          </div>
        </div>
        
        <!-- 文档列表 -->
        <div v-if="loading" class="loading">加载中...</div>
        
        <div v-else-if="filteredDocuments.length === 0" class="empty-state">
          <svg width="64" height="64" viewBox="0 0 24 24" fill="none">
            <path d="M13 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9z" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
            <polyline points="13 2 13 9 20 9" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
          </svg>
          <p>暂无文档</p>
        </div>
        
        <div v-else class="doc-list">
          <div 
            v-for="doc in filteredDocuments" 
            :key="doc.path"
            class="doc-item"
            :class="{ selected: selectedDoc?.path === doc.path }"
            @click="selectDocument(doc)"
          >
            <div class="doc-icon">
              {{ getFileIcon(doc.path) }}
            </div>
            <div class="doc-info">
              <div class="doc-name">{{ getFileName(doc.path) }}</div>
              <div class="doc-meta">
                <span>{{ doc.chunks }} 个分片</span>
                <span class="separator">·</span>
                <span>{{ doc.path }}</span>
              </div>
            </div>
            <div class="doc-actions">
              <button 
                class="btn-action" 
                @click.stop="previewDocument(doc)"
                title="预览"
              >
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
                  <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                  <circle cx="12" cy="12" r="3" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                </svg>
              </button>
              <button 
                class="btn-action btn-danger" 
                @click.stop="confirmDelete(doc)"
                title="删除"
              >
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
                  <polyline points="3 6 5 6 21 6" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                  <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                </svg>
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
    
    <!-- 预览模态窗口 -->
    <div v-if="previewDoc" class="preview-overlay" @click.self="previewDoc = null">
      <div class="preview-content">
        <div class="preview-header">
          <h4>{{ getFileName(previewDoc.path) }}</h4>
          <button class="close-btn" @click="previewDoc = null">×</button>
        </div>
        <div class="preview-body">
          <div v-if="loadingPreview" class="loading">加载中...</div>
          <div v-else class="chunks-list">
            <div v-for="(chunk, idx) in previewChunks" :key="idx" class="chunk-item">
              <div class="chunk-header">
                <span class="chunk-id">分片 #{{ chunk.chunk_id }}</span>
                <span class="chunk-size">{{ chunk.text?.length || 0 }} 字符</span>
              </div>
              <div class="chunk-text">{{ chunk.text }}</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, watch } from 'vue';
import api from '../api';

const emit = defineEmits(['close', 'refresh']);

const documents = ref([]);
const categories = ref([]);
const tags = ref([]);
const selectedCategory = ref('');
const selectedTag = ref('');
const selectedDoc = ref(null);
const searchQuery = ref('');
const loading = ref(false);
const previewDoc = ref(null);
const previewChunks = ref([]);
const loadingPreview = ref(false);

watch([selectedCategory, selectedTag], () => {
  loadDocuments();
});

const filteredDocuments = computed(() => {
  let list = documents.value;
  if (selectedCategory.value) {
    list = list.filter(doc => doc.category === selectedCategory.value);
  }
  if (selectedTag.value) {
    list = list.filter(doc => (doc.tags || []).includes(selectedTag.value));
  }
  if (!searchQuery.value.trim()) return list;
  const query = searchQuery.value.toLowerCase();
  return list.filter(doc => doc.path.toLowerCase().includes(query));
});

function getFileName(path) {
  return path.split('/').pop() || path;
}

function getFileIcon(path) {
  if (path.endsWith('.md')) return '📄';
  if (path.endsWith('.txt')) return '📃';
  if (path.endsWith('.pdf')) return '📕';
  return '📄';
}

function selectDocument(doc) {
  selectedDoc.value = doc;
}

async function loadDocuments() {
  loading.value = true;
  try {
    const res = await api.listDocuments({
      category: selectedCategory.value || undefined,
      tags: selectedTag.value || undefined
    });
    if (res.data.ok) {
      const docs = res.data.data?.documents || [];
      documents.value = docs.map(doc => ({
        path: doc.path,
        chunks: doc.chunk_count || 0,
        lastUpdated: doc.last_updated || new Date().toISOString(),
        tags: doc.tags || [],
        category: doc.category || ''
      }));
    }
  } catch (e) {
    console.error('加载文档列表失败:', e);
  } finally {
    loading.value = false;
  }
}

async function loadFilters() {
  try {
    const [tagsRes, categoriesRes] = await Promise.all([api.listTags(), api.listCategories()]);
    if (tagsRes.data.ok) {
      tags.value = tagsRes.data.data?.tags || [];
    }
    if (categoriesRes.data.ok) {
      categories.value = categoriesRes.data.data?.categories || [];
    }
  } catch (e) {
    console.error('加载文档筛选项失败:', e);
  }
}

async function previewDocument(doc) {
  previewDoc.value = doc;
  loadingPreview.value = true;
  previewChunks.value = [];
  
  try {
    const res = await api.exportPath(doc.path);
    if (res.data.ok) {
      previewChunks.value = res.data.chunks || [];
    }
  } catch (e) {
    console.error('加载文档预览失败:', e);
  } finally {
    loadingPreview.value = false;
  }
}

async function confirmDelete(doc) {
  if (!confirm(`确认删除文档 "${getFileName(doc.path)}"？\n\n这将删除该文档的所有 ${doc.chunks} 个分片。`)) {
    return;
  }
  
  try {
    const res = await api.deleteDoc(doc.path);
    if (res.data.ok) {
      // 从列表中移除
      const index = documents.value.findIndex(d => d.path === doc.path);
      if (index !== -1) {
        documents.value.splice(index, 1);
      }
      
      if (selectedDoc.value?.path === doc.path) {
        selectedDoc.value = null;
      }
      
      emit('refresh');
    } else {
      alert(`删除失败: ${res.data.error}`);
    }
  } catch (e) {
    alert(`删除失败: ${e.message}`);
  }
}

onMounted(async () => {
  await Promise.all([loadDocuments(), loadFilters()]);
});
</script>

<style scoped>
.modal-overlay {
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
  backdrop-filter: blur(4px);
  animation: fadeIn 0.15s ease-out;
}

@keyframes fadeIn {
  from { opacity: 0; }
  to { opacity: 1; }
}

.modal-content {
  background: var(--bg-primary);
  border-radius: 16px;
  box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
  max-width: 800px;
  width: 90%;
  max-height: 85vh;
  display: flex;
  flex-direction: column;
  overflow: hidden;
  animation: slideUp 0.2s ease-out;
}

@keyframes slideUp {
  from {
    transform: translateY(20px);
    opacity: 0;
  }
  to {
    transform: translateY(0);
    opacity: 1;
  }
}

.modal-header {
  padding: 20px 24px;
  border-bottom: 1px solid var(--border-primary);
  display: flex;
  justify-content: space-between;
  align-items: center;
  background: var(--bg-secondary);
}

.modal-header h3 {
  font-size: 18px;
  font-weight: 600;
  color: var(--text-primary);
  margin: 0;
}

.close-btn {
  background: transparent;
  border: none;
  font-size: 24px;
  color: var(--text-secondary);
  cursor: pointer;
  transition: color 0.2s;
  padding: 0;
  width: 32px;
  height: 32px;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 6px;
}

.close-btn:hover {
  color: var(--text-primary);
  background: var(--bg-tertiary);
}

.modal-body {
  padding: 20px 24px;
  overflow-y: auto;
  flex: 1;
}

.search-bar {
  display: flex;
  gap: 8px;
  margin-bottom: 16px;
}

.search-input {
  flex: 1;
  padding: 10px 14px;
  border: 1.5px solid var(--border-primary);
  border-radius: 8px;
  font-size: 14px;
  transition: all 0.2s;
  background: var(--bg-primary);
  color: var(--text-primary);
}

.search-input:focus {
  outline: none;
  border-color: var(--accent-primary);
  box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
}

.btn-refresh {
  padding: 10px;
  background: var(--bg-secondary);
  border: 1.5px solid var(--border-primary);
  border-radius: 8px;
  cursor: pointer;
  transition: all 0.2s;
  display: flex;
  align-items: center;
  justify-content: center;
  color: var(--text-secondary);
}

.btn-refresh:hover {
  background: var(--bg-primary);
  color: var(--text-primary);
}

.filter-bar {
  display: flex;
  gap: 8px;
  margin-bottom: 12px;
}
.filter-select {
  flex: 1;
  padding: 8px 10px;
  border: 1.5px solid var(--border-primary);
  border-radius: 8px;
  background: var(--bg-primary);
  color: var(--text-primary);
  font-size: 13px;
}

.stats-bar {
  display: flex;
  gap: 16px;
  padding: 12px 16px;
  background: var(--bg-secondary);
  border-radius: 8px;
  margin-bottom: 16px;
}

.stat-item {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.stat-label {
  font-size: 12px;
  color: var(--text-secondary);
  font-weight: 500;
}

.stat-value {
  font-size: 20px;
  font-weight: 700;
  color: var(--text-primary);
}

.loading {
  text-align: center;
  padding: 40px;
  color: var(--text-secondary);
  font-size: 14px;
}

.empty-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 60px 20px;
  color: var(--text-tertiary);
}

.empty-state svg {
  margin-bottom: 16px;
  opacity: 0.5;
}

.empty-state p {
  font-size: 14px;
  margin: 0;
}

.doc-list {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.doc-item {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 12px 16px;
  background: var(--bg-primary);
  border: 1.5px solid var(--border-primary);
  border-radius: 10px;
  cursor: pointer;
  transition: all 0.2s;
}

.doc-item:hover {
  border-color: var(--accent-primary);
  background: var(--bg-secondary);
  box-shadow: 0 2px 8px rgba(59, 130, 246, 0.1);
}

.doc-item.selected {
  border-color: var(--accent-primary);
  background: var(--bg-secondary);
}

.doc-icon {
  font-size: 24px;
  flex-shrink: 0;
}

.doc-info {
  flex: 1;
  min-width: 0;
}

.doc-name {
  font-size: 14px;
  font-weight: 600;
  color: var(--text-primary);
  margin-bottom: 4px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.doc-meta {
  font-size: 12px;
  color: var(--text-secondary);
  display: flex;
  align-items: center;
  gap: 6px;
}

.separator {
  color: var(--border-secondary);
}

.doc-actions {
  display: flex;
  gap: 4px;
  opacity: 0;
  transition: opacity 0.2s;
}

.doc-item:hover .doc-actions {
  opacity: 1;
}

.btn-action {
  padding: 8px;
  background: transparent;
  border: 1px solid var(--border-primary);
  border-radius: 6px;
  cursor: pointer;
  transition: all 0.2s;
  display: flex;
  align-items: center;
  justify-content: center;
  color: var(--text-secondary);
}

.btn-action:hover {
  background: var(--bg-secondary);
  border-color: var(--border-secondary);
  color: var(--text-primary);
}

.btn-action.btn-danger:hover {
  background: rgba(239, 68, 68, 0.15);
  border-color: var(--danger);
  color: var(--danger);
}

/* 预览窗口样式 */
.preview-overlay {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0, 0, 0, 0.6);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1001;
  backdrop-filter: blur(4px);
}

.preview-content {
  background: #ffffff;
  border-radius: 16px;
  box-shadow: 0 20px 60px rgba(0, 0, 0, 0.4);
  max-width: 900px;
  width: 90%;
  max-height: 90vh;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

.preview-header {
  padding: 20px 24px;
  border-bottom: 1px solid var(--border-primary);
  display: flex;
  justify-content: space-between;
  align-items: center;
  background: var(--bg-secondary);
}

.preview-header h4 {
  font-size: 16px;
  font-weight: 600;
  color: var(--text-primary);
  margin: 0;
}

.preview-body {
  padding: 20px 24px;
  overflow-y: auto;
  flex: 1;
}

.chunks-list {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.chunk-item {
  border: 1px solid #e5e7eb;
  border-radius: 10px;
  overflow: hidden;
  transition: all 0.2s;
}

.chunk-item:hover {
  border-color: var(--border-secondary);
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
}

.chunk-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 10px 14px;
  background: var(--bg-secondary);
  border-bottom: 1px solid var(--border-primary);
}

.chunk-id {
  font-size: 13px;
  font-weight: 600;
  color: var(--accent-primary);
}

.chunk-size {
  font-size: 12px;
  color: var(--text-secondary);
}

.chunk-text {
  padding: 14px;
  font-size: 14px;
  line-height: 1.6;
  color: var(--text-primary);
  white-space: pre-wrap;
  word-wrap: break-word;
}
</style>



