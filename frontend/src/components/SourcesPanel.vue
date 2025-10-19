<template>
  <aside class="sources-panel">
    <div class="aside-header">
      <h3>引用片段</h3>
      <button class="btn subtle" @click="collapsed = !collapsed">{{ collapsed ? '展开' : '折叠' }}</button>
    </div>
    
    <div v-if="!collapsed" id="sources"></div>
    
    <div v-if="!collapsed" class="doc-list">
      <div style="display:flex;align-items:center;gap:8px;margin:8px 0;">
        <strong class="muted">已入库路径</strong>
        <button class="btn" @click="$emit('refresh-paths')">刷新</button>
      </div>
      <ul>
        <li v-for="p in paths" :key="p">{{ p }}</li>
      </ul>
    </div>
    
    <h3 v-if="!collapsed" style="margin-top:16px">文档管理</h3>
    
    <!-- 批量导入本地知识库 -->
    <div v-if="!collapsed" class="batch-upload">
      <h4>📁 批量导入本地知识库</h4>
      <p class="muted">支持同时选择多个 .md 和 .txt 文件</p>
      <div class="upload-area" @click="triggerBatchUpload" @drop.prevent="handleDrop" @dragover.prevent>
        <input 
          ref="batchFileInput" 
          type="file" 
          multiple 
          accept=".txt,.md" 
          @change="handleBatchFiles"
          style="display:none"
        />
        <div class="upload-prompt">
          <span class="upload-icon">📂</span>
          <p>点击选择文件或拖拽文件到此处</p>
          <p class="muted small">支持 .md / .txt 格式</p>
        </div>
      </div>
      
      <!-- 文件列表 -->
      <div v-if="batchFiles.length > 0" class="file-list">
        <div class="file-item" v-for="(file, idx) in batchFiles" :key="idx">
          <span class="file-icon">{{ file.name.endsWith('.md') ? '📝' : '📄' }}</span>
          <span class="file-name">{{ file.name }}</span>
          <span class="file-size">{{ formatFileSize(file.size) }}</span>
          <button class="btn-remove" @click="removeFile(idx)">✕</button>
        </div>
      </div>
      
      <div v-if="batchFiles.length > 0" class="batch-actions">
        <button class="btn primary" @click="uploadBatch">
          上传 {{ batchFiles.length }} 个文件
        </button>
        <button class="btn" @click="clearBatch">清空</button>
      </div>
      
      <div v-if="batchMsg" class="batch-msg" :class="{ error: batchError }">
        {{ batchMsg }}
      </div>
    </div>
    
    <hr v-if="!collapsed" style="margin: 24px 0; border: none; border-top: 1px solid #e5e7eb;" />
    
    <!-- 单个文档上传 -->
    <div v-if="!collapsed" class="doc-mgr">
      <h4>📄 单个文档上传</h4>
      <input v-model="docPath" type="text" placeholder="文档路径(标识) 例如 data/docs/sample.txt" />
      <input type="file" accept=".txt,.md,.pdf" @change="handleFile" />
      <textarea v-model="docText" placeholder="或直接粘贴文本(二选一)"></textarea>
      <div class="doc-actions">
        <button class="btn" @click="uploadDoc">上传/入库</button>
        <button class="btn danger" @click="deleteDoc">删除该文档</button>
      </div>
      <div class="muted">{{ docMsg }}</div>
    </div>
  </aside>
</template>

<script setup>
import { ref } from 'vue';
import api from '../api';

const props = defineProps(['sources', 'paths']);
const emit = defineEmits(['refresh-paths']);

const collapsed = ref(false);
const docPath = ref('');
const docText = ref('');
const docFile = ref(null);
const docMsg = ref('');

// 批量上传
const batchFileInput = ref(null);
const batchFiles = ref([]);
const batchMsg = ref('');
const batchError = ref(false);

function handleFile(e) {
  docFile.value = e.target.files[0];
}

async function uploadDoc() {
  if (!docPath.value.trim()) {
    docMsg.value = '请填写文档路径';
    return;
  }
  
  try {
    let res;
    if (docFile.value) {
      const fd = new FormData();
      fd.append('file', docFile.value);
      fd.append('path', docPath.value);
      res = await api.uploadDoc(fd);
    } else if (docText.value.trim()) {
      res = await api.uploadDoc({ path: docPath.value, text: docText.value });
    } else {
      docMsg.value = '请选择文件或填写文本';
      return;
    }
    
    if (res.data.ok) {
      docMsg.value = `入库完成，新增分片：${res.data.added_chunks}`;
      docText.value = '';
      docFile.value = null;
    } else {
      docMsg.value = `失败：${res.data.error}`;
    }
  } catch (e) {
    docMsg.value = `失败：${e.message}`;
  }
}

async function deleteDoc() {
  if (!docPath.value.trim()) {
    docMsg.value = '请填写要删除的文档路径';
    return;
  }
  
  try {
    const res = await api.deleteDoc(docPath.value);
    if (res.data.ok) {
      docMsg.value = `已删除分片：${res.data.deleted}`;
      emit('refresh-paths');
    } else {
      docMsg.value = `失败：${res.data.error}`;
    }
  } catch (e) {
    docMsg.value = `失败：${e.message}`;
  }
}

// 批量上传功能
function triggerBatchUpload() {
  batchFileInput.value?.click();
}

function handleBatchFiles(e) {
  const files = Array.from(e.target.files).filter(f => 
    f.name.endsWith('.md') || f.name.endsWith('.txt')
  );
  batchFiles.value.push(...files);
  batchMsg.value = '';
  e.target.value = ''; // 清空 input，允许重复选择相同文件
}

function handleDrop(e) {
  const files = Array.from(e.dataTransfer.files).filter(f => 
    f.name.endsWith('.md') || f.name.endsWith('.txt')
  );
  batchFiles.value.push(...files);
  batchMsg.value = '';
}

function removeFile(idx) {
  batchFiles.value.splice(idx, 1);
}

function clearBatch() {
  batchFiles.value = [];
  batchMsg.value = '';
  batchError.value = false;
}

function formatFileSize(bytes) {
  if (bytes < 1024) return bytes + ' B';
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
  return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

async function uploadBatch() {
  if (batchFiles.value.length === 0) {
    batchMsg.value = '请先选择文件';
    batchError.value = true;
    return;
  }
  
  batchMsg.value = `正在上传 ${batchFiles.value.length} 个文件...`;
  batchError.value = false;
  
  let successCount = 0;
  let failCount = 0;
  const results = [];
  
  for (const file of batchFiles.value) {
    try {
      // 读取文件内容
      const text = await file.text();
      
      // 构造文档路径：使用文件名作为标识
      const docPath = `knowledge/${file.name}`;
      
      // 上传文档
      const res = await api.uploadDoc({ 
        path: docPath, 
        text: text 
      });
      
      if (res.data.ok) {
        successCount++;
        results.push(`✓ ${file.name}: ${res.data.added_chunks} 个分片`);
      } else {
        failCount++;
        results.push(`✗ ${file.name}: ${res.data.error}`);
      }
    } catch (e) {
      failCount++;
      results.push(`✗ ${file.name}: ${e.message}`);
    }
  }
  
  batchMsg.value = `上传完成！成功: ${successCount} 个，失败: ${failCount} 个\n${results.join('\n')}`;
  batchError.value = failCount > 0;
  
  if (successCount > 0) {
    emit('refresh-paths');
    // 成功后清空文件列表
    setTimeout(() => {
      if (successCount === batchFiles.value.length) {
        clearBatch();
      }
    }, 3000);
  }
}
</script>

<style scoped>
.sources-panel {
  background: #ffffff;
  border-left: 1px solid #e5e7eb;
  padding: 24px;
  height: 100vh;
  overflow-y: auto;
  overflow-x: hidden;
}

.aside-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }
h3 { font-size: 16px; font-weight: 700; color: #111827; }

.doc-list ul { list-style: none; padding: 0; margin: 10px 0; }
.doc-list li { 
  padding: 10px 14px; 
  background: #f9fafb; 
  margin-bottom: 6px; 
  border-radius: 8px; 
  font-size: 13px;
  border: 1px solid #e5e7eb;
  transition: all 0.2s;
  color: #374151;
}
.doc-list li:hover {
  background: #f3f4f6;
  border-color: #d1d5db;
  transform: translateX(2px);
}

.doc-mgr { display: grid; gap: 10px; }
.doc-mgr input[type="text"], .doc-mgr textarea {
  padding: 10px 14px;
  border-radius: 8px;
  border: 1px solid #d1d5db;
  background: #f9fafb;
  color: #111827;
}

.doc-mgr input[type="text"]:focus, .doc-mgr textarea:focus {
  outline: none;
  border-color: #9ca3af;
  background: #fff;
  box-shadow: 0 0 0 3px rgba(156,163,175,0.1);
}

.doc-mgr textarea { min-height: 90px; }
.doc-actions { display: flex; gap: 8px; }

.btn {
  padding: 8px 16px;
  border-radius: 8px;
  border: 1px solid #d1d5db;
  background: #fff;
  color: #374151;
  cursor: pointer;
  transition: all 0.2s;
  font-weight: 500;
}

.btn:hover { 
  background: #f9fafb; 
  border-color: #9ca3af;
  box-shadow: 0 2px 8px rgba(0,0,0,0.08);
}
.btn.danger { 
  background: #fee2e2; 
  border-color: #fca5a5;
  color: #dc2626;
}
.btn.danger:hover {
  background: #fecaca;
  border-color: #f87171;
}
.btn.subtle { 
  background: #f3f4f6; 
  border-color: #e5e7eb; 
  color: #6b7280; 
}
.muted { color: #9ca3af; font-size: 12px; }
.muted.small { font-size: 11px; margin-top: 4px; }

h4 { 
  font-size: 14px; 
  font-weight: 600; 
  color: #374151; 
  margin-bottom: 8px; 
}

/* 批量上传样式 */
.batch-upload {
  margin: 16px 0;
  padding: 16px;
  background: #f9fafb;
  border-radius: 10px;
  border: 1px solid #e5e7eb;
}

.upload-area {
  margin: 12px 0;
  padding: 32px 20px;
  border: 2px dashed #d1d5db;
  border-radius: 10px;
  background: #ffffff;
  cursor: pointer;
  transition: all 0.3s;
  text-align: center;
}

.upload-area:hover {
  border-color: #9ca3af;
  background: #f9fafb;
  box-shadow: 0 4px 12px rgba(0,0,0,0.05);
}

.upload-prompt {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 8px;
}

.upload-icon {
  font-size: 48px;
  opacity: 0.6;
}

.upload-prompt p {
  margin: 0;
  color: #6b7280;
  font-size: 14px;
}

.file-list {
  margin: 12px 0;
  max-height: 300px;
  overflow-y: auto;
}

.file-item {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 10px 12px;
  background: #ffffff;
  border: 1px solid #e5e7eb;
  border-radius: 8px;
  margin-bottom: 6px;
  transition: all 0.2s;
}

.file-item:hover {
  background: #f9fafb;
  border-color: #d1d5db;
}

.file-icon {
  font-size: 20px;
  flex-shrink: 0;
}

.file-name {
  flex: 1;
  font-size: 13px;
  color: #374151;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.file-size {
  font-size: 11px;
  color: #9ca3af;
  flex-shrink: 0;
}

.btn-remove {
  background: transparent;
  border: none;
  color: #dc2626;
  font-size: 16px;
  cursor: pointer;
  padding: 4px 8px;
  border-radius: 4px;
  transition: all 0.2s;
  flex-shrink: 0;
}

.btn-remove:hover {
  background: #fee2e2;
}

.batch-actions {
  display: flex;
  gap: 8px;
  margin-top: 12px;
}

.btn.primary {
  background: #3b82f6;
  color: #ffffff;
  border-color: #3b82f6;
}

.btn.primary:hover {
  background: #2563eb;
  border-color: #2563eb;
  box-shadow: 0 4px 12px rgba(59,130,246,0.3);
}

.batch-msg {
  margin-top: 12px;
  padding: 12px;
  background: #dbeafe;
  border: 1px solid #93c5fd;
  border-radius: 8px;
  font-size: 12px;
  color: #1e40af;
  white-space: pre-wrap;
  line-height: 1.6;
}

.batch-msg.error {
  background: #fee2e2;
  border-color: #fca5a5;
  color: #dc2626;
}
</style>

