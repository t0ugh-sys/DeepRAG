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
    <div v-if="!collapsed" class="doc-mgr">
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

const collapsed = ref(false);
const docPath = ref('');
const docText = ref('');
const docFile = ref(null);
const docMsg = ref('');

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
    } else {
      docMsg.value = `失败：${res.data.error}`;
    }
  } catch (e) {
    docMsg.value = `失败：${e.message}`;
  }
}
</script>

<style scoped>
.sources-panel {
  background: #ffffff;
  border: 1px solid #e5e7eb;
  border-radius: 12px;
  padding: 24px;
  box-shadow: 0 1px 3px rgba(0,0,0,0.05);
  position: sticky;
  top: 100px;
  height: calc(100vh - 150px);
  overflow: auto;
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
</style>

