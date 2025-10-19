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
  background: #101623;
  border: 1px solid #1d2639;
  border-radius: 12px;
  padding: 16px;
  box-shadow: 0 8px 20px rgba(0,0,0,0.25);
  position: sticky;
  top: 76px;
  height: calc(100vh - 120px);
  overflow: auto;
}

.aside-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }
h3 { font-size: 16px; }

.doc-list ul { list-style: none; padding: 0; margin: 8px 0; }
.doc-list li { padding: 6px; background: #0f1524; margin-bottom: 4px; border-radius: 6px; font-size: 12px; }

.doc-mgr { display: grid; gap: 8px; }
.doc-mgr input[type="text"], .doc-mgr textarea {
  padding: 10px 12px;
  border-radius: 8px;
  border: 1px solid #27324a;
  background: #0e1421;
  color: #e6e8eb;
}

.doc-mgr textarea { min-height: 90px; }
.doc-actions { display: flex; gap: 8px; }

.btn {
  padding: 8px 14px;
  border-radius: 8px;
  border: 1px solid #2a3a66;
  background: #161f35;
  color: #e6e8eb;
  cursor: pointer;
}

.btn:hover { background: #1b2746; }
.btn.danger { background: #b84343; }
.btn.subtle { background: #0f1524; border-color: #27324a; color: #b7c0d9; }
.muted { opacity: 0.7; font-size: 12px; }
</style>

