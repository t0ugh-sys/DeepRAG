<template>
  <header class="app-header">
    <h1>DeepRAG-Milvus</h1>
    <div class="controls">
      <input v-model="ns" @change="savePrefs" type="text" placeholder="Namespace" style="width:140px" />
      <input v-model="apikey" @change="savePrefs" type="password" placeholder="API Key" style="width:200px" />
      <input v-model="model" @change="savePrefs" type="text" placeholder="模型(可选)" style="width:220px" />
      <button class="btn" @click="$emit('toggle-theme')">主题</button>
      <button class="btn" @click="$emit('export-conv')">导出</button>
      <button class="btn danger" @click="$emit('clear-conv')">清空</button>
      <div class="status">{{ status }}</div>
    </div>
  </header>
</template>

<script setup>
import { ref, onMounted } from 'vue';
import api from '../api';

const ns = ref(localStorage.getItem('ns') || '');
const apikey = ref(localStorage.getItem('apikey') || '');
const model = ref(localStorage.getItem('model') || '');
const status = ref('Checking...');

function savePrefs() {
  localStorage.setItem('ns', ns.value);
  localStorage.setItem('apikey', apikey.value);
  localStorage.setItem('model', model.value);
}

async function checkHealth() {
  try {
    const res = await api.healthz();
    if (res.data.ok) {
      const d = res.data.details;
      status.value = `OK | ${d.vector_backend_active || 'unknown'} | ${d.milvus_entities || 0} docs`;
    } else {
      status.value = 'Unhealthy';
    }
  } catch {
    status.value = 'Offline';
  }
}

onMounted(() => {
  checkHealth();
});
</script>

<style scoped>
.app-header {
  position: sticky;
  top: 0;
  z-index: 10;
  background: rgba(11,15,23,0.9);
  backdrop-filter: blur(6px);
  border-bottom: 1px solid #1d2639;
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 12px 24px;
}

h1 { font-size: 22px; }
.controls { display: flex; gap: 8px; align-items: center; }
input { padding: 8px 10px; border-radius: 8px; border: 1px solid #27324a; background: #0e1421; color: #e6e8eb; }
.btn { padding: 8px 14px; border-radius: 8px; border: 1px solid #2a3a66; background: #161f35; color: #e6e8eb; cursor: pointer; }
.btn:hover { background: #1b2746; }
.btn.danger { background: #b84343; }
.status { font-size: 12px; opacity: 0.8; }
</style>

