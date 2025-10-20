<template>
  <div class="modal-overlay" @click.self="$emit('close')">
    <div class="modal-content">
      <div class="modal-header">
        <h2>📁 导入知识库文档</h2>
        <button class="close-btn" @click="$emit('close')">&times;</button>
      </div>
      
      <div class="modal-body">
        <!-- 上传区域 -->
        <div 
          class="upload-area" 
          @click="triggerFileInput" 
          @drop.prevent="handleDrop" 
          @dragover.prevent
          :class="{ 'dragging': isDragging }"
          @dragenter="isDragging = true"
          @dragleave="isDragging = false"
        >
          <input 
            ref="fileInput" 
            type="file" 
            multiple 
            accept=".txt,.md" 
            @change="handleFiles"
            style="display:none"
          />
          <div class="upload-prompt">
            <span class="upload-icon">📂</span>
            <p class="upload-title">点击选择文件或拖拽到此处</p>
            <p class="upload-hint">支持 .md 和 .txt 格式，可多选</p>
          </div>
        </div>
        
        <!-- 文件列表 -->
        <div v-if="files.length > 0" class="file-list">
          <div class="file-list-header">
            <span>已选择 {{ files.length }} 个文件</span>
            <button class="text-btn" @click="clearFiles">全部清除</button>
          </div>
          <div class="file-item" v-for="(file, idx) in files" :key="idx">
            <span class="file-icon">{{ file.name.endsWith('.md') ? '📝' : '📄' }}</span>
            <span class="file-name" :title="file.name">{{ file.name }}</span>
            <span class="file-size">{{ formatSize(file.size) }}</span>
            <button class="remove-btn" @click="removeFile(idx)" title="移除">✕</button>
          </div>
        </div>
        
        <!-- 上传结果 -->
        <div v-if="message" class="message" :class="{ error: isError }">
          {{ message }}
        </div>
      </div>
      
      <div class="modal-footer">
        <button class="btn btn-secondary" @click="$emit('close')">取消</button>
        <button 
          class="btn btn-primary" 
          @click="uploadFiles" 
          :disabled="files.length === 0 || uploading"
        >
          {{ uploading ? '上传中...' : `上传 ${files.length} 个文件` }}
        </button>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue';
import api from '../api';

const emit = defineEmits(['close', 'refresh']);

const fileInput = ref(null);
const files = ref([]);
const message = ref('');
const isError = ref(false);
const uploading = ref(false);
const isDragging = ref(false);

function triggerFileInput() {
  fileInput.value?.click();
}

function handleFiles(e) {
  const selectedFiles = Array.from(e.target.files).filter(f => 
    f.name.endsWith('.md') || f.name.endsWith('.txt')
  );
  files.value.push(...selectedFiles);
  message.value = '';
  e.target.value = '';
  isDragging.value = false;
}

function handleDrop(e) {
  const droppedFiles = Array.from(e.dataTransfer.files).filter(f => 
    f.name.endsWith('.md') || f.name.endsWith('.txt')
  );
  files.value.push(...droppedFiles);
  message.value = '';
  isDragging.value = false;
}

function removeFile(idx) {
  files.value.splice(idx, 1);
}

function clearFiles() {
  files.value = [];
  message.value = '';
}

function formatSize(bytes) {
  if (bytes < 1024) return bytes + ' B';
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
  return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

async function uploadFiles() {
  if (files.value.length === 0) return;
  
  uploading.value = true;
  message.value = `正在上传 ${files.value.length} 个文件...`;
  isError.value = false;
  
  let successCount = 0;
  let failCount = 0;
  const errors = [];
  
  for (const file of files.value) {
    try {
      const text = await file.text();
      const res = await api.uploadDoc({ 
        path: `knowledge/${file.name}`, 
        text: text 
      });
      
      if (res.data.ok) {
        successCount++;
      } else {
        failCount++;
        errors.push(`${file.name}: ${res.data.error || '未知错误'}`);
      }
    } catch (e) {
      failCount++;
      errors.push(`${file.name}: ${e.message || '请求失败'}`);
      console.error(`上传失败 ${file.name}:`, e);
    }
  }
  
  uploading.value = false;
  
  if (failCount === 0) {
    message.value = `✅ 上传成功！共 ${successCount} 个文件已添加到知识库`;
    isError.value = false;
    emit('refresh');
    
    // 3秒后自动关闭
    setTimeout(() => {
      emit('close');
    }, 2000);
  } else {
    message.value = `上传完成。成功: ${successCount} 个，失败: ${failCount} 个\n\n错误详情:\n${errors.join('\n')}`;
    isError.value = true;
  }
}
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
  from {
    opacity: 0;
  }
  to {
    opacity: 1;
  }
}

.modal-content {
  background: #ffffff;
  border-radius: 16px;
  box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
  max-width: 600px;
  width: 90%;
  max-height: 80vh;
  display: flex;
  flex-direction: column;
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
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 24px 24px 16px;
  border-bottom: 1px solid #f0f0f0;
}

.modal-header h2 {
  font-size: 20px;
  font-weight: 600;
  color: #111827;
  margin: 0;
}

.close-btn {
  width: 32px;
  height: 32px;
  border-radius: 8px;
  border: none;
  background: transparent;
  color: #9ca3af;
  font-size: 24px;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all 0.15s;
}

.close-btn:hover {
  background: #f3f4f6;
  color: #111827;
}

.modal-body {
  padding: 24px;
  overflow-y: auto;
  flex: 1;
}

.upload-area {
  border: 2px dashed #d1d5db;
  border-radius: 12px;
  padding: 48px 24px;
  text-align: center;
  cursor: pointer;
  transition: all 0.2s;
  background: #fafafa;
}

.upload-area:hover, .upload-area.dragging {
  border-color: #3b82f6;
  background: #eff6ff;
}

.upload-prompt {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 12px;
}

.upload-icon {
  font-size: 64px;
  opacity: 0.6;
}

.upload-title {
  font-size: 16px;
  font-weight: 500;
  color: #374151;
  margin: 0;
}

.upload-hint {
  font-size: 14px;
  color: #9ca3af;
  margin: 0;
}

.file-list {
  margin-top: 24px;
}

.file-list-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 12px;
  font-size: 14px;
  font-weight: 500;
  color: #6b7280;
}

.text-btn {
  background: none;
  border: none;
  color: #3b82f6;
  font-size: 13px;
  cursor: pointer;
  padding: 4px 8px;
}

.text-btn:hover {
  text-decoration: underline;
}

.file-item {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 12px;
  background: #f9fafb;
  border: 1px solid #e5e7eb;
  border-radius: 8px;
  margin-bottom: 8px;
  transition: all 0.15s;
}

.file-item:hover {
  background: #f3f4f6;
  border-color: #d1d5db;
}

.file-icon {
  font-size: 24px;
  flex-shrink: 0;
}

.file-name {
  flex: 1;
  font-size: 14px;
  color: #374151;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.file-size {
  font-size: 12px;
  color: #9ca3af;
  flex-shrink: 0;
}

.remove-btn {
  width: 24px;
  height: 24px;
  border-radius: 4px;
  border: none;
  background: transparent;
  color: #dc2626;
  font-size: 16px;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all 0.15s;
  flex-shrink: 0;
}

.remove-btn:hover {
  background: #fee2e2;
}

.message {
  margin-top: 16px;
  padding: 12px 16px;
  border-radius: 8px;
  font-size: 14px;
  line-height: 1.6;
  background: #dbeafe;
  color: #1e40af;
  border: 1px solid #93c5fd;
}

.message.error {
  background: #fee2e2;
  color: #dc2626;
  border-color: #fca5a5;
}

.modal-footer {
  display: flex;
  align-items: center;
  justify-content: flex-end;
  gap: 12px;
  padding: 16px 24px;
  border-top: 1px solid #f0f0f0;
}

.btn {
  padding: 10px 20px;
  border-radius: 8px;
  border: none;
  font-size: 14px;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.15s;
}

.btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.btn-secondary {
  background: #f3f4f6;
  color: #374151;
}

.btn-secondary:hover:not(:disabled) {
  background: #e5e7eb;
}

.btn-primary {
  background: #3b82f6;
  color: #ffffff;
}

.btn-primary:hover:not(:disabled) {
  background: #2563eb;
  box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
}
</style>

