import axios from 'axios';

const BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: BASE_URL,
  timeout: 60000,
});

// Attach API key + namespace to every request / 每次请求附加 API Key 和命名空间
api.interceptors.request.use((config) => {
  const apiKey = localStorage.getItem('apikey');
  const ns = localStorage.getItem('ns');
  if (apiKey) config.headers['X-API-Key'] = apiKey;
  if (ns) config.params = { ...config.params, namespace: ns };
  return config;
});

export default {
  // Health check / 健康检查
  healthz: () => api.get('/healthz'),

  // Available models / 可用模型
  getModels: () => api.get('/models'),

  // Streaming ask / 流式问答
  askStream: (question, model, top_k = 4, system_prompt = null, web_enabled = null, web_top_k = null, signal) =>
    fetch(`${BASE_URL}/ask_stream?namespace=${localStorage.getItem('ns') || ''}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-API-Key': localStorage.getItem('apikey') || '',
      },
      body: JSON.stringify({ question, model, top_k, system_prompt, web_enabled, web_top_k }),
      signal,
    }),

  // Namespace management / 命名空间管理
  createNS: (namespace) => api.post('/namespaces/create', {}, { params: { namespace } }),
  clearNS: (namespace) => api.post('/namespaces/clear', {}, { params: { namespace } }),
  deleteNS: (namespace) => api.delete('/namespaces', { params: { namespace } }),

  // Document management / 文档管理
  uploadDoc: (formData) => api.post('/docs', formData),
  deleteDoc: (path) => api.delete(`/docs?path=${encodeURIComponent(path)}`),
  listPaths: (limit = 1000) => api.get(`/docs/paths?limit=${limit}`),
  listDocuments: (params = {}) => api.get('/documents/list', { params }),
  listTags: () => api.get('/documents/tags'),
  listCategories: () => api.get('/documents/categories'),
  listConversations: (params = {}) => api.get('/conversations', { params }),
  deleteConversation: (id) => api.delete(`/conversations/${id}`),
  exportPath: (path) => api.get(`/export?path=${encodeURIComponent(path)}`),
};
