import axios from 'axios';

const BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: BASE_URL,
  timeout: 60000,
});

// 请求拦截器：自动带上 API Key 与 Namespace
api.interceptors.request.use((config) => {
  const apiKey = localStorage.getItem('apikey');
  const ns = localStorage.getItem('ns');
  if (apiKey) config.headers['X-API-Key'] = apiKey;
  if (ns) config.params = { ...config.params, namespace: ns };
  return config;
});

export default {
  // 健康检查
  healthz: () => api.get('/healthz'),
  
  // 提问（流式）
  askStream: (question, model, top_k = 4) => 
    fetch(`${BASE_URL}/ask_stream?namespace=${localStorage.getItem('ns') || ''}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-API-Key': localStorage.getItem('apikey') || '',
      },
      body: JSON.stringify({ question, model, top_k }),
    }),

  // 命名空间管理
  createNS: (namespace) => api.post('/namespaces/create', {}, { params: { namespace } }),
  clearNS: (namespace) => api.post('/namespaces/clear', {}, { params: { namespace } }),
  deleteNS: (namespace) => api.delete('/namespaces', { params: { namespace } }),

  // 文档管理
  uploadDoc: (formData) => api.post('/docs', formData),
  deleteDoc: (path) => api.delete(`/docs?path=${encodeURIComponent(path)}`),
  listPaths: (limit = 1000) => api.get(`/docs/paths?limit=${limit}`),
  exportPath: (path) => api.get(`/export?path=${encodeURIComponent(path)}`),
};

