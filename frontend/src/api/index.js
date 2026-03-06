import axios from 'axios';

const BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';
const ADMIN_KEY_STORAGE_KEY = 'admin_api_key';

const api = axios.create({
  baseURL: BASE_URL,
  timeout: 60000,
});

function getNamespace() {
  return localStorage.getItem('ns') || '';
}

function getApiKey() {
  return localStorage.getItem('apikey') || '';
}

function getAdminKey() {
  return localStorage.getItem(ADMIN_KEY_STORAGE_KEY) || getApiKey();
}

function getAuthHeaders(isAdmin = false) {
  const key = isAdmin ? getAdminKey() : getApiKey();
  return key ? { 'X-API-Key': key } : {};
}

// Attach API key + namespace to every request / 每次请求附加 API Key 和命名空间
api.interceptors.request.use((config) => {
  const apiKey = getApiKey();
  const ns = getNamespace();
  if (apiKey) config.headers['X-API-Key'] = apiKey;
  if (ns) config.params = { ...config.params, namespace: ns };
  return config;
});

export default {
  // Health check / 健康检查
  healthz: () => api.get('/v1/healthz'),

  // Available models / 可用模型
  getModels: () => api.get('/v1/models'),

  // Streaming ask / 流式问答
  askStream: (question, model, top_k = 4, system_prompt = null, web_enabled = null, web_top_k = null, signal) =>
    fetch(`${BASE_URL}/v1/ask_stream?namespace=${getNamespace()}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...getAuthHeaders(false),
      },
      body: JSON.stringify({ question, model, top_k, system_prompt, web_enabled, web_top_k }),
      signal,
    }),

  // Namespace management / 命名空间管理
  createNS: (namespace) => api.post('/admin/namespaces/create', {}, { params: { namespace }, headers: getAuthHeaders(true) }),
  clearNS: (namespace) => api.post('/admin/namespaces/clear', {}, { params: { namespace }, headers: getAuthHeaders(true) }),
  deleteNS: (namespace) => api.delete('/admin/namespaces', { params: { namespace }, headers: getAuthHeaders(true) }),

  // Document management / 文档管理
  uploadDoc: (formData) => api.post('/admin/docs', formData, { headers: getAuthHeaders(true) }),
  deleteDoc: (path) => api.delete('/admin/docs', { params: { path }, headers: getAuthHeaders(true) }),
  listPaths: (limit = 1000) => api.get('/v1/docs/paths', { params: { limit } }),
  listDocuments: (params = {}) => api.get('/documents/list', { params }),
  listTags: () => api.get('/documents/tags'),
  listCategories: () => api.get('/documents/categories'),
  listConversations: (params = {}) => api.get('/conversations', { params }),
  deleteConversation: (id) => api.delete(`/conversations/${id}`, { headers: getAuthHeaders(true) }),
  exportPath: (path) => api.get('/v1/export', { params: { path } }),
};
