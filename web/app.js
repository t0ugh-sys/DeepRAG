const $ = (sel) => document.querySelector(sel);
const messagesEl = $('#messages');
const sourcesEl = $('#sources');
const statusEl = $('#status');
let conversation = [];

function commonHeaders() {
  const key = $('#apikey').value.trim();
  const h = {};
  if (key) h['X-API-Key'] = key;
  return h;
}

function nsParam(url) {
  const ns = $('#ns').value.trim();
  if (!ns) return url;
  const hasQ = url.includes('?');
  return url + (hasQ ? '&' : '?') + 'namespace=' + encodeURIComponent(ns);
}

async function checkHealth() {
  try {
    const r = await fetch('/healthz', { headers: commonHeaders() });
    const j = await r.json();
    if (j.ok) {
      const be = j.details.vector_backend_active || j.details.vector_backend_config || 'unknown';
      statusEl.textContent = `OK | collection: ${j.details.milvus_collection} | entities: ${j.details.milvus_entities} | backend: ${be}`;
    } else {
      statusEl.textContent = 'Unhealthy';
    }
  } catch (e) {
    statusEl.textContent = 'Health check failed';
  }
}

function appendMsg(role, text, useMarkdown=false) {
  const div = document.createElement('div');
  div.className = `msg ${role}`;
  if (useMarkdown && window.marked) {
    // 渲染 markdown，并为代码块添加复制按钮
    const html = marked.parse(text);
    div.innerHTML = html;
    // 延迟到下一个宏任务，确保节点已插入
    setTimeout(() => {
      div.querySelectorAll('pre').forEach((pre) => {
        if (pre.dataset.enhanced) return;
        pre.dataset.enhanced = '1';
        const btn = document.createElement('button');
        btn.className = 'copy-btn';
        btn.textContent = '复制';
        btn.addEventListener('click', async () => {
          const code = pre.innerText || '';
          try { await navigator.clipboard.writeText(code); btn.textContent = '已复制'; setTimeout(() => btn.textContent = '复制', 1200); } catch {}
        });
        pre.prepend(btn);
      });
    });
  } else {
    div.textContent = text;
  }
  messagesEl.appendChild(div);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function buildHighlighter(question) {
  const tokens = Array.from(new Set(question.split(/\s+|[\p{P}\p{S}]+/u).filter(Boolean)))
    .filter(t => t.length >= 2)
    .sort((a, b) => b.length - a.length);
  if (!tokens.length) return (s) => s;
  const re = new RegExp(`(${tokens.map(t => t.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('|')})`, 'gi');
  return (s) => s.replace(re, (m) => `<span class="hl">${m}</span>`);
}

async function ask(question) {
  appendMsg('user', question);
  // 记录到历史
  conversation.push({ role: 'user', content: question, ts: Date.now() });
  const highlight = buildHighlighter(question);
  // UI: loading
  $('#askBtn').disabled = true;
  $('#typing').style.display = '';
  const ctrl = new AbortController();
  const t0 = performance.now();
  const res = await fetch(nsParam('/ask_stream'), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...commonHeaders() },
    body: JSON.stringify({ question, top_k: 4, model: ($('#model')?.value || '').trim() || undefined }),
    signal: ctrl.signal
  });
  let answer = '';
  const srcPayloads = [];
  // progressive render: create a live message element
  const live = document.createElement('div');
  live.className = 'msg assistant';
  messagesEl.appendChild(live);
  const reader = res.body.getReader();
  const decoder = new TextDecoder('utf-8');
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    const chunk = decoder.decode(value, { stream: true });
    const lines = chunk.split('\n\n');
    for (const line of lines) {
      if (!line) continue;
      if (line.startsWith('event: source')) {
        const dataLine = line.split('\n').find(x => x.startsWith('data: '));
        if (dataLine) {
          const payload = JSON.parse(dataLine.slice(6));
          srcPayloads.push(payload);
          const el = document.createElement('div');
          el.className = 'source-item';
          el.innerHTML = `<div class="muted">${payload.path} #${payload.chunk_id} · score ${payload.score.toFixed(4)}</div><div>${highlight(payload.snippet)}</div>`;
          sourcesEl.appendChild(el);
        }
        continue;
      }
      if (line.startsWith('data: ')) {
        const token = line.slice(6);
        answer += token;
        if (window.marked) {
          live.innerHTML = marked.parse(answer);
        } else {
          live.textContent = answer;
        }
      }
    }
  }
  // 保存助手回答到历史
  conversation.push({ role: 'assistant', content: answer, ts: Date.now(), sources: srcPayloads });
  saveConversation();
  const ms = Math.round(performance.now() - t0);
  appendMsg('assistant', `耗时 ${(ms/1000).toFixed(2)}s`, false);
  // UI: end loading
  $('#askBtn').disabled = false;
  $('#typing').style.display = 'none';
}

$('#askBtn').addEventListener('click', async () => {
  const q = $('#question').value.trim();
  if (!q) return;
  $('#question').value = '';
  sourcesEl.innerHTML = '';
  await ask(q);
});

$('#question').addEventListener('keydown', async (e) => {
  if (e.key === 'Enter') $('#askBtn').click();
});

checkHealth();

async function refreshPaths() {
  const r = await fetch(nsParam('/docs/paths'));
  const j = await r.json();
  const ul = document.querySelector('#paths');
  ul.innerHTML = '';
  if (!j.ok) return;
  for (const p of j.paths) {
    const li = document.createElement('li');
    li.textContent = p;
    // add export button per path
    const btn = document.createElement('button');
    btn.textContent = '导出';
    btn.style.marginLeft = '6px';
    btn.addEventListener('click', () => exportPath(p));
    li.appendChild(btn);
    ul.appendChild(li);
  }
}

document.querySelector('#refreshDocs').addEventListener('click', refreshPaths);

// Persist ns/apikey
function loadPrefs() {
  try {
    const ns = localStorage.getItem('ns');
    const key = localStorage.getItem('apikey');
    const model = localStorage.getItem('model');
    const theme = localStorage.getItem('theme') || 'dark';
    const conv = localStorage.getItem('conv');
    if (ns) $('#ns').value = ns;
    if (key) $('#apikey').value = key;
    if (model) $('#model').value = model;
    conversation = conv ? JSON.parse(conv) : [];
    applyTheme(theme);
    renderConversation();
  } catch {}
}

function savePrefs() {
  try {
    localStorage.setItem('ns', $('#ns').value.trim());
    localStorage.setItem('apikey', $('#apikey').value.trim());
    localStorage.setItem('model', ($('#model')?.value || '').trim());
  } catch {}
}

$('#ns').addEventListener('change', () => { savePrefs(); checkHealth(); });
$('#apikey').addEventListener('change', () => { savePrefs(); checkHealth(); });
$('#model').addEventListener('change', () => { savePrefs(); });
loadPrefs();
checkHealth();

// Namespace management buttons
async function nsAction(action) {
  const url = action === 'create' ? '/namespaces/create' : action === 'clear' ? '/namespaces/clear' : '/namespaces';
  const method = action === 'delete' ? 'DELETE' : 'POST';
  const r = await fetch(nsParam(url), { method, headers: { 'Content-Type': 'application/json', ...commonHeaders() } });
  const j = await r.json();
  if (!j.ok) { alert('操作失败: ' + (j.error || '')); return; }
  alert('操作成功');
  refreshPaths();
}

document.querySelector('#btnCreateNS').addEventListener('click', () => nsAction('create'));
document.querySelector('#btnClearNS').addEventListener('click', () => nsAction('clear'));
document.querySelector('#btnDeleteNS').addEventListener('click', () => nsAction('delete'));

// Doc upload/delete
async function uploadDoc() {
  const path = $('#docPath').value.trim();
  const file = $('#docFile').files[0];
  const text = $('#docText').value;
  if (!path) { $('#docMsg').textContent = '请填写文档路径'; return; }
  let res;
  if (file) {
    const fd = new FormData();
    fd.append('file', file);
    fd.append('path', path);
    res = await fetch(nsParam('/docs'), { method: 'POST', body: fd, headers: commonHeaders() });
  } else if (text && text.trim()) {
    res = await fetch(nsParam('/docs'), { method: 'POST', headers: { 'Content-Type': 'application/json', ...commonHeaders() }, body: JSON.stringify({ path, text }) });
  } else {
    $('#docMsg').textContent = '请选择文件或填写文本';
    return;
  }
  const j = await res.json();
  $('#docMsg').textContent = j.ok ? `入库完成，新增分片：${j.added_chunks}` : `失败：${j.error}`;
  $('#docText').value = '';
  $('#docFile').value = '';
}

// Drag & Drop
messagesEl.addEventListener('dragover', (e) => { e.preventDefault(); });
messagesEl.addEventListener('drop', async (e) => {
  e.preventDefault();
  const files = e.dataTransfer.files;
  if (!files || !files.length) return;
  $('#docFile').files = files; // set file input
});

async function deleteDoc() {
  const path = $('#docPath').value.trim();
  if (!path) { $('#docMsg').textContent = '请填写要删除的文档路径'; return; }
  const res = await fetch(nsParam(`/docs?path=${encodeURIComponent(path)}`), { method: 'DELETE', headers: commonHeaders() });
  const j = await res.json();
  $('#docMsg').textContent = j.ok ? `已删除分片：${j.deleted}` : `失败：${j.error}`;
}

$('#uploadBtn').addEventListener('click', uploadDoc);
$('#deleteBtn').addEventListener('click', deleteDoc);

async function exportPath(p) {
  const r = await fetch(nsParam(`/export?path=${encodeURIComponent(p)}`), { headers: commonHeaders() });
  const j = await r.json();
  if (!j.ok) { alert('导出失败'); return; }
  const blob = new Blob([JSON.stringify(j, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = (p.replace(/[^a-zA-Z0-9._-]+/g, '_')) + '.json';
  a.click();
  URL.revokeObjectURL(url);
}

// 复制最新回答与清空会话
$('#copyBtn').addEventListener('click', async () => {
  const last = [...document.querySelectorAll('.messages .msg.assistant')].pop();
  if (!last) return;
  const text = last.textContent || '';
  try { await navigator.clipboard.writeText(text); alert('已复制'); } catch { }
});

$('#clearConvBtn').addEventListener('click', () => {
  if (!confirm('确认清空会话？')) return;
  conversation = [];
  saveConversation();
  renderConversation();
  sourcesEl.innerHTML = '';
});

// 折叠引用区
$('#toggleSources').addEventListener('click', () => {
  const panel = document.getElementById('sourcesPanel');
  panel.classList.toggle('collapsed');
  document.getElementById('toggleSources').textContent = panel.classList.contains('collapsed') ? '展开' : '折叠';
});

// 对话历史渲染/保存/导出
function renderConversation() {
  messagesEl.innerHTML = '';
  for (const m of conversation) {
    const useMarkdown = m.role === 'assistant';
    appendMsg(m.role, m.content, useMarkdown);
  }
}

function saveConversation() {
  try { localStorage.setItem('conv', JSON.stringify(conversation)); } catch {}
}

function exportConversationJSON() {
  const blob = new Blob([JSON.stringify({ items: conversation }, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'conversation.json';
  a.click();
  URL.revokeObjectURL(url);
}

function toMarkdown(conv) {
  const lines = [];
  lines.push('# Conversation Export');
  for (const m of conv) {
    const time = new Date(m.ts || Date.now()).toISOString();
    lines.push(`\n## ${m.role} | ${time}`);
    lines.push('');
    lines.push(m.role === 'assistant' ? (m.content || '') : `> ${m.content || ''}`);
    if (m.sources && m.sources.length) {
      lines.push('\n### sources');
      for (const s of m.sources) {
        lines.push(`- ${s.path} #${s.chunk_id} (score ${Number(s.score).toFixed(4)})`);
      }
    }
  }
  return lines.join('\n');
}

function exportConversationMarkdown() {
  const md = toMarkdown(conversation);
  const blob = new Blob([md], { type: 'text/markdown' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'conversation.md';
  a.click();
  URL.revokeObjectURL(url);
}

$('#exportBtn').addEventListener('click', () => {
  const choice = window.prompt('导出格式: 输入 json 或 md', 'json');
  if (!choice) return;
  if (choice.toLowerCase().startsWith('m')) exportConversationMarkdown();
  else exportConversationJSON();
});

// 主题切换
function applyTheme(theme) {
  const isLight = theme === 'light';
  document.body.classList.toggle('light', isLight);
  try { localStorage.setItem('theme', isLight ? 'light' : 'dark'); } catch {}
}

$('#themeBtn').addEventListener('click', () => {
  const now = document.body.classList.contains('light') ? 'dark' : 'light';
  applyTheme(now);
});

