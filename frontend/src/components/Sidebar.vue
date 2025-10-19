<template>
  <aside class="sidebar">
    <div class="sidebar-header">
      <div class="muted" style="font-weight:600">会话历史</div>
    </div>
    <ul class="conv-list">
      <li>示例占位 - 待实现多会话管理</li>
    </ul>
  </aside>
</template>

<script setup>
</script>

<style scoped>
.sidebar {
  background: #101623;
  border: 1px solid #1d2639;
  border-radius: 12px;
  padding: 12px;
  position: sticky;
  top: 76px;
  height: calc(100vh - 120px);
  overflow: auto;
}

.sidebar-header { display: flex; justify-content: space-between; margin-bottom: 8px; }
.muted { opacity: 0.7; }
.conv-list { list-style: none; padding: 0; margin: 8px 0 0; }
.conv-list li {
  padding: 8px 10px;
  border-radius: 8px;
  margin-bottom: 6px;
  background: #0f1524;
  border: 1px solid #27324a;
}
</style>

