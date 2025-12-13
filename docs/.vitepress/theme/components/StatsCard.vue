<template>
  <div class="stats-container">
    <div
      v-for="stat in stats"
      :key="stat.label"
      class="stat-item"
      :style="{ '--stat-color': stat.color }"
    >
      <div class="stat-icon">{{ stat.icon }}</div>
      <div class="stat-number">{{ stat.value }}</div>
      <div class="stat-label">{{ stat.label }}</div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'

const stats = ref([
  { icon: '📓', value: '113+', label: 'Jupyter Notebooks', color: '#f97316' },
  { icon: '🧠', value: '9', label: '核心模块', color: '#8b5cf6' },
  { icon: '🚀', value: '19', label: '实战项目', color: '#22c55e' },
  { icon: '📝', value: '149k+', label: '代码行数', color: '#3b82f6' },
  { icon: '🏆', value: 'Gold', label: 'Kaggle 金牌', color: '#fbbf24' }
])
</script>

<style scoped>
.stats-container {
  display: flex;
  justify-content: center;
  gap: 32px;
  flex-wrap: wrap;
  padding: 48px 0;
}

.stat-item {
  text-align: center;
  padding: 24px 32px;
  background: var(--vp-c-bg-soft);
  border: 1px solid var(--vp-c-border);
  border-radius: 16px;
  min-width: 140px;
  transition: all 0.3s ease;
  position: relative;
  overflow: hidden;
}

.stat-item::after {
  content: '';
  position: absolute;
  bottom: 0;
  left: 0;
  right: 0;
  height: 3px;
  background: var(--stat-color);
  transform: scaleX(0);
  transition: transform 0.3s ease;
}

.stat-item:hover::after {
  transform: scaleX(1);
}

.stat-item:hover {
  border-color: var(--stat-color);
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.1);
  transform: translateY(-4px);
}

.stat-icon {
  font-size: 2rem;
  margin-bottom: 12px;
}

.stat-number {
  font-size: 2.5rem;
  font-weight: 800;
  background: linear-gradient(135deg, var(--stat-color) 0%, color-mix(in srgb, var(--stat-color) 70%, white) 100%);
  -webkit-background-clip: text;
  background-clip: text;
  -webkit-text-fill-color: transparent;
  line-height: 1.2;
}

.stat-label {
  font-size: 0.875rem;
  color: var(--vp-c-text-2);
  margin-top: 8px;
  font-weight: 500;
}

@media (max-width: 640px) {
  .stats-container {
    gap: 16px;
  }

  .stat-item {
    min-width: calc(50% - 16px);
    padding: 20px;
  }

  .stat-number {
    font-size: 2rem;
  }
}
</style>
