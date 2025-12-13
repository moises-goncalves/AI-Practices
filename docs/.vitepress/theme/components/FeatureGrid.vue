<template>
  <div class="feature-grid">
    <div
      v-for="feature in features"
      :key="feature.title"
      class="feature-item"
      :style="{ '--feature-color': feature.color }"
    >
      <div class="feature-icon">{{ feature.icon }}</div>
      <h3 class="feature-title">{{ feature.title }}</h3>
      <p class="feature-description">{{ feature.description }}</p>
    </div>
  </div>
</template>

<script setup lang="ts">
defineProps<{
  features: Array<{
    icon: string
    title: string
    description: string
    color: string
  }>
}>()
</script>

<style scoped>
.feature-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 20px;
  margin: 32px 0;
}

@media (max-width: 960px) {
  .feature-grid {
    grid-template-columns: repeat(2, 1fr);
  }
}

@media (max-width: 640px) {
  .feature-grid {
    grid-template-columns: 1fr;
  }
}

.feature-item {
  background: var(--vp-c-bg-soft);
  border: 1px solid var(--vp-c-border);
  border-radius: 12px;
  padding: 24px;
  text-align: center;
  transition: all 0.3s ease;
}

.feature-item:hover {
  border-color: var(--feature-color);
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.1);
  transform: translateY(-4px);
}

.feature-icon {
  font-size: 2.5rem;
  margin-bottom: 12px;
}

.feature-title {
  font-size: 1.1rem;
  font-weight: 600;
  color: var(--vp-c-text-1);
  margin-bottom: 8px;
}

.feature-description {
  font-size: 0.875rem;
  color: var(--vp-c-text-2);
  line-height: 1.5;
}
</style>
