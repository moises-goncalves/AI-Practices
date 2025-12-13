<template>
  <a :href="href" class="module-card" :style="{ '--module-color': color }">
    <div class="card-header">
      <span class="card-icon">{{ icon }}</span>
      <span class="card-badge">{{ badge }}</span>
    </div>

    <h3 class="card-title">{{ title }}</h3>
    <p class="card-subtitle">{{ subtitle }}</p>
    <p class="card-description">{{ description }}</p>

    <div class="card-topics" v-if="topics && topics.length">
      <span v-for="topic in topics" :key="topic" class="topic-tag">
        {{ topic }}
      </span>
    </div>

    <div class="card-footer">
      <span class="card-stat">
        <span class="stat-icon">📓</span>
        {{ notebooks }} Notebooks
      </span>
      <span class="card-arrow">→</span>
    </div>
  </a>
</template>

<script setup lang="ts">
defineProps<{
  icon: string
  title: string
  subtitle: string
  description: string
  badge: string
  color: string
  href: string
  topics?: string[]
  notebooks?: number
}>()
</script>

<style scoped>
.module-card {
  display: block;
  background: var(--vp-c-bg-soft);
  border: 1px solid var(--vp-c-border);
  border-radius: 16px;
  padding: 24px;
  text-decoration: none;
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  position: relative;
  overflow: hidden;
}

.module-card::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  height: 4px;
  background: var(--module-color);
  transform: scaleX(0);
  transform-origin: left;
  transition: transform 0.3s ease;
}

.module-card:hover::before {
  transform: scaleX(1);
}

.module-card:hover {
  border-color: var(--module-color);
  box-shadow: 0 12px 32px rgba(0, 0, 0, 0.1);
  transform: translateY(-6px);
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 16px;
}

.card-icon {
  font-size: 2.5rem;
}

.card-badge {
  font-size: 0.75rem;
  font-weight: 600;
  color: var(--module-color);
  background: color-mix(in srgb, var(--module-color) 15%, transparent);
  padding: 4px 12px;
  border-radius: 999px;
}

.card-title {
  font-size: 1.25rem;
  font-weight: 700;
  color: var(--vp-c-text-1);
  margin-bottom: 4px;
}

.card-subtitle {
  font-size: 0.875rem;
  color: var(--module-color);
  font-weight: 500;
  margin-bottom: 12px;
}

.card-description {
  font-size: 0.875rem;
  color: var(--vp-c-text-2);
  line-height: 1.6;
  margin-bottom: 16px;
}

.card-topics {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  margin-bottom: 16px;
}

.topic-tag {
  font-size: 0.75rem;
  padding: 4px 10px;
  background: var(--vp-c-bg);
  color: var(--vp-c-text-2);
  border-radius: 6px;
}

.card-footer {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding-top: 16px;
  border-top: 1px solid var(--vp-c-border);
}

.card-stat {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 0.75rem;
  color: var(--vp-c-text-3);
}

.stat-icon {
  font-size: 1rem;
}

.card-arrow {
  font-size: 1.25rem;
  color: var(--module-color);
  transition: transform 0.2s ease;
}

.module-card:hover .card-arrow {
  transform: translateX(4px);
}
</style>
