<template>
  <div class="goals-wrapper">
    <h2 class="goals-title">{{ title }}</h2>

    <div class="goals-grid">
      <div
        v-for="goal in goals"
        :key="goal.id"
        class="goal-card"
        :style="{ '--goal-color': goal.color }"
      >
        <div class="goal-icon-wrapper">
          <span class="goal-icon">{{ goal.icon }}</span>
        </div>

        <h3 class="goal-title">{{ goal.title }}</h3>
        <p class="goal-subtitle">{{ goal.subtitle }}</p>
        <p class="goal-description">{{ goal.description }}</p>

        <ul class="goal-features">
          <li v-for="feature in goal.features" :key="feature">
            <span class="feature-check">✓</span>
            {{ feature }}
          </li>
        </ul>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'

defineProps<{
  title?: string
}>()

const goals = ref([
  {
    id: 'theory',
    icon: '📐',
    title: '理论先行',
    subtitle: 'Theory First',
    color: '#6366f1',
    description: '每个模块均以数学原理和算法理论为基础，确保深入理解',
    features: [
      '数学推导与直觉解释',
      '算法原理深度剖析',
      '30+ 激活函数详解',
      '损失函数全景速查'
    ]
  },
  {
    id: 'implementation',
    icon: '💻',
    title: '代码实现',
    subtitle: 'Implementation',
    color: '#8b5cf6',
    description: '从零实现核心算法，深入理解底层机制和工程细节',
    features: [
      '核心算法手写实现',
      '113+ Jupyter Notebooks',
      '详细的中文注释',
      '可复现的实验代码'
    ]
  },
  {
    id: 'framework',
    icon: '🛠️',
    title: '框架应用',
    subtitle: 'Framework',
    color: '#a855f7',
    description: '掌握 TensorFlow/PyTorch 等主流框架的工程实践',
    features: [
      'TensorFlow 2.x 全栈',
      'PyTorch 最佳实践',
      'Hugging Face 生态',
      '工程化模板代码'
    ]
  },
  {
    id: 'practice',
    icon: '🏆',
    title: '项目实战',
    subtitle: 'Practice',
    color: '#22c55e',
    description: '通过真实项目和 Kaggle 竞赛检验学习成果',
    features: [
      '19 个端到端项目',
      'Kaggle 金牌方案',
      'CV/NLP/时序全覆盖',
      '从入门到竞赛'
    ]
  }
])
</script>

<style scoped>
.goals-wrapper {
  padding: 40px 0;
}

.goals-title {
  text-align: center;
  font-size: 1.75rem;
  font-weight: 700;
  margin-bottom: 48px;
  background: linear-gradient(135deg, #6366f1 0%, #a855f7 50%, #ec4899 100%);
  -webkit-background-clip: text;
  background-clip: text;
  -webkit-text-fill-color: transparent;
}

.goals-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 24px;
}

@media (max-width: 768px) {
  .goals-grid {
    grid-template-columns: 1fr;
  }
}

.goal-card {
  background: var(--vp-c-bg-soft);
  border: 1px solid var(--vp-c-border);
  border-radius: 20px;
  padding: 32px;
  position: relative;
  overflow: hidden;
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

.goal-card::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  height: 4px;
  background: var(--goal-color);
}

.goal-card::after {
  content: '';
  position: absolute;
  top: 0;
  right: 0;
  width: 120px;
  height: 120px;
  background: var(--goal-color);
  opacity: 0.05;
  border-radius: 0 0 0 100%;
  transition: all 0.3s ease;
}

.goal-card:hover {
  border-color: var(--goal-color);
  box-shadow: 0 12px 32px rgba(0, 0, 0, 0.12);
  transform: translateY(-6px);
}

.goal-card:hover::after {
  width: 160px;
  height: 160px;
  opacity: 0.1;
}

.goal-icon-wrapper {
  width: 64px;
  height: 64px;
  background: linear-gradient(135deg, var(--goal-color) 0%, color-mix(in srgb, var(--goal-color) 70%, white) 100%);
  border-radius: 16px;
  display: flex;
  align-items: center;
  justify-content: center;
  margin-bottom: 20px;
  box-shadow: 0 4px 12px color-mix(in srgb, var(--goal-color) 30%, transparent);
}

.goal-icon {
  font-size: 2rem;
}

.goal-title {
  font-size: 1.5rem;
  font-weight: 700;
  color: var(--vp-c-text-1);
  margin-bottom: 4px;
}

.goal-subtitle {
  font-size: 0.875rem;
  color: var(--goal-color);
  font-weight: 600;
  margin-bottom: 12px;
  letter-spacing: 1px;
}

.goal-description {
  font-size: 0.9375rem;
  color: var(--vp-c-text-2);
  line-height: 1.6;
  margin-bottom: 20px;
}

.goal-features {
  list-style: none;
  padding: 0;
  margin: 0;
}

.goal-features li {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 8px 0;
  font-size: 0.875rem;
  color: var(--vp-c-text-2);
  border-top: 1px solid var(--vp-c-border);
}

.goal-features li:first-child {
  border-top: none;
}

.feature-check {
  width: 20px;
  height: 20px;
  background: color-mix(in srgb, var(--goal-color) 15%, transparent);
  color: var(--goal-color);
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 0.75rem;
  font-weight: 700;
  flex-shrink: 0;
}
</style>
