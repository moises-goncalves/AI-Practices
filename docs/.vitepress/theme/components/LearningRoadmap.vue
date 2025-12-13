<template>
  <div class="roadmap-wrapper">
    <h2 class="roadmap-title">{{ title }}</h2>

    <div class="roadmap-container">
      <!-- 中心线 -->
      <div class="timeline-line"></div>

      <!-- 路线节点 -->
      <div
        v-for="(item, index) in roadmapItems"
        :key="item.phase"
        class="roadmap-item"
        :class="{ 'item-right': index % 2 === 1 }"
        :style="{ '--item-color': item.color }"
      >
        <!-- 时间点 -->
        <div class="timeline-dot">
          <span class="dot-icon">{{ item.icon }}</span>
        </div>

        <!-- 内容卡片 -->
        <div class="roadmap-card">
          <div class="card-header">
            <span class="phase-badge">{{ item.phase }}</span>
            <span class="phase-duration">{{ item.duration }}</span>
          </div>

          <h3 class="card-title">{{ item.title }}</h3>
          <p class="card-description">{{ item.description }}</p>

          <div class="card-modules">
            <div
              v-for="module in item.modules"
              :key="module.id"
              class="mini-module"
            >
              <span class="mini-icon">{{ module.icon }}</span>
              <span class="mini-name">{{ module.name }}</span>
            </div>
          </div>

          <div class="card-skills">
            <span
              v-for="skill in item.skills"
              :key="skill"
              class="skill-tag"
            >
              {{ skill }}
            </span>
          </div>
        </div>

        <!-- 连接线 -->
        <div class="connector-line"></div>
      </div>
    </div>

    <!-- 完成标记 -->
    <div class="completion-marker">
      <span class="completion-icon">🎓</span>
      <span class="completion-text">AI 全栈工程师</span>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'

defineProps<{
  title?: string
}>()

const roadmapItems = ref([
  {
    phase: 'PHASE 1',
    title: 'Foundation',
    icon: '🎯',
    color: '#6366f1',
    duration: '3-4 周',
    description: '建立坚实的机器学习理论基础，掌握经典算法的原理与实现',
    modules: [
      { id: '01', name: 'Foundations', icon: '📚' }
    ],
    skills: ['scikit-learn', 'XGBoost', 'Pandas', 'NumPy']
  },
  {
    phase: 'PHASE 2',
    title: 'Core',
    icon: '🧠',
    color: '#8b5cf6',
    duration: '5-7 周',
    description: '深入学习神经网络与深度学习核心技术，选择 CV 或 NLP 方向深入',
    modules: [
      { id: '02', name: 'Neural Networks', icon: '🧠' },
      { id: '03', name: 'Computer Vision', icon: '👁️' },
      { id: '04', name: 'Sequence Models', icon: '📝' }
    ],
    skills: ['TensorFlow', 'Keras', 'PyTorch', 'CNN', 'RNN']
  },
  {
    phase: 'PHASE 3',
    title: 'Advanced',
    icon: '⚡',
    color: '#a855f7',
    duration: '4-6 周',
    description: '探索高级专题，包括生成式模型和强化学习等前沿领域',
    modules: [
      { id: '05', name: 'Advanced Topics', icon: '⚡' },
      { id: '06', name: 'Generative Models', icon: '🎨' },
      { id: '07', name: 'Reinforcement Learning', icon: '🎮' }
    ],
    skills: ['GAN', 'Transformer', 'DQN', 'PPO', 'Optuna']
  },
  {
    phase: 'PHASE 4',
    title: 'Practice',
    icon: '🏆',
    color: '#22c55e',
    duration: '4-8 周',
    description: '通过真实项目和 Kaggle 竞赛检验学习成果，积累实战经验',
    modules: [
      { id: '09', name: 'Practical Projects', icon: '🏆' }
    ],
    skills: ['Kaggle', 'MLOps', 'Docker', 'Git']
  }
])
</script>

<style scoped>
.roadmap-wrapper {
  padding: 40px 0;
}

.roadmap-title {
  text-align: center;
  font-size: 1.75rem;
  font-weight: 700;
  margin-bottom: 48px;
  background: linear-gradient(135deg, #6366f1 0%, #a855f7 50%, #ec4899 100%);
  -webkit-background-clip: text;
  background-clip: text;
  -webkit-text-fill-color: transparent;
}

.roadmap-container {
  position: relative;
  max-width: 900px;
  margin: 0 auto;
  padding: 0 20px;
}

.timeline-line {
  position: absolute;
  left: 50%;
  top: 0;
  bottom: 60px;
  width: 4px;
  background: linear-gradient(180deg, #6366f1 0%, #8b5cf6 33%, #a855f7 66%, #22c55e 100%);
  transform: translateX(-50%);
  border-radius: 2px;
}

.roadmap-item {
  display: flex;
  align-items: center;
  margin-bottom: 60px;
  position: relative;
}

.roadmap-item:last-child {
  margin-bottom: 0;
}

/* 左侧项目 */
.roadmap-item:not(.item-right) {
  flex-direction: row;
  padding-right: calc(50% + 40px);
}

.roadmap-item:not(.item-right) .roadmap-card {
  margin-right: auto;
}

.roadmap-item:not(.item-right) .connector-line {
  right: calc(50% - 60px);
  left: auto;
}

/* 右侧项目 */
.roadmap-item.item-right {
  flex-direction: row-reverse;
  padding-left: calc(50% + 40px);
}

.roadmap-item.item-right .roadmap-card {
  margin-left: auto;
}

.roadmap-item.item-right .connector-line {
  left: calc(50% - 60px);
  right: auto;
}

.timeline-dot {
  position: absolute;
  left: 50%;
  transform: translateX(-50%);
  width: 48px;
  height: 48px;
  background: var(--item-color);
  border: 4px solid var(--vp-c-bg);
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 2;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
}

.dot-icon {
  font-size: 1.5rem;
}

.connector-line {
  position: absolute;
  top: 50%;
  width: 40px;
  height: 2px;
  background: var(--item-color);
}

.roadmap-card {
  background: var(--vp-c-bg-soft);
  border: 1px solid var(--vp-c-border);
  border-radius: 16px;
  padding: 24px;
  max-width: 380px;
  transition: all 0.3s ease;
  position: relative;
}

.roadmap-card::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  height: 4px;
  background: var(--item-color);
  border-radius: 16px 16px 0 0;
}

.roadmap-card:hover {
  border-color: var(--item-color);
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.12);
  transform: translateY(-4px);
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 12px;
}

.phase-badge {
  font-size: 0.75rem;
  font-weight: 700;
  color: var(--item-color);
  background: color-mix(in srgb, var(--item-color) 15%, transparent);
  padding: 4px 12px;
  border-radius: 999px;
  letter-spacing: 1px;
}

.phase-duration {
  font-size: 0.75rem;
  color: var(--vp-c-text-3);
}

.card-title {
  font-size: 1.25rem;
  font-weight: 700;
  color: var(--vp-c-text-1);
  margin-bottom: 8px;
}

.card-description {
  font-size: 0.875rem;
  color: var(--vp-c-text-2);
  line-height: 1.6;
  margin-bottom: 16px;
}

.card-modules {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-bottom: 16px;
}

.mini-module {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 6px 12px;
  background: var(--vp-c-bg);
  border-radius: 8px;
  font-size: 0.75rem;
}

.mini-icon {
  font-size: 1rem;
}

.mini-name {
  color: var(--vp-c-text-2);
  font-weight: 500;
}

.card-skills {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}

.skill-tag {
  font-size: 0.7rem;
  padding: 3px 8px;
  background: var(--vp-c-brand-soft);
  color: var(--vp-c-brand-1);
  border-radius: 4px;
  font-weight: 500;
}

.completion-marker {
  display: flex;
  flex-direction: column;
  align-items: center;
  margin-top: 40px;
  padding: 24px;
  background: linear-gradient(135deg, #6366f1 0%, #a855f7 50%, #22c55e 100%);
  border-radius: 16px;
  max-width: 300px;
  margin-left: auto;
  margin-right: auto;
}

.completion-icon {
  font-size: 3rem;
  margin-bottom: 8px;
}

.completion-text {
  font-size: 1.25rem;
  font-weight: 700;
  color: white;
}

@media (max-width: 768px) {
  .timeline-line {
    left: 30px;
  }

  .roadmap-item,
  .roadmap-item.item-right {
    flex-direction: row;
    padding-left: 80px;
    padding-right: 0;
  }

  .roadmap-item .roadmap-card,
  .roadmap-item.item-right .roadmap-card {
    margin: 0;
    max-width: 100%;
  }

  .timeline-dot {
    left: 30px;
  }

  .connector-line {
    display: none;
  }
}
</style>
