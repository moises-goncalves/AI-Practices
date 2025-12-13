<template>
  <div class="architecture-container">
    <h2 class="architecture-title">{{ title }}</h2>

    <div class="architecture-diagram">
      <!-- 阶段流程 -->
      <div class="phases-flow">
        <div
          v-for="(phase, index) in phases"
          :key="phase.id"
          class="phase-block"
          :style="{ '--phase-color': phase.color }"
        >
          <div class="phase-header">
            <span class="phase-icon">{{ phase.icon }}</span>
            <span class="phase-name">{{ phase.name }}</span>
          </div>

          <div class="phase-modules">
            <div
              v-for="module in phase.modules"
              :key="module.id"
              class="module-item"
              @click="$emit('module-click', module)"
            >
              <span class="module-icon">{{ module.icon }}</span>
              <div class="module-info">
                <span class="module-id">{{ module.id }}</span>
                <span class="module-name">{{ module.name }}</span>
              </div>
            </div>
          </div>

          <!-- 连接箭头 -->
          <div v-if="index < phases.length - 1" class="phase-arrow">
            <svg width="40" height="24" viewBox="0 0 40 24">
              <path
                d="M0 12 L30 12 M25 6 L32 12 L25 18"
                stroke="currentColor"
                stroke-width="2"
                fill="none"
              />
            </svg>
          </div>
        </div>
      </div>

      <!-- 支撑模块 -->
      <div class="support-modules">
        <div class="support-line"></div>
        <div class="support-items">
          <div
            v-for="support in supportModules"
            :key="support.id"
            class="support-item"
          >
            <span class="support-icon">{{ support.icon }}</span>
            <span class="support-name">{{ support.name }}</span>
            <span class="support-desc">{{ support.description }}</span>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'

defineProps<{
  title?: string
}>()

defineEmits<{
  'module-click': [module: any]
}>()

const phases = ref([
  {
    id: 'phase1',
    name: 'PHASE 1: Foundation',
    icon: '🎯',
    color: '#6366f1',
    modules: [
      { id: '01', name: '机器学习基础', icon: '📚' }
    ]
  },
  {
    id: 'phase2',
    name: 'PHASE 2: Core',
    icon: '🧠',
    color: '#8b5cf6',
    modules: [
      { id: '02', name: '神经网络', icon: '🧠' },
      { id: '03', name: '计算机视觉', icon: '👁️' },
      { id: '04', name: '序列模型', icon: '📝' }
    ]
  },
  {
    id: 'phase3',
    name: 'PHASE 3: Advanced',
    icon: '⚡',
    color: '#a855f7',
    modules: [
      { id: '05', name: '高级专题', icon: '⚡' },
      { id: '06', name: '生成式模型', icon: '🎨' },
      { id: '07', name: '强化学习', icon: '🎮' }
    ]
  },
  {
    id: 'phase4',
    name: 'PHASE 4: Practice',
    icon: '🏆',
    color: '#22c55e',
    modules: [
      { id: '09', name: '实战项目', icon: '🏆' }
    ]
  }
])

const supportModules = ref([
  {
    id: '08',
    name: '理论笔记',
    icon: '📖',
    description: '全程理论支撑'
  },
  {
    id: 'utils',
    name: '工具模块',
    icon: '🔧',
    description: '通用工具函数'
  }
])
</script>

<style scoped>
.architecture-container {
  background: var(--vp-c-bg-soft);
  border: 1px solid var(--vp-c-border);
  border-radius: 20px;
  padding: 40px;
  margin: 32px 0;
}

.architecture-title {
  text-align: center;
  font-size: 1.5rem;
  font-weight: 700;
  margin-bottom: 40px;
  background: linear-gradient(135deg, #6366f1 0%, #a855f7 50%, #ec4899 100%);
  -webkit-background-clip: text;
  background-clip: text;
  -webkit-text-fill-color: transparent;
}

.phases-flow {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: 16px;
  margin-bottom: 40px;
  flex-wrap: wrap;
}

.phase-block {
  flex: 1;
  min-width: 200px;
  position: relative;
}

.phase-header {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 12px 16px;
  background: var(--phase-color);
  color: white;
  border-radius: 12px 12px 0 0;
  font-weight: 600;
}

.phase-icon {
  font-size: 1.25rem;
}

.phase-name {
  font-size: 0.875rem;
}

.phase-modules {
  background: var(--vp-c-bg);
  border: 2px solid var(--phase-color);
  border-top: none;
  border-radius: 0 0 12px 12px;
  padding: 16px;
}

.module-item {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 12px;
  margin-bottom: 8px;
  background: var(--vp-c-bg-soft);
  border-radius: 8px;
  cursor: pointer;
  transition: all 0.2s ease;
}

.module-item:last-child {
  margin-bottom: 0;
}

.module-item:hover {
  background: var(--vp-c-brand-soft);
  transform: translateX(4px);
}

.module-icon {
  font-size: 1.5rem;
}

.module-info {
  display: flex;
  flex-direction: column;
}

.module-id {
  font-size: 0.75rem;
  color: var(--vp-c-text-3);
  font-weight: 600;
}

.module-name {
  font-size: 0.875rem;
  color: var(--vp-c-text-1);
  font-weight: 500;
}

.phase-arrow {
  position: absolute;
  right: -28px;
  top: 50%;
  transform: translateY(-50%);
  color: var(--vp-c-text-3);
  z-index: 1;
}

.support-modules {
  position: relative;
  padding-top: 32px;
}

.support-line {
  position: absolute;
  top: 0;
  left: 10%;
  right: 10%;
  height: 2px;
  background: linear-gradient(90deg, transparent 0%, var(--vp-c-border) 20%, var(--vp-c-border) 80%, transparent 100%);
}

.support-line::before {
  content: '全程支撑';
  position: absolute;
  left: 50%;
  top: -10px;
  transform: translateX(-50%);
  background: var(--vp-c-bg-soft);
  padding: 0 16px;
  font-size: 0.75rem;
  color: var(--vp-c-text-3);
}

.support-items {
  display: flex;
  justify-content: center;
  gap: 32px;
  margin-top: 24px;
}

.support-item {
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 20px 32px;
  background: var(--vp-c-bg);
  border: 1px dashed var(--vp-c-border);
  border-radius: 12px;
  transition: all 0.2s ease;
}

.support-item:hover {
  border-style: solid;
  border-color: var(--vp-c-brand-1);
  box-shadow: 0 4px 12px rgba(99, 102, 241, 0.15);
}

.support-icon {
  font-size: 2rem;
  margin-bottom: 8px;
}

.support-name {
  font-size: 0.875rem;
  font-weight: 600;
  color: var(--vp-c-text-1);
}

.support-desc {
  font-size: 0.75rem;
  color: var(--vp-c-text-3);
  margin-top: 4px;
}

@media (max-width: 768px) {
  .phases-flow {
    flex-direction: column;
  }

  .phase-block {
    width: 100%;
  }

  .phase-arrow {
    display: none;
  }

  .support-items {
    flex-direction: column;
    align-items: center;
  }
}
</style>
