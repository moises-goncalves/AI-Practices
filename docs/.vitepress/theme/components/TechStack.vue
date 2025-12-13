<template>
  <div class="tech-stack-wrapper">
    <h2 v-if="title" class="tech-title">{{ title }}</h2>

    <div class="tech-categories">
      <div
        v-for="category in categories"
        :key="category.name"
        class="tech-category"
      >
        <h3 class="category-name">{{ category.icon }} {{ category.name }}</h3>
        <div class="tech-items">
          <a
            v-for="tech in category.items"
            :key="tech.name"
            :href="tech.url"
            target="_blank"
            rel="noopener"
            class="tech-item"
            :style="{ '--tech-color': tech.color }"
          >
            <img
              v-if="tech.logo"
              :src="`https://img.shields.io/badge/${encodeURIComponent(tech.name)}-${tech.color.replace('#', '')}?style=flat-square&logo=${tech.logo}&logoColor=white`"
              :alt="tech.name"
              class="tech-badge"
            />
            <span v-else class="tech-name">{{ tech.name }}</span>
          </a>
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

const categories = ref([
  {
    name: 'Deep Learning',
    icon: '🤖',
    items: [
      { name: 'TensorFlow', color: '#FF6F00', logo: 'tensorflow', url: 'https://tensorflow.org' },
      { name: 'Keras', color: '#D00000', logo: 'keras', url: 'https://keras.io' },
      { name: 'PyTorch', color: '#EE4C2C', logo: 'pytorch', url: 'https://pytorch.org' },
      { name: 'Transformers', color: '#FFD21E', logo: 'huggingface', url: 'https://huggingface.co' }
    ]
  },
  {
    name: 'Machine Learning',
    icon: '📊',
    items: [
      { name: 'scikit-learn', color: '#F7931E', logo: 'scikit-learn', url: 'https://scikit-learn.org' },
      { name: 'XGBoost', color: '#189FDD', logo: 'xgboost', url: 'https://xgboost.ai' },
      { name: 'LightGBM', color: '#2980B9', logo: '', url: 'https://lightgbm.readthedocs.io' },
      { name: 'Optuna', color: '#0078D4', logo: '', url: 'https://optuna.org' }
    ]
  },
  {
    name: 'Data Science',
    icon: '🔬',
    items: [
      { name: 'NumPy', color: '#013243', logo: 'numpy', url: 'https://numpy.org' },
      { name: 'Pandas', color: '#150458', logo: 'pandas', url: 'https://pandas.pydata.org' },
      { name: 'Matplotlib', color: '#11557C', logo: '', url: 'https://matplotlib.org' },
      { name: 'Seaborn', color: '#4C72B0', logo: '', url: 'https://seaborn.pydata.org' }
    ]
  },
  {
    name: 'Development',
    icon: '🛠️',
    items: [
      { name: 'Python', color: '#3776AB', logo: 'python', url: 'https://python.org' },
      { name: 'Jupyter', color: '#F37626', logo: 'jupyter', url: 'https://jupyter.org' },
      { name: 'Git', color: '#F05032', logo: 'git', url: 'https://git-scm.com' },
      { name: 'Docker', color: '#2496ED', logo: 'docker', url: 'https://docker.com' }
    ]
  }
])
</script>

<style scoped>
.tech-stack-wrapper {
  padding: 40px 0;
}

.tech-title {
  text-align: center;
  font-size: 1.75rem;
  font-weight: 700;
  margin-bottom: 40px;
  background: linear-gradient(135deg, #6366f1 0%, #a855f7 50%, #ec4899 100%);
  -webkit-background-clip: text;
  background-clip: text;
  -webkit-text-fill-color: transparent;
}

.tech-categories {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 24px;
}

@media (max-width: 768px) {
  .tech-categories {
    grid-template-columns: 1fr;
  }
}

.tech-category {
  background: var(--vp-c-bg-soft);
  border: 1px solid var(--vp-c-border);
  border-radius: 16px;
  padding: 24px;
}

.category-name {
  font-size: 1rem;
  font-weight: 600;
  color: var(--vp-c-text-1);
  margin-bottom: 16px;
  padding-bottom: 12px;
  border-bottom: 1px solid var(--vp-c-border);
}

.tech-items {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
}

.tech-item {
  display: inline-block;
  transition: transform 0.2s ease;
}

.tech-item:hover {
  transform: scale(1.05);
}

.tech-badge {
  height: 24px;
  border-radius: 4px;
}

.tech-name {
  display: inline-block;
  padding: 4px 12px;
  background: var(--tech-color);
  color: white;
  font-size: 0.75rem;
  font-weight: 500;
  border-radius: 4px;
}
</style>
