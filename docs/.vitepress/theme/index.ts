import { h } from 'vue'
import type { Theme } from 'vitepress'
import DefaultTheme from 'vitepress/theme'

// 导入自定义组件
import ArchitectureDiagram from './components/ArchitectureDiagram.vue'
import LearningRoadmap from './components/LearningRoadmap.vue'
import DesignGoals from './components/DesignGoals.vue'
import ModuleCard from './components/ModuleCard.vue'
import StatsCard from './components/StatsCard.vue'
import TechStack from './components/TechStack.vue'
import FeatureGrid from './components/FeatureGrid.vue'

// 导入自定义样式
import './styles/custom.css'

export default {
  extends: DefaultTheme,

  Layout: () => {
    return h(DefaultTheme.Layout, null, {
      // 可以在这里添加插槽
    })
  },

  enhanceApp({ app, router, siteData }) {
    // 注册全局组件
    app.component('ArchitectureDiagram', ArchitectureDiagram)
    app.component('LearningRoadmap', LearningRoadmap)
    app.component('DesignGoals', DesignGoals)
    app.component('ModuleCard', ModuleCard)
    app.component('StatsCard', StatsCard)
    app.component('TechStack', TechStack)
    app.component('FeatureGrid', FeatureGrid)
  }
} satisfies Theme
