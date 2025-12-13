import { defineConfig } from 'vitepress'

export default defineConfig({
  title: 'AI-Practices',
  description: 'A Comprehensive Full-Stack AI Learning Laboratory',

  lang: 'zh-CN',
  lastUpdated: true,

  head: [
    ['link', { rel: 'icon', type: 'image/svg+xml', href: '/logo.svg' }],
    ['meta', { name: 'theme-color', content: '#6366f1' }],
    ['meta', { name: 'og:type', content: 'website' }],
    ['meta', { name: 'og:site_name', content: 'AI-Practices' }],
    ['meta', { name: 'og:title', content: 'AI-Practices - Full-Stack AI Learning Laboratory' }],
    ['meta', { name: 'og:description', content: 'From Zero to Hero: A Comprehensive AI Learning Journey' }],
  ],

  themeConfig: {
    logo: '/logo.svg',

    nav: [
      { text: '首页', link: '/' },
      { text: '快速开始', link: '/guide/getting-started' },
      {
        text: '课程模块',
        items: [
          { text: '01 - 机器学习基础', link: '/modules/01-foundations' },
          { text: '02 - 神经网络', link: '/modules/02-neural-networks' },
          { text: '03 - 计算机视觉', link: '/modules/03-computer-vision' },
          { text: '04 - 序列模型', link: '/modules/04-sequence-models' },
          { text: '05 - 高级专题', link: '/modules/05-advanced-topics' },
          { text: '06 - 生成式模型', link: '/modules/06-generative-models' },
          { text: '07 - 强化学习', link: '/modules/07-reinforcement-learning' },
          { text: '08 - 理论笔记', link: '/modules/08-theory-notes' },
          { text: '09 - 实战项目', link: '/modules/09-practical-projects' },
        ]
      },
      { text: '学习路径', link: '/roadmap' },
      { text: 'API', link: '/api/' },
      {
        text: '更多',
        items: [
          { text: '贡献指南', link: '/contributing' },
          { text: '代码规范', link: '/guide/code-style' },
          { text: 'FAQ', link: '/faq' },
          { text: '许可证', link: '/license' },
          { text: '行为准则', link: '/code-of-conduct' },
        ]
      }
    ],

    sidebar: {
      '/guide/': [
        {
          text: '入门指南',
          items: [
            { text: '项目介绍', link: '/guide/introduction' },
            { text: '快速开始', link: '/guide/getting-started' },
            { text: '环境配置', link: '/guide/installation' },
            { text: '项目结构', link: '/guide/project-structure' },
          ]
        },
        {
          text: '核心概念',
          items: [
            { text: '设计理念', link: '/guide/design-philosophy' },
            { text: '学习方法', link: '/guide/learning-method' },
          ]
        },
        {
          text: '开发规范',
          items: [
            { text: '代码规范', link: '/guide/code-style' },
          ]
        }
      ],
      '/modules/': [
        {
          text: '基础阶段',
          items: [
            { text: '01 - 机器学习基础', link: '/modules/01-foundations' },
            { text: '02 - 神经网络', link: '/modules/02-neural-networks' },
          ]
        },
        {
          text: '核心阶段',
          items: [
            { text: '03 - 计算机视觉', link: '/modules/03-computer-vision' },
            { text: '04 - 序列模型', link: '/modules/04-sequence-models' },
          ]
        },
        {
          text: '进阶阶段',
          items: [
            { text: '05 - 高级专题', link: '/modules/05-advanced-topics' },
            { text: '06 - 生成式模型', link: '/modules/06-generative-models' },
            { text: '07 - 强化学习', link: '/modules/07-reinforcement-learning' },
          ]
        },
        {
          text: '参考资料',
          items: [
            { text: '08 - 理论笔记', link: '/modules/08-theory-notes' },
          ]
        },
        {
          text: '实战阶段',
          items: [
            { text: '09 - 实战项目', link: '/modules/09-practical-projects' },
          ]
        }
      ],
    },

    socialLinks: [
      { icon: 'github', link: 'https://github.com/zimingttkx/AI-Practices' }
    ],

    footer: {
      message: 'Released under the MIT License.',
      copyright: 'Copyright © 2024 zimingttkx'
    },

    editLink: {
      pattern: 'https://github.com/zimingttkx/AI-Practices/edit/main/docs/:path',
      text: '在 GitHub 上编辑此页'
    },

    search: {
      provider: 'local',
      options: {
        translations: {
          button: {
            buttonText: '搜索文档',
            buttonAriaLabel: '搜索文档'
          },
          modal: {
            noResultsText: '无法找到相关结果',
            resetButtonTitle: '清除查询条件',
            footer: {
              selectText: '选择',
              navigateText: '切换'
            }
          }
        }
      }
    },

    outline: {
      level: [2, 3],
      label: '页面导航'
    },

    docFooter: {
      prev: '上一页',
      next: '下一页'
    },

    lastUpdatedText: '最后更新于',

    returnToTopLabel: '回到顶部',
    sidebarMenuLabel: '菜单',
    darkModeSwitchLabel: '主题',
  },

  markdown: {
    theme: {
      light: 'github-light',
      dark: 'one-dark-pro'
    },
    lineNumbers: true,
  },

  vite: {
    css: {
      preprocessorOptions: {
        scss: {
          additionalData: `@import "@/styles/variables.scss";`
        }
      }
    }
  }
})
