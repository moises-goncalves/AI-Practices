const dayjs = require('dayjs')

module.exports = {
  title: 'AI-Practices',
  description: '系统化、工程化的人工智能学习与研究平台',

  // GitHub Pages 子路径配置
  base: '/AI-Practices/',

  // 主题配置
  theme: 'vdoing',

  // 语言配置
  locales: {
    '/': {
      lang: 'zh-CN',
      title: 'AI-Practices',
      description: '系统化、工程化的人工智能学习与研究平台',
    },
    '/en/': {
      lang: 'en-US',
      title: 'AI-Practices',
      description: 'A Systematic Approach to AI Research & Engineering',
    },
  },

  head: [
    ['link', { rel: 'icon', href: '/AI-Practices/logo.svg' }],
    ['meta', { name: 'theme-color', content: '#11a8cd' }],
    ['meta', { name: 'author', content: 'zimingttkx' }],
    ['meta', { name: 'keywords', content: 'AI, Machine Learning, Deep Learning, Neural Networks, Computer Vision, NLP, PyTorch, TensorFlow' }],
    // 移动端优化
    ['meta', { name: 'viewport', content: 'width=device-width,initial-scale=1,user-scalable=no' }],
  ],

  // Markdown 配置
  markdown: {
    lineNumbers: true,
    extractHeaders: ['h2', 'h3', 'h4'],
  },

  // 主题配置
  themeConfig: {
    // 导航栏 Logo
    logo: '/logo.svg',

    // 站点名称
    siteTitle: 'AI-Practices',

    // 导航栏配置
    nav: [
      { text: '首页', link: '/' },
      {
        text: '入门指南',
        link: '/guide/',
        items: [
          { text: '快速开始', link: '/guide/getting-started/' },
          { text: '安装配置', link: '/guide/installation/' },
          { text: '项目架构', link: '/guide/architecture/' },
        ],
      },
      {
        text: '学习模块',
        link: '/modules/',
        items: [
          { text: '模块概览', link: '/modules/' },
          { text: '01 - 机器学习基础', link: '/modules/01-foundations/' },
          { text: '02 - 神经网络', link: '/modules/02-neural-networks/' },
          { text: '03 - 计算机视觉', link: '/modules/03-computer-vision/' },
          { text: '04 - 序列模型', link: '/modules/04-sequence-models/' },
          { text: '05 - 高级专题', link: '/modules/05-advanced/' },
          { text: '06 - 生成模型', link: '/modules/06-generative/' },
          { text: '07 - 强化学习', link: '/modules/07-reinforcement-learning/' },
          { text: '08 - 理论笔记', link: '/modules/08-theory/' },
          { text: '09 - 实战项目', link: '/modules/09-projects/' },
        ],
      },
      { text: 'GitHub', link: 'https://github.com/zimingttkx/AI-Practices' },
    ],

    // 侧边栏配置 - 结构化目录
    sidebar: 'structuring',

    // 文章默认的作者信息
    author: {
      name: 'zimingttkx',
      link: 'https://github.com/zimingttkx',
    },

    // 博主信息
    blogger: {
      avatar: '/logo.svg',
      name: 'AI-Practices',
      slogan: '系统化 AI 学习平台',
    },

    // 社交图标
    social: {
      icons: [
        {
          iconClass: 'icon-github',
          title: 'GitHub',
          link: 'https://github.com/zimingttkx/AI-Practices',
        },
      ],
    },

    // 页脚配置
    footer: {
      createYear: 2024,
      copyrightInfo: 'zimingttkx | <a href="https://github.com/zimingttkx/AI-Practices/blob/main/LICENSE" target="_blank">MIT License</a>',
    },

    // 扩展自动生成 frontmatter
    extendFrontmatter: {
      author: {
        name: 'zimingttkx',
        link: 'https://github.com/zimingttkx',
      },
    },

    // 目录页配置
    category: true,
    tag: true,
    archive: true,

    // 文章信息配置 - 显示作者、创建时间、更新时间、阅读时间、字数
    articleInfo: ['author', 'createTime', 'updateTime', 'readingTime', 'word'],

    // 最近更新栏
    updateBar: {
      showToArticle: true,
      moreArticle: '/archives/',
    },

    // 右侧文章大纲
    rightMenuBar: true,

    // 页面风格 - 卡片风格
    pageStyle: 'card',

    // 内容区域宽度
    contentBgStyle: 1,

    // 代码块样式 - tomorrow 主题
    codeTheme: 'tomorrow',

    // 搜索配置
    searchMaxSuggestions: 10,

    // 最后更新时间
    lastUpdated: '上次更新',

    // 编辑链接
    repo: 'zimingttkx/AI-Practices',
    docsDir: 'docs',
    docsBranch: 'main',
    editLinks: true,
    editLinkText: '在 GitHub 上编辑此页',

    // 面包屑导航
    breadcrumb: true,

    // 页面标题前的图标
    titleBadge: true,

    // 文章标题前的图标
    titleBadgeIcons: [
      '📚', '🧪', '🏆', '🔬', '💡', '🎯', '🚀', '⚡', '🔥'
    ],

    // 侧边栏深度
    sidebarDepth: 2,
  },

  // 插件配置
  plugins: [
    // 代码复制
    [
      'one-click-copy',
      {
        copySelector: ['div[class*="language-"] pre', 'div[class*="aside-code"] aside'],
        copyMessage: '复制成功',
        duration: 1000,
        showInMobile: false,
      },
    ],
    // 放大图片
    [
      'vuepress-plugin-zooming',
      {
        selector: '.theme-vdoing-content img:not(.no-zoom)',
        options: {
          bgColor: 'rgba(0,0,0,0.6)',
        },
      },
    ],
    // 最后更新时间
    [
      '@vuepress/last-updated',
      {
        transformer: (timestamp) => {
          return dayjs(timestamp).format('YYYY/MM/DD, HH:mm:ss')
        },
      },
    ],
    // 全文搜索
    ['fulltext-search'],
    // 进度条
    ['@vuepress/nprogress'],
    // 数学公式
    [
      'vuepress-plugin-mathjax',
      {
        target: 'svg',
        macros: {
          '*': '\\times',
        },
      },
    ],
  ],
}
