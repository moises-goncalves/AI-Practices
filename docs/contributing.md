# 贡献指南

<script setup>
import { ref } from 'vue'
</script>

感谢你对 AI-Practices 项目的关注！我们欢迎任何形式的贡献。

## 🤝 行为准则

为了营造一个开放和友好的环境，我们承诺：

- 使用友好和包容的语言
- 尊重不同的观点和经验
- 优雅地接受建设性批评
- 关注对社区最有利的事情
- 对其他社区成员表示同情

## 💡 如何贡献

### 报告 Bug

如果你发现了 bug，请：

1. 检查 [Issues](https://github.com/zimingttkx/AI-Practices/issues) 确保该 bug 尚未被报告
2. 创建新 issue，包含：
   - 清晰的标题和描述
   - 重现步骤
   - 预期行为和实际行为
   - 屏幕截图（如果适用）
   - 环境信息（Python 版本、操作系统等）

### 建议新功能

如果你有新想法：

1. 先在 Issues 中讨论
2. 描述清楚功能需求和使用场景
3. 等待维护者反馈后再开始开发

### 提交代码

#### 第一次贡献？

1. Fork 本仓库
2. Clone 你的 fork 到本地
3. 创建新分支
4. 进行修改
5. 提交 Pull Request

## 🔄 开发流程

### 1. 准备环境

```bash
# Clone 仓库
git clone https://github.com/your-username/AI-Practices.git
cd AI-Practices

# 创建 conda 环境
conda env create -f environment.yml
conda activate ai-practices

# 或使用 pip
python -m venv venv
source venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

### 2. 创建分支

```bash
# 从 main 分支创建新分支
git checkout -b feature/your-feature-name

# 或修复 bug
git checkout -b fix/bug-description
```

**分支命名规范：**

| 前缀 | 用途 |
|:-----|:-----|
| `feature/` | 新功能 |
| `fix/` | Bug 修复 |
| `docs/` | 文档更新 |
| `refactor/` | 代码重构 |
| `test/` | 测试相关 |

### 3. 进行修改

在修改代码前，请阅读 [代码规范](/guide/code-style)。

### 4. 测试

确保你的修改：
- 代码能正常运行
- 所有 notebook 单元格可以顺序执行
- 没有引入新的错误或警告

### 5. 提交

```bash
# 添加修改的文件
git add .

# 提交（遵循提交规范）
git commit -m "feat: add new linear regression example"

# 推送到你的 fork
git push origin feature/your-feature-name
```

### 6. 创建 Pull Request

1. 在 GitHub 上打开你的 fork
2. 点击 "New Pull Request"
3. 填写 PR 模板
4. 等待 review

## ✉️ 提交规范

### Commit Message 格式

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Type 类型

| Type | 描述 |
|:-----|:-----|
| `feat` | 新功能 |
| `fix` | Bug 修复 |
| `docs` | 文档更新 |
| `style` | 代码格式调整 |
| `refactor` | 重构 |
| `test` | 测试相关 |
| `chore` | 构建/工具相关 |

### 示例

```bash
feat(ml-basics): add ridge regression example

- 添加 Ridge 回归的完整实现
- 包含交叉验证和超参数调优
- 添加可视化结果

Closes #123
```

## 📝 Pull Request 规范

### PR 标题

使用与 commit 相同的格式：

```
feat: add convolutional neural network tutorial
```

### PR 描述模板

```markdown
## 变更类型
- [ ] 新功能
- [ ] Bug 修复
- [ ] 文档更新
- [ ] 代码重构
- [ ] 其他

## 变更描述
简要描述你的变更

## 相关 Issue
Closes #issue_number

## 测试
- [ ] 代码已测试
- [ ] Notebook 可以完整运行
- [ ] 添加了必要的文档

## 检查清单
- [ ] 代码遵循项目规范
- [ ] 添加了必要的注释
- [ ] 更新了相关文档
- [ ] 所有测试通过
```

## 🎯 贡献重点领域

### 优先级高

1. **改进现有代码**
   - 添加详细注释
   - 优化代码结构
   - 修复 bug

2. **完善文档**
   - 补充理论说明
   - 添加使用示例
   - 翻译英文资源

3. **添加可视化**
   - 改进图表质量
   - 添加交互式可视化
   - 使用更好的配色方案

### 优先级中

4. **新增教程**
   - 填补知识空白
   - 添加实战项目
   - 补充高级主题

5. **性能优化**
   - 提高代码效率
   - 减少内存占用
   - 加快训练速度

### 优先级低

6. **工具改进**
   - 添加实用脚本
   - 改进开发工具
   - 自动化流程

## 📮 获取帮助

如有任何问题：

1. 查看 [FAQ](/faq)
2. 在 [Issues](https://github.com/zimingttkx/AI-Practices/issues) 中搜索
3. 创建新 issue 询问
4. 联系维护者

---

再次感谢你的贡献！🎉
