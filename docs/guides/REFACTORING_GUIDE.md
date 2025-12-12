# 🔧 项目重构指南

## 📋 重构概述

本指南将帮助你将项目从中文命名重构为专业的英文命名结构。

---

## 🎯 重构前后对比

### 当前结构（重构前）
```
AI-Practices/
├── 机器学习实战/
│   ├── 机器学习基础知识/
│   └── 神经网络和深度学习/
├── python深度学习红书/
├── 实战项目/
└── 激活函数与损失函数/
```

### 新结构（重构后）
```
AI-Practices/
├── 01-foundations/              # 机器学习基础
├── 02-neural-networks/          # 神经网络
├── 03-computer-vision/          # 计算机视觉
├── 04-sequence-models/          # 序列模型
├── 05-advanced-topics/          # 高级主题
├── 06-generative-models/        # 生成式模型
├── 07-projects/                 # 实战项目
└── 08-theory-notes/             # 理论笔记
```

---

## 🚀 快速开始

以下流程总结了我在重构过程中的实战经验。按照这些步骤执行即可稳定完成迁移。

---

## 📝 详细步骤

### 步骤1：创建备份 ⚠️

```bash
# 创建完整备份
cp -r /Users/apple/PycharmProjects/AI-Practices \
      /Users/apple/PycharmProjects/AI-Practices_backup_$(date +%Y%m%d)
```

### 步骤2：创建新文件夹结构

```bash
cd /Users/apple/PycharmProjects/AI-Practices

# 创建主要目录
mkdir -p 01-foundations/{01-training-models,02-classification,03-support-vector-machines,04-decision-trees,05-ensemble-learning,06-dimensionality-reduction,07-unsupervised-learning,08-end-to-end-project}

mkdir -p 02-neural-networks/{01-keras-introduction,02-training-deep-networks,03-custom-models-training,04-data-loading-preprocessing}

mkdir -p 03-computer-vision/{01-cnn-basics,02-classic-architectures,03-transfer-learning,04-object-detection,05-semantic-segmentation}

mkdir -p 04-sequence-models/{01-rnn-basics,02-lstm-gru,03-text-processing,04-time-series,05-sequence-to-sequence}

mkdir -p 05-advanced-topics/{01-functional-api,02-callbacks-tensorboard,03-hyperparameter-tuning,04-model-optimization,05-deployment}

mkdir -p 06-generative-models/{01-autoencoders,02-gans,03-vaes,04-text-generation}

mkdir -p 07-projects/{01-ml-basics,02-computer-vision,03-nlp,04-time-series,05-recommendation,06-generative}

mkdir -p 08-theory-notes/{activation-functions,loss-functions,optimizers,regularization,architectures}

mkdir -p docs/{guides,tutorials,references}
mkdir -p tests/{unit,integration}
```

### 步骤3：移动内容（使用git mv保留历史）

```bash
# 示例：移动训练模型章节
git mv "机器学习实战/机器学习基础知识/训练模型" \
       "01-foundations/01-training-models"

# 移动分类章节
git mv "机器学习实战/机器学习基础知识/分类" \
       "01-foundations/02-classification"

# 移动SVM章节
git mv "机器学习实战/机器学习基础知识/Support Vector Machine" \
       "01-foundations/03-support-vector-machines"

# ... 继续移动其他文件夹
```

**💡 提示**：完整的移动命令列表请参考 `REFACTORING_PLAN.md`

### 步骤4：为每个章节创建README

在每个新文件夹中创建 `README.md`：

```bash
# 示例：为训练模型章节创建README
cat > 01-foundations/01-training-models/README.md << 'EOF'
# Training Models

## 📚 Content Overview

This chapter covers the fundamentals of training machine learning models.

## 🎯 Learning Objectives

- Understand linear regression
- Master gradient descent
- Learn regularization techniques
- Apply polynomial regression

## 📖 Topics

1. Linear Regression
2. Gradient Descent
3. Polynomial Regression
4. Regularization (Ridge, Lasso, Elastic Net)

## 💻 Notebooks

See `notebooks/` directory for practical implementations.

## 📝 Notes

See `notes/` directory for detailed theory notes.
EOF
```

### 步骤5：组织每个章节的内容

为每个章节创建标准子目录结构：

```bash
# 示例：组织训练模型章节
cd 01-foundations/01-training-models

# 创建子目录
mkdir -p notebooks notes code data assets/images

# 移动notebook文件到notebooks/
mv *.ipynb notebooks/ 2>/dev/null || true

# 移动markdown笔记到notes/
mv *.md notes/ 2>/dev/null || true

# 移动Python脚本到code/
mv *.py code/ 2>/dev/null || true
```

### 步骤6：更新主README

```bash
# 备份旧README
mv README.md README_OLD.md

# 创建新README（使用脚本生成的模板）
cp README_NEW.md README.md
```

### 步骤7：清理旧文件夹

```bash
# 确认所有内容已迁移后，删除空的旧文件夹
# ⚠️ 请先确认备份已创建！

rm -rf "机器学习实战"
rm -rf "python深度学习红书"
rm -rf "实战项目"
rm -rf "激活函数与损失函数"
```

---

## ✅ 验证清单

重构完成后，请检查以下项目：

### 文件完整性
- [ ] 所有notebook文件都已移动
- [ ] 所有markdown笔记都已移动
- [ ] 所有Python脚本都已移动
- [ ] 数据文件都已移动

### 结构规范性
- [ ] 每个章节都有README.md
- [ ] 文件夹命名符合规范（小写+连字符）
- [ ] 目录层次清晰合理

### 功能验证
- [ ] 随机测试几个notebook能否正常运行
- [ ] 检查notebook中的相对路径是否需要更新
- [ ] 验证图片等资源文件的引用是否正确

### 文档更新
- [ ] 主README已更新
- [ ] CONTRIBUTING.md已更新
- [ ] 所有文档中的路径引用已更新

---

## 🔧 常见问题

### Q1: 重构后notebook无法运行怎么办？

**A**: 可能是相对路径问题。检查notebook中的路径引用：

```python
# 旧路径（可能失效）
data = pd.read_csv('../data/dataset.csv')

# 新路径（需要更新）
data = pd.read_csv('../../data/dataset.csv')
```

**解决方案**：在notebook开头添加路径设置：

```python
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path.cwd().parent.parent
sys.path.insert(0, str(project_root))
```

### Q2: Git历史会丢失吗？

**A**: 使用 `git mv` 命令可以保留文件的Git历史。如果已经用普通方式移动了文件，可以：

```bash
# Git会自动检测文件移动（相似度>50%）
git add -A
git commit -m "refactor: reorganize project structure"
```

### Q3: 如何回滚重构？

**A**: 如果重构出现问题：

```bash
# 方式1：使用备份恢复
rm -rf /Users/apple/PycharmProjects/AI-Practices
cp -r /Users/apple/PycharmProjects/AI-Practices_backup_YYYYMMDD \
      /Users/apple/PycharmProjects/AI-Practices

# 方式2：使用Git回滚（如果已提交）
git log --oneline  # 找到重构前的commit
git reset --hard <commit-hash>
```

### Q4: 可以分批重构吗？

**A**: 可以！建议按模块分批重构：

1. **第一批**：机器学习基础（01-foundations）
2. **第二批**：神经网络（02-neural-networks）
3. **第三批**：计算机视觉（03-computer-vision）
4. **第四批**：其他模块

每批重构后测试验证，确认无误再继续下一批。

---

## 📊 重构进度追踪

创建一个进度追踪文件：

```bash
cat > REFACTORING_PROGRESS.md << 'EOF'
# 重构进度追踪

## 总体进度：0%

### ✅ 已完成
- [ ] 创建备份
- [ ] 创建新文件夹结构

### 🔄 进行中
- [ ] 01-foundations (0/8)
- [ ] 02-neural-networks (0/4)
- [ ] 03-computer-vision (0/5)
- [ ] 04-sequence-models (0/5)
- [ ] 05-advanced-topics (0/5)
- [ ] 06-generative-models (0/4)
- [ ] 07-projects (0/6)
- [ ] 08-theory-notes (0/5)

### ⏳ 待处理
- [ ] 更新所有README
- [ ] 更新文档引用
- [ ] 测试验证
- [ ] 清理旧文件夹

## 详细进度

### 01-foundations
- [ ] 01-training-models
- [ ] 02-classification
- [ ] 03-support-vector-machines
- [ ] 04-decision-trees
- [ ] 05-ensemble-learning
- [ ] 06-dimensionality-reduction
- [ ] 07-unsupervised-learning
- [ ] 08-end-to-end-project

（继续列出其他模块...）
EOF
```

---

## 🎯 重构后的优势

完成重构后，你的项目将具有：

1. ✅ **专业性**：符合国际开源项目标准
2. ✅ **可读性**：清晰的英文命名，易于理解
3. ✅ **可维护性**：规范的目录结构，便于扩展
4. ✅ **可分享性**：适合放在GitHub等平台展示
5. ✅ **可导航性**：逻辑清晰的层次结构

---

## 📞 需要帮助？

如果在重构过程中遇到问题：

1. 查看 `REFACTORING_PLAN.md` 了解详细映射关系
2. 检查 `migration_log.json`（如果使用了自动脚本）
3. 参考本指南的常见问题部分

---

## 🎉 完成重构

重构完成后，建议：

1. **提交到Git**：
```bash
git add -A
git commit -m "refactor: reorganize project structure with professional naming"
git push
```

2. **更新GitHub仓库描述**
3. **添加项目徽章**
4. **编写详细的README**

恭喜你完成了项目重构！🎊
