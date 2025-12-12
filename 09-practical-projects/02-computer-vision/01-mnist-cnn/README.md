# MNIST手写数字识别 - CNN实现

**难度**: ⭐⭐☆☆☆ (入门)

基于卷积神经网络的手写数字识别系统，实现99%+准确率。

## 📋 项目简介

本项目实现了完整的深度学习工作流程，从数据加载、模型构建、训练优化到模型评估，
提供了工业级的代码实现和详细的技术文档。

## 🎯 技术特点

- **完整的工程实现**: 模块化设计，代码规范，易于扩展
- **多种模型架构**: Simple CNN、Improved CNN、Deep CNN
- **训练优化策略**: 批标准化、Dropout、学习率调度、早停
- **性能评估**: 混淆矩阵、分类报告、训练曲线可视化

## 📁 项目结构

```
01-mnist-cnn/
├── data/              # 数据目录
├── src/              # 源代码
│   ├── data.py       # 数据加载和预处理
│   ├── model.py      # 模型定义
│   └── train.py      # 训练脚本
├── models/           # 保存的模型
├── results/          # 结果输出
├── README.md         # 项目说明
└── requirements.txt  # 依赖包
```

## 🚀 快速开始

### 环境配置

```bash
# 安装依赖
pip install -r requirements.txt
```

### 训练模型

```bash
# 基础模型
python src/train.py --model simple_cnn --epochs 20

# 改进模型
python src/train.py --model improved_cnn --epochs 30

# 深度模型
python src/train.py --model deep_cnn --epochs 40
```

## 📊 实验结果

| 模型 | 测试准确率 | 参数量 | 训练时间 |
|------|-----------|--------|---------|
| Simple CNN | 98.5% | 225K | ~5分钟 |
| Improved CNN | 99.4% | 280K | ~10分钟 |
| Deep CNN | 99.5% | 320K | ~15分钟 |

## 📝 核心技术

- **卷积神经网络**: 特征提取和分类
- **批标准化**: 加速训练和提高稳定性
- **数据增强**: 提升模型泛化能力
- **正则化**: Dropout防止过拟合
- **学习率调度**: 动态优化训练过程

## 🔗 参考资料

- [MNIST数据集](http://yann.lecun.com/exdb/mnist/)
- [深度学习卷积神经网络](http://www.deeplearningbook.org/)
- [LeCun et al. (1998) - Gradient-Based Learning Applied to Document Recognition](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf)

## 📧 技术支持

如有问题，欢迎提Issue讨论。

---

**研究级深度学习实践项目**
