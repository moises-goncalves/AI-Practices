# AI 实战项目集合

> **学习价值**：⭐⭐⭐⭐⭐ | **实战难度**：⭐⭐⭐⭐
> **最后更新**：2025-11-30

本目录包含了从基础到高级的完整AI实战项目，涵盖机器学习、计算机视觉、自然语言处理、时间序列分析和Kaggle竞赛等多个领域。

---

## 📋 目录结构

```
09-practical-projects/
├── 01_机器学习基础项目/          # 机器学习入门项目
│   ├── 01_鸢尾花分类_入门/
│   ├── 02_房价预测_回归/
│   ├── 03_信用卡欺诈检测_不平衡/
│   └── 04_客户细分_聚类/
├── 02_计算机视觉项目/            # 计算机视觉实战
│   └── 01_图像分类_MNIST/
├── 03_自然语言处理项目/          # NLP实战
│   ├── 01_情感分析_基础/
│   ├── 02_文本分类_新闻/
│   ├── 03_命名实体识别_NER/
│   └── 04_文本生成_GPT/
├── 04_时间序列项目/              # 时间序列分析
│   ├── 01_销售预测_ARIMA/
│   ├── 02_股票价格预测_LSTM高级/
│   └── 03_异常检测_时序/
└── 05_Kaggle竞赛项目/            # Kaggle顶级方案
    ├── 01-American-Express-Default-Prediction/    # 金融风控
    ├── 02-Feedback-ELL-1st-Place/                 # NLP评分
    ├── 03-RSNA-2023-1st-Place/                    # 医学影像
    └── 04-RSNA-2024-Lumbar-Spine/                 # 医学影像
```

---

## 🎯 项目分类

### 1️⃣ 机器学习基础项目

**适合人群**：机器学习初学者
**难度等级**：⭐⭐
**学习时长**：1-2周

#### 项目列表

| 项目 | 任务类型 | 核心技术 | 数据集 |
|------|---------|---------|--------|
| 鸢尾花分类 | 多分类 | Logistic回归、决策树 | Iris |
| 房价预测 | 回归 | 线性回归、随机森林 | Boston Housing |
| 信用卡欺诈检测 | 二分类（不平衡） | SMOTE、XGBoost | Credit Card Fraud |
| 客户细分 | 聚类 | K-Means、层次聚类 | Mall Customers |

#### 学习要点
- ✅ 数据预处理（缺失值、异常值、特征缩放）
- ✅ 特征工程（特征选择、特征构造）
- ✅ 模型选择（分类、回归、聚类算法）
- ✅ 模型评估（交叉验证、评估指标）
- ✅ 不平衡数据处理（SMOTE、类别权重）

---

### 2️⃣ 计算机视觉项目

**适合人群**：有深度学习基础
**难度等级**：⭐⭐⭐
**学习时长**：2-3周

#### 项目列表

| 项目 | 任务类型 | 核心技术 | 数据集 |
|------|---------|---------|--------|
| 图像分类 | 多分类 | CNN、ResNet | MNIST/CIFAR-10 |

#### 学习要点
- ✅ 卷积神经网络（CNN）架构
- ✅ 数据增强（旋转、翻转、裁剪）
- ✅ 迁移学习（预训练模型）
- ✅ 模型优化（Batch Normalization、Dropout）

---

### 3️⃣ 自然语言处理项目

**适合人群**：有NLP基础
**难度等级**：⭐⭐⭐⭐
**学习时长**：3-4周

#### 项目列表

| 项目 | 任务类型 | 核心技术 | 数据集 |
|------|---------|---------|--------|
| 情感分析 | 二分类 | LSTM、BERT | IMDB Reviews |
| 文本分类 | 多分类 | TF-IDF、FastText | 20 Newsgroups |
| 命名实体识别 | 序列标注 | BiLSTM-CRF | CoNLL-2003 |
| 文本生成 | 生成任务 | GPT、Transformer | Custom |

#### 学习要点
- ✅ 词嵌入（Word2Vec、GloVe、BERT）
- ✅ 序列模型（RNN、LSTM、GRU）
- ✅ 注意力机制（Self-Attention、Multi-Head）
- ✅ 预训练模型微调（BERT、GPT）
- ✅ 文本预处理（分词、去停用词）

---

### 4️⃣ 时间序列项目

**适合人群**：有统计学基础
**难度等级**：⭐⭐⭐
**学习时长**：2-3周

#### 项目列表

| 项目 | 任务类型 | 核心技术 | 数据集 |
|------|---------|---------|--------|
| 销售预测 | 时序预测 | ARIMA、Prophet | Store Sales |
| 股票价格预测 | 时序预测 | LSTM、GRU | Stock Prices |
| 异常检测 | 异常检测 | Isolation Forest | Time Series |

#### 学习要点
- ✅ 时间序列分析（趋势、季节性、周期性）
- ✅ 统计模型（ARIMA、SARIMA、Prophet）
- ✅ 深度学习模型（LSTM、GRU、Transformer）
- ✅ 特征工程（滞后特征、滚动统计）
- ✅ 异常检测（统计方法、机器学习方法）

---

### 5️⃣ Kaggle竞赛项目

**适合人群**：有实战经验，追求顶级方案
**难度等级**：⭐⭐⭐⭐⭐
**学习时长**：4-8周

#### 项目列表

| 项目 | 排名 | 任务类型 | 核心技术 | 奖金 |
|------|------|---------|---------|------|
| American Express违约预测 | 🥇 1st | 金融风控 | LightGBM集成 | $100,000 |
| Feedback ELL评分 | 🥇 1st | NLP回归 | DeBERTa+伪标签 | $50,000 |
| RSNA 2023腹部创伤 | 🥇 1st | 医学影像 | 3D CNN+分割 | $50,000 |
| RSNA 2024腰椎疾病 | 🏅 7th | 医学影像 | 多视图学习 | - |

#### 学习要点
- ✅ 高级特征工程（时间序列聚合、文本增强）
- ✅ 模型集成（Stacking、Blending、Voting）
- ✅ 超参数优化（Optuna、贝叶斯优化）
- ✅ 伪标签策略（半监督学习）
- ✅ 数据增强（Mixup、CutMix、回译）
- ✅ 3D医学影像处理（CT、MRI）
- ✅ 多任务学习（分类+分割）

---

## 🚀 学习路径建议

### 初学者路径（0-6个月）

```
第1阶段：机器学习基础（1-2个月）
├── 01_鸢尾花分类_入门
├── 02_房价预测_回归
└── 04_客户细分_聚类

第2阶段：深度学习入门（2-3个月）
├── 02_计算机视觉项目/01_图像分类_MNIST
└── 03_自然语言处理项目/01_情感分析_基础

第3阶段：实战进阶（3-4个月）
├── 03_信用卡欺诈检测_不平衡
├── 04_时间序列项目/01_销售预测_ARIMA
└── 03_自然语言处理项目/02_文本分类_新闻
```

### 进阶路径（6-12个月）

```
第1阶段：深度学习进阶（6-8个月）
├── 03_自然语言处理项目/03_命名实体识别_NER
├── 03_自然语言处理项目/04_文本生成_GPT
└── 04_时间序列项目/02_股票价格预测_LSTM高级

第2阶段：Kaggle实战（8-12个月）
├── 05_Kaggle竞赛项目/01-American-Express（表格数据）
├── 05_Kaggle竞赛项目/02-Feedback-ELL（NLP）
└── 05_Kaggle竞赛项目/03-RSNA-2023（医学影像）
```

### 专家路径（12个月+）

```
深度研究Kaggle顶级方案
├── 完整复现所有Kaggle项目
├── 参加实际Kaggle竞赛
├── 开发自己的创新方法
└── 发表论文或开源项目
```

---

## 💻 环境配置

### 基础环境

```bash
# Python版本
Python 3.8+

# 基础库
pip install numpy pandas matplotlib seaborn scikit-learn

# 深度学习框架
pip install torch torchvision tensorflow keras

# NLP库
pip install transformers nltk spacy

# 时间序列库
pip install statsmodels prophet

# 梯度提升库
pip install lightgbm xgboost catboost

# 可视化库
pip install plotly bokeh

# 工具库
pip install jupyter notebook tqdm
```

### GPU环境（推荐）

```bash
# CUDA版本：11.8+
# cuDNN版本：8.6+

# PyTorch（GPU版本）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# TensorFlow（GPU版本）
pip install tensorflow-gpu
```

### 硬件要求

| 项目类型 | CPU | RAM | GPU | 存储 |
|---------|-----|-----|-----|------|
| 机器学习基础 | 4核+ | 8GB+ | 可选 | 10GB+ |
| 计算机视觉 | 8核+ | 16GB+ | 8GB+ | 50GB+ |
| 自然语言处理 | 8核+ | 16GB+ | 8GB+ | 50GB+ |
| 时间序列 | 4核+ | 16GB+ | 可选 | 20GB+ |
| Kaggle竞赛 | 16核+ | 32GB+ | 16GB+ | 100GB+ |

---

## 📚 学习资源

### 在线课程
1. **Coursera - Machine Learning** by Andrew Ng
2. **Fast.ai - Practical Deep Learning**
3. **DeepLearning.AI - Deep Learning Specialization**
4. **Kaggle Learn** - 免费实战课程

### 推荐书籍
1. **《Python机器学习》** - Sebastian Raschka
2. **《深度学习》** - Ian Goodfellow
3. **《动手学深度学习》** - 李沐
4. **《Kaggle竞赛宝典》** - 实战技巧

### 在线资源
1. [Kaggle Competitions](https://www.kaggle.com/competitions)
2. [Papers with Code](https://paperswithcode.com/)
3. [Hugging Face](https://huggingface.co/)
4. [TensorFlow Hub](https://tfhub.dev/)

---

## 🎓 学习建议

### 1. 循序渐进
- 从简单项目开始，逐步提升难度
- 不要跳过基础项目，打好基础很重要
- 每个项目都要完整实现，不要浅尝辄止

### 2. 动手实践
- 不要只看代码，要自己动手写
- 尝试修改参数，观察效果变化
- 在不同数据集上测试模型

### 3. 理解原理
- 不要只会调用API，要理解算法原理
- 阅读相关论文，深入理解技术细节
- 尝试从零实现核心算法

### 4. 记录总结
- 记录每个项目的学习笔记
- 总结遇到的问题和解决方案
- 分享学习心得，帮助他人

### 5. 参与社区
- 加入Kaggle社区，参与讨论
- 在GitHub上分享代码
- 参加线上/线下技术交流会

---

## 🔧 项目使用指南

### 基础项目使用

```bash
# 1. 进入项目目录
cd 01_机器学习基础项目/01_鸢尾花分类_入门/

# 2. 安装依赖
pip install -r requirements.txt

# 3. 运行Jupyter Notebook
jupyter notebook

# 4. 或直接运行Python脚本
python src/train.py
```

### Kaggle项目使用

```bash
# 1. 进入Kaggle项目目录
cd 05_Kaggle竞赛项目/01-American-Express-Default-Prediction/

# 2. 阅读中文README
cat README_CN.md

# 3. 下载数据（需要Kaggle账号）
# 参考README_CN.md中的数据准备部分

# 4. 运行训练脚本
sh run.sh
```

---

## ⚠️ 注意事项

### 数据使用
- 遵守数据集的使用协议
- Kaggle数据仅用于学习，不得商用
- 医学影像数据需特别注意隐私保护

### 计算资源
- 深度学习项目建议使用GPU
- 可以使用Google Colab免费GPU
- 大型项目可考虑云平台（AWS、GCP、阿里云）

### 代码规范
- 遵循PEP 8代码规范
- 添加必要的注释和文档
- 使用版本控制（Git）

### 学习心态
- 不要急于求成，稳扎稳打
- 遇到困难是正常的，坚持下去
- 多与他人交流，共同进步

---

## 🤝 贡献指南

欢迎贡献更多优质实战项目！

### 贡献要求
- 项目代码完整可运行
- 有详细的README说明
- 代码注释清晰
- 提供数据集或数据获取方式

### 提交流程
1. Fork本项目
2. 创建新的项目目录
3. 添加完整的代码和文档
4. 提交Pull Request

---

## 📞 联系方式

如有问题或建议，欢迎通过以下方式联系：
- GitHub Issues
- Email: your-email@example.com

---

## 📊 项目统计

- **总项目数**：15+
- **覆盖领域**：5个（机器学习、CV、NLP、时序、Kaggle）
- **难度范围**：入门到专家
- **代码行数**：50,000+
- **文档字数**：100,000+

---

**祝学习愉快！在AI的道路上不断前进！🚀**
