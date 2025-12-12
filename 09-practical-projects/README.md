# 09-Practical Projects | 实战项目

> 从基础到高级的完整 AI 实战项目集合

---

## 目录结构

```
09-practical-projects/
├── 01-ml-basics/              # 机器学习基础
│   ├── 01-titanic-survival-xgboost/   # Titanic 生存预测
│   ├── 02-otto-classification-xgboost/ # Otto 多分类
│   ├── 03-svm-text-classification/    # SVM 文本分类
│   └── 04-xgboost-advanced/           # XGBoost 高级技巧
├── 02-computer-vision/        # 计算机视觉
│   └── 01-mnist-cnn/                  # MNIST CNN
├── 03-nlp/                    # 自然语言处理
│   ├── 01-sentiment-analysis-lstm/    # LSTM 情感分析
│   ├── 02-transformer-text-classification/ # Transformer 文本分类
│   ├── 03-transformer-ner/            # 命名实体识别
│   └── 04-transformer-translation/    # 机器翻译
├── 04-time-series/            # 时间序列
│   ├── 01-temperature-prediction-lstm/ # 温度预测
│   └── 02-stock-prediction-lstm/      # 股票预测
└── 05-kaggle-competitions/    # Kaggle 竞赛
    ├── 01-American-Express-Default-Prediction/
    ├── 02-Feedback-ELL-1st-Place/
    ├── 03-RSNA-2023-1st-Place/
    └── 04-RSNA-2024-Lumbar-Spine/
```

---

## 项目分类

### 入门级 (1-2 周)
| 项目 | 类型 | 核心技术 |
|------|------|----------|
| Titanic | 分类 | XGBoost、特征工程 |
| MNIST | 图像分类 | CNN |
| 情感分析 | NLP | LSTM |

### 中级 (2-4 周)
| 项目 | 类型 | 核心技术 |
|------|------|----------|
| Otto 分类 | 多分类 | XGBoost、模型集成 |
| SVM 文本 | NLP | SVM、TF-IDF |
| Transformer NER | 序列标注 | Transformer |
| 温度预测 | 时序 | LSTM |

### 高级 (4+ 周)
| 项目 | 类型 | 核心技术 |
|------|------|----------|
| 机器翻译 | Seq2Seq | Transformer |
| 股票预测 | 时序 | LSTM、特征工程 |
| Kaggle 竞赛 | 综合 | 顶级方案复现 |

---

## 学习路径

```
入门: Titanic → MNIST → 情感分析
进阶: Otto → SVM文本 → NER → 温度预测
高级: 翻译 → 股票 → Kaggle方案
```

---

## 项目结构模板

```
project-name/
├── README.md          # 项目说明
├── data/              # 数据目录
│   └── README.md      # 数据说明
├── src/               # 源代码
│   ├── data.py        # 数据处理
│   ├── model.py       # 模型定义
│   ├── train.py       # 训练脚本
│   └── evaluate.py    # 评估脚本
└── notebooks/         # Jupyter notebooks
```

---

[返回主页](../README.md) | [项目实施指南](../docs/guides/PROJECTS_GUIDE.md)
