# 贡献指南

感谢你对 AI-Practices 项目的关注！我们欢迎任何形式的贡献。

## 📋 目录

- [行为准则](#行为准则)
- [如何贡献](#如何贡献)
- [开发流程](#开发流程)
- [代码规范](#代码规范)
- [提交规范](#提交规范)
- [文档规范](#文档规范)

## 🤝 行为准则

### 我们的承诺

为了营造一个开放和友好的环境，我们承诺：

- 使用友好和包容的语言
- 尊重不同的观点和经验
- 优雅地接受建设性批评
- 关注对社区最有利的事情
- 对其他社区成员表示同情

## 💡 如何贡献

### 报告Bug

如果你发现了bug，请：

1. 检查 [Issues](https://github.com/yourusername/AI-Practices/issues) 确保该bug尚未被报告
2. 创建新issue，包含：
   - 清晰的标题和描述
   - 重现步骤
   - 预期行为和实际行为
   - 屏幕截图（如果适用）
   - 环境信息（Python版本、操作系统等）

### 建议新功能

如果你有新想法：

1. 先在Issues中讨论
2. 描述清楚功能需求和使用场景
3. 等待维护者反馈后再开始开发

### 提交代码

#### 第一次贡献？

1. Fork 本仓库
2. Clone 你的fork到本地
3. 创建新分支
4. 进行修改
5. 提交Pull Request

## 🔄 开发流程

### 1. 准备环境

```bash
# Clone仓库
git clone https://github.com/yourusername/AI-Practices.git
cd AI-Practices

# 创建conda环境
conda env create -f environment.yml
conda activate ai-practices

# 或使用pip
python -m venv venv
source venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

### 2. 创建分支

```bash
# 从main分支创建新分支
git checkout -b feature/your-feature-name

# 或修复bug
git checkout -b fix/bug-description
```

分支命名规范：
- `feature/` - 新功能
- `fix/` - Bug修复
- `docs/` - 文档更新
- `refactor/` - 代码重构
- `test/` - 测试相关

### 3. 进行修改

在修改代码前，请阅读[代码规范](#代码规范)。

### 4. 测试

确保你的修改：
- 代码能正常运行
- 所有notebook单元格可以顺序执行
- 没有引入新的错误或警告

### 5. 提交

```bash
# 添加修改的文件
git add .

# 提交（遵循提交规范）
git commit -m "feat: add new linear regression example"

# 推送到你的fork
git push origin feature/your-feature-name
```

### 6. 创建Pull Request

1. 在GitHub上打开你的fork
2. 点击 "New Pull Request"
3. 填写PR模板
4. 等待review

## 📝 代码规范

### Python代码规范

遵循 [PEP 8](https://www.python.org/dev/peps/pep-0008/) 风格指南：

```python
# 好的例子
def train_model(X_train, y_train, epochs=100, batch_size=32):
    """
    训练神经网络模型

    参数:
        X_train: 训练数据，shape (n_samples, n_features)
        y_train: 训练标签，shape (n_samples,)
        epochs: 训练轮数，默认100
        batch_size: 批次大小，默认32

    返回:
        history: 训练历史对象
    """
    model = create_model()
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2
    )
    return history
```

### Jupyter Notebook规范

#### 1. 文件结构

每个notebook应包含：

```markdown
# 标题

## 1. 概述
简要介绍本notebook的内容和目标

## 2. 导入库
import numpy as np
import matplotlib.pyplot as plt

## 3. 理论背景
详细的理论说明

## 4. 代码实现
具体的代码和注释

## 5. 结果分析
对结果的解释和分析

## 6. 总结
关键要点总结

## 7. 参考资料
相关资料链接
```

#### 2. 代码单元格

```python
# === 数据准备 ===
# 生成模拟数据
np.random.seed(42)  # 设置随机种子以确保可重复性
X = 2 * np.random.rand(100, 1)  # 100个样本，1个特征
y = 4 + 3 * X + np.random.randn(100, 1)  # y = 4 + 3x + 噪声

# === 模型训练 ===
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X, y)

print(f"截距: {model.intercept_[0]:.2f}")
print(f"斜率: {model.coef_[0][0]:.2f}")
```

#### 3. 可视化

```python
# === 结果可视化 ===
plt.figure(figsize=(10, 6))
plt.scatter(X, y, alpha=0.5, label='数据点')
plt.plot(X, model.predict(X), 'r-', linewidth=2, label='拟合线')
plt.xlabel('X', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.title('线性回归拟合结果', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

#### 4. Markdown单元格

使用清晰的标题层次：

```markdown
## 主标题

### 二级标题

**重点内容加粗**

核心公式：
$$y = wx + b$$

重要提示：
> 💡 这是一个重要的概念

代码说明：
- 第一步：数据准备
- 第二步：模型训练
- 第三步：结果评估
```

### 注释规范

#### 单行注释

```python
# 计算均方误差
mse = np.mean((y_pred - y_true) ** 2)
```

#### 多行注释

```python
"""
这是一个复杂的函数，需要详细说明：
1. 首先进行数据标准化
2. 然后构建模型
3. 最后返回训练好的模型和历史
"""
```

#### 代码块注释

```python
# === 数据预处理 ===
# 1. 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
```

## 📄 文档规范

### Markdown文档

#### 文件命名

- 使用中文描述性名称
- 例如：`决策树算法详解.md`, `Keras使用指南.md`

#### 内容结构

```markdown
# 文档标题

## 目录
- [概述](#概述)
- [核心概念](#核心概念)
- [实现方法](#实现方法)
- [常见问题](#常见问题)

## 概述

简要介绍

## 核心概念

### 概念1
详细说明

### 概念2
详细说明

## 实现方法

### 方法1
代码示例

### 方法2
代码示例

## 常见问题

**Q: 问题1？**
A: 回答

**Q: 问题2？**
A: 回答

## 参考资料

- [链接1](url)
- [链接2](url)
```

### 数学公式

使用LaTeX格式：

```markdown
行内公式：$f(x) = wx + b$

独立公式：
$$
L = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2
$$
```

## ✉️ 提交规范

### Commit Message格式

使用以下格式：

```
<type>(<scope>): <subject>

<body>

<footer>
```

#### Type类型

- `feat`: 新功能
- `fix`: Bug修复
- `docs`: 文档更新
- `style`: 代码格式调整
- `refactor`: 重构
- `test`: 测试相关
- `chore`: 构建/工具相关

#### 示例

```bash
feat(ml-basics): add ridge regression example

- 添加Ridge回归的完整实现
- 包含交叉验证和超参数调优
- 添加可视化结果

Closes #123
```

### Pull Request

#### PR标题

使用与commit相同的格式：

```
feat: add convolutional neural network tutorial
```

#### PR描述模板

```markdown
## 变更类型
- [ ] 新功能
- [ ] Bug修复
- [ ] 文档更新
- [ ] 代码重构
- [ ] 其他

## 变更描述
简要描述你的变更

## 相关Issue
Closes #issue_number

## 测试
- [ ] 代码已测试
- [ ] Notebook可以完整运行
- [ ] 添加了必要的文档

## 截图（如适用）
添加相关截图

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
   - 修复bug

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

1. 查看 [Issues](https://github.com/yourusername/AI-Practices/issues)
2. 创建新issue询问
3. 联系维护者

## 🙏 致谢

感谢所有贡献者的付出！

你的名字将出现在贡献者列表中。

---

再次感谢你的贡献！🎉
