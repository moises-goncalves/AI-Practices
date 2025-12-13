# 快速开始

本指南将帮助你快速上手 AI-Practices 项目。

## 环境要求

- **Python**: 3.10+
- **操作系统**: Windows / macOS / Linux
- **GPU** (可选): NVIDIA GPU with CUDA 12.1+

## 安装步骤

### 方式一：使用 Conda (推荐)

```bash
# 1. 克隆仓库
git clone https://github.com/zimingttkx/AI-Practices.git
cd AI-Practices

# 2. 创建 Conda 环境
conda create -n ai-practices python=3.10 -y
conda activate ai-practices

# 3. 安装依赖
pip install -r requirements.txt
```

### 方式二：使用 Docker

```bash
# 构建镜像
docker build -t ai-practices .

# 运行容器 (GPU 支持)
docker run -it --gpus all -v $(pwd):/workspace ai-practices
```

### 方式三：使用 environment.yml

```bash
conda env create -f environment.yml
conda activate ai-practices
```

## GPU 支持 (可选)

```bash
# PyTorch with CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# TensorFlow with CUDA
pip install tensorflow[and-cuda]
```

## 验证安装

```bash
# 验证 TensorFlow
python -c "import tensorflow as tf; print(f'TensorFlow: {tf.__version__}')"

# 验证 PyTorch
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# 验证 scikit-learn
python -c "import sklearn; print(f'scikit-learn: {sklearn.__version__}')"
```

## 运行第一个示例

```bash
# 进入 MNIST 项目
cd 09-practical-projects/02-computer-vision/01-mnist-cnn

# 下载数据
python src/data.py --download

# 训练模型
python src/train.py --model improved_cnn --epochs 10

# 查看结果
python src/evaluate.py --checkpoint runs/improved_cnn.best.pt
```

## 启动 JupyterLab

```bash
# 启动 JupyterLab
jupyter lab --port=8888

# 在浏览器中打开 http://localhost:8888
```

## 下一步

- 📚 [项目介绍](/guide/introduction) - 了解项目的设计理念
- 🗂️ [项目结构](/guide/project-structure) - 熟悉目录组织方式
- 🧭 [学习路径](/roadmap) - 规划你的学习路线
- 📖 [01-Foundations](/modules/01-foundations) - 开始学习机器学习基础
