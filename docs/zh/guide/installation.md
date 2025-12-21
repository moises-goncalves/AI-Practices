# 安装配置

详细的安装指南，涵盖多种安装方式。

## 方式一：Conda (推荐)

```bash
# 安装 Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b

# 克隆仓库
git clone https://github.com/zimingttkx/AI-Practices.git
cd AI-Practices

# 创建环境
conda create -n ai-practices python=3.10 -y
conda activate ai-practices

# 安装依赖
pip install -r requirements.txt
```

## 方式二：Docker

```bash
# 构建镜像
docker build -t ai-practices .

# 运行容器 (GPU)
docker run -it --gpus all -v $(pwd):/workspace ai-practices
```

## GPU 配置

### NVIDIA GPU

```bash
# 安装驱动
sudo apt install nvidia-driver-535

# 验证
nvidia-smi
```

### PyTorch GPU

```bash
# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### TensorFlow GPU

```bash
pip install tensorflow[and-cuda]
```

### Apple Silicon

```bash
pip install tensorflow-macos tensorflow-metal
```

## 验证脚本

```python
import tensorflow as tf
import torch

# TensorFlow
print(f"TensorFlow: {tf.__version__}")
print(f"GPU: {tf.config.list_physical_devices('GPU')}")

# PyTorch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
```

## 常见问题

### CUDA 不可用

检查 CUDA 和 cuDNN 版本是否匹配框架要求。

### 内存不足

```python
# TensorFlow
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
```

### 包版本冲突

```bash
conda env remove -n ai-practices
conda create -n ai-practices python=3.10 -y
pip install -r requirements.txt
```
