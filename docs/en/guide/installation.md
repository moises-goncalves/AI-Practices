# Installation

Detailed installation guide covering multiple methods.

## Method 1: Conda (Recommended)

```bash
# Clone repository
git clone https://github.com/zimingttkx/AI-Practices.git
cd AI-Practices

# Create environment
conda create -n ai-practices python=3.10 -y
conda activate ai-practices

# Install dependencies
pip install -r requirements.txt
```

## Method 2: Docker

```bash
# Build image
docker build -t ai-practices .

# Run container (GPU)
docker run -it --gpus all -v $(pwd):/workspace ai-practices
```

## GPU Setup

### NVIDIA GPU

```bash
# Install driver
sudo apt install nvidia-driver-535

# Verify
nvidia-smi
```

### PyTorch with CUDA

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### TensorFlow with CUDA

```bash
pip install tensorflow[and-cuda]
```

### Apple Silicon

```bash
pip install tensorflow-macos tensorflow-metal
```

## Verification

```python
import tensorflow as tf
import torch

print(f"TensorFlow: {tf.__version__}")
print(f"GPU: {tf.config.list_physical_devices('GPU')}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
```
