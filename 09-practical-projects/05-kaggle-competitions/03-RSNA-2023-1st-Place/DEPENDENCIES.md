# 依赖安装指南

## 快速开始

### 基础依赖 (必需)
```bash
pip install numpy>=1.21.0 pandas>=1.3.0 tqdm>=4.60.0
```

### 深度学习依赖
```bash
pip install torch>=2.0.0 torchvision>=0.15.0
pip install timm>=0.9.0
pip install transformers>=4.30.0
```

### 医学图像处理
```bash
pip install nibabel>=3.2.0
pip install pydicom>=2.3.0
pip install opencv-python>=4.5.0
pip install albumentations>=1.3.0
pip install segmentation-models-pytorch>=0.3.0
```

### dicomsdl (特殊依赖)
```bash
# 需要从Kaggle数据集或GitHub安装
pip install dicomsdl==0.109.2
```

## 完整requirements.txt

```text
# requirements.txt
# Core dependencies
numpy>=1.21.0,<2.0.0
pandas>=1.3.0
scipy>=1.7.0
tqdm>=4.60.0

# Deep Learning
torch>=2.0.0
torchvision>=0.15.0
timm>=0.9.0
transformers>=4.30.0

# Image Processing
opencv-python>=4.5.0
albumentations>=1.3.0
Pillow>=9.0.0

# Medical Imaging
nibabel>=3.2.0
pydicom>=2.3.0
dicomsdl>=0.109.2

# ML Libraries
segmentation-models-pytorch>=0.3.0
scikit-learn>=1.0.0

# Utilities
matplotlib>=3.5.0
```

## 验证安装

```bash
python -c "import torch; import numpy; import pandas; print('Core dependencies OK')"
python -c "import cv2; import albumentations; print('Image processing OK')"
python -c "import nibabel; import pydicom; print('Medical imaging OK')"
python test_basic.py
```

## 故障排除

### 问题: torch导入失败
```bash
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 问题: dicomsdl安装失败
从Kaggle数据集手动安装或使用预编译wheel

### 问题: OpenCV导入失败
```bash
pip uninstall opencv-python opencv-python-headless
pip install opencv-python-headless==4.5.5.64
```
