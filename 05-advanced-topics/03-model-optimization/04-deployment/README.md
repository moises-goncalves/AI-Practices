# 模型部署 (Model Deployment)

## 概述

将训练好的模型部署到生产环境，包括格式转换、推理优化和跨平台部署。

## 核心概念

| 技术 | 用途 | 加速比 | 平台 |
|:-----|:-----|:------:|:-----|
| **ONNX** | 跨框架格式 | 1-2x | 通用 |
| **TensorRT** | GPU 推理优化 | 2-5x | NVIDIA GPU |
| **TorchScript** | PyTorch 部署 | 1.5-2x | 通用 |
| **Mobile** | 移动端部署 | - | iOS/Android |

## 学习路径

| 序号 | Notebook | 内容 | 难度 |
|:----:|:---------|:-----|:----:|
| 1 | `onnx_export.ipynb` | ONNX 导出、验证、优化 | ⭐⭐⭐ |
| 2 | `tensorrt_optimization.ipynb` | TensorRT 加速、INT8 量化 | ⭐⭐⭐⭐ |
| 3 | `torchscript_deployment.ipynb` | TorchScript、JIT 编译 | ⭐⭐⭐ |

## 部署流程

```
训练模型 → 模型优化 → 格式转换 → 推理引擎 → 生产部署
   ↓          ↓          ↓          ↓          ↓
PyTorch   量化/剪枝    ONNX     TensorRT   Docker/K8s
```

## 参考文献

- [ONNX Documentation](https://onnx.ai/onnx/)
- [TensorRT Developer Guide](https://docs.nvidia.com/deeplearning/tensorrt/)
- [TorchScript Documentation](https://pytorch.org/docs/stable/jit.html)
