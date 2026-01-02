# 微调方法 (Fine-tuning Methods)

参数高效微调技术的实现，涵盖 LoRA、QLoRA、Adapter 等方法。

## 核心概念

**参数高效微调 (PEFT)**: 只更新少量参数，实现高效适配

$$\Delta W = BA, \quad B \in \mathbb{R}^{d \times r}, A \in \mathbb{R}^{r \times k}, r \ll \min(d, k)$$

## 目录结构

```
03-fine-tuning/
├── src/                          # 源代码
│   ├── lora.py                   # LoRA实现
│   ├── qlora.py                  # QLoRA量化微调
│   └── trainer.py                # 微调训练器
├── notebooks/                    # 交互式教程
│   ├── lora_finetuning.ipynb     # LoRA微调教程
│   └── qlora_finetuning.ipynb    # QLoRA微调教程
├── knowledge_points.md           # 知识点详解
└── README.md
```

## 核心组件

| 组件 | 文件 | 说明 |
|------|------|------|
| **LoRA** | `lora.py` | 低秩适配，高效微调 |
| **QLoRA** | `qlora.py` | 4-bit量化 + LoRA |
| **Trainer** | `trainer.py` | 统一训练接口 |

## 方法对比

| 方法 | 可训练参数 | 显存占用 | 性能 |
|------|-----------|---------|------|
| 全量微调 | 100% | 高 | 最佳 |
| LoRA | 0.1-1% | 中 | 接近全量 |
| QLoRA | 0.1-1% | 低 | 接近LoRA |
| Adapter | 1-5% | 中 | 良好 |

## 快速开始

```python
from src.lora import LoRAConfig, apply_lora_to_model
from src.trainer import FineTuneTrainer

# 配置LoRA
lora_config = LoRAConfig(r=8, alpha=16, target_modules=["q_proj", "v_proj"])

# 应用LoRA
model = apply_lora_to_model(base_model, lora_config)

# 训练
trainer = FineTuneTrainer(model, train_dataset, eval_dataset)
trainer.train()
```

## 参考文献

1. Hu et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models
2. Dettmers et al. (2023). QLoRA: Efficient Finetuning of Quantized LLMs
3. Houlsby et al. (2019). Parameter-Efficient Transfer Learning for NLP
