# Getting Started

This guide will help you get started with AI-Practices.

## Prerequisites

| Component | Minimum | Recommended |
|:----------|:--------|:------------|
| **Python** | 3.10 | 3.10 ~ 3.11 |
| **CPU** | 4 cores | 8+ cores |
| **RAM** | 8 GB | 32 GB |
| **GPU** | GTX 1060 | RTX 3080+ |
| **Storage** | 50 GB | 200 GB SSD |

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/zimingttkx/AI-Practices.git
cd AI-Practices
```

### 2. Create Environment

```bash
conda create -n ai-practices python=3.10 -y
conda activate ai-practices
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify Installation

```bash
python -c "import tensorflow as tf; print(f'TensorFlow: {tf.__version__}')"
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
```

### 5. Launch Jupyter

```bash
jupyter lab
```

## Run First Experiment

```bash
cd 09-practical-projects/02-computer-vision/01-mnist-cnn
python src/train.py --epochs 20 --batch_size 64
```

**Expected Output:**
```
Epoch 20/20 - loss: 0.0234 - accuracy: 0.9921
Test Accuracy: 99.12%
```

## Learning Path

### Beginner (8-12 weeks)

```
Week 1-2:  01-foundations (Part 1)
Week 3-4:  01-foundations (Part 2)
Week 5-6:  02-neural-networks
Week 7-8:  03-computer-vision
Week 9-10: 04-sequence-models
Week 11-12: 09-practical-projects
```

### Advanced (4-6 weeks)

```
Week 1-2: 05-advanced-topics
Week 3-4: 06-generative-models
Week 5-6: 07-reinforcement-learning
```

## Next Steps

- [Installation](./installation) - Detailed installation guide
- [Architecture](./architecture) - System architecture
- [Module 01](../modules/01-foundations) - Start learning
