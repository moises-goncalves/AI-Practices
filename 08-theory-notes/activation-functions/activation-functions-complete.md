# Activation Functions - Complete Guide

> **知识密度**：⭐⭐⭐⭐⭐ | **实战价值**：⭐⭐⭐⭐⭐
> **最后更新**：2025-11-30

---



## 📋 本章知识图谱

```
Activation Functions - Complete Guide
├── 核心概念
│   ├── 基本原理
│   ├── 数学基础
│   └── 应用场景
├── 算法详解
│   ├── 算法流程
│   ├── 时间复杂度
│   └── 空间复杂度
├── 实战技巧
│   ├── 参数调优
│   ├── 性能优化
│   └── 常见陷阱
└── 代码实现
    ├── 基础实现
    ├── 高级技巧
    └── 完整示例
```

---

## 📚 Overview

Activation functions are the "switches" of neural networks, determining whether and how signals are transmitted through neurons. This comprehensive guide covers 30+ activation functions with theory, mathematics, use cases, and practical recommendations.

## 🎯 Table of Contents

1. [What Are Activation Functions?](#what-are-activation-functions)
2. [Why Do We Need Them?](#why-do-we-need-them)
3. [Classic Activation Functions](#classic-activation-functions)
4. [ReLU Family](#relu-family)
5. [Modern High-Performance Functions](#modern-high-performance-functions)
6. [Gated Mechanisms](#gated-mechanisms)
7. [Transformer & LLM Specialized](#transformer--llm-specialized)
8. [Lightweight & Edge Device](#lightweight--edge-device)
9. [Special Purpose & Research](#special-purpose--research)
10. [Selection Guide](#selection-guide)
11. [Best Practices](#best-practices)

---

## What Are Activation Functions?

### Definition

An **activation function** is a mathematical function applied to the output of a neuron that introduces non-linearity into the network, enabling it to learn complex patterns.

**Mathematical Form**:
```
output = activation(weighted_sum + bias)
output = f(Σ(w_i × x_i) + b)
```

### Key Properties

1. **Non-linearity**: Enables learning of complex, non-linear relationships
2. **Differentiability**: Required for backpropagation (gradient-based learning)
3. **Range**: Output bounds affect network stability
4. **Monotonicity**: Whether function always increases/decreases
5. **Zero-centered**: Whether output is centered around zero

---

## Why Do We Need Them?

### The Problem with Linear Functions

Without activation functions (or with only linear activations), a neural network is equivalent to a single-layer linear model, regardless of depth:

```
Layer 1: y₁ = W₁x + b₁
Layer 2: y₂ = W₂y₁ + b₂ = W₂(W₁x + b₁) + b₂ = (W₂W₁)x + (W₂b₁ + b₂)
```

This collapses to: `y = Wx + b` (a single linear transformation)

### What Non-linearity Provides

1. **Universal Approximation**: Can approximate any continuous function
2. **Feature Hierarchy**: Learn increasingly abstract representations
3. **Decision Boundaries**: Create complex, non-linear decision boundaries
4. **Expressiveness**: Model real-world phenomena (which are rarely linear)

---

## Classic Activation Functions

### 1. Sigmoid

**Formula**:
```
σ(x) = 1 / (1 + e^(-x))
```

**Derivative**:
```
σ'(x) = σ(x) × (1 - σ(x))
```

**Properties**:
- **Range**: (0, 1)
- **Zero-centered**: No
- **Monotonic**: Yes (strictly increasing)
- **Differentiable**: Yes (everywhere)

**When to Use**:
- ✅ Binary classification output layer (probability interpretation)
- ✅ Gate mechanisms (LSTM, GRU)
- ❌ Hidden layers (causes vanishing gradient)

**Advantages**:
- Smooth, continuous output
- Clear probabilistic interpretation
- Bounded output prevents explosion

**Disadvantages**:
- **Vanishing gradient**: Saturates at extremes (gradient → 0)
- **Not zero-centered**: Causes zig-zagging in gradient descent
- **Computationally expensive**: Exponential operation

**Historical Note**: Dominant in early neural networks (1980s-2000s), now largely replaced by ReLU in hidden layers.

---

### 2. Tanh (Hyperbolic Tangent)

**Formula**:
```
tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
       = 2σ(2x) - 1
```

**Derivative**:
```
tanh'(x) = 1 - tanh²(x)
```

**Properties**:
- **Range**: (-1, 1)
- **Zero-centered**: Yes
- **Monotonic**: Yes
- **Differentiable**: Yes

**When to Use**:
- ✅ RNN/LSTM hidden states
- ✅ Output layer for regression in [-1, 1]
- ⚠️ Hidden layers (better than sigmoid, but still vanishing gradient)

**Advantages**:
- Zero-centered (better than sigmoid)
- Stronger gradients than sigmoid
- Symmetric around origin

**Disadvantages**:
- Still suffers from vanishing gradient
- Computationally expensive
- Saturates at extremes

---

### 3. Softmax

**Formula** (for vector input):
```
softmax(x_i) = e^(x_i) / Σⱼ e^(x_j)
```

**Properties**:
- **Range**: (0, 1) with Σ outputs = 1
- **Output**: Probability distribution
- **Differentiable**: Yes

**When to Use**:
- ✅ Multi-class classification output layer (REQUIRED)
- ❌ Never in hidden layers

**Advantages**:
- Converts logits to probabilities
- Differentiable
- Interpretable as confidence scores

**Disadvantages**:
- Numerically unstable without proper implementation
- Sensitive to outliers
- Computationally expensive for large number of classes

**Implementation Tip**:
```python
# Numerically stable softmax
def softmax_stable(x):
    exp_x = np.exp(x - np.max(x))  # Subtract max for stability
    return exp_x / np.sum(exp_x)
```

---

## ReLU Family

### 1. ReLU (Rectified Linear Unit)

**Formula**:
```
ReLU(x) = max(0, x)
```

**Derivative**:
```
ReLU'(x) = 1 if x > 0 else 0
```

**Properties**:
- **Range**: [0, ∞)
- **Zero-centered**: No
- **Monotonic**: Yes
- **Differentiable**: Almost everywhere (not at x=0)

**When to Use**:
- ✅ Default choice for hidden layers in CNNs, MLPs
- ✅ When training speed is critical
- ✅ Deep networks (doesn't saturate for positive values)

**Advantages**:
- **Computationally efficient**: Simple max operation
- **No vanishing gradient** for positive inputs
- **Sparse activation**: ~50% neurons are zero
- **Biological plausibility**: Similar to neuron firing

**Disadvantages**:
- **Dying ReLU problem**: Neurons can permanently "die" (always output 0)
- **Not zero-centered**: Can slow convergence
- **Unbounded**: Can lead to exploding activations

**The Dying ReLU Problem**:
- Occurs when large negative bias pushes neuron into negative region
- Gradient is always 0, so weights never update
- Can affect 10-40% of neurons in practice
- Solutions: Leaky ReLU, He initialization, lower learning rates

---

### 2. Leaky ReLU

**Formula**:
```
LeakyReLU(x) = x if x > 0 else αx
```
where α ≈ 0.01 (typically)

**Derivative**:
```
LeakyReLU'(x) = 1 if x > 0 else α
```

**When to Use**:
- ✅ When experiencing dying ReLU problem
- ✅ As default alternative to ReLU
- ✅ Deep networks

**Advantages**:
- Fixes dying ReLU (always has gradient)
- Nearly as efficient as ReLU
- Allows negative values (better gradient flow)

**Disadvantages**:
- Hyperparameter α needs tuning
- Not always better than ReLU in practice

---

### 3. PReLU (Parametric ReLU)

**Formula**:
```
PReLU(x) = x if x > 0 else αx
```
where α is **learnable** (different from Leaky ReLU)

**When to Use**:
- ✅ When you have enough data to learn α
- ✅ Small to medium networks
- ⚠️ Risk of overfitting on small datasets

**Advantages**:
- Adaptive negative slope
- Can outperform fixed-slope variants
- Minimal computational overhead

**Disadvantages**:
- Extra parameters to learn
- Can overfit
- Inconsistent across different channels

---

### 4. ELU (Exponential Linear Unit)

**Formula**:
```
ELU(x) = x if x > 0 else α(e^x - 1)
```

**Derivative**:
```
ELU'(x) = 1 if x > 0 else ELU(x) + α
```

**Properties**:
- **Range**: (-α, ∞)
- **Zero-centered**: Approximately (mean activation ≈ 0)
- **Smooth**: Continuous derivative

**When to Use**:
- ✅ Deep networks (better gradient flow)
- ✅ When training stability is important
- ✅ RNNs and autoencoders

**Advantages**:
- Negative values push mean activation toward zero
- Smooth everywhere (better optimization)
- Reduces bias shift
- No dying neuron problem

**Disadvantages**:
- Computationally expensive (exponential)
- Hyperparameter α needs tuning
- Slower than ReLU

---

### 5. SELU (Scaled Exponential Linear Unit)

**Formula**:
```
SELU(x) = λ × (x if x > 0 else α(e^x - 1))
```
where λ ≈ 1.0507, α ≈ 1.6733 (specific values for self-normalization)

**Properties**:
- **Self-normalizing**: Maintains mean ≈ 0, variance ≈ 1
- **Requires**: Specific initialization (LeCun normal) and architecture

**When to Use**:
- ✅ Fully connected networks (Self-Normalizing Neural Networks)
- ❌ CNNs (doesn't work well)
- ❌ With Batch Normalization (conflicts)

**Advantages**:
- Automatic normalization without BatchNorm
- Can train very deep networks
- Theoretical guarantees

**Disadvantages**:
- Strict requirements (initialization, architecture)
- Doesn't work with dropout or BatchNorm
- Limited to specific use cases

---

### 6. GELU (Gaussian Error Linear Unit)

**Formula** (exact):
```
GELU(x) = x × Φ(x) = x × (1/2)[1 + erf(x/√2)]
```

**Approximation** (commonly used):
```
GELU(x) ≈ 0.5x(1 + tanh[√(2/π)(x + 0.044715x³)])
```

**Properties**:
- **Range**: (-0.17, ∞)
- **Smooth**: Infinitely differentiable
- **Non-monotonic**: Has a small dip near x=0

**When to Use**:
- ✅ Transformers (BERT, GPT, etc.)
- ✅ Large language models
- ✅ Vision Transformers
- ✅ Modern architectures

**Advantages**:
- Smooth, probabilistic interpretation
- Better performance than ReLU in many tasks
- Theoretically motivated (stochastic regularization)
- State-of-the-art in NLP

**Disadvantages**:
- Computationally expensive
- Approximation needed for efficiency
- Not as simple as ReLU

**Why GELU Works**:
- Stochastically drops inputs based on their value
- Combines dropout-like regularization with activation
- Smooth gradients improve optimization

---

## Modern High-Performance Functions

### 1. Swish / SiLU

**Formula**:
```
Swish(x) = x × σ(βx)
SiLU(x) = x × σ(x)  # β = 1
```

**Properties**:
- **Range**: (-0.28, ∞)
- **Smooth**: Infinitely differentiable
- **Self-gated**: Output modulated by input

**When to Use**:
- ✅ Deep networks (alternative to ReLU)
- ✅ Mobile networks (MobileNet, EfficientNet)
- ✅ When performance matters more than speed

**Advantages**:
- Often outperforms ReLU
- Smooth (better optimization)
- Unbounded above, bounded below
- Self-gating mechanism

**Disadvantages**:
- More expensive than ReLU
- Requires sigmoid computation

**Discovery**: Found by Google using neural architecture search (AutoML)

---

### 2. Mish

**Formula**:
```
Mish(x) = x × tanh(softplus(x)) = x × tanh(ln(1 + e^x))
```

**Properties**:
- **Range**: (-0.31, ∞)
- **Smooth**: Infinitely differentiable
- **Self-regularizing**: Small negative values allowed

**When to Use**:
- ✅ Computer vision tasks
- ✅ When maximum performance is needed
- ⚠️ Research/experimental (computationally heavy)

**Advantages**:
- Often outperforms Swish and ReLU
- Smooth, unbounded above
- Strong empirical results

**Disadvantages**:
- Very computationally expensive
- Difficult to deploy
- Marginal gains over Swish

---

## Gated Mechanisms

### GLU (Gated Linear Unit)

**Formula**:
```
GLU(x) = x ⊙ σ(Wx + b)
```
where ⊙ is element-wise multiplication

**Concept**: Split input into two parts - one is the signal, other is the gate

**When to Use**:
- ✅ Sequence modeling
- ✅ Language models
- ✅ Transformer FFN layers

**Advantages**:
- Dynamic information flow control
- Better than standard activations in many tasks
- Flexible gating mechanism

**Disadvantages**:
- Doubles parameter count
- More complex than standard activations

---

## Transformer & LLM Specialized

### 1. GeGLU (Gated GELU)

**Formula**:
```
GeGLU(x) = x ⊙ GELU(Wx + b)
```

**When to Use**:
- ✅ Transformer FFN layers
- ✅ Large language models (T5, PaLM)
- ✅ Vision Transformers

**Used In**: T5, PaLM, Chinchilla, many modern LLMs

---

### 2. SwiGLU (Swish-Gated Linear Unit)

**Formula**:
```
SwiGLU(x) = x ⊙ Swish(Wx + b)
```

**When to Use**:
- ✅ **Current best practice for LLMs**
- ✅ Llama, Llama2, Llama3
- ✅ Modern transformer architectures

**Why It's Popular**:
- Best performance among GLU variants
- Smooth, stable training
- Proven at scale (billions of parameters)

**Used In**: Llama series, Phi-2, Falcon, many state-of-the-art LLMs

---

### 3. ReGLU (ReLU-Gated Linear Unit)

**Formula**:
```
ReGLU(x) = x ⊙ ReLU(Wx + b)
```

**When to Use**:
- ✅ When computational efficiency is critical
- ⚠️ Generally outperformed by GeGLU/SwiGLU

**Advantages**:
- Faster than GeGLU/SwiGLU
- Simpler computation

**Disadvantages**:
- Dying neuron problem
- Less smooth than alternatives

---

## Lightweight & Edge Device

### 1. Hard Swish

**Formula**:
```
HardSwish(x) = x × ReLU6(x + 3) / 6
```

**Properties**:
- Piecewise linear approximation of Swish
- No exponential operations
- Hardware-friendly

**When to Use**:
- ✅ Mobile deployment (MobileNetV3)
- ✅ Edge devices
- ✅ Quantized models
- ✅ Resource-constrained environments

**Advantages**:
- Much faster than Swish
- Quantization-friendly
- Near-Swish performance
- Low power consumption

**Disadvantages**:
- Slight accuracy loss vs Swish
- Piecewise nature (not smooth)

---

### 2. Hard Sigmoid

**Formula**:
```
HardSigmoid(x) = clip(0.2x + 0.5, 0, 1)
```

**When to Use**:
- ✅ Mobile/embedded systems
- ✅ Quantized networks
- ✅ LSTM/GRU gates on edge devices

**Advantages**:
- Very fast (no exponential)
- Easy to quantize
- Sufficient for many tasks

---

### 3. QuantReLU

**Concept**: ReLU with quantized outputs (discrete levels)

**When to Use**:
- ✅ Quantization-aware training (QAT)
- ✅ INT8/INT4 deployment
- ✅ Edge AI accelerators

**Advantages**:
- Enables low-bit inference
- Hardware-friendly
- Reduces memory and compute

**Disadvantages**:
- Accuracy loss
- Requires careful training

---

## Special Purpose & Research

### 1. Softplus

**Formula**:
```
Softplus(x) = ln(1 + e^x)
```

**Properties**:
- Smooth approximation of ReLU
- Always positive output
- Differentiable everywhere

**When to Use**:
- ✅ VAE (variance parameters)
- ✅ Reinforcement learning (policy networks)
- ✅ When positive outputs required

---

### 2. Gaussian

**Formula**:
```
Gaussian(x) = e^(-x²)
```

**When to Use**:
- ✅ Radial Basis Function (RBF) networks
- ✅ Local sensitivity modeling
- ❌ General deep learning (vanishing gradient)

---

### 3. Sine/Cosine

**Formula**:
```
Sine(x) = sin(x)
Cosine(x) = cos(x)
```

**When to Use**:
- ✅ Neural implicit representations (SIREN)
- ✅ Periodic signal modeling
- ✅ Fourier feature networks
- ❌ Standard classification/regression

**Special Use Case**: SIREN (Sinusoidal Representation Networks) for representing images, audio, 3D shapes

---

## Selection Guide

### Decision Tree

```
START
│
├─ Output Layer?
│  ├─ Binary Classification → Sigmoid
│  ├─ Multi-class Classification → Softmax
│  ├─ Regression (unbounded) → Linear
│  └─ Regression (bounded) → Tanh or Sigmoid
│
├─ Transformer/LLM?
│  ├─ FFN Layer → SwiGLU (best) or GeGLU
│  └─ Attention → Softmax
│
├─ Mobile/Edge Device?
│  ├─ Yes → Hard Swish or ReLU6
│  └─ No → Continue
│
├─ Maximum Performance?
│  ├─ NLP/Transformer → GELU or SwiGLU
│  ├─ Computer Vision → Swish or Mish
│  └─ General → GELU or Swish
│
├─ Computational Efficiency Critical?
│  ├─ Yes → ReLU or Leaky ReLU
│  └─ No → Continue
│
├─ Dying ReLU Problem?
│  ├─ Yes → Leaky ReLU or ELU
│  └─ No → ReLU
│
└─ Default → ReLU (start here)
```

---

### Quick Reference Table

| Task | Recommended | Alternative | Avoid |
|------|-------------|-------------|-------|
| **CNN Hidden Layers** | ReLU, Swish | Leaky ReLU, ELU | Sigmoid, Tanh |
| **Transformer FFN** | SwiGLU, GeGLU | GELU | ReLU |
| **RNN/LSTM** | Tanh (hidden), Sigmoid (gates) | - | ReLU |
| **Binary Classification Output** | Sigmoid | - | ReLU, Tanh |
| **Multi-class Output** | Softmax | - | Any other |
| **Regression Output** | Linear | Tanh (bounded) | ReLU |
| **Mobile/Edge** | Hard Swish, ReLU6 | ReLU | Mish, GELU |
| **LLM (Llama-style)** | SwiGLU | GeGLU | ReLU |
| **Vision Transformer** | GELU | Swish | ReLU |
| **Deep Networks (>50 layers)** | ELU, SELU | Leaky ReLU | Sigmoid |

---

## Best Practices

### 1. Initialization Matters

Different activations require different initialization strategies:

**ReLU Family**:
```python
# He initialization (Kaiming)
nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
```

**Tanh/Sigmoid**:
```python
# Xavier/Glorot initialization
nn.init.xavier_normal_(layer.weight)
```

**SELU**:
```python
# LeCun initialization
nn.init.normal_(layer.weight, mean=0, std=1/sqrt(fan_in))
```

---

### 2. Batch Normalization Interaction

**Compatible**:
- ReLU, Leaky ReLU, Swish, GELU
- Use BatchNorm → Activation order

**Incompatible**:
- SELU (designed to work without normalization)
- EvoNorm (combines normalization and activation)

**Best Practice**:
```python
# Standard pattern
nn.Conv2d(in_channels, out_channels, kernel_size)
nn.BatchNorm2d(out_channels)
nn.ReLU()  # or other activation
```

---

### 3. Gradient Clipping

For activations prone to exploding gradients (ReLU, Linear):

```python
# PyTorch
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# TensorFlow
optimizer = tf.keras.optimizers.Adam(clipnorm=1.0)
```

---

### 4. Learning Rate Adjustment

Different activations may require different learning rates:

- **ReLU**: Standard learning rates (1e-3 to 1e-4)
- **GELU/Swish**: Slightly lower (5e-4 to 1e-4)
- **SELU**: Specific learning rate schedules
- **Sigmoid/Tanh**: Lower learning rates (1e-4 to 1e-5)

---

### 5. Debugging Dead Neurons

**Check activation statistics**:
```python
def check_dead_neurons(activations):
    """Check percentage of dead neurons (always zero)"""
    dead = (activations == 0).all(dim=0).float().mean()
    print(f"Dead neurons: {dead.item()*100:.2f}%")

# During training
activations = model.get_activations(x)
check_dead_neurons(activations)
```

**Solutions**:
- Lower learning rate
- Use Leaky ReLU or ELU
- Better initialization
- Reduce batch size

---

### 6. Mixed Activation Strategies

You can use different activations in different parts of the network:

```python
class HybridNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3)
        self.relu = nn.ReLU()  # Fast for early layers

        self.conv2 = nn.Conv2d(64, 128, 3)
        self.swish = nn.SiLU()  # Better for deeper layers

        self.fc = nn.Linear(128, 10)
        # No activation (logits for softmax)
```

---

## Common Pitfalls

### ❌ Don't Do This

1. **Using Sigmoid/Tanh in deep hidden layers**
   - Causes severe vanishing gradient
   - Use ReLU or modern alternatives

2. **Forgetting activation in output layer**
   ```python
   # Wrong for classification
   output = nn.Linear(hidden, num_classes)(x)

   # Correct
   logits = nn.Linear(hidden, num_classes)(x)
   output = nn.Softmax(dim=-1)(logits)
   ```

3. **Using ReLU after final layer for regression**
   - Limits output to positive values
   - Use Linear (no activation) for unbounded regression

4. **Mixing SELU with BatchNorm**
   - SELU is self-normalizing, conflicts with BatchNorm
   - Use one or the other, not both

5. **Using wrong initialization with activation**
   - ReLU with Xavier init → suboptimal
   - Use He init for ReLU, Xavier for Tanh

---

## Performance Comparison

### Computational Cost (Relative to ReLU = 1.0)

| Activation | Relative Cost | Memory |
|------------|---------------|--------|
| ReLU | 1.0× | Low |
| Leaky ReLU | 1.1× | Low |
| ELU | 2.5× | Low |
| GELU | 3.0× | Low |
| Swish | 2.8× | Low |
| Mish | 4.5× | Low |
| Hard Swish | 1.3× | Low |
| SwiGLU | 3.5× | High (2× params) |

### Accuracy (Typical Improvement over ReLU)

| Task | GELU | Swish | Mish | SwiGLU |
|------|------|-------|------|--------|
| Image Classification | +0.5% | +0.3% | +0.7% | N/A |
| Language Modeling | +1.2% | +0.8% | +0.5% | +1.5% |
| Object Detection | +0.4% | +0.6% | +0.9% | N/A |

*Note: Improvements vary by architecture and dataset*

---

## 📖 References

### Seminal Papers

1. **ReLU**: Nair & Hinton (2010) - "Rectified Linear Units Improve Restricted Boltzmann Machines"
2. **ELU**: Clevert et al. (2015) - "Fast and Accurate Deep Network Learning by Exponential Linear Units"
3. **SELU**: Klambauer et al. (2017) - "Self-Normalizing Neural Networks"
4. **Swish**: Ramachandran et al. (2017) - "Searching for Activation Functions"
5. **GELU**: Hendrycks & Gimpel (2016) - "Gaussian Error Linear Units"
6. **Mish**: Misra (2019) - "Mish: A Self Regularized Non-Monotonic Activation Function"
7. **GLU Variants**: Shazeer (2020) - "GLU Variants Improve Transformer"

### Books

1. **"Deep Learning"** - Goodfellow, Bengio, Courville (Chapter 6.3)
2. **"Hands-On Machine Learning"** - Aurélien Géron (Chapter 11)
3. **"Deep Learning with Python"** - François Chollet (Chapter 4)

### Online Resources

1. [PyTorch Activation Functions](https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity)
2. [TensorFlow Activations](https://www.tensorflow.org/api_docs/python/tf/keras/activations)
3. [Papers with Code - Activation Functions](https://paperswithcode.com/methods/category/activation-functions)

---

## 🎯 Key Takeaways

1. **Start with ReLU** - It's the default for good reason (simple, effective, fast)

2. **Upgrade strategically**:
   - Transformers → GELU or SwiGLU
   - Mobile → Hard Swish or ReLU6
   - Deep networks → ELU or Leaky ReLU
   - Maximum performance → Swish or Mish

3. **Match initialization to activation**:
   - ReLU → He initialization
   - Tanh/Sigmoid → Xavier initialization
   - SELU → LeCun initialization

4. **Output layer is special**:
   - Binary classification → Sigmoid
   - Multi-class → Softmax
   - Regression → Linear (usually)

5. **Modern = Better (usually)**:
   - GELU > ReLU for Transformers
   - Swish > ReLU for many tasks
   - SwiGLU > standard FFN for LLMs

6. **Context matters**:
   - Research: Try Mish, GELU, Swish
   - Production: ReLU, Hard Swish (speed matters)
   - LLMs: SwiGLU (proven at scale)

7. **Don't overthink it**:
   - Activation choice matters, but less than architecture, data, and training
   - ReLU is still excellent for most tasks
   - Upgrade only when you have evidence it helps

---

## ✅ Practice Exercises

### Beginner

1. Implement ReLU, Sigmoid, and Tanh from scratch using only NumPy
2. Visualize activation functions and their derivatives
3. Compare training speed of ReLU vs Sigmoid on MNIST

### Intermediate

1. Implement Swish and GELU from scratch
2. Compare ReLU, Leaky ReLU, and ELU on a deep network (>20 layers)
3. Analyze dead neuron percentage with different activations
4. Implement proper initialization for different activations

### Advanced

1. Implement SwiGLU and compare with standard FFN in a Transformer
2. Design a custom activation function and test it
3. Analyze gradient flow through different activations
4. Implement quantization-friendly activation (Hard Swish) and measure speedup

---

*Last updated: 2025-11-29*
*Related notebook: `ActivationFunctions.ipynb`*
*Comparison table: See notebook for comprehensive 30+ function comparison*


## ✅ 最佳实践

### 使用建议
1. **数据预处理**：
   - ⚠️ 注意事项1
   - ✅ 推荐做法1

2. **参数选择**：
   - ⚠️ 注意事项2
   - ✅ 推荐做法2

3. **性能优化**：
   - ⚠️ 注意事项3
   - ✅ 推荐做法3

### 常见陷阱

| 陷阱 | 原因 | 解决方案 |
|------|------|----------|
| 陷阱1 | 原因说明 | 解决方法 |
| 陷阱2 | 原因说明 | 解决方法 |
| 陷阱3 | 原因说明 | 解决方法 |

---
