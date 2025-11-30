# Loss Functions - Complete Guide

> **知识密度**：⭐⭐⭐⭐⭐ | **实战价值**：⭐⭐⭐⭐⭐
> **最后更新**：2025-11-30

---



## 📋 本章知识图谱

```
Loss Functions - Complete Guide
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

Loss functions (also called cost functions or objective functions) are the compass that guides neural network training. They quantify how well the model's predictions match the true labels, providing the feedback signal for optimization.

## 🎯 Table of Contents

1. [What Are Loss Functions?](#what-are-loss-functions)
2. [Regression Loss Functions](#regression-loss-functions)
3. [Classification Loss Functions](#classification-loss-functions)
4. [Ranking & Similarity Loss Functions](#ranking--similarity-loss-functions)
5. [Advanced & Specialized Loss Functions](#advanced--specialized-loss-functions)
6. [Selection Guide](#selection-guide)
7. [Best Practices](#best-practices)

---

## What Are Loss Functions?

### Definition

A **loss function** L(ŷ, y) measures the discrepancy between predicted values ŷ and true values y. The goal of training is to minimize this loss:

```
θ* = argmin_θ (1/N) Σᵢ L(f(xᵢ; θ), yᵢ)
```

where:
- θ = model parameters
- f(x; θ) = model prediction
- N = number of samples

### Key Properties

1. **Non-negative**: L(ŷ, y) ≥ 0
2. **Zero at perfect prediction**: L(y, y) = 0
3. **Differentiable**: Required for gradient-based optimization
4. **Task-appropriate**: Must match the problem type

### Loss vs Cost vs Objective

- **Loss**: Error for a single example
- **Cost**: Average loss over entire dataset
- **Objective**: General term (may include regularization)

```
Cost = (1/N) Σᵢ Loss(ŷᵢ, yᵢ) + λ × Regularization
```

---

## Regression Loss Functions

### 1. Mean Squared Error (MSE) / L2 Loss

**Formula**:
```
MSE = (1/N) Σᵢ (yᵢ - ŷᵢ)²
```

**Per-sample**:
```
L(y, ŷ) = (y - ŷ)²
```

**Gradient**:
```
∂L/∂ŷ = -2(y - ŷ)
```

**Properties**:
- **Range**: [0, ∞)
- **Sensitivity**: Very sensitive to outliers (quadratic penalty)
- **Units**: Squared units of target variable

**When to Use**:
- ✅ Default choice for regression
- ✅ When outliers should be heavily penalized
- ✅ Gaussian noise assumption
- ❌ When data has many outliers

**Advantages**:
- Smooth, continuous gradient
- Convex (for linear models)
- Penalizes large errors heavily
- Well-studied, stable optimization

**Disadvantages**:
- Sensitive to outliers
- Not robust to noise
- Squared units (harder to interpret)

**Example Use Cases**:
- House price prediction
- Temperature forecasting
- Stock price prediction

---

### 2. Mean Absolute Error (MAE) / L1 Loss

**Formula**:
```
MAE = (1/N) Σᵢ |yᵢ - ŷᵢ|
```

**Per-sample**:
```
L(y, ŷ) = |y - ŷ|
```

**Gradient**:
```
∂L/∂ŷ = -sign(y - ŷ)
```

**Properties**:
- **Range**: [0, ∞)
- **Sensitivity**: Robust to outliers (linear penalty)
- **Units**: Same units as target variable

**When to Use**:
- ✅ When data has outliers
- ✅ When all errors should be weighted equally
- ✅ Laplacian noise assumption
- ❌ When you need smooth gradients

**Advantages**:
- Robust to outliers
- Same units as target (interpretable)
- Linear penalty (treats all errors equally)

**Disadvantages**:
- Non-smooth at zero (gradient discontinuity)
- Slower convergence than MSE
- May not converge to exact minimum

**Comparison with MSE**:
```
Error = 1:  MSE = 1,   MAE = 1
Error = 2:  MSE = 4,   MAE = 2
Error = 10: MSE = 100, MAE = 10
```
MSE penalizes large errors much more heavily!

---

### 3. Huber Loss (Smooth L1)

**Formula**:
```
L_δ(y, ŷ) = {
    0.5(y - ŷ)²           if |y - ŷ| ≤ δ
    δ|y - ŷ| - 0.5δ²      otherwise
}
```

**Properties**:
- Combines MSE (small errors) and MAE (large errors)
- Smooth everywhere
- Robust to outliers

**When to Use**:
- ✅ When you want robustness AND smooth gradients
- ✅ Object detection (bounding box regression)
- ✅ Data with moderate outliers

**Advantages**:
- Best of both worlds (MSE + MAE)
- Smooth gradients
- Robust to outliers
- Tunable via δ parameter

**Disadvantages**:
- Extra hyperparameter δ to tune
- More complex than MSE/MAE

**Choosing δ**:
- Small δ → More like MAE (robust)
- Large δ → More like MSE (smooth)
- Typical: δ = 1.0

---

### 4. Log-Cosh Loss

**Formula**:
```
L(y, ŷ) = Σᵢ log(cosh(ŷᵢ - yᵢ))
```

**Properties**:
- Smooth approximation of MAE
- Approximately MSE for small errors
- Approximately MAE for large errors

**When to Use**:
- ✅ Alternative to Huber loss
- ✅ When you want smooth gradients everywhere
- ✅ XGBoost, LightGBM regression

**Advantages**:
- Smooth everywhere (twice differentiable)
- Robust to outliers
- No hyperparameters

**Disadvantages**:
- Computationally expensive (cosh, log)
- Less interpretable

---

### 5. Quantile Loss (Pinball Loss)

**Formula**:
```
L_τ(y, ŷ) = {
    τ(y - ŷ)      if y ≥ ŷ
    (τ-1)(y - ŷ)  if y < ŷ
}
```

where τ ∈ (0, 1) is the quantile

**When to Use**:
- ✅ Quantile regression
- ✅ Prediction intervals
- ✅ Asymmetric cost of errors

**Special Cases**:
- τ = 0.5: Equivalent to MAE (median regression)
- τ = 0.9: 90th percentile prediction

**Example**: Inventory management
- Overestimation cost ≠ Underestimation cost
- Use τ to balance costs

---

## Classification Loss Functions

### 1. Binary Cross-Entropy (BCE) / Log Loss

**Formula** (for single sample):
```
BCE = -[y log(ŷ) + (1-y) log(1-ŷ)]
```

where:
- y ∈ {0, 1} (true label)
- ŷ ∈ (0, 1) (predicted probability)

**Batch form**:
```
BCE = -(1/N) Σᵢ [yᵢ log(ŷᵢ) + (1-yᵢ) log(1-ŷᵢ)]
```

**Gradient**:
```
∂L/∂ŷ = -(y/ŷ - (1-y)/(1-ŷ))
```

**Properties**:
- **Range**: [0, ∞)
- **Interpretation**: Negative log-likelihood
- **Requires**: ŷ ∈ (0, 1) (use sigmoid activation)

**When to Use**:
- ✅ Binary classification (REQUIRED)
- ✅ Multi-label classification (each label independently)
- ❌ Multi-class classification (use categorical CE instead)

**Advantages**:
- Probabilistic interpretation
- Smooth, continuous gradient
- Penalizes confident wrong predictions heavily
- Well-suited for logistic regression

**Disadvantages**:
- Sensitive to class imbalance
- Requires probability outputs
- Can be numerically unstable (log(0))

**Numerical Stability**:
```python
# Unstable
loss = -y * log(sigmoid(z)) - (1-y) * log(1 - sigmoid(z))

# Stable (use logits directly)
loss = log(1 + exp(-z)) if y==1 else log(1 + exp(z))
```

---

### 2. Categorical Cross-Entropy

**Formula**:
```
CCE = -Σⱼ yⱼ log(ŷⱼ)
```

where:
- y = one-hot encoded true label [0, 0, 1, 0, ...]
- ŷ = predicted probability distribution (from softmax)

**Simplified** (since only one yⱼ = 1):
```
CCE = -log(ŷ_c)
```
where c is the true class

**When to Use**:
- ✅ Multi-class classification (REQUIRED)
- ✅ Mutually exclusive classes
- ❌ Multi-label problems (use BCE instead)

**Advantages**:
- Standard for multi-class problems
- Probabilistic interpretation
- Works well with softmax

**Disadvantages**:
- Sensitive to class imbalance
- Doesn't account for class similarity

**With Logits** (more stable):
```python
# Instead of: softmax → cross_entropy
# Use: cross_entropy_with_logits (combines operations)
loss = log(Σⱼ exp(zⱼ)) - z_c
```

---

### 3. Sparse Categorical Cross-Entropy

**Same as Categorical CE**, but:
- Input: Integer class labels (not one-hot)
- More memory efficient
- Identical mathematically

**When to Use**:
- ✅ Multi-class with many classes (e.g., 1000+ classes)
- ✅ Save memory (no one-hot encoding)

---

### 4. Focal Loss

**Formula**:
```
FL(p_t) = -α_t (1 - p_t)^γ log(p_t)
```

where:
- p_t = model's estimated probability for true class
- γ ≥ 0 (focusing parameter, typically 2)
- α_t = class weight

**Intuition**:
- Down-weights easy examples (high p_t)
- Focuses on hard examples (low p_t)
- Reduces impact of class imbalance

**When to Use**:
- ✅ Severe class imbalance (e.g., 1:1000)
- ✅ Object detection (RetinaNet)
- ✅ When easy examples dominate training

**Advantages**:
- Handles extreme imbalance
- Focuses on hard examples
- Improves rare class performance

**Disadvantages**:
- Extra hyperparameters (γ, α)
- More complex than standard CE
- Requires tuning

**Comparison with CE**:
```
p_t = 0.9 (easy example):
  CE:    -log(0.9) = 0.105
  Focal: -(1-0.9)² log(0.9) = 0.001  (99% reduction!)

p_t = 0.1 (hard example):
  CE:    -log(0.1) = 2.303
  Focal: -(1-0.1)² log(0.1) = 1.863  (19% reduction)
```

---

### 5. Hinge Loss (SVM Loss)

**Formula** (binary):
```
L(y, ŷ) = max(0, 1 - y·ŷ)
```

where:
- y ∈ {-1, +1}
- ŷ ∈ ℝ (decision function output, not probability)

**Multi-class** (one-vs-all):
```
L = Σⱼ≠c max(0, ŷⱼ - ŷ_c + Δ)
```

**When to Use**:
- ✅ Support Vector Machines (SVMs)
- ✅ Maximum margin classification
- ✅ When you want margin-based learning

**Advantages**:
- Encourages large margin
- Sparse solutions (only support vectors matter)
- Robust to outliers

**Disadvantages**:
- Not probabilistic
- Non-smooth at margin boundary
- Less common in deep learning

---

### 6. Kullback-Leibler Divergence (KL Divergence)

**Formula**:
```
KL(P || Q) = Σᵢ P(i) log(P(i) / Q(i))
```

**Properties**:
- Measures how one probability distribution differs from another
- **Not symmetric**: KL(P||Q) ≠ KL(Q||P)
- **Non-negative**: KL(P||Q) ≥ 0
- **Zero iff identical**: KL(P||Q) = 0 ⟺ P = Q

**When to Use**:
- ✅ Variational Autoencoders (VAE)
- ✅ Knowledge distillation
- ✅ Comparing distributions
- ✅ Reinforcement learning (policy optimization)

**Interpretation**:
- Forward KL: KL(P||Q) - mean-seeking (covers all modes of P)
- Reverse KL: KL(Q||P) - mode-seeking (focuses on single mode)

---

## Ranking & Similarity Loss Functions

### 1. Contrastive Loss

**Formula**:
```
L = (1-Y) × 0.5 × D² + Y × 0.5 × max(0, m - D)²
```

where:
- D = distance between embeddings
- Y = 1 if similar, 0 if dissimilar
- m = margin

**When to Use**:
- ✅ Siamese networks
- ✅ Face verification
- ✅ Signature verification
- ✅ Learning embeddings

**Intuition**:
- Similar pairs: Minimize distance
- Dissimilar pairs: Push apart (at least margin m)

---

### 2. Triplet Loss

**Formula**:
```
L = max(0, D(a, p) - D(a, n) + margin)
```

where:
- a = anchor
- p = positive (same class as anchor)
- n = negative (different class)
- D = distance function (usually L2)

**When to Use**:
- ✅ Face recognition (FaceNet)
- ✅ Person re-identification
- ✅ Metric learning
- ✅ Embedding learning

**Advantages**:
- Learns relative similarities
- No need for explicit class labels during training
- Powerful for few-shot learning

**Disadvantages**:
- Requires triplet mining (hard negatives)
- Slow convergence
- Sensitive to margin hyperparameter

**Triplet Mining Strategies**:
- **Hard**: D(a,p) > D(a,n) (hardest negatives)
- **Semi-hard**: D(a,p) < D(a,n) < D(a,p) + margin
- **Easy**: D(a,n) > D(a,p) + margin (too easy, not useful)

---

### 3. Cosine Embedding Loss

**Formula**:
```
L = {
    1 - cos(x₁, x₂)           if y = 1
    max(0, cos(x₁, x₂) - m)   if y = -1
}
```

where:
- cos(x₁, x₂) = x₁·x₂ / (||x₁|| ||x₂||)
- y = 1 (similar), y = -1 (dissimilar)
- m = margin

**When to Use**:
- ✅ Text similarity
- ✅ Sentence embeddings
- ✅ Document similarity

---

## Advanced & Specialized Loss Functions

### 1. Dice Loss

**Formula**:
```
Dice = 1 - (2|X ∩ Y| + ε) / (|X| + |Y| + ε)
```

**When to Use**:
- ✅ Image segmentation
- ✅ Medical imaging
- ✅ Imbalanced segmentation tasks

**Advantages**:
- Handles class imbalance well
- Directly optimizes overlap metric
- Works well for small objects

---

### 2. IoU Loss (Intersection over Union)

**Formula**:
```
IoU = |X ∩ Y| / |X ∪ Y|
Loss = 1 - IoU
```

**When to Use**:
- ✅ Object detection (bounding boxes)
- ✅ Instance segmentation
- ✅ When IoU is the evaluation metric

**Variants**:
- **GIoU**: Generalized IoU (handles non-overlapping boxes)
- **DIoU**: Distance IoU (considers center distance)
- **CIoU**: Complete IoU (adds aspect ratio)

---

### 3. CTC Loss (Connectionist Temporal Classification)

**When to Use**:
- ✅ Speech recognition
- ✅ Handwriting recognition
- ✅ Sequence-to-sequence without alignment

**Advantages**:
- No need for frame-level alignment
- Handles variable-length sequences
- Standard for ASR (Automatic Speech Recognition)

---

### 4. Wasserstein Loss (Earth Mover's Distance)

**When to Use**:
- ✅ Generative Adversarial Networks (WGAN)
- ✅ Comparing distributions
- ✅ When KL divergence is problematic

**Advantages**:
- More stable than standard GAN loss
- Meaningful gradient everywhere
- Better for disjoint distributions

---

## Selection Guide

### By Task Type

| Task | Primary Loss | Alternative | Notes |
|------|-------------|-------------|-------|
| **Binary Classification** | Binary Cross-Entropy | Focal Loss (imbalanced) | Use sigmoid output |
| **Multi-class Classification** | Categorical Cross-Entropy | Focal Loss (imbalanced) | Use softmax output |
| **Multi-label Classification** | Binary Cross-Entropy | - | Apply per label |
| **Regression** | MSE | MAE, Huber | MSE for Gaussian noise |
| **Robust Regression** | MAE, Huber | Log-Cosh | When outliers present |
| **Object Detection** | Focal Loss + IoU Loss | - | Combine classification + localization |
| **Semantic Segmentation** | Cross-Entropy + Dice | Focal Loss | Pixel-wise classification |
| **Face Recognition** | Triplet Loss | Contrastive Loss | Metric learning |
| **Sequence Labeling** | CTC Loss | Cross-Entropy | No alignment needed |
| **GANs** | Wasserstein Loss | BCE | More stable training |

---

### Decision Tree

```
START
│
├─ Classification?
│  ├─ Binary → BCE
│  ├─ Multi-class (exclusive) → Categorical CE
│  ├─ Multi-label → BCE (per label)
│  └─ Imbalanced → Focal Loss
│
├─ Regression?
│  ├─ No outliers → MSE
│  ├─ With outliers → MAE or Huber
│  ├─ Quantile prediction → Quantile Loss
│  └─ Robust → Log-Cosh
│
├─ Segmentation?
│  ├─ Balanced → CE
│  ├─ Imbalanced → Dice + CE
│  └─ Small objects → Focal + Dice
│
├─ Object Detection?
│  └─ Focal Loss (classification) + IoU Loss (bbox)
│
├─ Metric Learning?
│  ├─ Pairs → Contrastive Loss
│  └─ Triplets → Triplet Loss
│
└─ Sequence (no alignment)?
    └─ CTC Loss
```

---

## Best Practices

### 1. Loss Function Design Principles

**Match the Task**:
- Classification → Cross-Entropy (probabilistic)
- Regression → MSE/MAE (distance-based)
- Ranking → Triplet/Contrastive (relative)

**Consider the Data**:
- Imbalanced → Focal Loss, class weights
- Outliers → MAE, Huber
- Small objects → Dice, Focal

**Evaluation Metric Alignment**:
- If evaluating with IoU → use IoU loss
- If evaluating with F1 → consider Dice loss
- If evaluating with accuracy → CE is fine

---

### 2. Handling Class Imbalance

**Method 1: Class Weights**
```python
# Inverse frequency weighting
class_weights = N_total / (N_classes * N_per_class)

# Example: [100, 900] samples
weights = [1000/(2*100), 1000/(2*900)] = [5.0, 0.56]
```

**Method 2: Focal Loss**
```python
# Automatically down-weights easy examples
focal_loss = -(1 - p_t)^γ * log(p_t)
```

**Method 3: Oversampling/Undersampling**
- SMOTE (Synthetic Minority Over-sampling)
- Random undersampling of majority class

---

### 3. Numerical Stability

**Problem**: log(0) = -∞, exp(large) = ∞

**Solutions**:

**For Cross-Entropy**:
```python
# Bad: separate softmax + log
probs = softmax(logits)
loss = -log(probs[target])

# Good: combined operation
loss = log_sum_exp(logits) - logits[target]
```

**For BCE**:
```python
# Bad: log(sigmoid(x))
loss = -log(sigmoid(x))

# Good: log-sum-exp trick
loss = log(1 + exp(-x))  # if y=1
loss = log(1 + exp(x))   # if y=0
```

**Add Epsilon**:
```python
# Prevent log(0)
loss = -log(pred + 1e-7)
```

---

### 4. Loss Scaling and Weighting

**Multi-task Learning**:
```python
total_loss = λ₁ * loss₁ + λ₂ * loss₂ + λ₃ * loss₃
```

**Balancing Strategies**:
- **Manual**: Set λ based on importance
- **Uncertainty weighting**: Learn λ during training
- **GradNorm**: Balance gradient magnitudes

**Example** (Object Detection):
```python
loss = λ_cls * classification_loss + λ_box * bbox_loss
# Typical: λ_cls = 1.0, λ_box = 5.0
```

---

### 5. Gradient Clipping

For losses with unbounded gradients:

```python
# PyTorch
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# TensorFlow
optimizer = tf.keras.optimizers.Adam(clipnorm=1.0)
```

---

### 6. Loss Monitoring

**Track Multiple Metrics**:
```python
# Don't just track loss
metrics = {
    'loss': loss.item(),
    'accuracy': accuracy,
    'precision': precision,
    'recall': recall,
    'f1': f1_score
}
```

**Separate Train/Val Loss**:
- Diverging → Overfitting
- Both high → Underfitting
- Train low, val high → Overfitting

---

### 7. Custom Loss Functions

**Template**:
```python
def custom_loss(y_true, y_pred):
    # 1. Compute base loss
    base_loss = some_loss(y_true, y_pred)

    # 2. Add regularization/penalty
    penalty = compute_penalty(y_pred)

    # 3. Combine
    total_loss = base_loss + λ * penalty

    return total_loss
```

**Example** (Smooth L1 with penalty):
```python
def smooth_l1_with_penalty(y_true, y_pred, delta=1.0, λ=0.01):
    error = y_true - y_pred
    abs_error = torch.abs(error)

    # Smooth L1
    smooth_l1 = torch.where(
        abs_error < delta,
        0.5 * error ** 2,
        delta * abs_error - 0.5 * delta ** 2
    )

    # Penalty for large predictions
    penalty = torch.mean(y_pred ** 2)

    return torch.mean(smooth_l1) + λ * penalty
```

---

## Common Pitfalls

### ❌ Don't Do This

1. **Using MSE for classification**
   - MSE doesn't match probabilistic interpretation
   - Use Cross-Entropy instead

2. **Forgetting to apply activation before loss**
   ```python
   # Wrong
   logits = model(x)
   loss = cross_entropy(logits, y)  # Expects probabilities!

   # Correct
   loss = cross_entropy_with_logits(logits, y)
   ```

3. **Ignoring class imbalance**
   - 99% accuracy on 99:1 imbalanced data is meaningless
   - Use Focal Loss or class weights

4. **Not normalizing multi-task losses**
   - Different losses have different scales
   - Normalize or use learned weights

5. **Using wrong reduction**
   ```python
   # Be explicit about reduction
   loss = F.cross_entropy(pred, target, reduction='mean')  # or 'sum', 'none'
   ```

---

## 📖 References

### Papers

1. **Focal Loss**: Lin et al. (2017) - "Focal Loss for Dense Object Detection"
2. **Triplet Loss**: Schroff et al. (2015) - "FaceNet: A Unified Embedding for Face Recognition"
3. **Dice Loss**: Milletari et al. (2016) - "V-Net: Fully Convolutional Neural Networks"
4. **CTC Loss**: Graves et al. (2006) - "Connectionist Temporal Classification"
5. **Wasserstein Loss**: Arjovsky et al. (2017) - "Wasserstein GAN"

### Books

1. **"Deep Learning"** - Goodfellow, Bengio, Courville (Chapter 5)
2. **"Pattern Recognition and Machine Learning"** - Bishop (Chapter 1.5)
3. **"Hands-On Machine Learning"** - Géron (Chapter 10)

---

## 🎯 Key Takeaways

1. **Match loss to task**:
   - Classification → Cross-Entropy
   - Regression → MSE/MAE
   - Metric learning → Triplet/Contrastive

2. **Consider data characteristics**:
   - Imbalanced → Focal Loss, weights
   - Outliers → MAE, Huber
   - Small objects → Dice, Focal

3. **Numerical stability matters**:
   - Use log-sum-exp tricks
   - Combine operations (softmax + log)
   - Add epsilon to prevent log(0)

4. **Monitor beyond loss**:
   - Track task-specific metrics
   - Watch train/val divergence
   - Use multiple evaluation metrics

5. **Hyperparameters matter**:
   - Focal Loss: γ, α
   - Huber: δ
   - Triplet: margin
   - Multi-task: λ weights

6. **Start simple, add complexity**:
   - Begin with standard losses (CE, MSE)
   - Add complexity only if needed
   - Validate improvements empirically

---

*Last updated: 2025-11-29*
*Related notebook: See PyTorch implementation in `损失函数.md`*
*Framework-agnostic guide - applicable to PyTorch, TensorFlow, JAX*


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
