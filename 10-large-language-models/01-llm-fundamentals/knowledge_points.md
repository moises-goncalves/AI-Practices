# LLM 基础知识点详解

> **知识密度**：⭐⭐⭐⭐⭐ | **实战价值**：⭐⭐⭐⭐⭐

---

## 目录

1. [Transformer 架构](#1-transformer-架构)
2. [自注意力机制](#2-自注意力机制)
3. [多头注意力](#3-多头注意力)
4. [位置编码](#4-位置编码)
5. [前馈神经网络](#5-前馈神经网络)
6. [层归一化](#6-层归一化)
7. [分词器](#7-分词器)
8. [Embedding层](#8-embedding层)
9. [训练技巧](#9-训练技巧)
10. [常见问题FAQ](#10-常见问题faq)

---

## 1. Transformer 架构

### 1.1 整体架构

Transformer 由 **编码器(Encoder)** 和 **解码器(Decoder)** 组成：

```
输入序列 → [Embedding + 位置编码] → [N × 编码器层] → 编码表示
                                                      ↓
输出序列 ← [线性层 + Softmax] ← [N × 解码器层] ←────────┘
```

**编码器层结构**：
```
输入 → 多头自注意力 → Add & Norm → 前馈网络 → Add & Norm → 输出
         ↑__________________|          ↑___________|
              残差连接                    残差连接
```

**解码器层结构**：
```
输入 → 掩码多头自注意力 → Add & Norm → 交叉注意力 → Add & Norm → FFN → Add & Norm
         ↑__________________|            ↑___________|        ↑________|
```

### 1.2 核心创新

| 创新点 | 解决的问题 | 效果 |
|--------|-----------|------|
| **自注意力** | RNN无法并行、长距离依赖 | O(1)距离，完全并行 |
| **多头注意力** | 单一注意力表达能力有限 | 多子空间表示 |
| **位置编码** | 注意力无位置感知 | 注入序列顺序信息 |
| **残差连接** | 深层网络梯度消失 | 梯度直接传播 |
| **层归一化** | 训练不稳定 | 稳定训练过程 |

### 1.3 编码器 vs 解码器

| 特性 | 编码器 | 解码器 |
|------|--------|--------|
| **注意力类型** | 双向自注意力 | 因果自注意力 + 交叉注意力 |
| **可见范围** | 全部输入 | 仅过去token |
| **典型应用** | BERT、文本分类 | GPT、文本生成 |
| **并行性** | 完全并行 | 训练并行，推理串行 |

---

## 2. 自注意力机制

### 2.1 核心公式

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

**分解理解**：
1. **Q (Query)**：查询向量，"我在找什么"
2. **K (Key)**：键向量，"我有什么特征"
3. **V (Value)**：值向量，"我的实际内容"

### 2.2 计算步骤

```python
def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q, K, V: [batch, seq_len, d_k]
    """
    d_k = Q.size(-1)
    
    # 1. 计算注意力分数: [batch, seq_len, seq_len]
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    
    # 2. 应用掩码（可选）
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    
    # 3. Softmax归一化
    attention_weights = F.softmax(scores, dim=-1)
    
    # 4. 加权求和
    output = torch.matmul(attention_weights, V)
    
    return output, attention_weights
```

### 2.3 为什么要缩放？

**问题**：当 $d_k$ 较大时，$QK^T$ 的方差为 $d_k$，导致softmax梯度消失

**数学推导**：
- 假设 $Q, K$ 的元素独立同分布，均值0，方差1
- $QK^T$ 的每个元素是 $d_k$ 个乘积之和
- 方差：$\text{Var}(QK^T) = d_k$
- 缩放后：$\text{Var}\left(\frac{QK^T}{\sqrt{d_k}}\right) = 1$

### 2.4 注意力掩码

**填充掩码 (Padding Mask)**：
```python
# 忽略填充位置
padding_mask = (input_ids != pad_token_id).unsqueeze(1).unsqueeze(2)
# [batch, 1, 1, seq_len] -> 广播到 [batch, heads, seq_len, seq_len]
```

**因果掩码 (Causal Mask)**：
```python
# 只能看到当前及之前的位置
seq_len = input_ids.size(1)
causal_mask = torch.tril(torch.ones(seq_len, seq_len))
# [[1, 0, 0],
#  [1, 1, 0],
#  [1, 1, 1]]
```

### 2.5 自注意力 vs 交叉注意力

| 类型 | Q来源 | K,V来源 | 应用场景 |
|------|-------|---------|---------|
| **自注意力** | 同一序列 | 同一序列 | 编码器、解码器自身 |
| **交叉注意力** | 解码器 | 编码器输出 | 解码器关注编码器 |

---

## 3. 多头注意力

### 3.1 核心思想

将注意力分成多个"头"，每个头学习不同的注意力模式：

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O$$

其中：
$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

### 3.2 参数维度

| 参数 | 维度 | 说明 |
|------|------|------|
| $W^Q, W^K, W^V$ | $[d_{model}, d_k]$ | 每个头的投影矩阵 |
| $W^O$ | $[h \cdot d_k, d_{model}]$ | 输出投影矩阵 |
| $d_k$ | $d_{model} / h$ | 每个头的维度 |

**典型配置**：
- BERT-base: $d_{model}=768$, $h=12$, $d_k=64$
- GPT-3: $d_{model}=12288$, $h=96$, $d_k=128$

### 3.3 实现代码

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # 合并的投影矩阵（更高效）
        self.W_qkv = nn.Linear(d_model, 3 * d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.size()
        
        # 1. 线性投影并分头
        qkv = self.W_qkv(x)  # [batch, seq, 3*d_model]
        qkv = qkv.reshape(batch_size, seq_len, 3, self.n_heads, self.d_k)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch, heads, seq, d_k]
        Q, K, V = qkv[0], qkv[1], qkv[2]
        
        # 2. 缩放点积注意力
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # 3. 加权求和
        context = torch.matmul(attn, V)  # [batch, heads, seq, d_k]
        
        # 4. 合并多头
        context = context.transpose(1, 2).reshape(batch_size, seq_len, self.d_model)
        
        # 5. 输出投影
        output = self.W_o(context)
        
        return output
```

### 3.4 多头的作用

不同的头可以学习不同的模式：
- **语法头**：关注语法结构（主谓宾）
- **语义头**：关注语义相关词
- **位置头**：关注相邻位置
- **长距离头**：关注远距离依赖

---

## 4. 位置编码

### 4.1 为什么需要位置编码？

自注意力是**置换不变的**（permutation invariant）：
- 打乱输入顺序，输出只是相应打乱
- 无法区分 "猫吃鱼" 和 "鱼吃猫"

### 4.2 正弦位置编码（原始Transformer）

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$
$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

**特点**：
- 固定编码，无需学习
- 可外推到更长序列
- 相对位置可通过线性变换表示

```python
def sinusoidal_position_encoding(max_len, d_model):
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len).unsqueeze(1).float()
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                         (-math.log(10000.0) / d_model))
    
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    
    return pe
```

### 4.3 可学习位置编码（BERT/GPT）

```python
self.position_embeddings = nn.Embedding(max_position, d_model)
```

**特点**：
- 通过训练学习
- 表达能力更强
- 无法外推到训练长度之外

### 4.4 旋转位置编码 RoPE（LLaMA）

**核心思想**：将位置信息编码为旋转矩阵

$$f_q(x_m, m) = (W_q x_m) e^{im\theta}$$
$$f_k(x_n, n) = (W_k x_n) e^{in\theta}$$

**优势**：
- 相对位置信息自然融入
- 良好的外推能力
- 计算高效

```python
def apply_rotary_pos_emb(q, k, cos, sin):
    # q, k: [batch, heads, seq_len, head_dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return torch.cat((-x2, x1), dim=-1)
```

### 4.5 ALiBi（Attention with Linear Biases）

**核心思想**：直接在注意力分数上加线性偏置

$$\text{softmax}(q_i K^T - m \cdot [0, 1, 2, ..., i])$$

**优势**：
- 无需额外参数
- 极强的外推能力
- 实现简单

### 4.6 位置编码对比

| 方法 | 参数量 | 外推能力 | 相对位置 | 代表模型 |
|------|--------|---------|---------|---------|
| **正弦编码** | 0 | 中等 | 隐式 | 原始Transformer |
| **可学习** | O(L·d) | 差 | 无 | BERT, GPT-2 |
| **RoPE** | 0 | 好 | 显式 | LLaMA, Qwen |
| **ALiBi** | 0 | 极好 | 显式 | BLOOM, MPT |

---

## 5. 前馈神经网络

### 5.1 结构

$$\text{FFN}(x) = \text{GELU}(xW_1 + b_1)W_2 + b_2$$

**维度变化**：
```
[batch, seq, d_model] → [batch, seq, d_ff] → [batch, seq, d_model]
```

通常 $d_{ff} = 4 \times d_{model}$

### 5.2 激活函数演进

| 激活函数 | 公式 | 特点 | 使用模型 |
|---------|------|------|---------|
| **ReLU** | $\max(0, x)$ | 简单，有死神经元 | 原始Transformer |
| **GELU** | $x \cdot \Phi(x)$ | 平滑，效果好 | BERT, GPT |
| **SwiGLU** | $\text{Swish}(xW_1) \otimes xW_2$ | 门控机制 | LLaMA, PaLM |

### 5.3 SwiGLU 实现

```python
class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)
    
    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
```

---

## 6. 层归一化

### 6.1 Layer Norm vs Batch Norm

| 特性 | Layer Norm | Batch Norm |
|------|-----------|------------|
| **归一化维度** | 特征维度 | 批次维度 |
| **依赖batch** | 否 | 是 |
| **适用场景** | NLP、变长序列 | CV、固定尺寸 |
| **推理一致性** | 训练=推理 | 需要running stats |

### 6.2 Pre-Norm vs Post-Norm

**Post-Norm（原始Transformer）**：
```
x → Attention → Add(x) → LayerNorm → FFN → Add → LayerNorm
```

**Pre-Norm（GPT-2+）**：
```
x → LayerNorm → Attention → Add(x) → LayerNorm → FFN → Add
```

**Pre-Norm优势**：
- 训练更稳定
- 可以训练更深的网络
- 不需要warmup

### 6.3 RMSNorm（LLaMA）

$$\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^d x_i^2 + \epsilon}} \cdot \gamma$$

**优势**：
- 计算更快（无需计算均值）
- 效果相当
- LLaMA、Qwen等模型采用

```python
class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps
    
    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight
```

---

## 7. 分词器

### 7.1 分词方法对比

| 方法 | 原理 | 优势 | 劣势 | 代表 |
|------|------|------|------|------|
| **Word** | 按空格分词 | 简单 | OOV问题严重 | 早期模型 |
| **Char** | 按字符分词 | 无OOV | 序列太长 | - |
| **BPE** | 频率合并 | 平衡 | 贪婪算法 | GPT |
| **WordPiece** | 似然合并 | 更优分割 | 计算复杂 | BERT |
| **Unigram** | 概率模型 | 多种分割 | 训练慢 | T5, XLNet |
| **SentencePiece** | 统一框架 | 语言无关 | - | LLaMA |

### 7.2 BPE 算法

```python
def train_bpe(corpus, vocab_size):
    # 1. 初始化：字符级词表
    vocab = set(char for word in corpus for char in word)
    
    # 2. 统计相邻pair频率
    while len(vocab) < vocab_size:
        pairs = count_pairs(corpus)
        best_pair = max(pairs, key=pairs.get)
        
        # 3. 合并最频繁的pair
        vocab.add(best_pair[0] + best_pair[1])
        corpus = merge_pair(corpus, best_pair)
    
    return vocab
```

### 7.3 特殊Token

| Token | 作用 | 示例 |
|-------|------|------|
| `[PAD]` | 填充 | 对齐序列长度 |
| `[UNK]` | 未知词 | OOV处理 |
| `[CLS]` | 分类 | BERT句子表示 |
| `[SEP]` | 分隔 | 句子边界 |
| `[MASK]` | 掩码 | MLM预训练 |
| `<s>`, `</s>` | 句首/句尾 | 生成任务 |
| `<|endoftext|>` | 文档结束 | GPT |

### 7.4 Tokenizer 使用

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# 编码
text = "Hello, world!"
tokens = tokenizer.tokenize(text)  # ['hello', ',', 'world', '!']
ids = tokenizer.encode(text)       # [101, 7592, 1010, 2088, 999, 102]

# 解码
decoded = tokenizer.decode(ids)    # "[CLS] hello, world! [SEP]"

# 批量处理
batch = tokenizer(
    ["Hello", "World"],
    padding=True,
    truncation=True,
    max_length=512,
    return_tensors="pt"
)
```

---

## 8. Embedding层

### 8.1 Token Embedding

将离散token映射到连续向量空间：

```python
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model
    
    def forward(self, x):
        # 缩放因子：稳定训练
        return self.embedding(x) * math.sqrt(self.d_model)
```

### 8.2 权重绑定（Weight Tying）

输入Embedding和输出层共享权重：

```python
class LM(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # 权重绑定
        self.lm_head.weight = self.embedding.weight
```

**优势**：
- 减少参数量
- 提升性能
- GPT、BERT等广泛使用

---

## 9. 训练技巧

### 9.1 学习率调度

**Warmup + Cosine Decay**：
```python
def get_lr(step, d_model, warmup_steps):
    if step < warmup_steps:
        return step / warmup_steps * base_lr
    else:
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return base_lr * 0.5 * (1 + math.cos(math.pi * progress))
```

### 9.2 梯度裁剪

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### 9.3 混合精度训练

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    output = model(input)
    loss = criterion(output, target)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### 9.4 梯度累积

```python
accumulation_steps = 4

for i, batch in enumerate(dataloader):
    loss = model(batch) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

---

## 10. 常见问题FAQ

### Q1: 为什么Transformer比RNN好？

| 方面 | RNN | Transformer |
|------|-----|-------------|
| **并行性** | 串行 | 完全并行 |
| **长距离依赖** | O(n)步传播 | O(1)直接连接 |
| **梯度问题** | 消失/爆炸 | 残差连接缓解 |
| **计算复杂度** | O(n) | O(n²) |

### Q2: 注意力复杂度O(n²)如何优化？

- **稀疏注意力**：Longformer, BigBird
- **线性注意力**：Linear Transformer, Performer
- **Flash Attention**：IO优化，不改变计算量

### Q3: 为什么用Layer Norm而不是Batch Norm？

- NLP序列长度变化大
- 小batch训练常见
- Layer Norm对batch size不敏感

### Q4: Pre-Norm和Post-Norm哪个好？

- **Pre-Norm**：训练稳定，适合深层网络
- **Post-Norm**：理论上表达能力更强，但难训练
- **实践**：大多数现代模型用Pre-Norm

### Q5: 如何选择位置编码？

- **短序列（<2K）**：可学习位置编码
- **长序列（>4K）**：RoPE或ALiBi
- **需要外推**：ALiBi最佳

---

## 参考文献

1. Vaswani et al. (2017). Attention Is All You Need
2. Devlin et al. (2019). BERT: Pre-training of Deep Bidirectional Transformers
3. Radford et al. (2019). Language Models are Unsupervised Multitask Learners
4. Su et al. (2021). RoFormer: Enhanced Transformer with Rotary Position Embedding
5. Press et al. (2022). Train Short, Test Long: Attention with Linear Biases

---

*最后更新：2026年1月*
