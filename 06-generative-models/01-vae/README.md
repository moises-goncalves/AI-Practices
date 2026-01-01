# Variational Autoencoders (VAE) 模块

本目录包含自编码器和变分自编码器的完整实现与理论推导。

## 目录结构

| 文件 | 内容 | 核心概念 |
|------|------|----------|
| `vanilla_ae.ipynb` | 普通自编码器 | 信息瓶颈、PCA关系、重建损失 |
| `variational_ae.ipynb` | 变分自编码器 | ELBO推导、重参数化技巧、KL散度 |
| `vq_vae.ipynb` | 向量量化VAE | 离散码本、直通估计器、承诺损失 |

## 核心数学

### 1. Vanilla Autoencoder

**目标函数:**
$$\mathcal{L}_{AE} = \|x - g(f(x))\|^2$$

**与PCA关系 (Bourlard & Kamp, 1988):**
线性自编码器的最优解等价于PCA主子空间。

### 2. Variational Autoencoder (VAE)

**ELBO (Evidence Lower Bound):**
$$\log p(x) \geq \mathbb{E}_{q(z|x)}[\log p(x|z)] - D_{KL}(q(z|x) \| p(z))$$

**重参数化技巧:**
$$z = \mu + \sigma \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

### 3. VQ-VAE

**量化操作:**
$$z_q = e_k, \quad k = \arg\min_j \|z_e - e_j\|_2$$

**损失函数:**
$$\mathcal{L} = \|x - \hat{x}\|^2 + \|\text{sg}[z_e] - e\|^2 + \beta\|z_e - \text{sg}[e]\|^2$$

## 算法对比

| 方法 | 潜在空间 | 生成能力 | 后验坍塌 |
|------|----------|----------|----------|
| AE | 连续/不规则 | 差 | N/A |
| VAE | 连续/正则化 | 好 | 常见 |
| VQ-VAE | 离散 | 优秀 | 避免 |

## 学习路径

1. **Vanilla AE** → 理解信息瓶颈和重建
2. **VAE** → 掌握概率生成模型和ELBO
3. **VQ-VAE** → 学习离散表示和码本

## 参考文献

- Kingma & Welling (2014). Auto-Encoding Variational Bayes
- van den Oord et al. (2017). Neural Discrete Representation Learning
- Bourlard & Kamp (1988). Auto-association by multilayer perceptrons
