# 学习优化奖励：知识芯片

> **从稀疏奖励困境到密集学习信号的系统性突破**

---

## 核心心法（The "Aha!" Moment）

**奖励优化 = 信息密度的艺术**

稀疏奖励是强化学习的"诅咒"，但也是"机遇"。所有奖励优化方法的本质是：**在不改变最优策略的前提下，增加学习信号的密度**。选择哪种方法，取决于你拥有什么信息（演示、模型、探索能力）。

---

## 第一部分：问题本质

### 稀疏奖励的信息论视角

**现象**：为什么智能体在稀疏奖励环境中学习困难？

**本质**：信息论中的"信号-噪声比"问题。

**数学直觉**：
- 随机策略成功概率：$P(\text{success}) \approx (\frac{1}{|A|})^T$
- 100步任务，4个动作：$P \approx (0.25)^{100} \approx 10^{-60}$
- 需要 $10^{60}$ 次尝试才能随机找到一条成功轨迹

**信息论分析**：
```
信息熵 H(π) = -Σ π(a|s) log π(a|s)
├─ 随机策略：H = log|A|（最大熵，最少信息）
├─ 最优策略：H ≈ 0（最小熵，最多信息）
└─ 学习过程：H 从高到低的递减过程
```

**工程含义**：
- 稀疏奖励 = 低信息密度 = 学习困难
- 解决方案 = 增加信息密度（不改变最优策略）

---

## 第二部分：四大方法论

### 1. 势能基奖励塑形（PBRS）

#### 深度原理

**现象**：如何在不改变最优策略的前提下加速学习？

**本质**：**望远镜求和**的魔法。

**核心定理**（Ng-Harada-Russell, 1999）：

$$R'(s, a, s') = R(s, a, s') + F(s, a, s')$$

**定理**：$M$ 和 $M'$ 具有相同最优策略 **当且仅当**：

$$\boxed{F(s, a, s') = \gamma \Phi(s') - \Phi(s)}$$

**证明直觉**：

对轨迹 $\tau = (s_0, a_0, s_1, ..., s_T)$ 的塑形奖励求和：

$$\sum_{t=0}^{T-1} \gamma^t F(s_t, a_t, s_{t+1}) = \sum_{t=0}^{T-1} \gamma^t [\gamma\Phi(s_{t+1}) - \Phi(s_t)]$$

**望远镜求和**（相邻项相消）：

$$= \gamma^T \Phi(s_T) - \Phi(s_0)$$

由于 $|\Phi|$ 有界，当 $T \to \infty$ 时，$\gamma^T \Phi(s_T) \to 0$。

**关键洞察**：塑形奖励的累积是一个**有界常数**，不影响策略排序！

#### 架构陷阱

| 陷阱 | 症状 | 根因 | 修复 |
|:----:|:-----|:-----|:-----|
| **终止状态处理错误** | 学习不稳定 | $\Phi(s_{terminal}) \neq 0$ | 强制设置为 0 |
| **势能函数不合适** | 收敛慢或陷入局部最优 | 势能与真实值函数差异大 | 用学习势能或子目标 |
| **权重衰减不当** | 最终策略次优 | 塑形权重过高导致偏差 | 自适应权重衰减 |

#### 前沿演进

- **经典**：Ng et al. (1999) 势能基塑形
- **深化**：Wiewiora (2003) 证明与 Q 值初始化等价
- **前沿**：Devlin & Kudenko (2012) 动态势能基塑形

#### 交互式思考

**问题 1**：为什么 $\Phi(s_{terminal}) = 0$ 是必要的？如果设为其他值会怎样？

**问题 2**：在代码中修改势能函数（如从距离改为子目标），观察收敛速度的变化。

---

### 2. 逆强化学习（IRL）

#### 深度原理

**现象**：为什么从演示学习奖励比手工设计更好？

**本质**：**模糊性的解决**与**信息的提取**。

**IRL 的模糊性**：

存在**无穷多个**奖励函数与同一策略一致！

```
R(s) = 0        → 所有策略都最优
R(s) = c        → 同上（常数不改变排序）
R(s) = θᵀφ(s)  → 任何使专家最优的 θ
```

**解决方案**：添加额外约束

#### 三大 IRL 方法的对比

| 方法 | 核心思想 | 优点 | 缺点 |
|:----:|:-----|:-----|:-----|
| **Max-Margin** | 最大化专家与其他策略的差距 | 直观、有理论保证 | 需要解 forward RL |
| **MaxEnt** | 在一致奖励中选最大熵 | 处理随机专家 | 计算复杂 |
| **GAIL** | 对抗学习，直接匹配行为分布 | 无需显式恢复奖励 | 训练不稳定 |

#### 最大熵 IRL 的数学直觉

**建模假设**：专家行为服从 Boltzmann 分布

$$P(\tau | \theta) = \frac{1}{Z(\theta)} \exp\left(\sum_t \theta^T \phi(s_t, a_t)\right)$$

**梯度**（简洁形式）：

$$\nabla_\theta \mathcal{L} = \mu_E - \mu_\theta$$

- $\mu_E$：专家特征期望（我们想要的）
- $\mu_\theta$：当前奖励下策略的特征期望（我们有的）

**直觉**：增加专家访问的特征的奖励，减少当前策略访问的特征的奖励。

#### 架构陷阱

| 陷阱 | 症状 | 修复 |
|:----:|:-----|:-----|
| **特征设计不当** | 学不到有意义的奖励 | 用神经网络（Deep IRL）或 RBF 特征 |
| **演示数据不足** | 过拟合到演示 | 增加演示数量或正则化 |
| **GAIL 训练不稳定** | 判别器崩溃 | 用 Wasserstein 距离或梯度惩罚 |

#### 前沿演进

- **经典**：Ng & Russell (2000) 最大边际 IRL
- **深化**：Ziebart et al. (2008) 最大熵 IRL
- **前沿**：Fu et al. (2018) AIRL（可迁移奖励）

#### 交互式思考

**问题 1**：为什么 MaxEnt IRL 比 Max-Margin 更能处理随机专家？

**问题 2**：在代码中对比 MaxMarginIRL 和 MaxEntropyIRL 的学习曲线。

---

### 3. 好奇心驱动探索

#### 深度原理

**现象**：为什么预测误差能驱动有效探索？

**本质**：**信息增益**的最大化。

**核心公式**：

$$r_{total} = r_{extrinsic} + \beta \cdot r_{intrinsic}$$

$$r_{intrinsic} = \eta \cdot \|f(s') - \hat{f}(s, a)\|^2$$

**信息论视角**：

```
预测误差 = 信息增益
├─ 高误差 = 新颖状态 = 高信息增益
├─ 低误差 = 已知状态 = 低信息增益
└─ 学习过程 = 逐步降低预测误差
```

#### ICM 的三层架构

```
观测 s → [特征编码器 φ] → f(s)
         ↓
[前向模型 g] → f̂(s') = g(f(s), a)
         ↓
内在奖励 = η · ||f(s') - f̂(s')||²

[逆向模型 h] → â = h(f(s), f(s'))
         ↓
辅助损失 = -log P(a | h(...))
```

**为什么需要逆向模型？**

- 强制编码器学习**动作相关**的特征
- 过滤掉不可控的环境变化（如背景运动）
- 提高前向预测的信噪比

#### RND vs ICM 的对比

| 维度 | ICM | RND |
|:----:|:-----|:-----|
| **内在信号** | 学习的前向预测误差 | 固定随机网络的预测误差 |
| **特征学习** | 自适应编码器 | 无编码器 |
| **随机环境鲁棒性** | 中等（逆向模型过滤） | 低（可能奖励噪声） |
| **计算开销** | 高（三个网络） | 中等（两个网络） |
| **实现复杂度** | 高 | 低 |

#### 架构陷阱

| 陷阱 | 症状 | 修复 |
|:----:|:-----|:-----|
| **内在奖励过大** | 忽视外在奖励 | 调整 β 或归一化内在奖励 |
| **特征编码器崩溃** | 内在奖励变为常数 | 增加逆向模型权重 |
| **随机环境中失效** | 奖励无意义的随机性 | 用 ICM 而非 RND |

#### 前沿演进

- **经典**：Pathak et al. (2017) ICM
- **深化**：Burda et al. (2019) RND
- **前沿**：Raileanu & Rocktäschel (2020) 集成不一致性

#### 交互式思考

**问题 1**：为什么 RND 在随机环境中会失效？设计一个实验验证。

**问题 2**：修改 ICM 的逆向模型权重，观察特征编码器的学习质量变化。

---

### 4. 后见经验回放（HER）

#### 深度原理

**现象**：为什么失败的轨迹也能提供学习信号？

**本质**：**目标空间的重标注**。

**核心洞察**：

失败轨迹 $(s_0, a_0, ..., s_T)$ 虽然没有达到原始目标 $g$，但**达到了某个中间状态** $s_t$。

如果我们把 $s_t$ 当作新目标 $g'$，那么这条轨迹就变成了"成功"！

#### 数学形式

**原始转移**：$(s, a, g, r=0, s')$（失败）

**重标注转移**：$(s, a, g'=\text{achieved\_goal}(s'), r'=1, s')$（成功！）

**样本效率分析**：

```
原始：P(r > 0) ≈ 0（几乎全是失败）
HER后：P(r > 0) ≈ k/(k+1)（大部分是"成功"）

例：k=4 个后见目标
原始：1 个成功样本 + 99 个失败样本
HER后：1 个原始 + 4 个重标注 = 5 个成功样本
```

#### 目标选择策略的对比

| 策略 | 描述 | 覆盖范围 | 计算开销 |
|:----:|:-----|:-----|:-----|
| **final** | 使用轨迹最终达成的目标 | 小 | O(1) |
| **future** | 从未来达成的目标中采样 | 中 | O(k) |
| **episode** | 从整个轨迹的达成目标采样 | 大 | O(T) |
| **random** | 从所有历史达成目标采样 | 最大 | O(T) |

**推荐**：future 策略（最常用，平衡覆盖与计算）

#### 架构陷阱

| 陷阱 | 症状 | 修复 |
|:----:|:-----|:-----|
| **目标定义不当** | 无法提取 achieved_goal | 重新设计状态表示 |
| **分布偏移** | 学到的策略不泛化 | 混合原始和重标注样本 |
| **目标空间不连续** | 某些目标无法达成 | 检查环境动力学 |

#### 前沿演进

- **经典**：Andrychowicz et al. (2017) HER
- **深化**：Racanière et al. (2020) 自动目标生成
- **前沿**：Racanière et al. (2020) 自动课程学习

#### 交互式思考

**问题 1**：为什么 HER 只适用于目标条件任务？能否推广到其他任务？

**问题 2**：在代码中对比不同目标选择策略的样本效率。

---

## 第三部分：方法选择决策树

```
是否有专家演示？
├── 是 → 逆强化学习/模仿学习
│         ├── 需要可迁移奖励？ → AIRL
│         └── 只需模仿行为？ → GAIL/BC
│
└── 否 → 是目标条件任务？
          ├── 是 → HER
          │       └── 探索困难？ → HER + Curiosity
          │
          └── 否 → 探索是主要困难？
                    ├── 是 → Curiosity (ICM/RND)
                    │       └── 随机环境？ → ICM
                    │
                    └── 否 → PBRS（如果有好的势能函数）
```

---

## 第四部分：工程实践

### 通用设计模式

#### 模式 1：奖励组合

```python
# 通用模式：多源奖励融合
r_total = w_ext * r_extrinsic + w_int * r_intrinsic + w_shape * r_shaping

# 权重调度
w_int(t) = w_int_0 * decay_rate^t  # 逐步减弱内在奖励
w_shape(t) = w_shape_0 * (1 - t/T)  # 线性衰减塑形
```

#### 模式 2：特征期望计算

```python
# 通用模式：计算特征期望
def compute_feature_expectations(demonstrations, feature_extractor, gamma):
    feature_sums = []
    for demo in demonstrations:
        traj_features = np.zeros(feature_dim)
        for t, state in enumerate(demo.states):
            discount = gamma ** t
            features = feature_extractor(state)
            traj_features += discount * features
        feature_sums.append(traj_features)
    return np.mean(feature_sums, axis=0)
```

#### 模式 3：势能函数设计

```python
# 三层势能函数设计
class PotentialFunction:
    # 第一层：距离基势能（简单）
    def distance_based(self, state, goal):
        return -np.linalg.norm(state - goal)

    # 第二层：子目标势能（中等）
    def subgoal_based(self, state, subgoals):
        achieved = sum(w for i, w in enumerate(weights)
                      if reached(state, subgoals[i]))
        next_dist = distance_to_next_subgoal(state, subgoals)
        return achieved - next_dist

    # 第三层：学习势能（复杂）
    def learned_potential(self, state):
        return self.neural_network(state)
```

### 代码验证清单

```python
# 1. PBRS 验证
assert abs(sum(shaped_returns) - sum(original_returns)) < threshold
# 验证：策略排序不变

# 2. IRL 验证
expert_features = compute_feature_expectations(demos, extractor, gamma)
learned_features = compute_feature_expectations(learned_policy, extractor, gamma)
assert np.linalg.norm(expert_features - learned_features) < threshold
# 验证：特征期望匹配

# 3. Curiosity 验证
assert intrinsic_reward > 0 for novel_state
assert intrinsic_reward ≈ 0 for visited_state
# 验证：新颖性检测

# 4. HER 验证
assert len(relabeled_buffer) > len(original_buffer)
assert success_rate(relabeled_buffer) > success_rate(original_buffer)
# 验证：样本效率提升
```

---

## 第五部分：快速复习清单

### 概念理解

- [ ] PBRS 为什么保证策略不变性（望远镜求和）
- [ ] IRL 的模糊性来源与解决方案
- [ ] ICM 中逆向模型的作用
- [ ] HER 如何从失败中学习
- [ ] 四种方法的适用场景

### 公式推导

- [ ] 贝尔曼期望方程 → 塑形奖励的累积
- [ ] 最大熵 IRL 的梯度推导
- [ ] ICM 的前向和逆向模型损失
- [ ] HER 的目标重标注机制

### 代码实现

- [ ] 势能函数的正确实现
- [ ] 特征期望的计算
- [ ] 内在奖励的归一化
- [ ] 目标选择策略的实现

### 工程应用

- [ ] 如何选择合适的方法
- [ ] 如何调整超参数
- [ ] 如何处理常见问题
- [ ] 如何组合多种方法

---

## 第六部分：迁移学习指南

### 从 MDP 到其他问题

#### 迁移 1：多任务强化学习

**相同点**：每个任务仍然是 MDP

**不同点**：
- 状态包含任务标识
- 奖励函数依赖任务
- 需要学习任务间的知识转移

**迁移方法**：
```python
# 单任务
r = reward_function(s, a)

# 多任务：用 HER 的思想
r = reward_function(s, a, task_id)
# 重标注：改变 task_id 而非目标
```

#### 迁移 2：分层强化学习

**相同点**：底层仍然是 MDP

**不同点**：
- 高层策略生成子目标
- 低层策略达成子目标
- 需要子目标的奖励塑形

**迁移方法**：
```python
# 用 PBRS 的思想
r_low = r_env + shaping_bonus(s, subgoal)
# 势能函数 = 到子目标的距离
```

#### 迁移 3：元学习

**相同点**：每个任务的学习过程仍然是 MDP

**不同点**：
- 需要快速适应新任务
- 需要学习学习算法本身
- 需要任务间的知识转移

**迁移方法**：
```python
# 用 IRL 的思想
# 从演示学习任务的奖励函数
# 然后快速适应新任务
```

---

## 总结：知识芯片的三个层次

### 第一层：现象理解
- 为什么稀疏奖励困难？
- 为什么需要多种方法？
- 什么时候用哪种方法？

### 第二层：本质掌握
- PBRS = 望远镜求和的魔法
- IRL = 模糊性的解决
- Curiosity = 信息增益的最大化
- HER = 目标空间的重标注

### 第三层：工程应用
- 如何实现这些方法？
- 如何调试和优化？
- 如何组合多种方法？
- 如何迁移到新问题？

**学习路径**：现象 → 本质 → 应用 → 迁移

---

*最后更新：2025-12-20*

*相关代码：`src/` 和 `core/` 目录*

*交互式实验：修改 `main.py` 中的参数进行验证*
