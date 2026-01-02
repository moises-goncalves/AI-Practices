"""
枚举类型定义

定义DQN算法中使用的各种枚举类型，包括网络架构、损失函数和探索策略。
"""

from enum import Enum, auto


class NetworkType(Enum):
    """
    Q网络架构类型
    
    - STANDARD: 标准MLP架构，直接输出所有动作的Q值
    - DUELING: 对偶架构，分离状态价值V(s)和动作优势A(s,a)
    """
    STANDARD = "standard"
    DUELING = "dueling"


class LossType(Enum):
    """
    损失函数类型
    
    - MSE: 均方误差损失 L = (y - Q)²
      简单但对异常值（大TD误差）敏感
    
    - HUBER: Huber损失（平滑L1），对异常值更鲁棒
      L = 0.5 * (y - Q)² if |y - Q| < 1 else |y - Q| - 0.5
    """
    MSE = "mse"
    HUBER = "huber"


class ExplorationStrategy(Enum):
    """
    探索策略类型
    
    - EPSILON_GREEDY: ε-贪婪策略，以ε概率随机探索
    - BOLTZMANN: 玻尔兹曼探索，基于Q值的softmax分布
    - UCB: 上置信界探索，考虑不确定性
    - NOISY: 噪声网络，通过参数噪声实现探索
    """
    EPSILON_GREEDY = "epsilon_greedy"
    BOLTZMANN = "boltzmann"
    UCB = "ucb"
    NOISY = "noisy"
