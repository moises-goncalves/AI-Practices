"""
Deep Q-Network (DQN) Model

DQN是一种结合深度学习和Q-Learning的强化学习算法。
核心思想：使用神经网络来近似Q函数，从而处理高维状态空间（如图像）。

网络结构说明：
- 输入：4帧连续的游戏画面（84x84x4）
- 卷积层：提取图像特征
- 全连接层：输出每个动作的Q值
"""

import torch
import torch.nn as nn


class DQN(nn.Module):
    """
    Deep Q-Network for Flappy Bird
    
    网络架构：
    Conv1: 4通道 -> 32通道, 8x8卷积核, 步长4
    Conv2: 32通道 -> 64通道, 4x4卷积核, 步长2  
    Conv3: 64通道 -> 64通道, 3x3卷积核, 步长1
    FC1: 7*7*64 -> 512
    FC2: 512 -> 2 (动作数量)
    """
    
    def __init__(self, input_channels=4, num_actions=2):
        super(DQN, self).__init__()
        
        # 卷积层：从图像中提取空间特征
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True)
        )
        
        # 全连接层：将特征映射到Q值
        self.fc1 = nn.Sequential(
            nn.Linear(7 * 7 * 64, 512),
            nn.ReLU(inplace=True)
        )
        self.fc2 = nn.Linear(512, num_actions)
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, -0.01, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入状态 (batch_size, 4, 84, 84)
        
        Returns:
            Q值 (batch_size, num_actions)
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)  # 展平
        x = self.fc1(x)
        q_values = self.fc2(x)
        return q_values
    
    def select_action(self, state, epsilon=0.0):
        """
        epsilon-greedy策略选择动作
        
        Args:
            state: 当前状态
            epsilon: 探索概率
        
        Returns:
            选择的动作索引
        """
        if torch.rand(1).item() < epsilon:
            return torch.randint(0, 2, (1,)).item()
        
        with torch.no_grad():
            q_values = self.forward(state)
            return q_values.argmax(dim=1).item()
