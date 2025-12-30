"""
工具函数模块

包含图像预处理和经验回放缓冲区的实现。
"""

import numpy as np
import cv2
from collections import deque
import random


def preprocess_frame(frame, target_size=(84, 84)):
    """
    预处理游戏画面
    
    处理步骤：
    1. 转换为灰度图（减少计算量）
    2. 裁剪感兴趣区域
    3. 缩放到目标尺寸
    4. 归一化到[0,1]
    
    Args:
        frame: 原始游戏画面 (H, W, C)
        target_size: 目标尺寸 (height, width)
    
    Returns:
        处理后的图像 (1, H, W)
    """
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    else:
        gray = frame
    
    resized = cv2.resize(gray, target_size, interpolation=cv2.INTER_AREA)
    normalized = resized.astype(np.float32) / 255.0
    
    return normalized[np.newaxis, :, :]


class ReplayBuffer:
    """
    经验回放缓冲区
    
    DQN的关键技术之一：
    - 打破样本之间的时间相关性
    - 提高数据利用效率
    - 稳定训练过程
    
    存储格式：(state, action, reward, next_state, done)
    """
    
    def __init__(self, capacity=50000):
        """
        Args:
            capacity: 缓冲区最大容量
        """
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """添加一条经验"""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """
        随机采样一批经验
        
        Args:
            batch_size: 批次大小
        
        Returns:
            states, actions, rewards, next_states, dones
        """
        batch = random.sample(self.buffer, min(len(self.buffer), batch_size))
        
        states = np.array([exp[0] for exp in batch])
        actions = np.array([exp[1] for exp in batch])
        rewards = np.array([exp[2] for exp in batch], dtype=np.float32)
        next_states = np.array([exp[3] for exp in batch])
        dones = np.array([exp[4] for exp in batch], dtype=np.float32)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)


class FrameStack:
    """
    帧堆叠器
    
    将连续4帧画面堆叠在一起，让网络能够感知运动信息。
    这是处理部分可观测问题的常用技术。
    """
    
    def __init__(self, num_frames=4):
        self.num_frames = num_frames
        self.frames = deque(maxlen=num_frames)
    
    def reset(self, initial_frame):
        """重置并用初始帧填充"""
        self.frames.clear()
        for _ in range(self.num_frames):
            self.frames.append(initial_frame)
        return self.get_state()
    
    def push(self, frame):
        """添加新帧"""
        self.frames.append(frame)
        return self.get_state()
    
    def get_state(self):
        """获取堆叠后的状态"""
        return np.concatenate(list(self.frames), axis=0)
