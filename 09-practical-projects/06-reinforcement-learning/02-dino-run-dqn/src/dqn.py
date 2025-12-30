"""
Chrome Dino游戏DQN实现

基于Keras/TensorFlow的DQN网络，用于训练Chrome恐龙游戏AI。
"""

import numpy as np

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
    from tensorflow.keras.optimizers import Adam
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


def build_dqn_model(input_shape=(80, 80, 4), num_actions=2, learning_rate=1e-4):
    """
    构建DQN网络
    
    网络结构：
    - 3层卷积 + 池化：提取图像特征
    - 2层全连接：输出Q值
    
    Args:
        input_shape: 输入图像尺寸 (height, width, channels)
        num_actions: 动作数量
        learning_rate: 学习率
    
    Returns:
        编译好的Keras模型
    """
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow is required. Install with: pip install tensorflow")
    
    model = Sequential([
        Conv2D(32, (8, 8), strides=(4, 4), padding='same', 
               activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),
        
        Conv2D(64, (4, 4), strides=(2, 2), padding='same', activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        
        Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        
        Flatten(),
        Dense(512, activation='relu'),
        Dense(num_actions, activation='linear')
    ])
    
    model.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate))
    return model


class DQNAgent:
    """
    DQN智能体
    
    封装了DQN的核心逻辑：
    - epsilon-greedy动作选择
    - 经验回放
    - 网络训练
    """
    
    def __init__(
        self,
        state_shape=(80, 80, 4),
        num_actions=2,
        learning_rate=1e-4,
        gamma=0.99,
        epsilon=0.1,
        epsilon_min=0.0001,
        epsilon_decay=0.9999,
        memory_size=50000,
        batch_size=32
    ):
        self.state_shape = state_shape
        self.num_actions = num_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        
        self.model = build_dqn_model(state_shape, num_actions, learning_rate)
        self.memory = []
        self.memory_size = memory_size
    
    def select_action(self, state):
        """epsilon-greedy策略选择动作"""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.num_actions)
        
        q_values = self.model.predict(state[np.newaxis], verbose=0)
        return np.argmax(q_values[0])
    
    def remember(self, state, action, reward, next_state, done):
        """存储经验"""
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > self.memory_size:
            self.memory.pop(0)
    
    def replay(self):
        """经验回放训练"""
        if len(self.memory) < self.batch_size:
            return 0
        
        indices = np.random.choice(len(self.memory), self.batch_size, replace=False)
        batch = [self.memory[i] for i in indices]
        
        states = np.array([exp[0] for exp in batch])
        actions = np.array([exp[1] for exp in batch])
        rewards = np.array([exp[2] for exp in batch])
        next_states = np.array([exp[3] for exp in batch])
        dones = np.array([exp[4] for exp in batch])
        
        current_q = self.model.predict(states, verbose=0)
        next_q = self.model.predict(next_states, verbose=0)
        
        for i in range(self.batch_size):
            if dones[i]:
                current_q[i][actions[i]] = rewards[i]
            else:
                current_q[i][actions[i]] = rewards[i] + self.gamma * np.max(next_q[i])
        
        history = self.model.fit(states, current_q, epochs=1, verbose=0)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return history.history['loss'][0]
    
    def save(self, path):
        """保存模型"""
        self.model.save_weights(path)
    
    def load(self, path):
        """加载模型"""
        self.model.load_weights(path)
