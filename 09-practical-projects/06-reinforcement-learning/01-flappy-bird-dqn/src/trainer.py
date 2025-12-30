"""
DQN训练器

实现DQN算法的核心训练逻辑，包括：
- epsilon-greedy探索策略
- 经验回放
- 目标网络（可选）
- 训练循环
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .dqn import DQN
from .utils import ReplayBuffer, preprocess_frame, FrameStack


class DQNTrainer:
    """
    DQN训练器
    
    核心算法流程：
    1. 使用epsilon-greedy策略选择动作
    2. 执行动作，获取奖励和下一状态
    3. 将经验存入回放缓冲区
    4. 从缓冲区采样，计算TD目标
    5. 更新网络参数
    """
    
    def __init__(
        self,
        env,
        learning_rate=1e-6,
        gamma=0.99,
        epsilon_start=0.1,
        epsilon_end=1e-4,
        epsilon_decay_steps=1000000,
        replay_size=50000,
        batch_size=32,
        device=None
    ):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.batch_size = batch_size
        
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        
        self.model = DQN().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        
        self.replay_buffer = ReplayBuffer(capacity=replay_size)
        self.frame_stack = FrameStack(num_frames=4)
        
        self.total_steps = 0
        self.episode_rewards = []
    
    def _update_epsilon(self):
        """线性衰减epsilon"""
        decay = (self.epsilon_start - self.epsilon_end) / self.epsilon_decay_steps
        self.epsilon = max(self.epsilon_end, self.epsilon - decay)
    
    def _preprocess_and_stack(self, frame, reset=False):
        """预处理并堆叠帧"""
        processed = preprocess_frame(frame[:self.env.screen_width, :int(self.env.base_y)])
        if reset:
            return self.frame_stack.reset(processed)
        return self.frame_stack.push(processed)
    
    def train_step(self):
        """执行一次训练更新"""
        if len(self.replay_buffer) < self.batch_size:
            return 0.0
        
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        current_q = self.model(states)
        current_q = current_q.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        with torch.no_grad():
            next_q = self.model(next_states).max(1)[0]
            target_q = rewards + self.gamma * next_q * (1 - dones)
        
        loss = self.criterion(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def train(self, num_iterations=2000000, save_interval=100000, save_path="models"):
        """
        主训练循环
        
        Args:
            num_iterations: 总训练步数
            save_interval: 保存模型的间隔
            save_path: 模型保存路径
        """
        os.makedirs(save_path, exist_ok=True)
        
        frame = self.env.reset()
        state = self._preprocess_and_stack(frame, reset=True)
        
        episode_reward = 0
        episode_count = 0
        
        for iteration in range(num_iterations):
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action = self.model.select_action(state_tensor, self.epsilon)
            
            next_frame, reward, done, info = self.env.step(action)
            next_state = self._preprocess_and_stack(next_frame)
            
            self.replay_buffer.push(state, action, reward, next_state, done)
            
            loss = self.train_step()
            self._update_epsilon()
            
            episode_reward += reward
            state = next_state
            self.total_steps += 1
            
            if done:
                frame = self.env.reset()
                state = self._preprocess_and_stack(frame, reset=True)
                self.episode_rewards.append(episode_reward)
                episode_count += 1
                
                if episode_count % 10 == 0:
                    avg_reward = np.mean(self.episode_rewards[-10:])
                    print(f"Episode {episode_count}, Avg Reward: {avg_reward:.2f}, "
                          f"Epsilon: {self.epsilon:.4f}, Score: {info['score']}")
                
                episode_reward = 0
            
            if (iteration + 1) % save_interval == 0:
                self.save(f"{save_path}/dqn_{iteration+1}.pth")
                print(f"Model saved at iteration {iteration+1}")
            
            if iteration % 1000 == 0:
                print(f"Iter: {iteration}, Loss: {loss:.4f}, Epsilon: {self.epsilon:.4f}")
        
        self.save(f"{save_path}/dqn_final.pth")
    
    def save(self, path):
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'total_steps': self.total_steps,
        }, path)
    
    def load(self, path):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon_end)
        self.total_steps = checkpoint.get('total_steps', 0)
