"""
Flappy Bird DQN 单元测试

测试核心组件的功能正确性，使用简化参数进行快速验证。
"""

import os
import sys
import unittest
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dqn import DQN
from src.utils import ReplayBuffer, FrameStack, preprocess_frame


class TestDQN(unittest.TestCase):
    """测试DQN网络"""
    
    def setUp(self):
        self.model = DQN(input_channels=4, num_actions=2)
    
    def test_forward_shape(self):
        """测试前向传播输出形状"""
        batch_size = 4
        x = torch.randn(batch_size, 4, 84, 84)
        output = self.model(x)
        self.assertEqual(output.shape, (batch_size, 2))
    
    def test_select_action_greedy(self):
        """测试贪婪动作选择"""
        state = torch.randn(1, 4, 84, 84)
        action = self.model.select_action(state, epsilon=0.0)
        self.assertIn(action, [0, 1])
    
    def test_select_action_random(self):
        """测试随机探索"""
        state = torch.randn(1, 4, 84, 84)
        actions = [self.model.select_action(state, epsilon=1.0) for _ in range(100)]
        self.assertTrue(0 in actions and 1 in actions)


class TestReplayBuffer(unittest.TestCase):
    """测试经验回放缓冲区"""
    
    def setUp(self):
        self.buffer = ReplayBuffer(capacity=100)
    
    def test_push_and_len(self):
        """测试添加经验"""
        state = np.zeros((4, 84, 84))
        for i in range(50):
            self.buffer.push(state, 0, 0.1, state, False)
        self.assertEqual(len(self.buffer), 50)
    
    def test_capacity_limit(self):
        """测试容量限制"""
        state = np.zeros((4, 84, 84))
        for i in range(150):
            self.buffer.push(state, 0, 0.1, state, False)
        self.assertEqual(len(self.buffer), 100)
    
    def test_sample(self):
        """测试采样"""
        state = np.zeros((4, 84, 84))
        for i in range(50):
            self.buffer.push(state, i % 2, 0.1, state, False)
        
        states, actions, rewards, next_states, dones = self.buffer.sample(16)
        self.assertEqual(states.shape, (16, 4, 84, 84))
        self.assertEqual(actions.shape, (16,))
        self.assertEqual(rewards.shape, (16,))


class TestFrameStack(unittest.TestCase):
    """测试帧堆叠器"""
    
    def setUp(self):
        self.frame_stack = FrameStack(num_frames=4)
    
    def test_reset(self):
        """测试重置"""
        frame = np.random.rand(1, 84, 84).astype(np.float32)
        state = self.frame_stack.reset(frame)
        self.assertEqual(state.shape, (4, 84, 84))
    
    def test_push(self):
        """测试添加新帧"""
        frame = np.random.rand(1, 84, 84).astype(np.float32)
        self.frame_stack.reset(frame)
        
        new_frame = np.ones((1, 84, 84), dtype=np.float32)
        state = self.frame_stack.push(new_frame)
        self.assertEqual(state.shape, (4, 84, 84))
        np.testing.assert_array_equal(state[3], new_frame[0])


class TestPreprocessFrame(unittest.TestCase):
    """测试图像预处理"""
    
    def test_preprocess_rgb(self):
        """测试RGB图像预处理"""
        frame = np.random.randint(0, 255, (512, 288, 3), dtype=np.uint8)
        processed = preprocess_frame(frame)
        self.assertEqual(processed.shape, (1, 84, 84))
        self.assertTrue(0 <= processed.min() <= processed.max() <= 1)
    
    def test_preprocess_gray(self):
        """测试灰度图像预处理"""
        frame = np.random.randint(0, 255, (512, 288), dtype=np.uint8)
        processed = preprocess_frame(frame)
        self.assertEqual(processed.shape, (1, 84, 84))


class TestTrainingStep(unittest.TestCase):
    """测试训练步骤（简化版）"""
    
    def test_loss_computation(self):
        """测试损失计算"""
        model = DQN()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        criterion = torch.nn.MSELoss()
        
        states = torch.randn(4, 4, 84, 84)
        actions = torch.randint(0, 2, (4,))
        rewards = torch.randn(4)
        next_states = torch.randn(4, 4, 84, 84)
        dones = torch.zeros(4)
        
        current_q = model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q = model(next_states).max(1)[0]
            target_q = rewards + 0.99 * next_q * (1 - dones)
        
        loss = criterion(current_q, target_q)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        self.assertIsInstance(loss.item(), float)
        self.assertFalse(torch.isnan(loss))


if __name__ == '__main__':
    unittest.main(verbosity=2)
