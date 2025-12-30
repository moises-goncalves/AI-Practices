"""
Chrome Dino DQN 单元测试
"""

import os
import sys
import unittest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import process_image, FrameBuffer
from src.game_env import DinoGameSimulator


class TestUtils(unittest.TestCase):
    """测试工具函数"""
    
    def test_process_image_rgb(self):
        """测试RGB图像处理"""
        image = np.random.randint(0, 255, (150, 600, 3), dtype=np.uint8)
        processed = process_image(image, target_size=(80, 80))
        self.assertEqual(processed.shape, (80, 80))
        self.assertTrue(0 <= processed.min() <= processed.max() <= 1)
    
    def test_process_image_gray(self):
        """测试灰度图像处理"""
        image = np.random.randint(0, 255, (150, 600), dtype=np.uint8)
        processed = process_image(image, target_size=(80, 80))
        self.assertEqual(processed.shape, (80, 80))


class TestFrameBuffer(unittest.TestCase):
    """测试帧缓冲区"""
    
    def setUp(self):
        self.buffer = FrameBuffer(num_frames=4, frame_size=(80, 80))
    
    def test_reset(self):
        """测试重置"""
        frame = np.random.randint(0, 255, (150, 600), dtype=np.uint8)
        state = self.buffer.reset(frame)
        self.assertEqual(state.shape, (80, 80, 4))
    
    def test_add_frame(self):
        """测试添加帧"""
        frame = np.random.randint(0, 255, (150, 600), dtype=np.uint8)
        self.buffer.reset(frame)
        
        new_frame = np.random.randint(0, 255, (150, 600), dtype=np.uint8)
        state = self.buffer.add_frame(new_frame)
        self.assertEqual(state.shape, (80, 80, 4))


class TestDinoSimulator(unittest.TestCase):
    """测试游戏模拟器"""
    
    def setUp(self):
        self.env = DinoGameSimulator()
    
    def test_reset(self):
        """测试重置"""
        screen = self.env.reset()
        self.assertEqual(screen.shape, (150, 600))
    
    def test_step_no_action(self):
        """测试不操作"""
        self.env.reset()
        screen, reward, done, info = self.env.step(0)
        self.assertEqual(screen.shape, (150, 600))
        self.assertIsInstance(reward, float)
        self.assertIsInstance(done, bool)
        self.assertIn('score', info)
    
    def test_step_jump(self):
        """测试跳跃"""
        self.env.reset()
        screen, reward, done, info = self.env.step(1)
        self.assertEqual(screen.shape, (150, 600))
    
    def test_multiple_steps(self):
        """测试多步执行"""
        self.env.reset()
        for _ in range(100):
            action = np.random.randint(0, 2)
            screen, reward, done, info = self.env.step(action)
            if done:
                self.env.reset()


class TestDQNAgent(unittest.TestCase):
    """测试DQN智能体"""
    
    def test_agent_creation(self):
        """测试智能体创建"""
        try:
            from src.dqn import DQNAgent
            agent = DQNAgent(state_shape=(80, 80, 4), num_actions=2)
            self.assertIsNotNone(agent.model)
        except ImportError:
            self.skipTest("TensorFlow not available")
    
    def test_select_action(self):
        """测试动作选择"""
        try:
            from src.dqn import DQNAgent
            agent = DQNAgent(state_shape=(80, 80, 4), num_actions=2)
            state = np.random.rand(80, 80, 4).astype(np.float32)
            action = agent.select_action(state)
            self.assertIn(action, [0, 1])
        except ImportError:
            self.skipTest("TensorFlow not available")
    
    def test_remember_and_replay(self):
        """测试经验存储和回放"""
        try:
            from src.dqn import DQNAgent
            agent = DQNAgent(state_shape=(80, 80, 4), num_actions=2, batch_size=4)
            
            state = np.random.rand(80, 80, 4).astype(np.float32)
            for _ in range(10):
                agent.remember(state, 0, 0.1, state, False)
            
            loss = agent.replay()
            self.assertIsInstance(loss, (int, float))
        except ImportError:
            self.skipTest("TensorFlow not available")


if __name__ == '__main__':
    unittest.main(verbosity=2)
