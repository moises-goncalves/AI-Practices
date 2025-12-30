"""
股票交易强化学习单元测试
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import generate_sample_data, add_technical_indicators, split_data
from src.env import StockTradingEnv


class TestData(unittest.TestCase):
    """测试数据处理"""
    
    def test_generate_sample_data(self):
        """测试生成样本数据"""
        df = generate_sample_data('2020-01-01', '2020-12-31')
        self.assertIsInstance(df, pd.DataFrame)
        self.assertIn('close', df.columns)
        self.assertGreater(len(df), 0)
    
    def test_add_technical_indicators(self):
        """测试添加技术指标"""
        df = generate_sample_data('2020-01-01', '2020-12-31')
        df = add_technical_indicators(df)
        self.assertIn('ma5', df.columns)
        self.assertIn('rsi', df.columns)
        self.assertIn('macd', df.columns)
    
    def test_split_data(self):
        """测试数据划分"""
        df = generate_sample_data('2020-01-01', '2020-12-31')
        train_df, test_df = split_data(df, train_ratio=0.8)
        self.assertGreater(len(train_df), len(test_df))


class TestEnv(unittest.TestCase):
    """测试交易环境"""
    
    def setUp(self):
        df = generate_sample_data('2020-01-01', '2020-06-30')
        df = add_technical_indicators(df)
        self.env = StockTradingEnv(df, initial_balance=100000)
    
    def test_reset(self):
        """测试重置"""
        state = self.env.reset()
        self.assertIsInstance(state, np.ndarray)
        self.assertEqual(self.env.balance, 100000)
        self.assertEqual(self.env.shares_held, 0)
    
    def test_step_hold(self):
        """测试持有动作"""
        self.env.reset()
        state, reward, done, info = self.env.step(0)
        self.assertIsInstance(state, np.ndarray)
        self.assertIsInstance(reward, float)
        self.assertIsInstance(done, bool)
    
    def test_step_buy(self):
        """测试买入动作"""
        self.env.reset()
        initial_balance = self.env.balance
        state, reward, done, info = self.env.step(1)
        self.assertLessEqual(self.env.balance, initial_balance)
    
    def test_step_sell(self):
        """测试卖出动作"""
        self.env.reset()
        self.env.step(1)
        shares_before = self.env.shares_held
        if shares_before > 0:
            self.env.step(2)
            self.assertLessEqual(self.env.shares_held, shares_before)
    
    def test_full_episode(self):
        """测试完整回合"""
        self.env.reset()
        done = False
        steps = 0
        while not done and steps < 1000:
            action = np.random.randint(0, 3)
            state, reward, done, info = self.env.step(action)
            steps += 1
        self.assertIn('total_asset', info)


class TestAgents(unittest.TestCase):
    """测试智能体"""
    
    def test_dqn_agent(self):
        """测试DQN智能体"""
        try:
            from src.agents import DQNAgent
            agent = DQNAgent(state_dim=5, action_dim=3)
            state = np.random.rand(5).astype(np.float32)
            action = agent.select_action(state)
            self.assertIn(action, [0, 1, 2])
        except ImportError:
            self.skipTest("PyTorch not available")
    
    def test_dqn_train(self):
        """测试DQN训练"""
        try:
            from src.agents import DQNAgent
            agent = DQNAgent(state_dim=5, action_dim=3, batch_size=4)
            state = np.random.rand(5).astype(np.float32)
            for _ in range(10):
                agent.remember(state, 0, 0.1, state, False)
            loss = agent.train()
            self.assertIsInstance(loss, (int, float))
        except ImportError:
            self.skipTest("PyTorch not available")
    
    def test_a2c_agent(self):
        """测试A2C智能体"""
        try:
            from src.agents import A2CAgent
            agent = A2CAgent(state_dim=5, action_dim=3)
            state = np.random.rand(5).astype(np.float32)
            action = agent.select_action(state)
            self.assertIn(action, [0, 1, 2])
        except ImportError:
            self.skipTest("PyTorch not available")


if __name__ == '__main__':
    unittest.main(verbosity=2)
