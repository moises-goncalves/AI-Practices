"""
TD算法单元测试 (TD Algorithm Unit Tests)
=======================================
"""

import pytest
import numpy as np
from typing import Dict, Tuple, Any

# 导入核心模块
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import (
    TDConfig, TrainingMetrics, EligibilityTraceType,
    TD0ValueLearner, SARSA, QLearning, ExpectedSARSA,
    DoubleQLearning, NStepTD, TDLambda, SARSALambda, WatkinsQLambda,
    create_td_learner
)
from environments import CliffWalkingEnv, RandomWalk, WindyGridWorld


class TestTDConfig:
    """测试配置类。"""
    
    def test_default_config(self):
        config = TDConfig()
        assert config.alpha == 0.1
        assert config.gamma == 0.99
        assert config.epsilon == 0.1
    
    def test_invalid_alpha(self):
        with pytest.raises(ValueError):
            TDConfig(alpha=0)
        with pytest.raises(ValueError):
            TDConfig(alpha=1.5)
    
    def test_invalid_gamma(self):
        with pytest.raises(ValueError):
            TDConfig(gamma=-0.1)
        with pytest.raises(ValueError):
            TDConfig(gamma=1.5)


class TestTD0ValueLearner:
    """测试TD(0)预测。"""
    
    def test_initialization(self):
        config = TDConfig(alpha=0.1, gamma=1.0)
        learner = TD0ValueLearner(config)
        assert learner.config.alpha == 0.1
    
    def test_update(self):
        config = TDConfig(alpha=0.5, gamma=1.0)
        learner = TD0ValueLearner(config)
        learner.set_action_space([0, 1])
        
        td_error = learner.update(state=1, action=0, reward=1.0, 
                                   next_state=2, next_action=0, done=False)
        assert learner.get_value(1) == 0.5  # 0 + 0.5 * (1 + 0 - 0)


class TestSARSA:
    """测试SARSA算法。"""
    
    def test_update_requires_next_action(self):
        config = TDConfig(alpha=0.5, gamma=0.99)
        sarsa = SARSA(config)
        sarsa.set_action_space([0, 1, 2, 3])
        
        with pytest.raises(ValueError):
            sarsa.update(0, 0, -1.0, 1, None, False)
    
    def test_terminal_update(self):
        config = TDConfig(alpha=0.5, gamma=0.99)
        sarsa = SARSA(config)
        sarsa.set_action_space([0, 1, 2, 3])
        
        td_error = sarsa.update(0, 0, 10.0, 1, None, True)
        assert sarsa.get_q_value(0, 0) == 5.0  # 0 + 0.5 * (10 - 0)


class TestQLearning:
    """测试Q-Learning算法。"""
    
    def test_max_operation(self):
        config = TDConfig(alpha=1.0, gamma=1.0)
        qlearn = QLearning(config)
        qlearn.set_action_space([0, 1])
        
        # 设置下一状态的Q值
        qlearn._q_function[(1, 0)] = 5.0
        qlearn._q_function[(1, 1)] = 10.0
        
        qlearn.update(0, 0, 0.0, 1, None, False)
        assert qlearn.get_q_value(0, 0) == 10.0  # 使用max


class TestExpectedSARSA:
    """测试Expected SARSA算法。"""
    
    def test_expected_q_computation(self):
        config = TDConfig(alpha=1.0, gamma=1.0, epsilon=0.0)
        exp_sarsa = ExpectedSARSA(config)
        exp_sarsa.set_action_space([0, 1])
        
        exp_sarsa._q_function[(1, 0)] = 5.0
        exp_sarsa._q_function[(1, 1)] = 10.0
        
        # epsilon=0时应该等于Q-Learning
        exp_sarsa.update(0, 0, 0.0, 1, None, False)
        assert exp_sarsa.get_q_value(0, 0) == 10.0


class TestDoubleQLearning:
    """测试Double Q-Learning算法。"""
    
    def test_two_q_tables(self):
        config = TDConfig(alpha=0.5, gamma=0.99)
        double_q = DoubleQLearning(config)
        double_q.set_action_space([0, 1])
        
        assert hasattr(double_q, '_q_a')
        assert hasattr(double_q, '_q_b')


class TestFactory:
    """测试工厂函数。"""
    
    def test_create_sarsa(self):
        learner = create_td_learner('sarsa', alpha=0.1)
        assert isinstance(learner, SARSA)
    
    def test_create_q_learning(self):
        learner = create_td_learner('q_learning', alpha=0.1)
        assert isinstance(learner, QLearning)
    
    def test_invalid_algorithm(self):
        with pytest.raises(ValueError):
            create_td_learner('invalid_algo')


class TestEnvironments:
    """测试环境。"""
    
    def test_cliff_walking_reset(self):
        env = CliffWalkingEnv()
        state, info = env.reset()
        assert state == 36  # (3, 0) -> 3*12 + 0
    
    def test_cliff_walking_cliff_penalty(self):
        env = CliffWalkingEnv()
        env.reset()
        # 向右走入悬崖
        state, reward, done, _, info = env.step(1)
        assert reward == -100
        assert info.get('fell', False)
    
    def test_random_walk_true_values(self):
        env = RandomWalk(n_states=5)
        true_vals = env.get_true_values(gamma=1.0)
        # V(s) = s / (n+1) for gamma=1
        assert abs(true_vals[3] - 0.5) < 0.01
    
    def test_windy_grid_wind_effect(self):
        env = WindyGridWorld()
        env.reset()
        # 在有风的列向下走，应该被风向上推
        env._state = (3, 4)  # 风力=1的列
        state, _, _, _, info = env.step(2)  # DOWN
        assert info['wind'] == 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
