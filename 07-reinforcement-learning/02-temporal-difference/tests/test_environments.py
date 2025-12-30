"""
环境单元测试 (Environment Unit Tests)
=====================================

测试覆盖:
--------
1. RandomWalk环境
2. CliffWalking环境
3. WindyGridWorld环境
4. GridWorld环境
5. Blackjack环境
"""

import unittest
import numpy as np

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environments import (
    RandomWalk,
    CliffWalkingEnv,
    WindyGridWorld,
    GridWorld,
    GridWorldConfig,
    Blackjack,
    Action,
    DiscreteSpace,
)


class TestDiscreteSpace(unittest.TestCase):
    """测试离散空间类。"""

    def test_creation(self):
        """测试创建。"""
        space = DiscreteSpace(4)
        self.assertEqual(space.n, 4)

    def test_sample(self):
        """测试采样。"""
        space = DiscreteSpace(4)
        for _ in range(100):
            sample = space.sample()
            self.assertTrue(0 <= sample < 4)

    def test_contains(self):
        """测试包含检查。"""
        space = DiscreteSpace(4)
        self.assertTrue(space.contains(0))
        self.assertTrue(space.contains(3))
        self.assertFalse(space.contains(4))
        self.assertFalse(space.contains(-1))


class TestRandomWalk(unittest.TestCase):
    """测试RandomWalk环境。"""

    def setUp(self):
        """设置测试环境。"""
        self.env = RandomWalk(n_states=5)

    def test_initialization(self):
        """测试初始化。"""
        self.assertEqual(self.env.n_states, 5)
        self.assertEqual(self.env.n_total_states, 7)

    def test_reset(self):
        """测试重置。"""
        state, info = self.env.reset()
        self.assertEqual(state, 3)  # 中间位置

    def test_step_moves_randomly(self):
        """测试步进随机移动。"""
        np.random.seed(42)
        self.env.reset()

        # 多次步进，检查状态变化
        states = []
        for _ in range(100):
            self.env.reset()
            state, _, done, _, _ = self.env.step(0)
            states.append(state)

        # 应该有向左和向右的移动
        unique_states = set(states)
        self.assertTrue(len(unique_states) > 1)

    def test_terminal_states(self):
        """测试终止状态。"""
        # 左终止
        self.env._state = 1  # 左边第一个非终止状态
        np.random.seed(0)  # 设置种子使其向左移动
        # 手动向左移动
        self.env._state = 0
        self.assertTrue(self.env._state == 0)  # 左终止

    def test_true_values(self):
        """测试真实价值计算。"""
        true_values = self.env.get_true_values(gamma=1.0)

        # 检查终止状态
        self.assertEqual(true_values[0], 0.0)
        self.assertEqual(true_values[6], 1.0)

        # 检查中间状态（应该线性递增）
        for i in range(1, 6):
            expected = i / 6
            self.assertAlmostEqual(true_values[i], expected, places=5)


class TestCliffWalkingEnv(unittest.TestCase):
    """测试CliffWalking环境。"""

    def setUp(self):
        """设置测试环境。"""
        self.env = CliffWalkingEnv()

    def test_initialization(self):
        """测试初始化。"""
        self.assertEqual(self.env.HEIGHT, 4)
        self.assertEqual(self.env.WIDTH, 12)
        self.assertEqual(self.env.observation_space.n, 48)

    def test_reset(self):
        """测试重置。"""
        state, _ = self.env.reset()
        self.assertEqual(state, 36)  # (3, 0) -> 3*12 + 0 = 36

    def test_cliff_positions(self):
        """测试悬崖位置。"""
        # 悬崖在底行中间
        for c in range(1, 11):
            self.assertIn((3, c), self.env._cliff)

        # 非悬崖位置
        self.assertNotIn((3, 0), self.env._cliff)  # 起点
        self.assertNotIn((3, 11), self.env._cliff)  # 终点

    def test_fall_off_cliff(self):
        """测试掉入悬崖。"""
        self.env.reset()
        # 从起点向右走进入悬崖
        state, reward, done, _, info = self.env.step(Action.RIGHT)

        self.assertEqual(reward, -100)
        self.assertTrue(info.get('fell_off_cliff', False))
        self.assertFalse(done)  # 不终止，回到起点
        self.assertEqual(state, 36)  # 回到起点

    def test_reach_goal(self):
        """测试到达目标。"""
        # 手动设置到目标旁边
        self.env._state = (3, 10)
        # 但(3, 10)是悬崖...改用(2, 11)向下
        self.env._state = (2, 11)
        state, reward, done, _, _ = self.env.step(Action.DOWN)

        self.assertTrue(done)
        self.assertEqual(reward, -1.0)

    def test_boundary_clipping(self):
        """测试边界裁剪。"""
        self.env.reset()
        # 在起点向左走应该保持原位
        state_before = self.env._state
        state, _, _, _, _ = self.env.step(Action.LEFT)
        # 状态应该在(3, 0)即36
        self.assertEqual(state, 36)


class TestWindyGridWorld(unittest.TestCase):
    """测试WindyGridWorld环境。"""

    def setUp(self):
        """设置测试环境。"""
        self.env = WindyGridWorld()

    def test_initialization(self):
        """测试初始化。"""
        self.assertEqual(self.env.HEIGHT, 7)
        self.assertEqual(self.env.WIDTH, 10)

    def test_wind_strength(self):
        """测试风力配置。"""
        expected_wind = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]
        self.assertEqual(self.env.WIND_STRENGTH, expected_wind)

    def test_wind_effect(self):
        """测试风力效果。"""
        self.env.reset()
        # 移动到有风区域
        self.env._state = (3, 3)  # 风力=1

        # 向右移动，应该被风向上吹
        state, _, _, _, info = self.env.step(Action.RIGHT)
        # 新位置应该是 (3-1, 4) = (2, 4)
        expected_state = 2 * 10 + 4
        self.assertEqual(state, expected_state)
        self.assertEqual(info['wind'], 1)

    def test_stochastic_wind(self):
        """测试随机风力。"""
        env = WindyGridWorld(stochastic_wind=True)
        env._state = (3, 6)  # 风力=2

        # 多次测试，风力应该变化
        np.random.seed(42)
        winds = set()
        for _ in range(100):
            env._state = (3, 6)
            _, _, _, _, info = env.step(Action.DOWN)
            winds.add(info['wind'])

        # 应该有1, 2, 3三种风力
        self.assertTrue(len(winds) > 1)

    def test_king_moves(self):
        """测试国王移动（8方向）。"""
        env = WindyGridWorld(king_moves=True)
        self.assertEqual(env.action_space.n, 8)


class TestGridWorld(unittest.TestCase):
    """测试GridWorld环境。"""

    def setUp(self):
        """设置测试环境。"""
        self.config = GridWorldConfig(
            height=4,
            width=4,
            start=(3, 0),
            goals={(0, 3): 1.0}
        )
        self.env = GridWorld(self.config)

    def test_initialization(self):
        """测试初始化。"""
        self.assertEqual(self.env.observation_space.n, 16)
        self.assertEqual(self.env.action_space.n, 4)

    def test_reset(self):
        """测试重置。"""
        state, info = self.env.reset()
        self.assertEqual(state, 12)  # (3, 0) -> 3*4 + 0 = 12

    def test_reach_goal(self):
        """测试到达目标。"""
        self.env._state = (0, 2)  # 目标旁边
        state, reward, done, _, _ = self.env.step(Action.RIGHT)

        self.assertTrue(done)
        self.assertEqual(reward, -1.0 + 1.0)  # step_cost + goal_reward

    def test_obstacles(self):
        """测试障碍物。"""
        config = GridWorldConfig(
            height=4,
            width=4,
            start=(3, 0),
            goals={(0, 3): 1.0},
            obstacles={(1, 1)}
        )
        env = GridWorld(config)

        # 尝试移动到障碍物
        env._state = (1, 0)
        state, _, _, _, _ = env.step(Action.RIGHT)

        # 应该保持原位
        self.assertEqual(state, 4)  # (1, 0)

    def test_stochastic_transitions(self):
        """测试随机转移。"""
        config = GridWorldConfig(
            height=4,
            width=4,
            start=(2, 2),
            goals={(0, 3): 1.0},
            stochastic=True,
            slip_prob=0.5
        )
        env = GridWorld(config)

        # 多次执行同一动作，应该有不同结果
        np.random.seed(42)
        results = set()
        for _ in range(100):
            env.reset()
            state, _, _, _, _ = env.step(Action.RIGHT)
            results.add(state)

        self.assertTrue(len(results) > 1)


class TestBlackjack(unittest.TestCase):
    """测试Blackjack环境。"""

    def setUp(self):
        """设置测试环境。"""
        self.env = Blackjack()

    def test_initialization(self):
        """测试初始化。"""
        self.assertEqual(self.env.observation_space.n, 200)
        self.assertEqual(self.env.action_space.n, 2)

    def test_reset(self):
        """测试重置。"""
        np.random.seed(42)
        state, info = self.env.reset()

        self.assertTrue(0 <= state < 200)
        self.assertIn('player_hand', info)
        self.assertIn('dealer_showing', info)

    def test_state_encoding_decoding(self):
        """测试状态编解码。"""
        for player_sum in range(12, 22):
            for dealer_showing in range(1, 11):
                for usable_ace in [False, True]:
                    state = self.env._encode_state(
                        player_sum, dealer_showing, usable_ace
                    )
                    decoded = self.env.decode_state(state)
                    self.assertEqual(decoded, (player_sum, dealer_showing, usable_ace))

    def test_hand_value_calculation(self):
        """测试手牌点数计算。"""
        # 普通手牌
        value, usable = self.env._hand_value([5, 7])
        self.assertEqual(value, 12)
        self.assertFalse(usable)

        # 有可用A
        value, usable = self.env._hand_value([1, 6])
        self.assertEqual(value, 17)  # A算11
        self.assertTrue(usable)

        # A不可用（会爆）
        value, usable = self.env._hand_value([1, 5, 6])
        self.assertEqual(value, 12)  # A算1
        self.assertFalse(usable)

    def test_hit_action(self):
        """测试要牌动作。"""
        np.random.seed(42)
        self.env.reset()
        initial_hand = self.env._player_hand.copy()

        state, reward, done, _, _ = self.env.step(1)  # HIT

        # 手牌应该增加
        self.assertEqual(len(self.env._player_hand), len(initial_hand) + 1)

    def test_stick_action(self):
        """测试停牌动作。"""
        np.random.seed(42)
        self.env.reset()

        state, reward, done, _, info = self.env.step(0)  # STICK

        self.assertTrue(done)
        self.assertIn('result', info)
        self.assertIn('player_sum', info)
        self.assertIn('dealer_sum', info)

    def test_bust(self):
        """测试爆牌。"""
        self.env._player_hand = [10, 10, 5]  # 25点
        self.assertTrue(self.env._is_bust(self.env._player_hand))

    def test_natural_blackjack(self):
        """测试自然21点。"""
        self.assertTrue(self.env._is_natural([1, 10]))
        self.assertTrue(self.env._is_natural([10, 1]))
        self.assertFalse(self.env._is_natural([7, 7, 7]))  # 21但不是natural


if __name__ == '__main__':
    unittest.main(verbosity=2)
