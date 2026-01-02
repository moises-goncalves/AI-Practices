"""
神经网络策略模块单元测试 (Unit Tests)

本模块提供完整的单元测试，验证所有核心组件的正确性。
"""

import sys
import os
import unittest
import tempfile

import numpy as np
import torch
import torch.nn as nn

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestNetworkBase(unittest.TestCase):
    """测试基础网络组件"""

    def test_mlp_forward(self):
        """测试MLP前向传播"""
        from networks.base import MLP

        mlp = MLP(input_dim=4, output_dim=2, hidden_dims=[64, 64])
        x = torch.randn(32, 4)
        y = mlp(x)

        self.assertEqual(y.shape, (32, 2))

    def test_mlp_gradient_flow(self):
        """测试MLP梯度流"""
        from networks.base import MLP

        mlp = MLP(input_dim=4, output_dim=2, hidden_dims=[64, 64])
        x = torch.randn(8, 4, requires_grad=True)
        y = mlp(x)
        loss = y.sum()
        loss.backward()

        self.assertIsNotNone(x.grad)
        self.assertFalse(torch.isnan(x.grad).any())

    def test_activation_functions(self):
        """测试激活函数获取"""
        from networks.base import get_activation

        for name in ["relu", "tanh", "elu", "gelu"]:
            act = get_activation(name)
            x = torch.randn(10)
            y = act(x)
            self.assertEqual(y.shape, x.shape)


class TestPolicyNetworks(unittest.TestCase):
    """测试策略网络"""

    def test_discrete_policy_sample(self):
        """测试离散策略采样"""
        from networks.policy import DiscretePolicy

        policy = DiscretePolicy(state_dim=4, action_dim=3)
        state = torch.randn(32, 4)

        action, log_prob, entropy = policy.sample(state)

        self.assertEqual(action.shape, (32,))
        self.assertEqual(log_prob.shape, (32,))
        self.assertEqual(entropy.shape, (32,))
        self.assertTrue((action >= 0).all() and (action < 3).all())

    def test_discrete_policy_evaluate(self):
        """测试离散策略评估"""
        from networks.policy import DiscretePolicy

        policy = DiscretePolicy(state_dim=4, action_dim=3)
        state = torch.randn(32, 4)

        action, log_prob1, _ = policy.sample(state)
        log_prob2, _ = policy.evaluate(state, action)

        self.assertTrue(torch.allclose(log_prob1, log_prob2, atol=1e-5))

    def test_continuous_policy_sample(self):
        """测试连续策略采样"""
        from networks.policy import ContinuousPolicy

        policy = ContinuousPolicy(state_dim=3, action_dim=2)
        state = torch.randn(32, 3)

        action, log_prob, entropy = policy.sample(state)

        self.assertEqual(action.shape, (32, 2))
        self.assertEqual(log_prob.shape, (32,))

    def test_squashed_gaussian_bounds(self):
        """测试压缩高斯策略的动作边界"""
        from networks.policy import SquashedGaussianPolicy

        policy = SquashedGaussianPolicy(state_dim=3, action_dim=2)
        state = torch.randn(100, 3)

        action, _, _ = policy.sample(state)

        self.assertTrue((action >= -1).all())
        self.assertTrue((action <= 1).all())


class TestValueNetworks(unittest.TestCase):
    """测试价值网络"""

    def test_value_network_output(self):
        """测试价值网络输出"""
        from networks.value import ValueNetwork

        value_net = ValueNetwork(state_dim=4)
        state = torch.randn(32, 4)

        value = value_net(state)

        self.assertEqual(value.shape, (32, 1))

    def test_actor_critic_network(self):
        """测试Actor-Critic网络"""
        from networks.value import ActorCriticNetwork

        ac_net = ActorCriticNetwork(state_dim=4, action_dim=2, continuous=False)
        state = torch.randn(32, 4)

        action, log_prob, entropy, value = ac_net.get_action_and_value(state)

        self.assertEqual(action.shape, (32,))
        self.assertEqual(log_prob.shape, (32,))
        self.assertEqual(entropy.shape, (32,))
        self.assertEqual(value.shape, (32,))

    def test_actor_critic_evaluate(self):
        """测试Actor-Critic评估一致性"""
        from networks.value import ActorCriticNetwork

        ac_net = ActorCriticNetwork(state_dim=4, action_dim=2)
        state = torch.randn(32, 4)

        action, log_prob1, _, _ = ac_net.get_action_and_value(state)
        log_prob2, _, _ = ac_net.evaluate_actions(state, action)

        self.assertTrue(torch.allclose(log_prob1, log_prob2, atol=1e-5))


class TestBuffers(unittest.TestCase):
    """测试缓冲区"""

    def test_episode_buffer_store(self):
        """测试Episode缓冲区存储"""
        from buffers.trajectory import EpisodeBuffer

        buffer = EpisodeBuffer()

        for t in range(100):
            state = np.random.randn(4).astype(np.float32)
            action = np.random.randint(0, 2)
            reward = 1.0
            log_prob = -0.5
            value = 5.0
            done = (t == 99)

            buffer.store(state, action, reward, log_prob, value, done)

        self.assertEqual(len(buffer), 100)
        self.assertEqual(buffer.total_reward, 100.0)

    def test_rollout_buffer_gae(self):
        """测试Rollout缓冲区GAE计算"""
        from buffers.trajectory import RolloutBuffer

        buffer = RolloutBuffer(
            buffer_size=64,
            state_dim=4,
            action_dim=1,
            discrete=True,
        )

        for t in range(64):
            state = np.random.randn(4).astype(np.float32)
            action = np.random.randint(0, 2)
            reward = 1.0
            log_prob = -0.5
            value = 5.0
            done = (t == 63)

            buffer.add(state, action, reward, log_prob, value, done)

        buffer.compute_gae(next_value=0.0, gamma=0.99, gae_lambda=0.95)

        self.assertEqual(len(buffer), 64)
        self.assertFalse(np.isnan(buffer.advantages).any())
        self.assertFalse(np.isnan(buffer.returns).any())

    def test_rollout_buffer_normalize(self):
        """测试优势标准化"""
        from buffers.trajectory import RolloutBuffer

        buffer = RolloutBuffer(buffer_size=64, state_dim=4)

        for t in range(64):
            buffer.add(
                np.random.randn(4).astype(np.float32),
                np.random.randint(0, 2),
                np.random.randn(),
                -0.5,
                np.random.randn(),
                False
            )

        buffer.compute_gae(next_value=0.0)
        buffer.normalize_advantages()

        self.assertAlmostEqual(buffer.advantages.mean(), 0.0, places=5)
        self.assertAlmostEqual(buffer.advantages.std(), 1.0, places=5)


class TestCoreTypes(unittest.TestCase):
    """测试核心类型"""

    def test_trajectory_add(self):
        """测试轨迹添加"""
        from core.types import Trajectory

        traj = Trajectory()

        for t in range(10):
            traj.add(
                state=np.random.randn(4),
                action=np.random.randint(0, 2),
                reward=1.0,
                log_prob=-0.5,
                value=5.0,
                done=(t == 9)
            )

        self.assertEqual(traj.length, 10)
        self.assertEqual(traj.total_reward, 10.0)

    def test_trajectory_to_tensors(self):
        """测试轨迹转换为张量"""
        from core.types import Trajectory

        traj = Trajectory()

        for t in range(10):
            traj.add(
                state=np.random.randn(4),
                action=np.random.randint(0, 2),
                reward=1.0,
                log_prob=-0.5,
                value=5.0,
                done=False
            )

        tensors = traj.to_tensors()

        self.assertEqual(tensors["states"].shape, (10, 4))
        self.assertEqual(tensors["actions"].shape, (10,))
        self.assertEqual(tensors["rewards"].shape, (10,))


class TestConfig(unittest.TestCase):
    """测试配置"""

    def test_config_validation(self):
        """测试配置验证"""
        from core.config import TrainingConfig

        # 有效配置
        config = TrainingConfig()
        self.assertIsNotNone(config)

        # 无效gamma
        with self.assertRaises(ValueError):
            TrainingConfig(gamma=1.5)

        # 无效学习率
        with self.assertRaises(ValueError):
            TrainingConfig(learning_rate=-0.1)

    def test_config_serialization(self):
        """测试配置序列化"""
        from core.config import TrainingConfig

        config = TrainingConfig(
            env_name="CartPole-v1",
            learning_rate=1e-4,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "config.json")
            config.save(path)

            loaded = TrainingConfig.load(path)

            self.assertEqual(loaded.env_name, config.env_name)
            self.assertEqual(loaded.learning_rate, config.learning_rate)


class TestEnums(unittest.TestCase):
    """测试枚举类型"""

    def test_policy_type_enum(self):
        """测试策略类型枚举"""
        from core.enums import PolicyType

        self.assertIsNotNone(PolicyType.DISCRETE)
        self.assertIsNotNone(PolicyType.CONTINUOUS)
        self.assertIsNotNone(PolicyType.SQUASHED_GAUSSIAN)

    def test_advantage_estimator_enum(self):
        """测试优势估计器枚举"""
        from core.enums import AdvantageEstimator

        self.assertIsNotNone(AdvantageEstimator.MONTE_CARLO)
        self.assertIsNotNone(AdvantageEstimator.TD_ERROR)
        self.assertIsNotNone(AdvantageEstimator.GAE)


class TestIntegration(unittest.TestCase):
    """集成测试"""

    def test_policy_buffer_integration(self):
        """测试策略和缓冲区集成"""
        from networks.policy import DiscretePolicy
        from buffers.trajectory import EpisodeBuffer

        policy = DiscretePolicy(state_dim=4, action_dim=2)
        buffer = EpisodeBuffer()

        # 模拟收集数据
        for _ in range(10):
            state = torch.randn(1, 4)
            action, log_prob, entropy = policy.sample(state)

            buffer.store(
                state=state.squeeze().numpy(),
                action=action.item(),
                reward=1.0,
                log_prob=log_prob.item(),
                done=False
            )

        self.assertEqual(len(buffer), 10)

    def test_actor_critic_training_step(self):
        """测试Actor-Critic训练步骤"""
        from networks.value import ActorCriticNetwork
        from buffers.trajectory import RolloutBuffer

        ac_net = ActorCriticNetwork(state_dim=4, action_dim=2)
        optimizer = torch.optim.Adam(ac_net.parameters(), lr=3e-4)
        buffer = RolloutBuffer(buffer_size=32, state_dim=4)

        # 收集数据
        for t in range(32):
            state = np.random.randn(4).astype(np.float32)
            state_t = torch.tensor(state).unsqueeze(0)

            with torch.no_grad():
                action, log_prob, _, value = ac_net.get_action_and_value(state_t)

            buffer.add(
                state=state,
                action=action.item(),
                reward=1.0,
                log_prob=log_prob.item(),
                value=value.item(),
                done=(t == 31)
            )

        # 计算GAE
        buffer.compute_gae(next_value=0.0)
        buffer.normalize_advantages()

        # 训练步骤
        data = buffer.get_all()
        _, log_probs, entropy, values = ac_net.get_action_and_value(
            data["states"], action=data["actions"]
        )

        advantages = data["advantages"]
        returns = data["returns"]

        policy_loss = -(log_probs * advantages).mean()
        value_loss = ((values - returns) ** 2).mean()
        entropy_loss = -entropy.mean()

        loss = policy_loss + 0.5 * value_loss + 0.01 * entropy_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        self.assertFalse(torch.isnan(loss))


def run_tests():
    """运行所有测试"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # 添加所有测试类
    suite.addTests(loader.loadTestsFromTestCase(TestNetworkBase))
    suite.addTests(loader.loadTestsFromTestCase(TestPolicyNetworks))
    suite.addTests(loader.loadTestsFromTestCase(TestValueNetworks))
    suite.addTests(loader.loadTestsFromTestCase(TestBuffers))
    suite.addTests(loader.loadTestsFromTestCase(TestCoreTypes))
    suite.addTests(loader.loadTestsFromTestCase(TestConfig))
    suite.addTests(loader.loadTestsFromTestCase(TestEnums))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))

    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    print("=" * 70)
    print("神经网络策略模块 - 单元测试")
    print("=" * 70)

    success = run_tests()

    print("\n" + "=" * 70)
    if success:
        print("所有测试通过!")
    else:
        print("部分测试失败!")
    print("=" * 70)
