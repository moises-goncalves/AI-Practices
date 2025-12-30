#!/usr/bin/env python3
"""
时序差分学习主入口 (Main Entry Point)
====================================

提供命令行接口和快速演示功能。

使用方法:
--------
# 运行演示
python main.py --demo

# 训练SARSA
python main.py --algorithm sarsa --episodes 500

# 比较算法
python main.py --compare

# 运行测试
python main.py --test
"""

import argparse
import numpy as np
import sys
import os

# 添加模块路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core import (
    TDConfig,
    SARSA,
    QLearning,
    ExpectedSARSA,
    DoubleQLearning,
    create_td_learner,
)
from environments import (
    CliffWalkingEnv,
    RandomWalk,
    GridWorld,
    GridWorldConfig,
)
from utils import (
    plot_training_curves,
    plot_value_heatmap,
    compute_rmse,
    extract_greedy_policy,
)


def run_demo():
    """运行快速演示。"""
    print("=" * 60)
    print("时序差分学习演示")
    print("=" * 60)

    # 1. RandomWalk上的TD(0)预测
    print("\n1. RandomWalk环境上的TD(0)预测")
    print("-" * 40)

    from core import TD0ValueLearner

    env = RandomWalk(n_states=19)
    config = TDConfig(alpha=0.1, gamma=1.0)
    learner = TD0ValueLearner(config)

    print("训练中...")
    metrics = learner.train(env, n_episodes=100, log_interval=50)

    # 计算RMSE
    true_values = env.get_true_values(gamma=1.0)
    estimated = {s: learner._value_function.get(s, 0.0) for s in range(env.n_total_states)}
    rmse = compute_rmse(estimated, dict(enumerate(true_values)))
    print(f"RMSE: {rmse:.4f}")

    # 2. CliffWalking上的SARSA vs Q-Learning
    print("\n2. CliffWalking环境上的SARSA vs Q-Learning")
    print("-" * 40)

    env = CliffWalkingEnv()

    # SARSA
    print("训练SARSA...")
    sarsa = SARSA(TDConfig(alpha=0.5, gamma=1.0, epsilon=0.1))
    sarsa_metrics = sarsa.train(env, n_episodes=300, log_interval=150)

    # Q-Learning
    print("训练Q-Learning...")
    qlearn = QLearning(TDConfig(alpha=0.5, gamma=1.0, epsilon=0.1))
    qlearn_metrics = qlearn.train(env, n_episodes=300, log_interval=150)

    # 比较最后100回合的平均奖励
    sarsa_final = np.mean(sarsa_metrics.episode_rewards[-100:])
    qlearn_final = np.mean(qlearn_metrics.episode_rewards[-100:])

    print(f"\n最后100回合平均奖励:")
    print(f"  SARSA:      {sarsa_final:.2f}")
    print(f"  Q-Learning: {qlearn_final:.2f}")

    # 评估贪婪策略
    print("\n贪婪策略评估（无探索）:")
    sarsa_eval = sarsa.evaluate(env, n_episodes=100)
    qlearn_eval = qlearn.evaluate(env, n_episodes=100)
    print(f"  SARSA:      {sarsa_eval[0]:.2f} ± {sarsa_eval[1]:.2f}")
    print(f"  Q-Learning: {qlearn_eval[0]:.2f} ± {qlearn_eval[1]:.2f}")

    print("\n" + "=" * 60)
    print("演示完成!")
    print("=" * 60)


def train_algorithm(algorithm: str, n_episodes: int, env_name: str):
    """训练指定算法。"""
    print(f"训练 {algorithm} 在 {env_name} 上，{n_episodes} 回合")
    print("-" * 40)

    # 创建环境
    if env_name == 'cliff':
        env = CliffWalkingEnv()
    elif env_name == 'random_walk':
        env = RandomWalk(n_states=19)
    else:
        env = GridWorld(GridWorldConfig())

    # 创建学习器
    config = TDConfig(alpha=0.5, gamma=0.99, epsilon=0.1)
    learner = create_td_learner(algorithm, config)

    # 训练
    metrics = learner.train(env, n_episodes=n_episodes, log_interval=100)

    # 输出结果
    final_reward = np.mean(metrics.episode_rewards[-100:]) if len(metrics.episode_rewards) >= 100 else np.mean(metrics.episode_rewards)
    print(f"\n最终平均奖励: {final_reward:.2f}")

    # 评估
    mean_reward, std_reward = learner.evaluate(env, n_episodes=100)
    print(f"评估结果: {mean_reward:.2f} ± {std_reward:.2f}")


def compare_algorithms():
    """比较不同算法的性能。"""
    print("算法性能比较")
    print("=" * 60)

    env = CliffWalkingEnv()
    n_episodes = 500

    algorithms = ['sarsa', 'q_learning', 'expected_sarsa', 'double_q']
    results = {}

    for algo in algorithms:
        print(f"\n训练 {algo}...")
        learner = create_td_learner(algo, TDConfig(alpha=0.5, gamma=1.0, epsilon=0.1))
        metrics = learner.train(env, n_episodes=n_episodes, log_interval=n_episodes + 1)

        # 评估
        mean_reward, std_reward = learner.evaluate(env, n_episodes=100)
        results[algo] = {
            'metrics': metrics,
            'mean_reward': mean_reward,
            'std_reward': std_reward,
        }

    # 输出比较结果
    print("\n" + "=" * 60)
    print("评估结果比较:")
    print("-" * 40)
    print(f"{'算法':<20} {'平均奖励':<15} {'标准差':<10}")
    print("-" * 40)
    for algo, result in results.items():
        print(f"{algo:<20} {result['mean_reward']:<15.2f} {result['std_reward']:<10.2f}")

    # 尝试可视化
    try:
        metrics_list = [r['metrics'].__dict__ for r in results.values()]
        labels = list(results.keys())
        plot_training_curves(metrics_list, labels, title="算法比较")
    except Exception as e:
        print(f"\n可视化失败: {e}")


def run_tests():
    """运行单元测试。"""
    import unittest

    # 发现并运行测试
    loader = unittest.TestLoader()
    suite = loader.discover('tests', pattern='test_*.py')

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # 返回退出码
    return 0 if result.wasSuccessful() else 1


def main():
    """主函数。"""
    parser = argparse.ArgumentParser(
        description='时序差分学习模块命令行接口',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py --demo                    运行快速演示
  python main.py --algorithm sarsa         训练SARSA
  python main.py --compare                 比较算法性能
  python main.py --test                    运行单元测试
        """
    )

    parser.add_argument('--demo', action='store_true',
                        help='运行快速演示')
    parser.add_argument('--algorithm', type=str,
                        choices=['sarsa', 'q_learning', 'expected_sarsa', 'double_q'],
                        help='要训练的算法')
    parser.add_argument('--episodes', type=int, default=500,
                        help='训练回合数 (默认: 500)')
    parser.add_argument('--env', type=str, default='cliff',
                        choices=['cliff', 'random_walk', 'grid'],
                        help='训练环境 (默认: cliff)')
    parser.add_argument('--compare', action='store_true',
                        help='比较所有算法')
    parser.add_argument('--test', action='store_true',
                        help='运行单元测试')

    args = parser.parse_args()

    if args.demo:
        run_demo()
    elif args.algorithm:
        train_algorithm(args.algorithm, args.episodes, args.env)
    elif args.compare:
        compare_algorithms()
    elif args.test:
        sys.exit(run_tests())
    else:
        # 默认运行演示
        run_demo()


if __name__ == '__main__':
    main()
