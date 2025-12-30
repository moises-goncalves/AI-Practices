"""
时序差分学习演示程序 (TD Learning Demo)
======================================

用法:
    python main.py --algorithm sarsa --episodes 500
    python main.py --algorithm q_learning --env cliff_walking
    python main.py --compare sarsa q_learning
"""

import argparse
import numpy as np
from typing import List

from core import TDConfig, create_td_learner
from environments import CliffWalkingEnv, RandomWalk, WindyGridWorld
from utils import plot_training_curve, compare_algorithms


def get_env(name: str):
    """获取环境实例。"""
    envs = {
        'cliff_walking': CliffWalkingEnv,
        'random_walk': lambda: RandomWalk(n_states=19),
        'windy_grid': WindyGridWorld,
    }
    if name not in envs:
        raise ValueError(f"未知环境: {name}. 支持: {list(envs.keys())}")
    return envs[name]()


def train_single(args):
    """训练单个算法。"""
    config = TDConfig(
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon=args.epsilon,
        lambda_=args.lambda_
    )
    learner = create_td_learner(args.algorithm, config)
    env = get_env(args.env)
    
    print(f"训练 {args.algorithm} 在 {args.env} 环境...")
    metrics = learner.train(env, n_episodes=args.episodes, log_interval=args.log_interval)
    
    # 评估
    eval_reward, eval_std = learner.evaluate(env, n_episodes=100)
    print(f"\n评估结果: {eval_reward:.2f} ± {eval_std:.2f}")
    
    if args.plot:
        plot_training_curve(metrics.episode_rewards, title=f"{args.algorithm} Training")
        import matplotlib.pyplot as plt
        plt.show()


def compare_algorithms_demo(args):
    """比较多个算法。"""
    env = get_env(args.env)
    config = TDConfig(alpha=args.alpha, gamma=args.gamma, epsilon=args.epsilon)
    
    results = {}
    for algo in args.compare:
        print(f"\n训练 {algo}...")
        learner = create_td_learner(algo, config)
        metrics = learner.train(env, n_episodes=args.episodes, log_interval=args.log_interval)
        results[algo] = metrics.episode_rewards
        
        eval_reward, _ = learner.evaluate(env, n_episodes=100)
        print(f"  评估奖励: {eval_reward:.2f}")
    
    comparison = compare_algorithms(results)
    print("\n=== 算法对比 ===")
    for name, stats in comparison.items():
        print(f"{name}: 最终100回合平均 = {stats['final_100_mean']:.2f}")


def main():
    parser = argparse.ArgumentParser(description="时序差分学习演示")
    parser.add_argument('--algorithm', '-a', default='sarsa',
                        choices=['td0', 'sarsa', 'q_learning', 'expected_sarsa',
                                'double_q', 'n_step', 'td_lambda', 'sarsa_lambda'])
    parser.add_argument('--env', '-e', default='cliff_walking',
                        choices=['cliff_walking', 'random_walk', 'windy_grid'])
    parser.add_argument('--episodes', '-n', type=int, default=500)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--epsilon', type=float, default=0.1)
    parser.add_argument('--lambda_', type=float, default=0.9)
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--compare', nargs='+', help='比较多个算法')
    
    args = parser.parse_args()
    
    if args.compare:
        compare_algorithms_demo(args)
    else:
        train_single(args)


if __name__ == '__main__':
    main()
