#!/usr/bin/env python3
"""
Flappy Bird DQN 训练脚本

使用方法：
    python train.py                          # 默认参数训练
    python train.py --num_iters 100000       # 指定训练步数
    python train.py --lr 1e-5 --batch_size 64  # 自定义参数
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.game_env import FlappyBirdEnv
from src.trainer import DQNTrainer


def parse_args():
    parser = argparse.ArgumentParser(description='Train DQN on Flappy Bird')
    parser.add_argument('--num_iters', type=int, default=2000000,
                        help='Total training iterations')
    parser.add_argument('--lr', type=float, default=1e-6,
                        help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--replay_size', type=int, default=50000,
                        help='Replay buffer size')
    parser.add_argument('--epsilon_start', type=float, default=0.1,
                        help='Initial exploration rate')
    parser.add_argument('--epsilon_end', type=float, default=1e-4,
                        help='Final exploration rate')
    parser.add_argument('--save_interval', type=int, default=100000,
                        help='Save model every N iterations')
    parser.add_argument('--save_path', type=str, default='models',
                        help='Directory to save models')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 50)
    print("Flappy Bird DQN Training")
    print("=" * 50)
    print(f"Training iterations: {args.num_iters}")
    print(f"Learning rate: {args.lr}")
    print(f"Batch size: {args.batch_size}")
    print("=" * 50)
    
    assets_path = os.path.join(os.path.dirname(__file__), 'assets')
    env = FlappyBirdEnv(assets_path=assets_path)
    
    trainer = DQNTrainer(
        env=env,
        learning_rate=args.lr,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        replay_size=args.replay_size,
        batch_size=args.batch_size,
    )
    
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        trainer.load(args.resume)
    
    try:
        trainer.train(
            num_iterations=args.num_iters,
            save_interval=args.save_interval,
            save_path=args.save_path
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving checkpoint...")
        trainer.save(f"{args.save_path}/dqn_interrupted.pth")
    finally:
        env.close()


if __name__ == '__main__':
    main()
