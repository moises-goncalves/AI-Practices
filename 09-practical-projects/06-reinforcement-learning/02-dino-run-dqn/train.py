#!/usr/bin/env python3
"""
Chrome Dino DQN 训练脚本

支持两种模式：
1. 浏览器模式：控制真实Chrome浏览器（需要ChromeDriver）
2. 模拟模式：使用简化的游戏模拟器（用于测试）
"""

import argparse
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.dqn import DQNAgent
from src.game_env import DinoGameSimulator, DinoGameEnv
from src.utils import FrameBuffer


def parse_args():
    parser = argparse.ArgumentParser(description='Train DQN on Chrome Dino')
    parser.add_argument('--mode', type=str, default='simulator', choices=['browser', 'simulator'],
                        help='Game mode: browser (real Chrome) or simulator (for testing)')
    parser.add_argument('--episodes', type=int, default=1000,
                        help='Number of training episodes')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor')
    parser.add_argument('--epsilon', type=float, default=0.1,
                        help='Initial exploration rate')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--save_interval', type=int, default=100,
                        help='Save model every N episodes')
    parser.add_argument('--save_path', type=str, default='models',
                        help='Model save directory')
    parser.add_argument('--chrome_driver', type=str, default=None,
                        help='Path to ChromeDriver (for browser mode)')
    return parser.parse_args()


def train(args):
    os.makedirs(args.save_path, exist_ok=True)
    
    if args.mode == 'browser':
        env = DinoGameEnv(chrome_driver_path=args.chrome_driver)
    else:
        env = DinoGameSimulator()
    
    agent = DQNAgent(
        state_shape=(80, 80, 4),
        num_actions=2,
        learning_rate=args.lr,
        gamma=args.gamma,
        epsilon=args.epsilon,
        batch_size=args.batch_size
    )
    
    frame_buffer = FrameBuffer(num_frames=4, frame_size=(80, 80))
    
    scores = []
    
    print("=" * 50)
    print("Chrome Dino DQN Training")
    print(f"Mode: {args.mode}")
    print(f"Episodes: {args.episodes}")
    print("=" * 50)
    
    try:
        for episode in range(args.episodes):
            frame = env.reset()
            state = frame_buffer.reset(frame)
            
            total_reward = 0
            steps = 0
            done = False
            
            while not done:
                action = agent.select_action(state)
                next_frame, reward, done, info = env.step(action)
                next_state = frame_buffer.add_frame(next_frame)
                
                agent.remember(state, action, reward, next_state, done)
                loss = agent.replay()
                
                state = next_state
                total_reward += reward
                steps += 1
            
            scores.append(info['score'])
            
            if (episode + 1) % 10 == 0:
                avg_score = np.mean(scores[-10:])
                print(f"Episode {episode+1}/{args.episodes}, "
                      f"Score: {info['score']}, Avg: {avg_score:.1f}, "
                      f"Epsilon: {agent.epsilon:.4f}")
            
            if (episode + 1) % args.save_interval == 0:
                agent.save(f"{args.save_path}/dino_dqn_{episode+1}.h5")
                print(f"Model saved at episode {episode+1}")
        
        agent.save(f"{args.save_path}/dino_dqn_final.h5")
        print("Training completed!")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving model...")
        agent.save(f"{args.save_path}/dino_dqn_interrupted.h5")
    finally:
        env.close()


if __name__ == '__main__':
    args = parse_args()
    train(args)
