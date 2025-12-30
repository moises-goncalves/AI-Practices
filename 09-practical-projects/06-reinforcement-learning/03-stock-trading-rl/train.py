#!/usr/bin/env python3
"""
股票交易强化学习训练脚本
"""

import argparse
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data import generate_sample_data, add_technical_indicators, split_data
from src.env import StockTradingEnv
from src.agents import DQNAgent, A2CAgent


def parse_args():
    parser = argparse.ArgumentParser(description='Train RL agent for stock trading')
    parser.add_argument('--agent', type=str, default='dqn', choices=['dqn', 'a2c'],
                        help='Agent type')
    parser.add_argument('--episodes', type=int, default=500,
                        help='Training episodes')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor')
    parser.add_argument('--initial_balance', type=float, default=100000,
                        help='Initial balance')
    parser.add_argument('--save_path', type=str, default='models',
                        help='Model save directory')
    parser.add_argument('--data_file', type=str, default=None,
                        help='Stock data CSV file (optional)')
    return parser.parse_args()


def train(args):
    os.makedirs(args.save_path, exist_ok=True)
    
    if args.data_file and os.path.exists(args.data_file):
        import pandas as pd
        df = pd.read_csv(args.data_file)
    else:
        print("Using sample data for training...")
        df = generate_sample_data('2020-01-01', '2023-12-31')
    
    df = add_technical_indicators(df)
    train_df, test_df = split_data(df, train_ratio=0.8)
    
    env = StockTradingEnv(train_df, initial_balance=args.initial_balance)
    
    state = env.reset()
    state_dim = len(state)
    action_dim = 3
    
    if args.agent == 'dqn':
        agent = DQNAgent(state_dim, action_dim, lr=args.lr, gamma=args.gamma)
    else:
        agent = A2CAgent(state_dim, action_dim, lr=args.lr, gamma=args.gamma)
    
    print("=" * 50)
    print(f"Stock Trading RL Training")
    print(f"Agent: {args.agent.upper()}")
    print(f"Episodes: {args.episodes}")
    print(f"Initial Balance: {args.initial_balance}")
    print("=" * 50)
    
    best_reward = float('-inf')
    rewards_history = []
    
    for episode in range(args.episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        states, actions, rewards, next_states, dones = [], [], [], [], []
        
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            
            if args.agent == 'dqn':
                agent.remember(state, action, reward, next_state, done)
                agent.train()
            else:
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                next_states.append(next_state)
                dones.append(done)
            
            state = next_state
            total_reward += reward
        
        if args.agent == 'a2c' and len(states) > 0:
            agent.train(states, actions, rewards, next_states, dones)
        
        if args.agent == 'dqn' and (episode + 1) % 10 == 0:
            agent.update_target()
        
        rewards_history.append(total_reward)
        
        if total_reward > best_reward:
            best_reward = total_reward
            agent.save(f"{args.save_path}/{args.agent}_best.pth")
        
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(rewards_history[-10:])
            print(f"Episode {episode+1}/{args.episodes}, "
                  f"Reward: {total_reward:.2f}, Avg: {avg_reward:.2f}, "
                  f"Final Asset: {info['total_asset']:.2f}")
    
    agent.save(f"{args.save_path}/{args.agent}_final.pth")
    print(f"\nTraining completed! Best reward: {best_reward:.2f}")
    
    print("\nEvaluating on test data...")
    test_env = StockTradingEnv(test_df, initial_balance=args.initial_balance)
    state = test_env.reset()
    done = False
    
    while not done:
        action = agent.select_action(state)
        state, _, done, info = test_env.step(action)
    
    profit = info['total_asset'] - args.initial_balance
    roi = profit / args.initial_balance * 100
    print(f"Test Results: Final Asset: {info['total_asset']:.2f}, "
          f"Profit: {profit:.2f}, ROI: {roi:.2f}%")


if __name__ == '__main__':
    args = parse_args()
    train(args)
