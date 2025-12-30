#!/usr/bin/env python3
"""
Flappy Bird DQN 测试脚本

使用训练好的模型玩游戏。

使用方法：
    python play.py --model models/dqn_final.pth
"""

import argparse
import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.game_env import FlappyBirdEnv
from src.dqn import DQN
from src.utils import preprocess_frame, FrameStack


def parse_args():
    parser = argparse.ArgumentParser(description='Play Flappy Bird with trained DQN')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model')
    parser.add_argument('--episodes', type=int, default=10,
                        help='Number of episodes to play')
    return parser.parse_args()


def main():
    args = parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = DQN().to(device)
    checkpoint = torch.load(args.model, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    assets_path = os.path.join(os.path.dirname(__file__), 'assets')
    env = FlappyBirdEnv(assets_path=assets_path)
    frame_stack = FrameStack(num_frames=4)
    
    scores = []
    
    for episode in range(args.episodes):
        frame = env.reset()
        processed = preprocess_frame(frame[:env.screen_width, :int(env.base_y)])
        state = frame_stack.reset(processed)
        
        done = False
        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                action = model(state_tensor).argmax(dim=1).item()
            
            next_frame, reward, done, info = env.step(action)
            processed = preprocess_frame(next_frame[:env.screen_width, :int(env.base_y)])
            state = frame_stack.push(processed)
        
        scores.append(info['score'])
        print(f"Episode {episode + 1}: Score = {info['score']}")
    
    print(f"\nAverage Score: {sum(scores) / len(scores):.2f}")
    print(f"Max Score: {max(scores)}")
    
    env.close()


if __name__ == '__main__':
    main()
