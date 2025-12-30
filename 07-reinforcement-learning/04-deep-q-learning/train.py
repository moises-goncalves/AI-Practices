"""
DQN训练脚本

提供完整的训练流程，支持命令行参数配置。

使用示例:
    python train.py --env CartPole-v1 --episodes 300 --double --dueling
"""

from __future__ import annotations

import argparse
import logging
import time
import warnings
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

try:
    import gymnasium as gym
    HAS_GYM = True
except ImportError:
    HAS_GYM = False
    warnings.warn("gymnasium未安装。安装: pip install gymnasium")

from agent import DQNAgent, create_dqn_agent
from core import DQNConfig
from utils.training import TrainingConfig, TrainingMetrics
from utils.visualization import plot_training_curves

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_dqn(
    agent: DQNAgent,
    env_name: str = "CartPole-v1",
    config: Optional[TrainingConfig] = None,
    verbose: bool = True,
) -> TrainingMetrics:
    """
    训练DQN Agent
    
    Parameters
    ----------
    agent : DQNAgent
        要训练的DQN agent
    env_name : str
        Gymnasium环境名称
    config : TrainingConfig
        训练配置
    verbose : bool
        是否打印训练日志
    
    Returns
    -------
    TrainingMetrics
        训练历史指标
    """
    if not HAS_GYM:
        raise ImportError("gymnasium未安装")
    
    if config is None:
        config = TrainingConfig()
    
    render_mode = "human" if config.render else None
    env = gym.make(env_name, render_mode=render_mode)
    
    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    metrics = TrainingMetrics()
    reward_window: Deque[float] = deque(maxlen=100)
    start_time = time.time()
    best_eval_reward = float("-inf")
    
    if verbose:
        print("=" * 60)
        print(f"DQN训练: {env_name}")
        print("=" * 60)
        print(f"状态维度: {agent.config.state_dim}")
        print(f"动作维度: {agent.config.action_dim}")
        print(f"Double DQN: {agent.config.double_dqn}")
        print(f"Dueling: {agent.config.dueling}")
        print(f"设备: {agent.device}")
        print("=" * 60)
    
    try:
        for episode in range(config.num_episodes):
            state, _ = env.reset()
            state = np.asarray(state, dtype=np.float32)
            
            episode_reward = 0.0
            episode_loss = 0.0
            loss_count = 0
            
            for step in range(config.max_steps_per_episode):
                action = agent.select_action(state, training=True)
                next_state, reward, terminated, truncated, _ = env.step(action)
                next_state = np.asarray(next_state, dtype=np.float32)
                done = terminated or truncated
                
                loss = agent.train_step(state, action, float(reward), next_state, done)
                
                if loss is not None:
                    episode_loss += loss
                    loss_count += 1
                
                episode_reward += reward
                state = next_state
                
                if done:
                    break
            
            avg_loss = episode_loss / loss_count if loss_count > 0 else None
            metrics.add_episode(episode_reward, step + 1, agent.epsilon, avg_loss)
            reward_window.append(episode_reward)
            
            # 日志
            if verbose and (episode + 1) % config.log_frequency == 0:
                avg_reward = np.mean(list(reward_window))
                loss_str = f"{avg_loss:.4f}" if avg_loss else "N/A"
                print(
                    f"回合 {episode + 1:4d} | "
                    f"奖励: {episode_reward:7.2f} | "
                    f"平均(100): {avg_reward:7.2f} | "
                    f"损失: {loss_str} | "
                    f"ε: {agent.epsilon:.3f}"
                )
            
            # 评估
            if (episode + 1) % config.eval_frequency == 0:
                eval_mean, eval_std = evaluate_agent(agent, env_name, config.eval_episodes)
                metrics.add_evaluation(episode + 1, eval_mean, eval_std)
                if verbose:
                    print(f"  [评估] 平均: {eval_mean:.2f} ± {eval_std:.2f}")
                
                if eval_mean > best_eval_reward:
                    best_eval_reward = eval_mean
                    agent.save(checkpoint_dir / "best_model.pt")
            
            # 保存检查点
            if (episode + 1) % config.save_frequency == 0:
                agent.save(checkpoint_dir / f"checkpoint_{episode + 1}.pt")
            
            # 早停
            if config.early_stopping_reward is not None:
                if len(reward_window) >= config.early_stopping_episodes:
                    recent = list(reward_window)[-config.early_stopping_episodes:]
                    if np.mean(recent) >= config.early_stopping_reward:
                        if verbose:
                            print(f"\n[早停] 达到目标奖励!")
                        break
    
    except KeyboardInterrupt:
        if verbose:
            print("\n[中断] 用户中断训练")
    finally:
        env.close()
    
    metrics.training_time = time.time() - start_time
    
    if verbose:
        stats = metrics.get_statistics()
        print("=" * 60)
        print("训练完成!")
        print(f"总回合: {stats['total_episodes']}")
        print(f"总步数: {stats['total_steps']}")
        print(f"训练时间: {metrics.training_time:.2f}秒")
        print(f"最终平均奖励: {stats['mean_reward']:.2f} ± {stats['std_reward']:.2f}")
        print("=" * 60)
    
    agent.save(checkpoint_dir / "final_model.pt")
    metrics.save(checkpoint_dir / "training_metrics.json")
    
    return metrics


def evaluate_agent(
    agent: DQNAgent,
    env_name: str,
    num_episodes: int = 10,
    max_steps: int = 1000,
    render: bool = False,
) -> Tuple[float, float]:
    """评估Agent性能"""
    if not HAS_GYM:
        raise ImportError("gymnasium未安装")
    
    render_mode = "human" if render else None
    env = gym.make(env_name, render_mode=render_mode)
    
    agent.set_eval_mode()
    rewards: List[float] = []
    
    try:
        for _ in range(num_episodes):
            state, _ = env.reset()
            state = np.asarray(state, dtype=np.float32)
            episode_reward = 0.0
            
            for _ in range(max_steps):
                action = agent.select_action(state, training=False)
                next_state, reward, terminated, truncated, _ = env.step(action)
                state = np.asarray(next_state, dtype=np.float32)
                episode_reward += reward
                if terminated or truncated:
                    break
            
            rewards.append(episode_reward)
    finally:
        env.close()
        agent.set_train_mode()
    
    return float(np.mean(rewards)), float(np.std(rewards))


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(description="DQN训练脚本")
    parser.add_argument("--env", type=str, default="CartPole-v1", help="环境名称")
    parser.add_argument("--episodes", type=int, default=300, help="训练回合数")
    parser.add_argument("--double", action="store_true", help="使用Double DQN")
    parser.add_argument("--dueling", action="store_true", help="使用Dueling架构")
    parser.add_argument("--per", action="store_true", help="使用优先经验回放")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--render", action="store_true", help="渲染环境")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints")
    args = parser.parse_args()
    
    if not HAS_GYM:
        print("错误: gymnasium未安装")
        return
    
    # 获取环境信息
    env = gym.make(args.env)
    state_dim = int(np.prod(env.observation_space.shape))
    action_dim = env.action_space.n
    env.close()
    
    # 创建agent
    agent = create_dqn_agent(
        state_dim=state_dim,
        action_dim=action_dim,
        double_dqn=args.double,
        dueling=args.dueling,
        prioritized_replay=args.per,
        seed=args.seed,
    )
    
    # 训练配置
    training_config = TrainingConfig(
        num_episodes=args.episodes,
        render=args.render,
        checkpoint_dir=args.checkpoint_dir,
    )
    
    # 训练
    metrics = train_dqn(agent, env_name=args.env, config=training_config)
    
    # 绘制曲线
    plot_training_curves(
        metrics.episode_rewards,
        metrics.losses,
        metrics.epsilon_history,
        metrics.eval_rewards,
        save_path="training_curves.png",
    )


if __name__ == "__main__":
    main()
