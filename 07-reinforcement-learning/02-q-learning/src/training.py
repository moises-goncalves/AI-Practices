"""Training Infrastructure for Tabular RL Agents.

This module provides a unified training interface for temporal-difference
control algorithms, supporting both custom environments and Gymnasium.

Core Components:
    - TrainingMetrics: Container for episode statistics
    - Trainer: Unified training loop with logging and checkpointing

Design Principles:
    - Algorithm-agnostic: Works with any BaseAgent subclass
    - Environment-agnostic: Gymnasium-compatible interface
    - Configurable logging and visualization
    - Checkpoint support for long training runs
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np


@dataclass
class TrainingMetrics:
    """Container for training statistics and performance metrics.

    Core Idea:
        Track episode-level statistics to monitor learning progress,
        diagnose issues, and compare algorithms.

    Attributes:
        episode_rewards: Cumulative reward per episode
        episode_lengths: Steps until termination per episode
        epsilon_history: Exploration rate trajectory
        td_errors: Mean TD error per episode (optional)
        eval_rewards: Periodic evaluation rewards (optional)
    """

    episode_rewards: List[float] = field(default_factory=list)
    episode_lengths: List[int] = field(default_factory=list)
    epsilon_history: List[float] = field(default_factory=list)
    td_errors: List[float] = field(default_factory=list)
    eval_rewards: List[float] = field(default_factory=list)

    def get_moving_average(self, window: int = 100) -> np.ndarray:
        """Compute moving average of episode rewards.

        Mathematical Formula:
            MA[i] = (1/w) Σ_{j=i}^{i+w-1} R[j]

        Args:
            window: Sliding window size

        Returns:
            Smoothed reward array of length max(0, len(rewards) - window + 1)

        Complexity: O(n) via convolution
        """
        if len(self.episode_rewards) < window:
            return np.array(self.episode_rewards, dtype=np.float32)

        return np.convolve(
            self.episode_rewards,
            np.ones(window, dtype=np.float32) / window,
            mode="valid",
        )

    def get_summary(self, last_n: int = 100) -> Dict[str, float]:
        """Get summary statistics for recent episodes.

        Args:
            last_n: Number of recent episodes to summarize

        Returns:
            Dictionary with mean, std, min, max for rewards and steps
        """
        recent_rewards = self.episode_rewards[-last_n:]
        recent_steps = self.episode_lengths[-last_n:]

        return {
            "mean_reward": float(np.mean(recent_rewards)) if recent_rewards else 0.0,
            "std_reward": float(np.std(recent_rewards)) if recent_rewards else 0.0,
            "min_reward": float(np.min(recent_rewards)) if recent_rewards else 0.0,
            "max_reward": float(np.max(recent_rewards)) if recent_rewards else 0.0,
            "mean_steps": float(np.mean(recent_steps)) if recent_steps else 0.0,
            "episodes": len(self.episode_rewards),
        }


class Trainer:
    """Unified training infrastructure for tabular RL agents.

    Core Idea:
        Encapsulate the training loop, providing consistent interface for
        different agents and environments while supporting logging,
        evaluation, and checkpointing.

    Architecture:
        - Environment wrapper handles both custom and Gymnasium interfaces
        - Callback system for custom logging/visualization
        - Periodic evaluation with greedy policy
        - Checkpoint saving at configurable intervals

    Complexity:
        Time: O(episodes × max_steps × agent_update_cost)
        Space: O(|S| × |A|) for Q-table (in agent)

    Example:
        >>> from src.agents import QLearningAgent
        >>> from src.environments import CliffWalkingEnv
        >>>
        >>> env = CliffWalkingEnv()
        >>> agent = QLearningAgent(n_actions=4)
        >>> trainer = Trainer(env, agent)
        >>> metrics = trainer.train(episodes=500)
    """

    def __init__(
        self,
        env: Any,
        agent: Any,
        eval_env: Optional[Any] = None,
    ) -> None:
        """Initialize trainer.

        Args:
            env: Training environment (Gymnasium-compatible interface)
            agent: RL agent with get_action() and update() methods
            eval_env: Optional separate environment for evaluation
        """
        self.env = env
        self.agent = agent
        self.eval_env = eval_env or env

        # Detect if this is SARSA (needs next_action in update)
        self._is_sarsa = hasattr(agent, "update") and "next_action" in str(
            agent.update.__code__.co_varnames
        )

    def train(
        self,
        episodes: int = 500,
        max_steps: int = 200,
        verbose: bool = True,
        log_interval: int = 100,
        eval_interval: Optional[int] = None,
        eval_episodes: int = 10,
        checkpoint_interval: Optional[int] = None,
        checkpoint_path: Optional[Union[str, Path]] = None,
        callback: Optional[Callable[[int, TrainingMetrics], None]] = None,
    ) -> TrainingMetrics:
        """Execute training loop.

        Algorithm:
            For each episode:
                1. Reset environment
                2. For each step:
                    a. Select action (with exploration)
                    b. Execute action, observe transition
                    c. Update Q-values
                3. Decay exploration rate
                4. Log metrics
                5. Optionally evaluate and checkpoint

        Args:
            episodes: Total training episodes
            max_steps: Maximum steps per episode
            verbose: Whether to print progress
            log_interval: Episodes between progress logs
            eval_interval: Episodes between evaluations (None to disable)
            eval_episodes: Number of episodes per evaluation
            checkpoint_interval: Episodes between checkpoints (None to disable)
            checkpoint_path: Path for checkpoint files
            callback: Function called after each episode with (episode, metrics)

        Returns:
            TrainingMetrics containing episode statistics
        """
        metrics = TrainingMetrics()

        for episode in range(episodes):
            episode_reward, episode_steps, mean_td = self._run_episode(
                max_steps=max_steps,
                training=True,
            )

            # Decay exploration
            self.agent.decay_epsilon()

            # Record metrics
            metrics.episode_rewards.append(episode_reward)
            metrics.episode_lengths.append(episode_steps)
            metrics.epsilon_history.append(self.agent.epsilon)
            if mean_td is not None:
                metrics.td_errors.append(mean_td)

            # Logging
            if verbose and (episode + 1) % log_interval == 0:
                self._log_progress(episode + 1, metrics, log_interval)

            # Evaluation
            if eval_interval and (episode + 1) % eval_interval == 0:
                eval_reward = self.evaluate(eval_episodes)
                metrics.eval_rewards.append(eval_reward)
                if verbose:
                    print(f"  [Eval] Mean reward: {eval_reward:.2f}")

            # Checkpointing
            if (
                checkpoint_interval
                and checkpoint_path
                and (episode + 1) % checkpoint_interval == 0
            ):
                path = Path(checkpoint_path)
                self.agent.save(path.with_stem(f"{path.stem}_ep{episode+1}"))

            # Callback
            if callback:
                callback(episode, metrics)

        # Final checkpoint
        if checkpoint_path:
            self.agent.save(checkpoint_path)

        return metrics

    def _run_episode(
        self,
        max_steps: int,
        training: bool = True,
    ) -> tuple[float, int, Optional[float]]:
        """Run a single episode.

        Args:
            max_steps: Maximum steps before truncation
            training: Whether to use exploration and update Q-values

        Returns:
            Tuple of (total_reward, steps, mean_td_error)
        """
        # Reset environment
        result = self.env.reset()
        state = result[0] if isinstance(result, tuple) else result

        total_reward = 0.0
        steps = 0
        td_errors = []

        # SARSA: pre-select initial action
        if self._is_sarsa and training:
            action = self.agent.get_action(state, training=True)

        for _ in range(max_steps):
            # Select action
            if not self._is_sarsa or not training:
                action = self.agent.get_action(state, training=training)

            # Execute action
            result = self.env.step(action)
            if len(result) == 3:
                next_state, reward, done = result
            else:
                # Gymnasium interface
                next_state, reward, terminated, truncated, _ = result
                done = terminated or truncated

            # Update Q-values
            if training:
                if self._is_sarsa:
                    # SARSA: need next action before update
                    next_action = self.agent.get_action(next_state, training=True)
                    td_error = self.agent.update(
                        state, action, reward, next_state, next_action, done
                    )
                    action = next_action  # Carry forward
                else:
                    # Q-Learning variants
                    td_error = self.agent.update(
                        state, action, reward, next_state, done
                    )
                td_errors.append(abs(td_error))

            state = next_state
            total_reward += reward
            steps += 1

            if done:
                break

        mean_td = float(np.mean(td_errors)) if td_errors else None
        return total_reward, steps, mean_td

    def evaluate(
        self,
        episodes: int = 10,
        max_steps: int = 200,
    ) -> float:
        """Evaluate agent with greedy policy.

        Args:
            episodes: Number of evaluation episodes
            max_steps: Maximum steps per episode

        Returns:
            Mean episode reward
        """
        rewards = []

        for _ in range(episodes):
            result = self.eval_env.reset()
            state = result[0] if isinstance(result, tuple) else result

            total_reward = 0.0

            for _ in range(max_steps):
                action = self.agent.get_action(state, training=False)

                result = self.eval_env.step(action)
                if len(result) == 3:
                    next_state, reward, done = result
                else:
                    next_state, reward, terminated, truncated, _ = result
                    done = terminated or truncated

                total_reward += reward
                state = next_state

                if done:
                    break

            rewards.append(total_reward)

        return float(np.mean(rewards))

    def _log_progress(
        self,
        episode: int,
        metrics: TrainingMetrics,
        interval: int,
    ) -> None:
        """Print training progress."""
        recent_rewards = metrics.episode_rewards[-interval:]
        recent_steps = metrics.episode_lengths[-interval:]

        avg_reward = np.mean(recent_rewards)
        avg_steps = np.mean(recent_steps)
        epsilon = metrics.epsilon_history[-1]

        print(
            f"Episode {episode:5d} | "
            f"Avg Reward: {avg_reward:8.2f} | "
            f"Avg Steps: {avg_steps:6.1f} | "
            f"ε: {epsilon:.4f}"
        )


def train_q_learning(
    env: Any,
    agent: Any,
    episodes: int = 500,
    max_steps: int = 200,
    verbose: bool = True,
    log_interval: int = 100,
) -> TrainingMetrics:
    """Convenience function for Q-Learning training.

    This is a simplified interface for basic training without
    evaluation or checkpointing.

    Args:
        env: Training environment
        agent: Q-Learning agent
        episodes: Number of training episodes
        max_steps: Maximum steps per episode
        verbose: Whether to print progress
        log_interval: Episodes between logs

    Returns:
        TrainingMetrics with episode statistics
    """
    trainer = Trainer(env, agent)
    return trainer.train(
        episodes=episodes,
        max_steps=max_steps,
        verbose=verbose,
        log_interval=log_interval,
    )


def train_sarsa(
    env: Any,
    agent: Any,
    episodes: int = 500,
    max_steps: int = 200,
    verbose: bool = True,
    log_interval: int = 100,
) -> TrainingMetrics:
    """Convenience function for SARSA training.

    SARSA requires special handling because the next action must
    be sampled before the update (on-policy requirement).

    Args:
        env: Training environment
        agent: SARSA agent
        episodes: Number of training episodes
        max_steps: Maximum steps per episode
        verbose: Whether to print progress
        log_interval: Episodes between logs

    Returns:
        TrainingMetrics with episode statistics
    """
    metrics = TrainingMetrics()

    for episode in range(episodes):
        result = env.reset()
        state = result[0] if isinstance(result, tuple) else result

        # SARSA: select initial action before loop
        action = agent.get_action(state, training=True)

        total_reward = 0.0
        steps = 0

        for _ in range(max_steps):
            result = env.step(action)
            if len(result) == 3:
                next_state, reward, done = result
            else:
                next_state, reward, terminated, truncated, _ = result
                done = terminated or truncated

            # SARSA: select next action before update
            next_action = agent.get_action(next_state, training=True)

            # Update with actual next action
            agent.update(state, action, reward, next_state, next_action, done)

            # Transition
            state = next_state
            action = next_action
            total_reward += reward
            steps += 1

            if done:
                break

        agent.decay_epsilon()

        metrics.episode_rewards.append(total_reward)
        metrics.episode_lengths.append(steps)
        metrics.epsilon_history.append(agent.epsilon)

        if verbose and (episode + 1) % log_interval == 0:
            avg_reward = np.mean(metrics.episode_rewards[-log_interval:])
            avg_steps = np.mean(metrics.episode_lengths[-log_interval:])
            print(
                f"Episode {episode + 1:5d} | "
                f"Avg Reward: {avg_reward:8.2f} | "
                f"Avg Steps: {avg_steps:6.1f} | "
                f"ε: {agent.epsilon:.4f}"
            )

    return metrics


if __name__ == "__main__":
    print("Training Infrastructure Unit Tests")
    print("=" * 60)

    # Import dependencies for testing
    import sys
    sys.path.insert(0, str(Path(__file__).parent))

    from agents import QLearningAgent, SARSAAgent
    from environments import CliffWalkingEnv

    # Test 1: TrainingMetrics moving average
    metrics = TrainingMetrics()
    metrics.episode_rewards = list(range(100))
    ma = metrics.get_moving_average(window=10)
    assert len(ma) == 91, "Moving average length incorrect"
    assert np.isclose(ma[0], 4.5), "Moving average value incorrect"
    print("✓ Test 1: TrainingMetrics moving average correct")

    # Test 2: TrainingMetrics summary
    metrics = TrainingMetrics()
    metrics.episode_rewards = [10.0, 20.0, 30.0, 40.0, 50.0]
    metrics.episode_lengths = [100, 90, 80, 70, 60]
    summary = metrics.get_summary(last_n=3)
    assert np.isclose(summary["mean_reward"], 40.0), "Summary mean incorrect"
    print("✓ Test 2: TrainingMetrics summary correct")

    # Test 3: Q-Learning training
    env = CliffWalkingEnv()
    agent = QLearningAgent(
        n_actions=4,
        learning_rate=0.5,
        epsilon=0.1,
        epsilon_decay=1.0,
        epsilon_min=0.1,
    )
    metrics = train_q_learning(env, agent, episodes=100, verbose=False)
    assert len(metrics.episode_rewards) == 100, "Should have 100 episodes"
    print("✓ Test 3: Q-Learning training runs correctly")

    # Test 4: SARSA training
    env = CliffWalkingEnv()
    agent = SARSAAgent(
        n_actions=4,
        learning_rate=0.5,
        epsilon=0.1,
        epsilon_decay=1.0,
        epsilon_min=0.1,
    )
    metrics = train_sarsa(env, agent, episodes=100, verbose=False)
    assert len(metrics.episode_rewards) == 100, "Should have 100 episodes"
    print("✓ Test 4: SARSA training runs correctly")

    # Test 5: Trainer class
    env = CliffWalkingEnv()
    agent = QLearningAgent(n_actions=4, epsilon=0.1, epsilon_decay=1.0)
    trainer = Trainer(env, agent)
    metrics = trainer.train(episodes=50, verbose=False)
    assert len(metrics.episode_rewards) == 50
    print("✓ Test 5: Trainer class works correctly")

    # Test 6: Trainer evaluation
    eval_reward = trainer.evaluate(episodes=10)
    assert isinstance(eval_reward, float)
    print(f"✓ Test 6: Trainer evaluation works (reward={eval_reward:.2f})")

    # Test 7: Training convergence
    env = CliffWalkingEnv()
    agent = QLearningAgent(
        n_actions=4,
        learning_rate=0.5,
        epsilon=0.1,
        epsilon_decay=1.0,
        epsilon_min=0.1,
    )
    metrics = train_q_learning(env, agent, episodes=300, verbose=False)
    final_avg = np.mean(metrics.episode_rewards[-50:])
    assert final_avg > -100, f"Training should converge (avg={final_avg})"
    print(f"✓ Test 7: Training converges (final avg reward: {final_avg:.2f})")

    print("=" * 60)
    print("All tests passed!")
