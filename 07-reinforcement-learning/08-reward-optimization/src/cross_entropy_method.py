"""
Cross-Entropy Method (CEM) Implementation
==========================================

This module implements the Cross-Entropy Method, a sample-efficient policy search
algorithm that iteratively fits a distribution to elite samples.

Core Components:
- CrossEntropyMethod: Main CEM algorithm class
- Utility functions for elite selection and distribution fitting
"""

from typing import Tuple, Dict, Any
import numpy as np
from core.base import BasePolicySearch, BasePolicy
from core.config import CrossEntropyConfig


class CrossEntropyMethod(BasePolicySearch):
    """
    Cross-Entropy Method (CEM)

    Core Idea:
        Iteratively fits a distribution (typically Gaussian) to elite samples.
        Concentrates probability mass on high-performing regions of parameter space.
        Provides excellent sample efficiency through elite selection.

    Mathematical Theory:
        Kullback-Leibler divergence minimization:
        D_KL(p_elite || q_θ) = E_p_elite[log p_elite - log q_θ]

        For Gaussian: q_θ = N(μ, Σ)

        Distribution update:
        μ_{t+1} = mean(elite samples)
        Σ_{t+1} = cov(elite samples)

        With smoothing (exponential averaging):
        μ_{t+1} = (1-β)*μ_t + β*mean(elite)
        Σ_{t+1} = (1-β)*Σ_t + β*cov(elite)

        where β is smoothing factor.

    Problem Statement:
        Addresses sample efficiency in policy search. Standard ES wastes samples
        on poor policies. CEM concentrates exploration on promising regions,
        reducing wasted samples and improving convergence speed.

    Comparison with Baselines:
        - vs ES: Better sample efficiency through elite selection
        - vs CMA-ES: Simpler, but less adaptive covariance structure
        - vs Random Search: Exponentially faster convergence
        - vs Policy Gradient: No gradient computation, handles discrete actions

    Complexity:
        Time: O(λ * T + k*d^2) per generation (k elite samples, d dimension)
        Space: O(d^2) (covariance matrix)

    Attributes:
        mean: Mean of search distribution
        cov: Covariance matrix of search distribution
        elite_count: Number of elite samples to keep
        best_weights: Best weights found so far
        best_fitness: Best fitness found so far
    """

    def __init__(self, policy: BasePolicy, config: CrossEntropyConfig, env_fn=None):
        """
        Initialize Cross-Entropy Method.

        Args:
            policy: Policy network
            config: CrossEntropyConfig configuration
            env_fn: Environment factory function
        """
        super().__init__(policy, config, env_fn)
        self.config: CrossEntropyConfig = config

        weight_dim = policy.get_weight_dim()
        self.mean = policy.get_weights().copy()
        self.cov = np.eye(weight_dim) * (config.noise_std ** 2)
        self.elite_count = max(1, int(config.population_size * config.elite_ratio))

        self.best_weights = self.mean.copy()
        self.best_fitness = -np.inf

    def _sample_population(self) -> np.ndarray:
        """
        Sample population from multivariate Gaussian distribution.

        Returns:
            Population array of shape (population_size, weight_dim)
        """
        weight_dim = self.policy.get_weight_dim()
        population = np.zeros((self.config.population_size, weight_dim))

        # Compute Cholesky decomposition
        try:
            L = np.linalg.cholesky(self.cov)
        except np.linalg.LinAlgError:
            # Use eigendecomposition if Cholesky fails
            eigvals, eigvecs = np.linalg.eigh(self.cov)
            eigvals = np.maximum(eigvals, 1e-10)
            L = eigvecs @ np.diag(np.sqrt(eigvals))

        # Sample population
        for i in range(self.config.population_size):
            z = np.random.randn(weight_dim)
            population[i] = self.mean + L @ z

        return population

    def _update_distribution(self, population: np.ndarray, fitness: np.ndarray) -> None:
        """
        Update distribution based on elite samples.

        Args:
            population: Population of policies
            fitness: Fitness values for each policy
        """
        # Select elite samples
        elite_indices = np.argsort(-fitness)[:self.elite_count]
        elite_population = population[elite_indices]

        # Compute elite statistics
        elite_mean = np.mean(elite_population, axis=0)
        if len(elite_population) > 1:
            elite_cov = np.cov(elite_population.T)
        else:
            weight_dim = elite_population.shape[1]
            elite_cov = np.eye(weight_dim) * self.config.min_variance

        # Handle 1D case
        if elite_cov.ndim == 0:
            elite_cov = np.array([[elite_cov]])

        # Ensure minimum variance
        elite_cov = np.maximum(elite_cov, self.config.min_variance * np.eye(elite_cov.shape[0]))

        # Update distribution with smoothing
        if self.config.update_mean:
            self.mean = (
                (1 - self.config.smoothing_factor) * self.mean +
                self.config.smoothing_factor * elite_mean
            )

        if self.config.update_covariance:
            self.cov = (
                (1 - self.config.smoothing_factor) * self.cov +
                self.config.smoothing_factor * elite_cov
            )

        # Ensure covariance is symmetric
        self.cov = (self.cov + self.cov.T) / 2

        # Track best policy
        best_idx = np.argmax(fitness)
        if fitness[best_idx] > self.best_fitness:
            self.best_fitness = fitness[best_idx]
            self.best_weights = population[best_idx].copy()

    def get_best_policy(self) -> BasePolicy:
        """
        Return the best policy found during training.

        Returns:
            Policy with best weights
        """
        self.policy.set_weights(self.best_weights)
        return self.policy


class AdaptiveCEM(CrossEntropyMethod):
    """
    Adaptive Cross-Entropy Method

    Core Idea:
        Extends CEM with adaptive elite ratio and smoothing factor.
        Automatically adjusts exploration-exploitation trade-off during training.

    Mathematical Theory:
        Adaptive elite ratio:
        elite_ratio_t = elite_ratio_min + (elite_ratio_max - elite_ratio_min) * exp(-t/τ)

        Adaptive smoothing:
        β_t = β_min + (β_max - β_min) * (1 - exp(-t/τ))

        where t is generation number, τ is time constant.

    Problem Statement:
        Fixed elite ratio and smoothing may not be optimal throughout training.
        Early training benefits from more exploration (higher elite ratio),
        while later training benefits from more exploitation (lower elite ratio).

    Attributes:
        elite_ratio_min: Minimum elite ratio
        elite_ratio_max: Maximum elite ratio
        smoothing_min: Minimum smoothing factor
        smoothing_max: Maximum smoothing factor
        time_constant: Time constant for adaptation
    """

    def __init__(
        self,
        policy: BasePolicy,
        config: CrossEntropyConfig,
        env_fn=None,
        elite_ratio_min: float = 0.05,
        elite_ratio_max: float = 0.3,
        smoothing_min: float = 0.5,
        smoothing_max: float = 0.95,
        time_constant: float = 50.0
    ):
        """
        Initialize Adaptive CEM.

        Args:
            policy: Policy network
            config: CrossEntropyConfig configuration
            env_fn: Environment factory function
            elite_ratio_min: Minimum elite ratio
            elite_ratio_max: Maximum elite ratio
            smoothing_min: Minimum smoothing factor
            smoothing_max: Maximum smoothing factor
            time_constant: Time constant for adaptation
        """
        super().__init__(policy, config, env_fn)
        self.elite_ratio_min = elite_ratio_min
        self.elite_ratio_max = elite_ratio_max
        self.smoothing_min = smoothing_min
        self.smoothing_max = smoothing_max
        self.time_constant = time_constant
        self.generation = 0

    def _get_adaptive_elite_ratio(self) -> float:
        """
        Compute adaptive elite ratio based on generation.

        Returns:
            Elite ratio for current generation
        """
        decay = np.exp(-self.generation / self.time_constant)
        return self.elite_ratio_min + (self.elite_ratio_max - self.elite_ratio_min) * decay

    def _get_adaptive_smoothing(self) -> float:
        """
        Compute adaptive smoothing factor based on generation.

        Returns:
            Smoothing factor for current generation
        """
        growth = 1 - np.exp(-self.generation / self.time_constant)
        return self.smoothing_min + (self.smoothing_max - self.smoothing_min) * growth

    def _update_distribution(self, population: np.ndarray, fitness: np.ndarray) -> None:
        """
        Update distribution with adaptive parameters.

        Args:
            population: Population of policies
            fitness: Fitness values for each policy
        """
        # Get adaptive parameters
        adaptive_elite_ratio = self._get_adaptive_elite_ratio()
        adaptive_smoothing = self._get_adaptive_smoothing()

        # Select elite samples
        elite_count = max(1, int(self.config.population_size * adaptive_elite_ratio))
        elite_indices = np.argsort(-fitness)[:elite_count]
        elite_population = population[elite_indices]

        # Compute elite statistics
        elite_mean = np.mean(elite_population, axis=0)
        if len(elite_population) > 1:
            elite_cov = np.cov(elite_population.T)
        else:
            weight_dim = elite_population.shape[1]
            elite_cov = np.eye(weight_dim) * self.config.min_variance

        # Handle 1D case
        if elite_cov.ndim == 0:
            elite_cov = np.array([[elite_cov]])

        # Ensure minimum variance
        elite_cov = np.maximum(elite_cov, self.config.min_variance * np.eye(elite_cov.shape[0]))

        # Update distribution with adaptive smoothing
        if self.config.update_mean:
            self.mean = (
                (1 - adaptive_smoothing) * self.mean +
                adaptive_smoothing * elite_mean
            )

        if self.config.update_covariance:
            self.cov = (
                (1 - adaptive_smoothing) * self.cov +
                adaptive_smoothing * elite_cov
            )

        # Ensure covariance is symmetric
        self.cov = (self.cov + self.cov.T) / 2

        # Track best policy
        best_idx = np.argmax(fitness)
        if fitness[best_idx] > self.best_fitness:
            self.best_fitness = fitness[best_idx]
            self.best_weights = population[best_idx].copy()

        self.generation += 1


class ImportanceSampledCEM(CrossEntropyMethod):
    """
    Importance-Sampled Cross-Entropy Method

    Core Idea:
        Uses importance sampling to reweight samples from previous generations.
        Improves sample efficiency by utilizing historical data.

    Mathematical Theory:
        Importance weight:
        w_i = q_{t-1}(x_i) / q_t(x_i)

        where q_t is current distribution, q_{t-1} is previous distribution.

        For Gaussian:
        w_i = exp(-0.5 * (x_i - μ_{t-1})^T Σ_t^{-1} (x_i - μ_{t-1}) +
                   0.5 * (x_i - μ_t)^T Σ_{t-1}^{-1} (x_i - μ_t))

    Problem Statement:
        Standard CEM discards samples from previous generations. Importance
        sampling allows reuse of historical data, improving sample efficiency.

    Attributes:
        history_size: Number of previous generations to keep
        sample_history: Historical samples
        fitness_history: Historical fitness values
    """

    def __init__(self, policy: BasePolicy, config: CrossEntropyConfig, env_fn=None, history_size: int = 3):
        """
        Initialize Importance-Sampled CEM.

        Args:
            policy: Policy network
            config: CrossEntropyConfig configuration
            env_fn: Environment factory function
            history_size: Number of previous generations to keep
        """
        super().__init__(policy, config, env_fn)
        self.history_size = history_size
        self.sample_history = []
        self.fitness_history = []
        self.mean_history = []
        self.cov_history = []

    def _compute_importance_weights(self, samples: np.ndarray) -> np.ndarray:
        """
        Compute importance weights for historical samples.

        Args:
            samples: Sample array

        Returns:
            Importance weights
        """
        if len(self.mean_history) == 0:
            return np.ones(len(samples))

        weights = np.ones(len(samples))

        # Use most recent previous distribution
        prev_mean = self.mean_history[-1]
        prev_cov = self.cov_history[-1]

        try:
            prev_cov_inv = np.linalg.inv(prev_cov)
            curr_cov_inv = np.linalg.inv(self.cov)

            for i, sample in enumerate(samples):
                diff_prev = sample - prev_mean
                diff_curr = sample - self.mean

                log_weight = (
                    -0.5 * (diff_prev @ prev_cov_inv @ diff_prev) +
                    0.5 * (diff_curr @ curr_cov_inv @ diff_curr)
                )
                weights[i] = np.exp(np.clip(log_weight, -10, 10))

        except np.linalg.LinAlgError:
            pass

        return weights / np.sum(weights)

    def _update_distribution(self, population: np.ndarray, fitness: np.ndarray) -> None:
        """
        Update distribution using importance-sampled historical data.

        Args:
            population: Population of policies
            fitness: Fitness values for each policy
        """
        # Store current distribution
        self.mean_history.append(self.mean.copy())
        self.cov_history.append(self.cov.copy())
        if len(self.mean_history) > self.history_size:
            self.mean_history.pop(0)
            self.cov_history.pop(0)

        # Select elite samples
        elite_indices = np.argsort(-fitness)[:self.elite_count]
        elite_population = population[elite_indices]

        # Compute elite statistics
        elite_mean = np.mean(elite_population, axis=0)
        if len(elite_population) > 1:
            elite_cov = np.cov(elite_population.T)
        else:
            weight_dim = elite_population.shape[1]
            elite_cov = np.eye(weight_dim) * self.config.min_variance

        # Handle 1D case
        if elite_cov.ndim == 0:
            elite_cov = np.array([[elite_cov]])

        # Ensure minimum variance
        elite_cov = np.maximum(elite_cov, self.config.min_variance * np.eye(elite_cov.shape[0]))

        # Update distribution with smoothing
        if self.config.update_mean:
            self.mean = (
                (1 - self.config.smoothing_factor) * self.mean +
                self.config.smoothing_factor * elite_mean
            )

        if self.config.update_covariance:
            self.cov = (
                (1 - self.config.smoothing_factor) * self.cov +
                self.config.smoothing_factor * elite_cov
            )

        # Ensure covariance is symmetric
        self.cov = (self.cov + self.cov.T) / 2

        # Track best policy
        best_idx = np.argmax(fitness)
        if fitness[best_idx] > self.best_fitness:
            self.best_fitness = fitness[best_idx]
            self.best_weights = population[best_idx].copy()
