"""
CMA-ES (Covariance Matrix Adaptation Evolution Strategy) Implementation
=======================================================================

This module implements the CMA-ES algorithm, a state-of-the-art evolution strategy
that adapts both the mean and covariance matrix of the search distribution.

Core Components:
- CMAES: Main CMA-ES algorithm class
- Utility functions for covariance matrix adaptation
"""

from typing import Tuple, Dict, Any
import numpy as np
from core.base import BasePolicySearch, BasePolicy
from core.config import CMAESConfig


class CMAES(BasePolicySearch):
    """
    CMA-ES (Covariance Matrix Adaptation Evolution Strategy)

    Core Idea:
        Adapts both mean and covariance matrix of search distribution.
        Automatically learns correlations between parameters for efficient exploration.
        Particularly effective on ill-conditioned optimization landscapes.

    Mathematical Theory:
        Distribution: x ~ N(m, C) where m is mean, C is covariance matrix

        Mean update:
        m_{t+1} = m_t + α_m * ∑_i w_i * (x_i - m_t)

        Covariance update (simplified):
        C_{t+1} = (1-c_c)*C_t + c_c*p_c*p_c^T + c_cov*∑_i w_i*(x_i-m_t)*(x_i-m_t)^T/σ^2

        Evolution path (momentum):
        p_c = (1-c_c)*p_c + √(c_c*(2-c_c))*C^{-1/2}*(m_{t+1}-m_t)/σ

        Step-size adaptation (cumulative step-size adaptation):
        σ_{t+1} = σ_t * exp((c_σ/d_σ) * (||p_σ||/E[||N(0,I)||] - 1))

    Problem Statement:
        Addresses ill-conditioned optimization where parameter correlations matter.
        Outperforms isotropic methods (like ES) on rotated/correlated problems.
        Provides automatic adaptation without manual tuning.

    Comparison with Baselines:
        - vs ES: Adapts covariance, better for correlated parameters
        - vs CEM: More sophisticated adaptation, better convergence
        - vs Gradient Methods: No gradient needed, handles non-smooth functions
        - vs Genetic Algorithms: Faster convergence, better scaling

    Complexity:
        Time: O(λ * T + d^2) per generation (covariance matrix operations)
        Space: O(d^2) (covariance matrix storage)

    Attributes:
        mean: Mean of search distribution
        cov: Covariance matrix
        step_size: Standard deviation of search distribution
        evolution_path_c: Evolution path for covariance adaptation
        evolution_path_sigma: Evolution path for step-size adaptation
        best_weights: Best weights found so far
        best_fitness: Best fitness found so far
    """

    def __init__(self, policy: BasePolicy, config: CMAESConfig, env_fn=None):
        """
        Initialize CMA-ES.

        Args:
            policy: Policy network
            config: CMAESConfig configuration
            env_fn: Environment factory function
        """
        super().__init__(policy, config, env_fn)
        self.config: CMAESConfig = config

        weight_dim = policy.get_weight_dim()
        self.mean = policy.get_weights().copy()
        self.step_size = config.initial_step_size
        self.cov = np.eye(weight_dim)
        self.evolution_path_c = np.zeros(weight_dim)
        self.evolution_path_sigma = np.zeros(weight_dim)

        self.best_weights = self.mean.copy()
        self.best_fitness = -np.inf

        # CMA-ES hyperparameters
        self.lambda_ = config.population_size
        self.mu = max(1, self.lambda_ // 2)
        self.weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights /= np.sum(self.weights)
        self.mueff = 1.0 / np.sum(self.weights ** 2)

        # Adaptation parameters
        self.c_m = config.c_m
        self.c_c = config.c_c
        self.c_cov = config.c_cov
        self.c_sigma = 0.3
        self.d_sigma = config.damps
        self.c_p = 0.4 / (weight_dim + 2)

        # Expectation of ||N(0,I)||
        self.chi_n = weight_dim ** 0.5 * (1 - 1 / (4 * weight_dim) + 1 / (21 * weight_dim ** 2))

        self.generation = 0

    def _sample_population(self) -> np.ndarray:
        """
        Sample population from multivariate Gaussian distribution.

        Returns:
            Population array of shape (population_size, weight_dim)
        """
        weight_dim = self.policy.get_weight_dim()
        population = np.zeros((self.lambda_, weight_dim))

        # Compute Cholesky decomposition of covariance matrix
        try:
            L = np.linalg.cholesky(self.cov)
        except np.linalg.LinAlgError:
            # If Cholesky fails, use eigendecomposition
            eigvals, eigvecs = np.linalg.eigh(self.cov)
            eigvals = np.maximum(eigvals, 1e-10)
            L = eigvecs @ np.diag(np.sqrt(eigvals))

        # Sample population
        for i in range(self.lambda_):
            z = np.random.randn(weight_dim)
            population[i] = self.mean + self.step_size * (L @ z)

        return population

    def _update_distribution(self, population: np.ndarray, fitness: np.ndarray) -> None:
        """
        Update mean, covariance, and step-size.

        Args:
            population: Population of policies
            fitness: Fitness values for each policy
        """
        weight_dim = self.policy.get_weight_dim()

        # Select elite samples
        elite_indices = np.argsort(-fitness)[:self.mu]
        elite_population = population[elite_indices]
        elite_fitness = fitness[elite_indices]

        # Update mean
        old_mean = self.mean.copy()
        self.mean = np.average(elite_population, axis=0, weights=self.weights)
        mean_step = (self.mean - old_mean) / self.step_size

        # Update evolution path for covariance
        self.evolution_path_c = (
            (1 - self.c_c) * self.evolution_path_c +
            np.sqrt(self.c_c * (2 - self.c_c) * self.mueff) * mean_step
        )

        # Update covariance matrix
        cov_update = np.zeros((weight_dim, weight_dim))
        for i, idx in enumerate(elite_indices):
            diff = (elite_population[i] - old_mean) / self.step_size
            cov_update += self.weights[i] * np.outer(diff, diff)

        self.cov = (
            (1 - self.c_cov) * self.cov +
            self.c_cov * (np.outer(self.evolution_path_c, self.evolution_path_c) + cov_update)
        )

        # Ensure covariance matrix is symmetric
        self.cov = (self.cov + self.cov.T) / 2

        # Update step-size using cumulative step-size adaptation
        self.evolution_path_sigma = (
            (1 - self.c_sigma) * self.evolution_path_sigma +
            np.sqrt(self.c_sigma * (2 - self.c_sigma) * self.mueff) * mean_step
        )

        norm_ps = np.linalg.norm(self.evolution_path_sigma)
        self.step_size *= np.exp(
            (self.c_sigma / self.d_sigma) * (norm_ps / self.chi_n - 1)
        )

        # Track best policy
        best_idx = np.argmax(fitness)
        if fitness[best_idx] > self.best_fitness:
            self.best_fitness = fitness[best_idx]
            self.best_weights = population[best_idx].copy()

        self.generation += 1

    def get_best_policy(self) -> BasePolicy:
        """
        Return the best policy found during training.

        Returns:
            Policy with best weights
        """
        self.policy.set_weights(self.best_weights)
        return self.policy


class SeparableCMAES(CMAES):
    """
    Separable CMA-ES (Diagonal Covariance Adaptation)

    Core Idea:
        Simplified version of CMA-ES that only adapts diagonal elements
        of covariance matrix. Reduces computational cost while maintaining
        most benefits for separable problems.

    Mathematical Theory:
        Covariance matrix restricted to diagonal:
        C = diag(σ_1^2, σ_2^2, ..., σ_d^2)

        Diagonal update:
        σ_i^2 ← σ_i^2 * exp(c_cov * (z_i^2 - 1))

        where z_i are normalized steps.

    Problem Statement:
        For separable problems (no parameter correlations), full covariance
        adaptation is unnecessary. Diagonal version is faster and more stable.

    Complexity:
        Time: O(λ * T + d) per generation (linear in dimension)
        Space: O(d) (only diagonal elements)

    Attributes:
        Same as CMAES, but cov is diagonal
    """

    def __init__(self, policy: BasePolicy, config: CMAESConfig, env_fn=None):
        """Initialize Separable CMA-ES."""
        super().__init__(policy, config, env_fn)
        # Use only diagonal covariance
        weight_dim = policy.get_weight_dim()
        self.cov = np.eye(weight_dim)
        self.sigma_diag = np.ones(weight_dim)

    def _sample_population(self) -> np.ndarray:
        """
        Sample population with diagonal covariance.

        Returns:
            Population array of shape (population_size, weight_dim)
        """
        weight_dim = self.policy.get_weight_dim()
        population = np.zeros((self.lambda_, weight_dim))

        for i in range(self.lambda_):
            z = np.random.randn(weight_dim)
            population[i] = self.mean + self.step_size * (self.sigma_diag * z)

        return population

    def _update_distribution(self, population: np.ndarray, fitness: np.ndarray) -> None:
        """
        Update mean, diagonal covariance, and step-size.

        Args:
            population: Population of policies
            fitness: Fitness values for each policy
        """
        weight_dim = self.policy.get_weight_dim()

        # Select elite samples
        elite_indices = np.argsort(-fitness)[:self.mu]
        elite_population = population[elite_indices]

        # Update mean
        old_mean = self.mean.copy()
        self.mean = np.average(elite_population, axis=0, weights=self.weights)
        mean_step = (self.mean - old_mean) / self.step_size

        # Update evolution path for covariance
        self.evolution_path_c = (
            (1 - self.c_c) * self.evolution_path_c +
            np.sqrt(self.c_c * (2 - self.c_c) * self.mueff) * mean_step
        )

        # Update diagonal covariance
        for i, idx in enumerate(elite_indices):
            diff = (elite_population[i] - old_mean) / self.step_size
            self.sigma_diag *= np.exp(self.c_cov * (diff ** 2 - 1) / 2)

        # Update step-size
        self.evolution_path_sigma = (
            (1 - self.c_sigma) * self.evolution_path_sigma +
            np.sqrt(self.c_sigma * (2 - self.c_sigma) * self.mueff) * mean_step
        )

        norm_ps = np.linalg.norm(self.evolution_path_sigma)
        self.step_size *= np.exp(
            (self.c_sigma / self.d_sigma) * (norm_ps / self.chi_n - 1)
        )

        # Track best policy
        best_idx = np.argmax(fitness)
        if fitness[best_idx] > self.best_fitness:
            self.best_fitness = fitness[best_idx]
            self.best_weights = population[best_idx].copy()

        self.generation += 1
