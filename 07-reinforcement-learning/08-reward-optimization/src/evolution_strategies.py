"""
Evolution Strategies (ES) Algorithm Implementation
==================================================

This module implements the Evolution Strategies algorithm, a population-based
black-box optimization method that uses Gaussian perturbations for exploration.

Core Components:
- EvolutionStrategies: Main ES algorithm class
- Utility functions for noise sampling and fitness ranking
"""

from typing import Tuple, Dict, Any
import numpy as np
from core.base import BasePolicySearch, BasePolicy
from core.config import ESConfig, NoiseType


class EvolutionStrategies(BasePolicySearch):
    """
    Evolution Strategies Algorithm

    Core Idea:
        Uses population-based optimization with Gaussian perturbations.
        Maintains a mean policy and explores by sampling perturbations.
        Updates mean based on fitness-weighted average of perturbations.

    Mathematical Theory:
        Parameter update:
        θ_{t+1} = θ_t + α * (1/λ) * ∑_i w_i * ε_i / σ

        where:
        - θ_t: current mean policy parameters
        - α: learning rate
        - λ: population size
        - w_i: fitness weight (rank or normalized fitness)
        - ε_i ~ N(0, I): Gaussian noise
        - σ: noise standard deviation

        With antithetic sampling:
        - Sample ε_i, then use both ε_i and -ε_i
        - Reduces variance, effective population size = 2λ

    Problem Statement:
        Addresses black-box optimization where:
        1. Gradients are unavailable or expensive
        2. Environment is non-differentiable
        3. Action space is discrete or mixed
        4. Parallelization is desired

    Comparison with Baselines:
        - vs Policy Gradient: No gradient computation, better for discrete actions
        - vs Genetic Algorithms: Simpler, faster convergence
        - vs CEM: Less sample-efficient but simpler
        - vs CMA-ES: Isotropic exploration, but faster per iteration

    Complexity:
        Time: O(λ * T) per generation (λ samples, T steps per episode)
        Space: O(λ * d) (λ samples of dimension d)

    Attributes:
        mean: Mean policy parameters
        noise_std: Current noise standard deviation
        best_weights: Best weights found so far
        best_fitness: Best fitness found so far
    """

    def __init__(self, policy: BasePolicy, config: ESConfig, env_fn=None):
        """
        Initialize Evolution Strategies.

        Args:
            policy: Policy network
            config: ESConfig configuration
            env_fn: Environment factory function
        """
        super().__init__(policy, config, env_fn)
        self.config: ESConfig = config
        self.mean = policy.get_weights().copy()
        self.noise_std = config.noise_std
        self.best_weights = self.mean.copy()
        self.best_fitness = -np.inf

    def _sample_population(self) -> np.ndarray:
        """
        Sample population using Gaussian perturbations.

        Returns:
            Population array of shape (population_size, weight_dim)
        """
        weight_dim = self.policy.get_weight_dim()
        population = np.zeros((self.config.population_size, weight_dim))

        if self.config.antithetic_sampling:
            # Use antithetic pairs for variance reduction
            half_pop = self.config.population_size // 2
            for i in range(half_pop):
                noise = self._sample_noise(weight_dim)
                population[2*i] = self.mean + self.noise_std * noise
                population[2*i + 1] = self.mean - self.noise_std * noise

            # Handle odd population size
            if self.config.population_size % 2 == 1:
                noise = self._sample_noise(weight_dim)
                population[-1] = self.mean + self.noise_std * noise
        else:
            # Standard sampling without antithetic pairs
            for i in range(self.config.population_size):
                noise = self._sample_noise(weight_dim)
                population[i] = self.mean + self.noise_std * noise

        return population

    def _sample_noise(self, dim: int) -> np.ndarray:
        """
        Sample noise according to configured distribution.

        Args:
            dim: Dimension of noise vector

        Returns:
            Noise vector
        """
        if self.config.noise_type == NoiseType.GAUSSIAN:
            return np.random.randn(dim)
        elif self.config.noise_type == NoiseType.UNIFORM:
            return np.random.uniform(-1, 1, dim)
        elif self.config.noise_type == NoiseType.CAUCHY:
            return np.random.standard_cauchy(dim)
        else:
            raise ValueError(f"Unknown noise type: {self.config.noise_type}")

    def _compute_fitness_weights(self, fitness: np.ndarray) -> np.ndarray:
        """
        Compute weights for fitness-weighted update.

        Args:
            fitness: Fitness values for population

        Returns:
            Weight vector for each sample
        """
        if self.config.fitness_shaping:
            # Rank-based fitness shaping
            ranks = np.argsort(np.argsort(-fitness))  # Descending rank
            weights = np.maximum(0, np.log(self.config.population_size / 2 + 1) - np.log(ranks + 1))
            weights /= np.sum(weights)
        else:
            # Direct fitness normalization
            fitness_centered = fitness - np.mean(fitness)
            if np.std(fitness_centered) > 1e-8:
                fitness_normalized = fitness_centered / np.std(fitness_centered)
            else:
                fitness_normalized = fitness_centered
            weights = np.maximum(0, fitness_normalized)
            if np.sum(weights) > 0:
                weights /= np.sum(weights)
            else:
                weights = np.ones_like(fitness) / len(fitness)

        return weights

    def _update_distribution(self, population: np.ndarray, fitness: np.ndarray) -> None:
        """
        Update mean policy based on fitness-weighted population.

        Args:
            population: Population of policies
            fitness: Fitness values for each policy
        """
        # Compute fitness weights
        weights = self._compute_fitness_weights(fitness)

        # Compute weighted average of perturbations
        perturbations = population - self.mean
        weighted_perturbation = np.average(perturbations, axis=0, weights=weights)

        # Update mean
        self.mean += self.config.learning_rate * weighted_perturbation

        # Adaptive noise (optional)
        if self.config.adaptive_noise:
            self.noise_std *= self.config.noise_decay

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


class NaturalEvolutionStrategies(EvolutionStrategies):
    """
    Natural Evolution Strategies (NES)

    Core Idea:
        Extends ES with natural gradient updates using Fisher information matrix.
        Provides invariance to reparameterization of the policy.

    Mathematical Theory:
        Natural gradient update:
        θ_{t+1} = θ_t + α * F^{-1} * ∇J(θ_t)

        where F is Fisher information matrix:
        F = E[(∇ log π_θ)(∇ log π_θ)^T]

        For Gaussian policy with diagonal covariance:
        F ≈ I (identity matrix for certain parameterizations)

    Problem Statement:
        Standard ES uses Euclidean gradient, which is not invariant to
        parameter reparameterization. Natural gradient provides better
        convergence properties.

    Comparison with Baselines:
        - vs ES: Better convergence rate, more principled updates
        - vs Policy Gradient: No explicit gradient computation needed
        - vs CMA-ES: Simpler than full covariance adaptation

    Attributes:
        Same as EvolutionStrategies
    """

    def _update_distribution(self, population: np.ndarray, fitness: np.ndarray) -> None:
        """
        Update using natural gradient.

        Args:
            population: Population of policies
            fitness: Fitness values for each policy
        """
        # Compute fitness weights
        weights = self._compute_fitness_weights(fitness)

        # Compute perturbations
        perturbations = population - self.mean
        weight_dim = self.policy.get_weight_dim()

        # Estimate Fisher information matrix (simplified)
        # For Gaussian noise: F ≈ (1/σ^2) * I
        fisher_inv = (self.noise_std ** 2) * np.eye(weight_dim)

        # Compute natural gradient
        weighted_perturbation = np.average(perturbations, axis=0, weights=weights)
        natural_gradient = fisher_inv @ weighted_perturbation / self.noise_std

        # Update mean
        self.mean += self.config.learning_rate * natural_gradient

        # Adaptive noise
        if self.config.adaptive_noise:
            self.noise_std *= self.config.noise_decay

        # Track best policy
        best_idx = np.argmax(fitness)
        if fitness[best_idx] > self.best_fitness:
            self.best_fitness = fitness[best_idx]
            self.best_weights = population[best_idx].copy()
