"""
Dynamics Models for World Prediction and Intrinsic Motivation

================================================================================
CORE IDEA
================================================================================
Dynamics models predict the consequences of actions in the environment.
In curiosity-driven exploration, prediction errors serve as intrinsic rewards:
states that are hard to predict are novel and worth exploring.

================================================================================
MATHEMATICAL THEORY
================================================================================
Forward Model: Predicts next state features from current state and action

    f̂(s_{t+1}) = g_ψ(f(s_t), a_t)

    Loss: L_fwd = ||f(s_{t+1}) - g_ψ(f(s_t), a_t)||²

Inverse Model: Predicts action from state transition (auxiliary task)

    â_t = h_ω(f(s_t), f(s_{t+1}))

    Loss: L_inv = -log P(a_t | h_ω(f(s_t), f(s_{t+1})))

The inverse model ensures the feature encoder captures action-relevant
information, not just any predictable aspects of the observation.

================================================================================
REFERENCES
================================================================================
[1] Pathak et al. (2017). Curiosity-driven exploration by self-supervised
    prediction. ICML. (ICM forward/inverse models)
[2] Chua et al. (2018). Deep RL in a Handful of Trials with Probabilistic
    Ensemble and Trajectory Sampling. (Ensemble dynamics)
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import numpy as np


class ForwardDynamicsModel:
    """Neural network predicting next state features from current state and action.

    ============================================================================
    CORE IDEA
    ============================================================================
    Learn to predict the consequences of actions in feature space. High
    prediction error indicates novelty: the agent hasn't learned this
    transition yet, so it should be explored.

    ============================================================================
    MATHEMATICAL THEORY
    ============================================================================
    The forward model g_ψ predicts next state features:

        f̂(s_{t+1}) = g_ψ(f(s_t), a_t)

    Training objective (MSE loss):

        L_fwd = E[||f(s_{t+1}) - g_ψ(f(s_t), a_t)||²]

    The prediction error serves as intrinsic reward:

        r_i = η · ||f(s_{t+1}) - f̂(s_{t+1})||²

    ============================================================================
    WHY FEATURE SPACE?
    ============================================================================
    Predicting in feature space rather than raw observation space:
    1. Ignores irrelevant noise (clouds moving, leaves rustling)
    2. Focuses on controllable aspects of the environment
    3. Reduces dimensionality for efficient prediction

    ============================================================================
    COMPLEXITY
    ============================================================================
    - Time: O(forward_pass) for prediction
    - Space: O(|ψ|) for network parameters
    """

    def __init__(
        self,
        feature_dim: int,
        action_dim: int,
        hidden_dims: Tuple[int, ...] = (128, 64),
        learning_rate: float = 0.001,
    ) -> None:
        """Initialize forward dynamics model.

        Args:
            feature_dim: Dimensionality of state features.
            action_dim: Dimensionality of action space.
                For discrete actions, this is the number of actions.
                For continuous actions, this is the action vector size.
            hidden_dims: Sizes of hidden layers.
            learning_rate: Learning rate for gradient descent updates.
        """
        self.feature_dim = feature_dim
        self.action_dim = action_dim
        self.hidden_dims = hidden_dims
        self.learning_rate = learning_rate

        self._weights: List[np.ndarray] = []
        self._biases: List[np.ndarray] = []
        self._init_network()

    def _init_network(self) -> None:
        """Initialize network parameters with Xavier/Glorot initialization."""
        input_dim = self.feature_dim + self.action_dim
        dims = [input_dim] + list(self.hidden_dims) + [self.feature_dim]

        for i in range(len(dims) - 1):
            fan_in, fan_out = dims[i], dims[i + 1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            self._weights.append(
                np.random.randn(fan_in, fan_out).astype(np.float64) * std
            )
            self._biases.append(np.zeros(fan_out, dtype=np.float64))

    def _encode_action(self, action: np.ndarray, batch_size: int) -> np.ndarray:
        """Encode action(s) to appropriate format.

        For discrete actions (scalars), creates one-hot encoding.
        For continuous actions (vectors), uses directly.

        Args:
            action: Action(s), shape varies.
            batch_size: Expected batch size.

        Returns:
            Encoded actions, shape (batch_size, action_dim).
        """
        action = np.atleast_1d(action)

        if action.ndim == 1 and len(action) == batch_size:
            action_encoded = np.zeros((batch_size, self.action_dim), dtype=np.float64)
            for i, a in enumerate(action):
                if isinstance(a, (int, np.integer)):
                    if 0 <= int(a) < self.action_dim:
                        action_encoded[i, int(a)] = 1.0
                else:
                    action_encoded[i] = np.atleast_1d(a)[: self.action_dim]
        elif action.ndim == 2:
            action_encoded = action[:, : self.action_dim].astype(np.float64)
        else:
            action_encoded = np.broadcast_to(
                np.atleast_2d(action), (batch_size, self.action_dim)
            ).astype(np.float64)

        return action_encoded

    def predict(
        self,
        current_features: np.ndarray,
        action: np.ndarray,
    ) -> np.ndarray:
        """Predict next state features given current features and action.

        Args:
            current_features: Current state features.
                Shape: (feature_dim,) or (batch, feature_dim).
            action: Action taken.
                Shape: scalar, (action_dim,), or (batch, action_dim).

        Returns:
            Predicted next features.
                Shape: (feature_dim,) or (batch, feature_dim).
        """
        current_features = np.atleast_2d(current_features).astype(np.float64)
        batch_size = len(current_features)
        single_input = current_features.shape[0] == 1

        action_encoded = self._encode_action(action, batch_size)
        x = np.concatenate([current_features, action_encoded], axis=1)

        for i, (w, b) in enumerate(zip(self._weights, self._biases)):
            x = x @ w + b
            if i < len(self._weights) - 1:
                x = np.maximum(0, x)

        return x.flatten() if single_input else x

    def compute_prediction_error(
        self,
        current_features: np.ndarray,
        action: np.ndarray,
        next_features: np.ndarray,
    ) -> float:
        """Compute squared prediction error (intrinsic reward signal).

        Args:
            current_features: Features of current state.
            action: Action taken.
            next_features: Actual features of next state.

        Returns:
            Squared L2 norm of prediction error.
        """
        predicted = self.predict(current_features, action)
        error = np.sum((next_features.flatten() - predicted.flatten()) ** 2)
        return float(error)

    def update(
        self,
        current_features: np.ndarray,
        action: np.ndarray,
        next_features: np.ndarray,
    ) -> float:
        """Update model parameters via gradient descent.

        Performs one step of backpropagation to minimize prediction error.

        Args:
            current_features: Current state features (batch).
            action: Actions taken (batch).
            next_features: Target next state features (batch).

        Returns:
            Loss value before update.
        """
        current_features = np.atleast_2d(current_features).astype(np.float64)
        next_features = np.atleast_2d(next_features).astype(np.float64)
        batch_size = len(current_features)

        action_encoded = self._encode_action(action, batch_size)
        x = np.concatenate([current_features, action_encoded], axis=1)

        activations = [x.copy()]
        for i, (w, b) in enumerate(zip(self._weights, self._biases)):
            x = x @ w + b
            if i < len(self._weights) - 1:
                x = np.maximum(0, x)
            activations.append(x.copy())

        loss = float(np.mean(np.sum((x - next_features) ** 2, axis=1)))

        delta = 2 * (x - next_features) / batch_size

        for i in range(len(self._weights) - 1, -1, -1):
            grad_w = activations[i].T @ delta
            grad_b = np.sum(delta, axis=0)

            self._weights[i] -= self.learning_rate * grad_w
            self._biases[i] -= self.learning_rate * grad_b

            if i > 0:
                delta = delta @ self._weights[i].T
                delta = delta * (activations[i] > 0)

        return loss

    def get_parameters(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Get network parameters."""
        return (
            [w.copy() for w in self._weights],
            [b.copy() for b in self._biases],
        )


class InverseDynamicsModel:
    """Neural network predicting action from state transition.

    ============================================================================
    CORE IDEA
    ============================================================================
    The inverse model predicts which action caused a state transition.
    This auxiliary task helps the encoder learn action-relevant features.

    ============================================================================
    MATHEMATICAL THEORY
    ============================================================================
    The inverse model h_ω predicts the action:

        â = h_ω(f(s_t), f(s_{t+1}))

    For discrete actions (classification):

        L_inv = -E[log P(a | h_ω(f(s_t), f(s_{t+1})))]

    For continuous actions (regression):

        L_inv = E[||a - h_ω(f(s_t), f(s_{t+1}))||²]

    ============================================================================
    WHY INVERSE MODEL?
    ============================================================================
    Without the inverse model, the encoder might learn features that are
    easy to predict but irrelevant to actions:
    - Static background (doesn't change with any action)
    - Random noise (unpredictable regardless of action)

    The inverse model forces features to capture what the agent can
    actually influence through its actions.

    ============================================================================
    COMPLEXITY
    ============================================================================
    - Time: O(forward_pass)
    - Space: O(|ω|) parameters
    """

    def __init__(
        self,
        feature_dim: int,
        action_dim: int,
        hidden_dims: Tuple[int, ...] = (128, 64),
        discrete_actions: bool = True,
        learning_rate: float = 0.001,
    ) -> None:
        """Initialize inverse dynamics model.

        Args:
            feature_dim: Dimensionality of state features.
            action_dim: Number of discrete actions or continuous action dim.
            hidden_dims: Hidden layer sizes.
            discrete_actions: Whether action space is discrete.
            learning_rate: Learning rate for updates.
        """
        self.feature_dim = feature_dim
        self.action_dim = action_dim
        self.hidden_dims = hidden_dims
        self.discrete_actions = discrete_actions
        self.learning_rate = learning_rate

        self._weights: List[np.ndarray] = []
        self._biases: List[np.ndarray] = []
        self._init_network()

    def _init_network(self) -> None:
        """Initialize network parameters."""
        input_dim = 2 * self.feature_dim
        dims = [input_dim] + list(self.hidden_dims) + [self.action_dim]

        for i in range(len(dims) - 1):
            fan_in, fan_out = dims[i], dims[i + 1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            self._weights.append(
                np.random.randn(fan_in, fan_out).astype(np.float64) * std
            )
            self._biases.append(np.zeros(fan_out, dtype=np.float64))

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Numerically stable softmax."""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def predict(
        self,
        current_features: np.ndarray,
        next_features: np.ndarray,
    ) -> np.ndarray:
        """Predict action from state transition.

        Args:
            current_features: Features of current state.
            next_features: Features of next state.

        Returns:
            For discrete: action probabilities, shape (action_dim,).
            For continuous: predicted action vector, shape (action_dim,).
        """
        current_features = np.atleast_2d(current_features).astype(np.float64)
        next_features = np.atleast_2d(next_features).astype(np.float64)
        single_input = current_features.shape[0] == 1

        x = np.concatenate([current_features, next_features], axis=1)

        for i, (w, b) in enumerate(zip(self._weights, self._biases)):
            x = x @ w + b
            if i < len(self._weights) - 1:
                x = np.maximum(0, x)

        if self.discrete_actions:
            x = self._softmax(x)

        return x.flatten() if single_input else x

    def compute_loss(
        self,
        current_features: np.ndarray,
        next_features: np.ndarray,
        action: np.ndarray,
    ) -> float:
        """Compute inverse model loss.

        Args:
            current_features: Features of current state.
            next_features: Features of next state.
            action: True action taken.

        Returns:
            Cross-entropy (discrete) or MSE (continuous) loss.
        """
        predicted = self.predict(current_features, next_features)

        if self.discrete_actions:
            action = np.atleast_1d(action)
            predicted = np.atleast_2d(predicted)
            predicted = np.clip(predicted, 1e-8, 1 - 1e-8)
            loss = 0.0
            for i, a in enumerate(action):
                loss -= np.log(predicted[i, int(a)] + 1e-10)
            return float(loss / len(action))
        else:
            action = np.atleast_2d(action).astype(np.float64)
            return float(np.mean(np.sum((predicted - action) ** 2, axis=-1)))

    def update(
        self,
        current_features: np.ndarray,
        next_features: np.ndarray,
        action: np.ndarray,
    ) -> float:
        """Update model parameters via gradient descent.

        Args:
            current_features: Batch of current state features.
            next_features: Batch of next state features.
            action: Batch of true actions.

        Returns:
            Loss before update.
        """
        current_features = np.atleast_2d(current_features).astype(np.float64)
        next_features = np.atleast_2d(next_features).astype(np.float64)
        batch_size = len(current_features)

        x = np.concatenate([current_features, next_features], axis=1)

        activations = [x.copy()]
        for i, (w, b) in enumerate(zip(self._weights, self._biases)):
            x = x @ w + b
            if i < len(self._weights) - 1:
                x = np.maximum(0, x)
            activations.append(x.copy())

        if self.discrete_actions:
            probs = self._softmax(x)
            action_int = np.atleast_1d(action).astype(int)

            loss = 0.0
            for i, a in enumerate(action_int):
                loss -= np.log(probs[i, a] + 1e-10)
            loss = float(loss / batch_size)

            target = np.zeros_like(probs)
            for i, a in enumerate(action_int):
                target[i, a] = 1.0
            delta = (probs - target) / batch_size
        else:
            action_arr = np.atleast_2d(action).astype(np.float64)
            loss = float(np.mean(np.sum((x - action_arr) ** 2, axis=-1)))
            delta = 2 * (x - action_arr) / batch_size

        for i in range(len(self._weights) - 1, -1, -1):
            grad_w = activations[i].T @ delta
            grad_b = np.sum(delta, axis=0)

            self._weights[i] -= self.learning_rate * grad_w
            self._biases[i] -= self.learning_rate * grad_b

            if i > 0:
                delta = delta @ self._weights[i].T
                delta = delta * (activations[i] > 0)

        return loss


class EnsembleDynamicsModel:
    """Ensemble of dynamics models for uncertainty-aware prediction.

    ============================================================================
    CORE IDEA
    ============================================================================
    Train multiple dynamics models with different initializations. The
    disagreement (variance) among their predictions indicates epistemic
    uncertainty: high disagreement → unexplored region → intrinsic reward.

    ============================================================================
    MATHEMATICAL THEORY
    ============================================================================
    Given K models {g_ψ₁, ..., g_ψₖ}, compute:

    Mean prediction:
        μ(s', a) = (1/K) Σᵢ g_ψᵢ(s, a)

    Epistemic uncertainty (disagreement):
        σ²(s', a) = (1/K) Σᵢ ||g_ψᵢ(s, a) - μ(s', a)||²

    Intrinsic reward from disagreement:
        r_i = η · σ(s', a)

    ============================================================================
    ADVANTAGES
    ============================================================================
    - Captures model uncertainty, not just prediction error
    - More robust to stochastic environments
    - High uncertainty in truly novel regions, not just noisy ones

    ============================================================================
    COMPLEXITY
    ============================================================================
    - Time: O(K × forward_pass) for prediction
    - Space: O(K × |ψ|) for K model copies
    """

    def __init__(
        self,
        feature_dim: int,
        action_dim: int,
        n_models: int = 5,
        hidden_dims: Tuple[int, ...] = (128, 64),
        learning_rate: float = 0.001,
    ) -> None:
        """Initialize ensemble dynamics model.

        Args:
            feature_dim: Dimensionality of state features.
            action_dim: Dimensionality of action space.
            n_models: Number of ensemble members.
            hidden_dims: Hidden layer sizes for each model.
            learning_rate: Learning rate for updates.
        """
        self.feature_dim = feature_dim
        self.action_dim = action_dim
        self.n_models = n_models
        self.hidden_dims = hidden_dims
        self.learning_rate = learning_rate

        self.models = [
            ForwardDynamicsModel(
                feature_dim=feature_dim,
                action_dim=action_dim,
                hidden_dims=hidden_dims,
                learning_rate=learning_rate,
            )
            for _ in range(n_models)
        ]

    def predict_all(
        self,
        current_features: np.ndarray,
        action: np.ndarray,
    ) -> np.ndarray:
        """Get predictions from all ensemble members.

        Args:
            current_features: Current state features.
            action: Action taken.

        Returns:
            Array of predictions, shape (n_models, feature_dim) or
            (n_models, batch, feature_dim).
        """
        predictions = np.array(
            [model.predict(current_features, action) for model in self.models]
        )
        return predictions

    def predict_mean(
        self,
        current_features: np.ndarray,
        action: np.ndarray,
    ) -> np.ndarray:
        """Get mean prediction across ensemble.

        Args:
            current_features: Current state features.
            action: Action taken.

        Returns:
            Mean predicted features.
        """
        predictions = self.predict_all(current_features, action)
        return np.mean(predictions, axis=0)

    def compute_disagreement(
        self,
        current_features: np.ndarray,
        action: np.ndarray,
    ) -> float:
        """Compute ensemble disagreement (epistemic uncertainty).

        Args:
            current_features: Current state features.
            action: Action taken.

        Returns:
            Standard deviation of predictions (scalar).
        """
        predictions = self.predict_all(current_features, action)
        mean_pred = np.mean(predictions, axis=0)

        variance = np.mean(np.sum((predictions - mean_pred) ** 2, axis=-1))
        return float(np.sqrt(variance))

    def compute_intrinsic_reward(
        self,
        current_features: np.ndarray,
        action: np.ndarray,
        scale: float = 0.01,
    ) -> float:
        """Compute disagreement-based intrinsic reward.

        Args:
            current_features: Current state features.
            action: Action taken.
            scale: Scaling factor for intrinsic reward.

        Returns:
            Scaled disagreement as intrinsic reward.
        """
        disagreement = self.compute_disagreement(current_features, action)
        return scale * disagreement

    def update(
        self,
        current_features: np.ndarray,
        action: np.ndarray,
        next_features: np.ndarray,
    ) -> float:
        """Update all ensemble members.

        Args:
            current_features: Batch of current features.
            action: Batch of actions.
            next_features: Batch of target next features.

        Returns:
            Average loss across ensemble.
        """
        losses = [
            model.update(current_features, action, next_features)
            for model in self.models
        ]
        return float(np.mean(losses))


if __name__ == "__main__":
    print("=" * 70)
    print("Dynamics Models - Validation Tests")
    print("=" * 70)

    np.random.seed(42)

    feature_dim = 32
    action_dim = 4

    # Test forward dynamics
    print("\n[Test 1] Forward Dynamics Model")
    forward = ForwardDynamicsModel(feature_dim, action_dim)

    curr = np.random.randn(feature_dim)
    act = np.array([2])
    pred = forward.predict(curr, act)
    print(f"  Prediction shape: {pred.shape}")
    assert pred.shape == (feature_dim,), "Shape mismatch"

    next_feat = np.random.randn(feature_dim)
    error = forward.compute_prediction_error(curr, act, next_feat)
    print(f"  Prediction error: {error:.4f}")
    print("  [PASS]")

    # Test inverse dynamics
    print("\n[Test 2] Inverse Dynamics Model")
    inverse = InverseDynamicsModel(feature_dim, action_dim)

    probs = inverse.predict(curr, next_feat)
    print(f"  Action probabilities: {probs.shape}, sum={probs.sum():.4f}")
    assert np.isclose(probs.sum(), 1.0), "Probabilities don't sum to 1"
    print("  [PASS]")

    # Test ensemble
    print("\n[Test 3] Ensemble Dynamics Model")
    ensemble = EnsembleDynamicsModel(feature_dim, action_dim, n_models=3)

    disagreement = ensemble.compute_disagreement(curr, act)
    print(f"  Ensemble disagreement: {disagreement:.4f}")

    intrinsic = ensemble.compute_intrinsic_reward(curr, act)
    print(f"  Intrinsic reward: {intrinsic:.6f}")
    print("  [PASS]")

    print("\n" + "=" * 70)
    print("All tests passed!")
    print("=" * 70)
