"""
Feature Encoding Networks for State Representation Learning

================================================================================
CORE IDEA
================================================================================
Feature encoders transform high-dimensional observations (images, sensor data)
into compact, learnable representations. The quality of these features directly
impacts the effectiveness of downstream reward learning algorithms.

================================================================================
MATHEMATICAL THEORY
================================================================================
An encoder φ: O → ℝᵈ maps observations to d-dimensional feature vectors:

    f = φ_θ(o)

Key properties of good representations:
1. Compression: d << dim(O) for complex observations
2. Task-relevance: Features capture information useful for decision-making
3. Stability: Similar states map to similar features

Training objectives vary by method:
- ICM: Inverse model forces action-predictive features
- RND: Fixed random targets define the feature space
- VAE: Reconstruction + KL regularization

================================================================================
REFERENCES
================================================================================
[1] Pathak et al. (2017). ICM feature learning
[2] Burda et al. (2019). RND feature spaces
[3] Kingma & Welling (2014). VAE for representation learning
"""

from __future__ import annotations

import abc
from typing import List, Optional, Tuple, Union

import numpy as np


class FeatureEncoder(abc.ABC):
    """Abstract base class for feature encoding networks.

    ============================================================================
    DESIGN RATIONALE
    ============================================================================
    Feature encoders serve as the foundation for curiosity-driven exploration
    and inverse RL. They must:

    1. Handle variable input formats (vectors, images, sequences)
    2. Support both single observations and batches
    3. Optionally normalize outputs for stable downstream learning
    4. Be trainable via backpropagation for joint optimization

    ============================================================================
    USAGE PATTERN
    ============================================================================
    ```python
    encoder = MLPFeatureEncoder(input_dim=84, feature_dim=64)

    # Single observation
    obs = env.reset()
    features = encoder.encode(obs)  # shape: (64,)

    # Batch processing
    batch_obs = np.stack([env.reset() for _ in range(32)])
    batch_features = encoder.encode(batch_obs)  # shape: (32, 64)
    ```
    """

    def __init__(
        self,
        input_dim: int,
        feature_dim: int,
        normalize_output: bool = True,
    ) -> None:
        """Initialize feature encoder.

        Args:
            input_dim: Dimensionality of input observations.
            feature_dim: Dimensionality of output features.
            normalize_output: Whether to L2-normalize output features.
                Normalization prevents feature collapse and stabilizes
                prediction error computation.
        """
        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self.normalize_output = normalize_output

    @abc.abstractmethod
    def encode(self, observation: np.ndarray) -> np.ndarray:
        """Encode observation(s) to feature representation(s).

        Args:
            observation: Input observation(s).
                Shape: (input_dim,) for single, (batch, input_dim) for batch.

        Returns:
            Feature vector(s).
                Shape: (feature_dim,) for single, (batch, feature_dim) for batch.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_parameters(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Get network parameters.

        Returns:
            Tuple of (weights, biases) lists.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def set_parameters(
        self,
        weights: List[np.ndarray],
        biases: List[np.ndarray],
    ) -> None:
        """Set network parameters.

        Args:
            weights: List of weight matrices.
            biases: List of bias vectors.
        """
        raise NotImplementedError

    def _normalize(self, features: np.ndarray) -> np.ndarray:
        """L2-normalize feature vectors along last axis."""
        if not self.normalize_output:
            return features
        norm = np.linalg.norm(features, axis=-1, keepdims=True)
        return features / (norm + 1e-8)


class MLPFeatureEncoder(FeatureEncoder):
    """Multi-layer perceptron feature encoder.

    ============================================================================
    MATHEMATICAL THEORY
    ============================================================================
    The MLP encoder applies a sequence of affine transformations and
    non-linearities:

        h₀ = o                          (input observation)
        hᵢ = σ(Wᵢ hᵢ₋₁ + bᵢ)           (hidden layers)
        f = [Wₙ hₙ₋₁ + bₙ] / ||...||   (output, optionally normalized)

    where σ is the activation function (ReLU by default).

    ============================================================================
    ARCHITECTURE NOTES
    ============================================================================
    - Xavier/Glorot initialization for stable gradient flow
    - ReLU activations for hidden layers (no vanishing gradients)
    - Optional L2 normalization on output for stable ICM/RND
    - Supports arbitrary depth via hidden_dims tuple

    ============================================================================
    COMPLEXITY
    ============================================================================
    - Time: O(Σᵢ dᵢ · dᵢ₊₁) = O(forward_pass)
    - Space: O(Σᵢ dᵢ · dᵢ₊₁) for parameters
    """

    def __init__(
        self,
        input_dim: int,
        feature_dim: int = 64,
        hidden_dims: Tuple[int, ...] = (128, 64),
        activation: str = "relu",
        normalize_output: bool = True,
    ) -> None:
        """Initialize MLP feature encoder.

        Args:
            input_dim: Input observation dimensionality.
            feature_dim: Output feature dimensionality.
            hidden_dims: Sizes of hidden layers.
            activation: Activation function ("relu", "tanh", "leaky_relu").
            normalize_output: Whether to L2-normalize output.
        """
        super().__init__(input_dim, feature_dim, normalize_output)
        self.hidden_dims = hidden_dims
        self.activation = activation

        self._weights: List[np.ndarray] = []
        self._biases: List[np.ndarray] = []
        self._init_network()

    def _init_network(self) -> None:
        """Initialize network with Xavier/Glorot initialization.

        Xavier initialization sets weights with variance:
            Var(W) = 2 / (fan_in + fan_out)

        This maintains gradient variance across layers, preventing
        vanishing/exploding gradients in deep networks.
        """
        dims = [self.input_dim] + list(self.hidden_dims) + [self.feature_dim]

        for i in range(len(dims) - 1):
            fan_in, fan_out = dims[i], dims[i + 1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            self._weights.append(
                np.random.randn(fan_in, fan_out).astype(np.float64) * std
            )
            self._biases.append(np.zeros(fan_out, dtype=np.float64))

    def _activation_fn(self, x: np.ndarray) -> np.ndarray:
        """Apply activation function element-wise."""
        if self.activation == "relu":
            return np.maximum(0, x)
        elif self.activation == "tanh":
            return np.tanh(x)
        elif self.activation == "leaky_relu":
            return np.where(x > 0, x, 0.01 * x)
        else:
            return x

    def encode(self, observation: np.ndarray) -> np.ndarray:
        """Encode observation(s) to features.

        Performs forward pass through the MLP, applying activations
        to all hidden layers but not the output layer.

        Args:
            observation: Input, shape (input_dim,) or (batch, input_dim).

        Returns:
            Features, shape (feature_dim,) or (batch, feature_dim).
        """
        x = np.atleast_2d(observation).astype(np.float64)
        single_input = observation.ndim == 1

        for i, (w, b) in enumerate(zip(self._weights, self._biases)):
            x = x @ w + b
            if i < len(self._weights) - 1:
                x = self._activation_fn(x)

        x = self._normalize(x)

        return x.flatten() if single_input else x

    def get_parameters(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Get network parameters.

        Returns:
            Tuple of (weights, biases) lists (copies).
        """
        return (
            [w.copy() for w in self._weights],
            [b.copy() for b in self._biases],
        )

    def set_parameters(
        self,
        weights: List[np.ndarray],
        biases: List[np.ndarray],
    ) -> None:
        """Set network parameters.

        Args:
            weights: Weight matrices matching layer dimensions.
            biases: Bias vectors matching layer dimensions.
        """
        self._weights = [np.asarray(w, dtype=np.float64) for w in weights]
        self._biases = [np.asarray(b, dtype=np.float64) for b in biases]


class CNNFeatureEncoder(FeatureEncoder):
    """Convolutional neural network feature encoder for image observations.

    ============================================================================
    CORE IDEA
    ============================================================================
    For visual observations, CNNs exploit spatial structure through:
    1. Local connectivity: Each neuron sees only a small receptive field
    2. Weight sharing: Same filter applied across spatial positions
    3. Hierarchical features: Low-level edges → high-level objects

    ============================================================================
    MATHEMATICAL THEORY
    ============================================================================
    Convolution operation (2D, single channel simplified):

        (f * g)(x, y) = Σᵢ Σⱼ f(i, j) · g(x-i, y-j)

    For image I and kernel K of size k×k:

        O(x, y) = Σᵢ₌₀ᵏ⁻¹ Σⱼ₌₀ᵏ⁻¹ K(i, j) · I(x+i, y+j) + b

    Output dimensions with stride s, padding p:
        H_out = (H_in + 2p - k) / s + 1

    ============================================================================
    ARCHITECTURE (Nature DQN-style)
    ============================================================================
    Input: 84×84×4 (4 stacked grayscale frames)
    Conv1: 32 filters, 8×8, stride 4 → 20×20×32
    Conv2: 64 filters, 4×4, stride 2 → 9×9×64
    Conv3: 64 filters, 3×3, stride 1 → 7×7×64
    Flatten: 3136
    FC: 512
    Output: feature_dim

    ============================================================================
    COMPLEXITY
    ============================================================================
    - Time: O(k² · C_in · C_out · H · W) per conv layer
    - Space: O(k² · C_in · C_out) parameters per conv layer
    """

    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (84, 84, 4),
        feature_dim: int = 512,
        normalize_output: bool = True,
    ) -> None:
        """Initialize CNN feature encoder.

        Args:
            input_shape: (height, width, channels) of input images.
            feature_dim: Output feature dimensionality.
            normalize_output: Whether to L2-normalize output.
        """
        input_dim = int(np.prod(input_shape))
        super().__init__(input_dim, feature_dim, normalize_output)
        self.input_shape = input_shape

        self._conv_weights: List[np.ndarray] = []
        self._conv_biases: List[np.ndarray] = []
        self._fc_weights: List[np.ndarray] = []
        self._fc_biases: List[np.ndarray] = []

        self._init_network()

    def _init_network(self) -> None:
        """Initialize convolutional and fully-connected layers."""
        h, w, c = self.input_shape

        conv_configs = [
            (c, 32, 8, 4),
            (32, 64, 4, 2),
            (64, 64, 3, 1),
        ]

        for in_c, out_c, k, s in conv_configs:
            std = np.sqrt(2.0 / (k * k * in_c))
            self._conv_weights.append(
                np.random.randn(out_c, in_c, k, k).astype(np.float64) * std
            )
            self._conv_biases.append(np.zeros(out_c, dtype=np.float64))

            h = (h - k) // s + 1
            w = (w - k) // s + 1

        flat_dim = h * w * 64
        fc_dims = [flat_dim, 512, self.feature_dim]

        for i in range(len(fc_dims) - 1):
            fan_in, fan_out = fc_dims[i], fc_dims[i + 1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            self._fc_weights.append(
                np.random.randn(fan_in, fan_out).astype(np.float64) * std
            )
            self._fc_biases.append(np.zeros(fan_out, dtype=np.float64))

    def _conv2d(
        self,
        x: np.ndarray,
        weight: np.ndarray,
        bias: np.ndarray,
        stride: int,
    ) -> np.ndarray:
        """2D convolution operation (simplified, for educational purposes).

        Args:
            x: Input, shape (batch, in_channels, height, width).
            weight: Filters, shape (out_channels, in_channels, k, k).
            bias: Bias, shape (out_channels,).
            stride: Stride for convolution.

        Returns:
            Output, shape (batch, out_channels, h_out, w_out).
        """
        batch, in_c, h, w = x.shape
        out_c, _, k, _ = weight.shape

        h_out = (h - k) // stride + 1
        w_out = (w - k) // stride + 1

        output = np.zeros((batch, out_c, h_out, w_out), dtype=np.float64)

        for b in range(batch):
            for oc in range(out_c):
                for i in range(h_out):
                    for j in range(w_out):
                        h_start = i * stride
                        w_start = j * stride
                        receptive_field = x[b, :, h_start : h_start + k, w_start : w_start + k]
                        output[b, oc, i, j] = np.sum(receptive_field * weight[oc]) + bias[oc]

        return output

    def encode(self, observation: np.ndarray) -> np.ndarray:
        """Encode image observation(s) to features.

        Args:
            observation: Image(s), shape (H, W, C) or (batch, H, W, C).

        Returns:
            Features, shape (feature_dim,) or (batch, feature_dim).
        """
        x = np.atleast_2d(observation.reshape(-1, *self.input_shape))
        single_input = observation.ndim == 3
        batch = len(x)

        x = x.transpose(0, 3, 1, 2)

        strides = [4, 2, 1]
        for w, b, s in zip(self._conv_weights, self._conv_biases, strides):
            x = self._conv2d(x, w, b, s)
            x = np.maximum(0, x)

        x = x.reshape(batch, -1)

        for i, (w, b) in enumerate(zip(self._fc_weights, self._fc_biases)):
            x = x @ w + b
            if i < len(self._fc_weights) - 1:
                x = np.maximum(0, x)

        x = self._normalize(x)

        return x.flatten() if single_input else x

    def get_parameters(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Get all network parameters."""
        weights = [w.copy() for w in self._conv_weights] + [
            w.copy() for w in self._fc_weights
        ]
        biases = [b.copy() for b in self._conv_biases] + [
            b.copy() for b in self._fc_biases
        ]
        return weights, biases

    def set_parameters(
        self,
        weights: List[np.ndarray],
        biases: List[np.ndarray],
    ) -> None:
        """Set all network parameters."""
        n_conv = len(self._conv_weights)
        self._conv_weights = [np.asarray(w, dtype=np.float64) for w in weights[:n_conv]]
        self._conv_biases = [np.asarray(b, dtype=np.float64) for b in biases[:n_conv]]
        self._fc_weights = [np.asarray(w, dtype=np.float64) for w in weights[n_conv:]]
        self._fc_biases = [np.asarray(b, dtype=np.float64) for b in biases[n_conv:]]


class RandomFeatureEncoder(FeatureEncoder):
    """Fixed random network for RND-style exploration.

    ============================================================================
    CORE IDEA
    ============================================================================
    In Random Network Distillation, the target network is a fixed random
    network. Its outputs define an arbitrary but consistent feature space.
    Novel states produce unpredictable outputs that a predictor network
    struggles to match, generating intrinsic reward.

    ============================================================================
    MATHEMATICAL THEORY
    ============================================================================
    The random network f_rand: S → ℝᵈ is initialized once and never trained:

        f_rand(s) = σ(Wₙ ... σ(W₁ s + b₁) ... + bₙ)

    Properties:
    - Deterministic: Same input always produces same output
    - High-dimensional: Captures many aspects of the input
    - Random projection: No bias toward any particular features

    For RND, a separate predictor f̂ is trained to match f_rand on visited
    states. Intrinsic reward = ||f_rand(s) - f̂(s)||².

    ============================================================================
    WHY IT WORKS
    ============================================================================
    Random projections preserve distance structure (Johnson-Lindenstrauss).
    The predictor learns the random features for frequently-visited states.
    Novel states have high prediction error because similar states weren't
    seen during training.

    ============================================================================
    COMPLEXITY
    ============================================================================
    - Time: O(forward_pass), same as MLP
    - Space: O(parameters), but never updated
    """

    def __init__(
        self,
        input_dim: int,
        feature_dim: int = 64,
        hidden_dims: Tuple[int, ...] = (128, 64),
        seed: Optional[int] = None,
        normalize_output: bool = True,
    ) -> None:
        """Initialize fixed random feature encoder.

        Args:
            input_dim: Input dimensionality.
            feature_dim: Output feature dimensionality.
            hidden_dims: Hidden layer sizes.
            seed: Random seed for reproducibility. Important for RND
                where target and predictor must use same random init.
            normalize_output: Whether to L2-normalize output.
        """
        super().__init__(input_dim, feature_dim, normalize_output)
        self.hidden_dims = hidden_dims
        self.seed = seed

        self._weights: List[np.ndarray] = []
        self._biases: List[np.ndarray] = []
        self._init_network()

    def _init_network(self) -> None:
        """Initialize fixed random weights (never updated)."""
        if self.seed is not None:
            np.random.seed(self.seed)

        dims = [self.input_dim] + list(self.hidden_dims) + [self.feature_dim]

        for i in range(len(dims) - 1):
            fan_in, fan_out = dims[i], dims[i + 1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            self._weights.append(
                np.random.randn(fan_in, fan_out).astype(np.float64) * std
            )
            self._biases.append(np.zeros(fan_out, dtype=np.float64))

    def encode(self, observation: np.ndarray) -> np.ndarray:
        """Encode observation through fixed random network.

        Args:
            observation: Input, shape (input_dim,) or (batch, input_dim).

        Returns:
            Random features, shape (feature_dim,) or (batch, feature_dim).
        """
        x = np.atleast_2d(observation).astype(np.float64)
        single_input = observation.ndim == 1

        for i, (w, b) in enumerate(zip(self._weights, self._biases)):
            x = x @ w + b
            if i < len(self._weights) - 1:
                x = np.maximum(0, x)

        x = self._normalize(x)

        return x.flatten() if single_input else x

    def get_parameters(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Get network parameters (read-only for RND target)."""
        return (
            [w.copy() for w in self._weights],
            [b.copy() for b in self._biases],
        )

    def set_parameters(
        self,
        weights: List[np.ndarray],
        biases: List[np.ndarray],
    ) -> None:
        """Set parameters (typically not called for random target)."""
        self._weights = [np.asarray(w, dtype=np.float64) for w in weights]
        self._biases = [np.asarray(b, dtype=np.float64) for b in biases]


if __name__ == "__main__":
    print("=" * 70)
    print("Feature Encoder Networks - Validation Tests")
    print("=" * 70)

    np.random.seed(42)

    # Test MLP encoder
    print("\n[Test 1] MLP Feature Encoder")
    encoder = MLPFeatureEncoder(input_dim=10, feature_dim=32)

    obs = np.random.randn(10)
    features = encoder.encode(obs)
    print(f"  Single input: {obs.shape} → {features.shape}")
    print(f"  Feature norm: {np.linalg.norm(features):.4f}")
    assert features.shape == (32,), "Shape mismatch"
    assert np.isclose(np.linalg.norm(features), 1.0, atol=0.01), "Not normalized"

    batch_obs = np.random.randn(16, 10)
    batch_features = encoder.encode(batch_obs)
    print(f"  Batch input: {batch_obs.shape} → {batch_features.shape}")
    assert batch_features.shape == (16, 32), "Batch shape mismatch"
    print("  [PASS]")

    # Test CNN encoder
    print("\n[Test 2] CNN Feature Encoder")
    cnn = CNNFeatureEncoder(input_shape=(84, 84, 4), feature_dim=512)

    img = np.random.randn(84, 84, 4)
    features = cnn.encode(img)
    print(f"  Single image: (84,84,4) → {features.shape}")
    assert features.shape == (512,), "CNN output shape mismatch"
    print("  [PASS]")

    # Test random encoder
    print("\n[Test 3] Random Feature Encoder")
    rand_enc = RandomFeatureEncoder(input_dim=10, feature_dim=32, seed=123)

    obs1 = np.random.randn(10)
    feat1 = rand_enc.encode(obs1)
    feat2 = rand_enc.encode(obs1)
    print(f"  Same input produces same output: {np.allclose(feat1, feat2)}")
    assert np.allclose(feat1, feat2), "Random encoder not deterministic"
    print("  [PASS]")

    print("\n" + "=" * 70)
    print("All tests passed successfully!")
    print("=" * 70)
