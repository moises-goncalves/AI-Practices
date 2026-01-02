"""
Neural Network Architectures for Reward Optimization

================================================================================
OVERVIEW
================================================================================
This module provides production-grade neural network implementations for
reward learning and intrinsic motivation. All networks use pure NumPy for
educational clarity while maintaining research-quality implementations.

Module Contents:
    - feature_encoders: State representation learning networks
    - dynamics_models: Forward/inverse dynamics predictors
    - discriminators: GAN-style reward discriminators
    - value_networks: Potential/value function approximators

================================================================================
DESIGN PRINCIPLES
================================================================================
1. Educational Transparency: Pure NumPy implementations with detailed comments
2. Modularity: Each network component is independently usable
3. Efficiency: Vectorized operations for batch processing
4. Extensibility: Easy to subclass and customize

================================================================================
REFERENCES
================================================================================
[1] Pathak et al. (2017). Curiosity-driven exploration (ICM architecture)
[2] Burda et al. (2019). Random Network Distillation
[3] Ho & Ermon (2016). GAIL discriminator design
"""

from .feature_encoders import (
    FeatureEncoder,
    CNNFeatureEncoder,
    MLPFeatureEncoder,
    RandomFeatureEncoder,
)

from .dynamics_models import (
    ForwardDynamicsModel,
    InverseDynamicsModel,
    EnsembleDynamicsModel,
)

from .discriminators import (
    GAILDiscriminator,
    AIRLDiscriminator,
    StateActionDiscriminator,
)

from .value_networks import (
    ValueNetwork,
    DuelingValueNetwork,
    PotentialNetwork,
)

__all__ = [
    # Feature Encoders
    "FeatureEncoder",
    "CNNFeatureEncoder",
    "MLPFeatureEncoder",
    "RandomFeatureEncoder",
    # Dynamics Models
    "ForwardDynamicsModel",
    "InverseDynamicsModel",
    "EnsembleDynamicsModel",
    # Discriminators
    "GAILDiscriminator",
    "AIRLDiscriminator",
    "StateActionDiscriminator",
    # Value Networks
    "ValueNetwork",
    "DuelingValueNetwork",
    "PotentialNetwork",
]
