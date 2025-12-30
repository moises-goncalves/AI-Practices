"""Unit tests for policy and value networks."""

import torch
import numpy as np

from ..networks.policy_networks import DiscretePolicy, ContinuousPolicy, GaussianPolicy
from ..networks.value_networks import ValueNetwork, DuelingValueNetwork


def test_discrete_policy():
    """Test discrete policy network."""
    print("Testing DiscretePolicy network...")

    state_dim = 4
    action_dim = 2
    batch_size = 8

    policy = DiscretePolicy(state_dim, action_dim, hidden_dims=(32, 32))

    # Test forward pass
    states = torch.randn(batch_size, state_dim)
    logits = policy.forward(states)
    assert logits.shape == (batch_size, action_dim)

    # Test sampling
    action, log_prob = policy.sample(states)
    assert action.shape == (batch_size,)
    assert log_prob.shape == (batch_size,)

    # Test evaluation
    actions = torch.randint(0, action_dim, (batch_size,))
    log_probs, entropy = policy.evaluate(states, actions)
    assert log_probs.shape == (batch_size, 1)
    assert entropy.shape == (batch_size,)

    print("  ✓ DiscretePolicy test passed")


def test_continuous_policy():
    """Test continuous policy network."""
    print("Testing ContinuousPolicy network...")

    state_dim = 4
    action_dim = 2
    batch_size = 8

    policy = ContinuousPolicy(state_dim, action_dim, hidden_dims=(32, 32), std=0.5)

    # Test forward pass
    states = torch.randn(batch_size, state_dim)
    means = policy.forward(states)
    assert means.shape == (batch_size, action_dim)

    # Test sampling
    action, log_prob = policy.sample(states)
    assert action.shape == (batch_size, action_dim)
    assert log_prob.shape == (batch_size,)

    # Test evaluation
    actions = torch.randn(batch_size, action_dim)
    log_probs, entropy = policy.evaluate(states, actions)
    assert log_probs.shape == (batch_size, 1)
    assert entropy.shape == (batch_size,)

    print("  ✓ ContinuousPolicy test passed")


def test_gaussian_policy():
    """Test Gaussian policy network."""
    print("Testing GaussianPolicy network...")

    state_dim = 4
    action_dim = 2
    batch_size = 8

    policy = GaussianPolicy(state_dim, action_dim, hidden_dims=(32, 32))

    # Test forward pass
    states = torch.randn(batch_size, state_dim)
    mean, log_std = policy.forward(states)
    assert mean.shape == (batch_size, action_dim)
    assert log_std.shape == (batch_size, action_dim)

    # Test sampling
    action, log_prob = policy.sample(states)
    assert action.shape == (batch_size, action_dim)
    assert log_prob.shape == (batch_size,)

    # Test evaluation
    actions = torch.randn(batch_size, action_dim)
    log_probs, entropy = policy.evaluate(states, actions)
    assert log_probs.shape == (batch_size, 1)
    assert entropy.shape == (batch_size,)

    print("  ✓ GaussianPolicy test passed")


def test_value_network():
    """Test value function network."""
    print("Testing ValueNetwork...")

    state_dim = 4
    batch_size = 8

    value_fn = ValueNetwork(state_dim, hidden_dims=(32, 32))

    # Test forward pass
    states = torch.randn(batch_size, state_dim)
    values = value_fn.forward(states)
    assert values.shape == (batch_size, 1)

    # Test loss computation
    targets = torch.randn(batch_size, 1)
    loss = value_fn.compute_loss(states, targets)
    assert loss.item() >= 0

    print("  ✓ ValueNetwork test passed")


def test_dueling_value_network():
    """Test dueling value function network."""
    print("Testing DuelingValueNetwork...")

    state_dim = 4
    batch_size = 8

    value_fn = DuelingValueNetwork(state_dim, hidden_dims=(32, 32))

    # Test forward pass
    states = torch.randn(batch_size, state_dim)
    values = value_fn.forward(states)
    assert values.shape == (batch_size, 1)

    # Test loss computation
    targets = torch.randn(batch_size, 1)
    loss = value_fn.compute_loss(states, targets)
    assert loss.item() >= 0

    print("  ✓ DuelingValueNetwork test passed")


def test_unbatched_input():
    """Test networks with unbatched input."""
    print("Testing unbatched input handling...")

    state_dim = 4
    action_dim = 2

    policy = DiscretePolicy(state_dim, action_dim)
    value_fn = ValueNetwork(state_dim)

    # Test with single state (unbatched)
    state = torch.randn(state_dim)
    action, log_prob = policy.sample(state)
    assert action.dim() == 0  # Scalar
    assert log_prob.dim() == 0  # Scalar

    value = value_fn.forward(state.unsqueeze(0))
    assert value.shape == (1, 1)

    print("  ✓ Unbatched input test passed")


if __name__ == "__main__":
    print("Running network tests...\n")
    test_discrete_policy()
    test_continuous_policy()
    test_gaussian_policy()
    test_value_network()
    test_dueling_value_network()
    test_unbatched_input()
    print("\n✓ All network tests passed!")
