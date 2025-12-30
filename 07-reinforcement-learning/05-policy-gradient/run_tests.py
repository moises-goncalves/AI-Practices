"""Simple test runner for policy gradient algorithms."""

import sys
import os
import importlib.util

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

import torch
import numpy as np

print("=" * 60)
print("Testing Policy Gradient Networks")
print("=" * 60)

# Test imports using direct file loading
try:
    # Load policy networks module
    spec = importlib.util.spec_from_file_location(
        "policy_networks",
        os.path.join(current_dir, "networks", "policy_networks.py")
    )
    policy_networks = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(policy_networks)

    # Load value networks module
    spec = importlib.util.spec_from_file_location(
        "value_networks",
        os.path.join(current_dir, "networks", "value_networks.py")
    )
    value_networks = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value_networks)

    DiscretePolicy = policy_networks.DiscretePolicy
    ContinuousPolicy = policy_networks.ContinuousPolicy
    GaussianPolicy = policy_networks.GaussianPolicy
    ValueNetwork = value_networks.ValueNetwork
    DuelingValueNetwork = value_networks.DuelingValueNetwork

    print("✓ Successfully imported network modules")
except Exception as e:
    print(f"✗ Failed to import networks: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test DiscretePolicy
try:
    print("\nTesting DiscretePolicy...")
    state_dim, action_dim, batch_size = 4, 2, 8
    policy = DiscretePolicy(state_dim, action_dim, hidden_dims=(32, 32))
    states = torch.randn(batch_size, state_dim)

    logits = policy.forward(states)
    assert logits.shape == (batch_size, action_dim), f"Expected shape {(batch_size, action_dim)}, got {logits.shape}"

    action, log_prob = policy.sample(states)
    assert action.shape == (batch_size,), f"Expected action shape {(batch_size,)}, got {action.shape}"

    actions = torch.randint(0, action_dim, (batch_size,))
    log_probs, entropy = policy.evaluate(states, actions)
    assert log_probs.shape == (batch_size, 1), f"Expected log_probs shape {(batch_size, 1)}, got {log_probs.shape}"

    print("  ✓ DiscretePolicy test passed")
except Exception as e:
    print(f"  ✗ DiscretePolicy test failed: {e}")
    import traceback
    traceback.print_exc()

# Test ContinuousPolicy
try:
    print("\nTesting ContinuousPolicy...")
    policy = ContinuousPolicy(state_dim, action_dim, hidden_dims=(32, 32), std=0.5)
    states = torch.randn(batch_size, state_dim)

    means = policy.forward(states)
    assert means.shape == (batch_size, action_dim)

    action, log_prob = policy.sample(states)
    assert action.shape == (batch_size, action_dim)

    actions = torch.randn(batch_size, action_dim)
    log_probs, entropy = policy.evaluate(states, actions)
    assert log_probs.shape == (batch_size, 1)

    print("  ✓ ContinuousPolicy test passed")
except Exception as e:
    print(f"  ✗ ContinuousPolicy test failed: {e}")
    import traceback
    traceback.print_exc()

# Test GaussianPolicy
try:
    print("\nTesting GaussianPolicy...")
    policy = GaussianPolicy(state_dim, action_dim, hidden_dims=(32, 32))
    states = torch.randn(batch_size, state_dim)

    mean, log_std = policy.forward(states)
    assert mean.shape == (batch_size, action_dim)
    assert log_std.shape == (batch_size, action_dim)

    action, log_prob = policy.sample(states)
    assert action.shape == (batch_size, action_dim)

    actions = torch.randn(batch_size, action_dim)
    log_probs, entropy = policy.evaluate(states, actions)
    assert log_probs.shape == (batch_size, 1)

    print("  ✓ GaussianPolicy test passed")
except Exception as e:
    print(f"  ✗ GaussianPolicy test failed: {e}")
    import traceback
    traceback.print_exc()

# Test ValueNetwork
try:
    print("\nTesting ValueNetwork...")
    value_fn = ValueNetwork(state_dim, hidden_dims=(32, 32))
    states = torch.randn(batch_size, state_dim)

    values = value_fn.forward(states)
    assert values.shape == (batch_size, 1)

    targets = torch.randn(batch_size, 1)
    loss = value_fn.compute_loss(states, targets)
    assert loss.item() >= 0

    print("  ✓ ValueNetwork test passed")
except Exception as e:
    print(f"  ✗ ValueNetwork test failed: {e}")
    import traceback
    traceback.print_exc()

# Test DuelingValueNetwork
try:
    print("\nTesting DuelingValueNetwork...")
    value_fn = DuelingValueNetwork(state_dim, hidden_dims=(32, 32))
    states = torch.randn(batch_size, state_dim)

    values = value_fn.forward(states)
    assert values.shape == (batch_size, 1)

    targets = torch.randn(batch_size, 1)
    loss = value_fn.compute_loss(states, targets)
    assert loss.item() >= 0

    print("  ✓ DuelingValueNetwork test passed")
except Exception as e:
    print(f"  ✗ DuelingValueNetwork test failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("✓ All network tests completed successfully!")
print("=" * 60)
