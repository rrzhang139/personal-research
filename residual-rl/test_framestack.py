"""Test how FrameStack handles observations"""
import gymnasium as gym
import numpy as np
import torch

# Simulate what happens in the code
print("="*60)
print("Simulating FrameStack Behavior")
print("="*60)

# Simulate CPUNumpyWrapper output
obs_dim = 43  # ManiSkill3 PegInsertionSide-v1 state dim
obs_horizon = 2

# Simulate a single observation from CPUNumpyWrapper (should be 1D)
single_obs = np.random.randn(obs_dim).astype(np.float32)
print(f"\n1. Single observation from CPUNumpyWrapper:")
print(f"   Shape: {single_obs.shape}")
print(f"   Type: {type(single_obs)}")

# FrameStack creates LazyFrames
# Let's simulate what FrameStack returns
from collections import deque
from gymnasium.wrappers.frame_stack import LazyFrames

# Simulate FrameStack with 2 frames
frames = deque([single_obs, single_obs], maxlen=obs_horizon)
lazy_frames = LazyFrames(list(frames), lz4_compress=False)

print(f"\n2. LazyFrames from FrameStack:")
print(f"   Type: {type(lazy_frames)}")
print(f"   Length: {len(lazy_frames)}")

# Convert to numpy - this is what happens internally
frames_array = np.array(lazy_frames)
print(f"\n3. LazyFrames converted to numpy:")
print(f"   Shape: {frames_array.shape}")
print(f"   Expected: ({obs_horizon}, {obs_dim})")

# Now let's see what happens when we convert to torch.Tensor
# This is what the evaluation code does: torch.Tensor(obs)
frames_tensor = torch.Tensor(frames_array)
print(f"\n4. Converted to torch.Tensor:")
print(f"   Shape: {frames_tensor.shape}")

# For vectorized environments (num_envs=10)
print(f"\n5. With VectorEnv (num_envs=10):")
vectorized_obs = np.array([frames_array for _ in range(10)])
print(f"   VectorEnv obs shape: {vectorized_obs.shape}")
print(f"   Expected: (10, {obs_horizon}, {obs_dim})")

vec_tensor = torch.Tensor(vectorized_obs)
print(f"   As torch.Tensor: {vec_tensor.shape}")

# Now let's check what the Agent expects
print(f"\n6. Agent expectations:")
print(f"   single_observation_space.shape should be: ({obs_horizon}, {obs_dim})")
print(f"   global_cond_dim = obs_horizon * obs_dim = {obs_horizon * obs_dim}")

# During inference
print(f"\n7. During evaluation:")
print(f"   obs from env: {vectorized_obs.shape}")
print(f"   torch.Tensor(obs): {vec_tensor.shape}")
print(f"   After flatten(start_dim=1): {vec_tensor.flatten(start_dim=1).shape}")
print(f"   Expected global_cond: (10, {obs_horizon * obs_dim})")

print("\n" + "="*60)
print("Checking if shapes match...")
print("="*60)

if vec_tensor.shape == (10, obs_horizon, obs_dim):
    print("✓ Shapes look correct!")
else:
    print(f"✗ Shape mismatch!")
    print(f"  Got: {vec_tensor.shape}")
    print(f"  Expected: (10, {obs_horizon}, {obs_dim})")

# Now let's test the ACTUAL issue - what if FrameStack is NOT applied correctly?
print("\n" + "="*60)
print("Testing potential bugs...")
print("="*60)

# Bug scenario 1: If CPUNumpyWrapper doesn't squeeze
print("\n1. If CPUNumpyWrapper fails to squeeze (1, obs_dim) -> (obs_dim):")
bad_obs = np.random.randn(1, obs_dim).astype(np.float32)
print(f"   Observation shape: {bad_obs.shape}")
bad_frames = deque([bad_obs, bad_obs], maxlen=obs_horizon)
bad_lazy = LazyFrames(list(bad_frames), lz4_compress=False)
bad_array = np.array(bad_lazy)
print(f"   After FrameStack: {bad_array.shape}")
print(f"   Expected: ({obs_horizon}, {obs_dim})")
print(f"   Actual: {bad_array.shape}")
if bad_array.shape != (obs_horizon, obs_dim):
    print(f"   ✗ BUG! Shape is {bad_array.shape} instead of ({obs_horizon}, {obs_dim})")
    print(f"   This would cause: global_cond_dim mismatch")
else:
    print(f"   ✓ No issue")

# Bug scenario 2: Check dimensions after vectorization
print(f"\n2. After vectorization with bad observations:")
bad_vec_obs = np.array([bad_array for _ in range(10)])
print(f"   VectorEnv obs shape: {bad_vec_obs.shape}")
print(f"   Expected: (10, {obs_horizon}, {obs_dim})")
if bad_vec_obs.shape == (10, obs_horizon, 1, obs_dim):
    print(f"   ✗ BUG! Extra dimension in the middle!")
    print(f"   When flattened: {bad_vec_obs.reshape(10, -1).shape}")
    print(f"   Expected global_cond_dim: {obs_horizon * obs_dim}")
    print(f"   Actual global_cond_dim: {obs_horizon * 1 * obs_dim}")
