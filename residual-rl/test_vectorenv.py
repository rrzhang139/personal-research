"""Test VectorEnv with wrappers"""
import gymnasium as gym
import numpy as np

class FakeMS3Env:
    """Simulate ManiSkill3 environment"""
    metadata = {}

    def __init__(self, seed=None):
        from gymnasium import spaces
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(43,), dtype=np.float32)
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(7,), dtype=np.float32)
        self._seed = seed

    def reset(self, seed=None, options=None):
        obs = np.random.randn(1, 43).astype(np.float32)
        return obs, {}

    def step(self, action):
        obs = np.random.randn(1, 43).astype(np.float32)
        rew = np.float32(0.0)
        return obs, rew, False, False, {'success': False}

class CPUNumpyWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_space = env.observation_space
        if hasattr(obs_space, 'shape') and len(obs_space.shape) == 2 and obs_space.shape[0] == 1:
            self.observation_space = gym.spaces.Box(
                low=obs_space.low.reshape(-1), high=obs_space.high.reshape(-1), dtype=obs_space.dtype)

    def observation(self, obs):
        if obs.ndim == 2 and obs.shape[0] == 1:
            obs = obs.squeeze(0)
        return np.asarray(obs, dtype=np.float32)

    def step(self, action):
        obs, rew, terminated, truncated, info = self.env.step(action)
        return self.observation(obs), float(rew), bool(terminated), bool(truncated), info

def make_env(seed):
    def thunk():
        env = FakeMS3Env(seed=seed)
        env = CPUNumpyWrapper(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.FrameStack(env, 2)  # obs_horizon=2
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env
    return thunk

print("="*70)
print("Testing VectorEnv with 10 environments")
print("="*70)

# Create vectorized environments
num_envs = 10
envs = gym.vector.SyncVectorEnv([make_env(i) for i in range(num_envs)])

print(f"\n1. VectorEnv created")
print(f"   envs.single_observation_space: {envs.single_observation_space}")
print(f"   envs.single_observation_space.shape: {envs.single_observation_space.shape}")
print(f"   envs.observation_space: {envs.observation_space}")

# Reset
obs, info = envs.reset(seed=42)
print(f"\n2. After reset:")
print(f"   obs type: {type(obs)}")
print(f"   obs dtype: {obs.dtype if hasattr(obs, 'dtype') else 'N/A'}")
if hasattr(obs, 'shape'):
    print(f"   obs shape: {obs.shape}")
else:
    print(f"   obs length: {len(obs)}")
    print(f"   obs[0] type: {type(obs[0])}")
    if hasattr(obs[0], 'shape'):
        print(f"   obs[0] shape: {obs[0].shape}")

# Take a step
actions = envs.action_space.sample()
obs, rew, terminated, truncated, info = envs.step(actions)
print(f"\n3. After step:")
print(f"   obs shape: {obs.shape if hasattr(obs, 'shape') else len(obs)}")
print(f"   rew shape: {rew.shape if hasattr(rew, 'shape') else len(rew)}")

print("\n" + "="*70)
print("Expected vs Actual")
print("="*70)
print(f"Expected obs shape: (10, 2, 43) = (num_envs, obs_horizon, obs_dim)")
if hasattr(obs, 'shape'):
    print(f"Actual obs shape: {obs.shape}")
    if obs.shape == (10, 2, 43):
        print("✓ CORRECT shape!")
    else:
        print(f"✗ WRONG shape! Expected (10, 2, 43), got {obs.shape}")
else:
    # LazyFrames in array
    print(f"Obs is array of LazyFrames, length: {len(obs)}")
    obs_array = np.array([np.array(o) for o in obs])
    print(f"As stacked array: {obs_array.shape}")
    if obs_array.shape == (10, 2, 43):
        print("✓ CORRECT shape!")
    else:
        print(f"✗ WRONG shape! Expected (10, 2, 43), got {obs_array.shape}")

# Test conversion to torch
import torch
print("\n" + "="*70)
print("Testing torch.Tensor conversion (as done in evaluate())")
print("="*70)
if hasattr(obs, 'shape'):
    obs_tensor = torch.Tensor(obs)
else:
    obs_tensor = torch.Tensor(np.array([np.array(o) for o in obs]))
print(f"torch.Tensor(obs) shape: {obs_tensor.shape}")
print(f"Expected: torch.Size([10, 2, 43])")
if obs_tensor.shape == torch.Size([10, 2, 43]):
    print("✓ Tensor conversion correct!")

    # Test flattening as done in the model
    flattened = obs_tensor.flatten(start_dim=1)
    print(f"\nAfter flatten(start_dim=1): {flattened.shape}")
    print(f"Expected: torch.Size([10, 86])")  # 2 * 43 = 86
    if flattened.shape == torch.Size([10, 86]):
        print("✓ Flattening correct!")
    else:
        print(f"✗ Flattening wrong! Expected torch.Size([10, 86]), got {flattened.shape}")
else:
    print(f"✗ Tensor conversion wrong! Expected torch.Size([10, 2, 43]), got {obs_tensor.shape}")

envs.close()
