"""Test SeqActionWrapper with VectorEnv"""
import gymnasium as gym
import numpy as np
import torch

class FakeMS3Env:
    metadata = {}
    def __init__(self, seed=None):
        from gymnasium import spaces
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(43,), dtype=np.float32)
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(7,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        return np.random.randn(1, 43).astype(np.float32), {}

    def step(self, action):
        print(f"      [FakeMS3Env.step] action shape: {action.shape if hasattr(action, 'shape') else 'N/A'}")
        obs = np.random.randn(1, 43).astype(np.float32)
        return obs, np.float32(0.0), False, False, {'success': False}

    def close(self):
        pass

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

class SeqActionWrapper(gym.Wrapper):
    def step(self, action_seq):
        print(f"    [SeqActionWrapper.step] action_seq shape: {action_seq.shape if hasattr(action_seq, 'shape') else 'N/A'}")
        rew_sum = 0
        for i, action in enumerate(action_seq):
            print(f"      [SeqActionWrapper.step] Executing action {i}/{len(action_seq)}")
            obs, rew, terminated, truncated, info = self.env.step(action)
            rew_sum += rew
            if terminated or truncated:
                break
        return obs, rew_sum, terminated, truncated, info

def make_env(seed):
    def thunk():
        env = FakeMS3Env(seed=seed)
        env = CPUNumpyWrapper(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.FrameStack(env, 2)
        env = SeqActionWrapper(env)  # AFTER FrameStack!
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env
    return thunk

print("="*70)
print("Testing SeqActionWrapper with VectorEnv")
print("="*70)

num_envs = 2  # Small number for visibility
envs = gym.vector.SyncVectorEnv([make_env(i) for i in range(num_envs)])

print(f"\n1. VectorEnv created")
print(f"   single_action_space: {envs.single_action_space}")
print(f"   single_action_space.shape: {envs.single_action_space.shape}")

# Reset
obs, info = envs.reset(seed=42)
print(f"\n2. After reset, obs shape: {obs.shape}")

# Simulate what the agent does
obs_horizon = 2
act_horizon = 4
act_dim = 7

# The agent returns actions of shape (num_envs, act_horizon, act_dim)
print(f"\n3. Agent produces actions:")
print(f"   Shape: ({num_envs}, {act_horizon}, {act_dim})")
actions = np.random.randn(num_envs, act_horizon, act_dim).astype(np.float32)
print(f"   actions.shape: {actions.shape}")

# Try to step - this is where the bug might be
print(f"\n4. Calling envs.step(actions)...")
try:
    obs, rew, terminated, truncated, info = envs.step(actions)
    print(f"   ✓ Step succeeded!")
    print(f"   obs shape: {obs.shape}")
    print(f"   rew shape: {rew.shape}")
except Exception as e:
    print(f"   ✗ Step failed with error: {e}")
    print(f"\n   THIS IS THE BUG!")
    print(f"   VectorEnv.step() expects actions of shape (num_envs, action_dim)")
    print(f"   But we're passing shape (num_envs, act_horizon, action_dim)")
    print(f"\n   SeqActionWrapper expects shape (act_horizon, action_dim)")
    print(f"   But VectorEnv passes it shape (action_dim,) for each env")

envs.close()

print("\n" + "="*70)
print("DIAGNOSIS")
print("="*70)
print("The issue is that SeqActionWrapper is designed for single environments")
print("but the code is using it with VectorEnv, which expects different shapes!")
print()
print("Single env: action_seq shape is (act_horizon, act_dim)")
print("Vector env: actions shape is (num_envs, ???)")
print()
print("With SeqActionWrapper in each env:")
print("  - Each env's SeqActionWrapper expects (act_horizon, act_dim)")
print("  - But VectorEnv distributes a (num_envs, X) array")
print("  - So each env gets actions[i] which should be (act_horizon, act_dim)")
print("  - But VectorEnv.step() expects just (act_dim,)!")
