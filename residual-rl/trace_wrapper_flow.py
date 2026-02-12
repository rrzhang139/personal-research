"""Trace exact flow of observations through wrappers"""
import gymnasium as gym
import numpy as np

class FakeMS3Env:
    """Simulate ManiSkill3 environment"""
    def __init__(self):
        from gymnasium import spaces
        # MS3 defines observation_space as 1D
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(43,), dtype=np.float32)
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(7,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        # But returns 2D observations
        obs = np.random.randn(1, 43).astype(np.float32)
        print(f"   [FakeMS3Env.reset] Returning obs shape: {obs.shape}")
        return obs, {}

    def step(self, action):
        obs = np.random.randn(1, 43).astype(np.float32)
        print(f"   [FakeMS3Env.step] Returning obs shape: {obs.shape}")
        return obs, 0.0, False, False, {}

class CPUNumpyWrapper(gym.ObservationWrapper):
    """From offline script"""
    def __init__(self, env):
        super().__init__(env)
        obs_space = env.observation_space
        print(f"   [CPUNumpyWrapper.__init__] obs_space.shape = {obs_space.shape}")
        if hasattr(obs_space, 'shape') and len(obs_space.shape) == 2 and obs_space.shape[0] == 1:
            print(f"   [CPUNumpyWrapper.__init__] SQUEEZING observation_space")
            self.observation_space = gym.spaces.Box(
                low=obs_space.low.reshape(-1), high=obs_space.high.reshape(-1), dtype=obs_space.dtype)
        else:
            print(f"   [CPUNumpyWrapper.__init__] NOT squeezing (condition failed)")
        print(f"   [CPUNumpyWrapper.__init__] Final obs_space.shape = {self.observation_space.shape}")

    def observation(self, obs):
        print(f"   [CPUNumpyWrapper.observation] Input: {obs.shape}")
        if obs.ndim == 2 and obs.shape[0] == 1:
            obs = obs.squeeze(0)
            print(f"   [CPUNumpyWrapper.observation] Squeezed to: {obs.shape}")
        result = np.asarray(obs, dtype=np.float32)
        print(f"   [CPUNumpyWrapper.observation] Returning: {result.shape}")
        return result

print("="*70)
print("Step 1: Create base env")
print("="*70)
env = FakeMS3Env()
print(f"observation_space.shape: {env.observation_space.shape}\n")

print("="*70)
print("Step 2: Wrap with CPUNumpyWrapper")
print("="*70)
env = CPUNumpyWrapper(env)
print(f"observation_space.shape: {env.observation_space.shape}\n")

print("="*70)
print("Step 3: Wrap with FrameStack(2)")
print("="*70)
env = gym.wrappers.FrameStack(env, 2)
print(f"observation_space.shape: {env.observation_space.shape}\n")

print("="*70)
print("Step 4: Reset and check observations")
print("="*70)
obs, info = env.reset()
print(f"\n→ Final obs type: {type(obs)}")
if hasattr(obs, 'shape'):
    print(f"→ Final obs shape: {obs.shape}")
else:
    # LazyFrames
    obs_array = np.array(obs)
    print(f"→ Final obs as array shape: {obs_array.shape}")

print("\n" + "="*70)
print("DIAGNOSIS")
print("="*70)
print(f"Expected shape: (2, 43) = (obs_horizon, obs_dim)")
if hasattr(obs, 'shape'):
    actual_shape = obs.shape
else:
    actual_shape = np.array(obs).shape
print(f"Actual shape: {actual_shape}")

if actual_shape == (2, 43):
    print("✓ CORRECT!")
elif actual_shape == (2, 1, 43):
    print("✗ BUG: Extra dimension from unsqueezed observations!")
elif actual_shape == tuple([2] + list(env.unwrapped.observation_space.shape)):
    print(f"✗ BUG: FrameStack used wrong observation_space!")
else:
    print(f"✗ UNEXPECTED shape!")
