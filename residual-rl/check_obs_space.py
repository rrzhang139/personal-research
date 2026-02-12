"""Check what observation space ManiSkill3 actually defines vs what it returns"""
import numpy as np

# Simulate what ManiSkill3 might do
class FakeMS3Env:
    """Simulate ManiSkill3 environment"""
    def __init__(self):
        # MS3 might define observation_space as 1D even though observations are 2D
        from gymnasium import spaces
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(43,),  # Defined as 1D!
            dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1,
            high=1,
            shape=(7,),
            dtype=np.float32
        )

    def reset(self, seed=None):
        # But actual observations returned are 2D torch tensors (1, 43)
        import torch
        obs = torch.randn(1, 43)  # Returns (1, obs_dim)!
        return obs, {}

    def step(self, action):
        import torch
        obs = torch.randn(1, 43)  # Returns (1, obs_dim)!
        return obs, 0.0, False, False, {}

# Now test the CPUNumpyWrapper logic
print("="*60)
print("Testing observation space detection issue")
print("="*60)

env = FakeMS3Env()
print(f"\n1. ManiSkill3 env.observation_space.shape: {env.observation_space.shape}")
obs, info = env.reset()
print(f"2. Actual observation from reset(): {obs.shape}")

print(f"\n3. CPUNumpyWrapper __init__ check:")
obs_space = env.observation_space
print(f"   hasattr(obs_space, 'shape'): {hasattr(obs_space, 'shape')}")
print(f"   len(obs_space.shape): {len(obs_space.shape)}")
print(f"   obs_space.shape: {obs_space.shape}")

if len(obs_space.shape) == 2:
    print(f"   ✓ len(obs_space.shape) == 2")
else:
    print(f"   ✗ len(obs_space.shape) == {len(obs_space.shape)}, not 2!")
    print(f"   Therefore, observation_space will NOT be squeezed in __init__")

print(f"\n4. CPUNumpyWrapper observation() check:")
print(f"   obs.ndim: {obs.ndim}")
if obs.ndim == 2:
    print(f"   obs.shape[0]: {obs.shape[0]}")
    if obs.shape[0] == 1:
        print(f"   ✓ Will squeeze in observation() method")
        squeezed = obs.cpu().numpy().squeeze(0)
        print(f"   After squeeze: {squeezed.shape}")
    else:
        print(f"   ✗ Won't squeeze")
else:
    print(f"   ✗ obs.ndim = {obs.ndim}, not 2")

print("\n" + "="*60)
print("THE BUG:")
print("="*60)
print("• ManiSkill3 defines observation_space with shape (43,)")
print("• But returns observations with shape (1, 43)")
print("• CPUNumpyWrapper.__init__() checks observation_space.shape")
print("  → len(shape) = 1, so condition fails")
print("  → observation_space is NOT updated")
print("• FrameStack wrapper is added AFTER CPUNumpyWrapper")
print("  → FrameStack sees observation_space.shape = (43,)")
print("  → But FrameStack receives observations with shape ??? ")
print()
print("Let's check what FrameStack actually gets...")
