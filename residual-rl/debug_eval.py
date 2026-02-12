"""Debug script to check ManiSkill3 observation shapes and eval behavior"""
import gymnasium as gym
import numpy as np
import sys
import os

# Add parent dir to path to import from the script
sys.path.insert(0, os.path.dirname(__file__))

# Register ManiSkill environments
import mani_skill.envs

# Import the wrappers from the training script
class CPUNumpyWrapper(gym.ObservationWrapper):
    """Convert ManiSkill3 torch tensor observations to numpy and squeeze the num_envs dimension."""
    def __init__(self, env):
        super().__init__(env)
        obs_space = env.observation_space
        print(f"[CPUNumpyWrapper.__init__] Input obs_space: {obs_space}")
        print(f"[CPUNumpyWrapper.__init__] Has shape attr: {hasattr(obs_space, 'shape')}")
        if hasattr(obs_space, 'shape'):
            print(f"[CPUNumpyWrapper.__init__] obs_space.shape: {obs_space.shape}")
            print(f"[CPUNumpyWrapper.__init__] len(shape): {len(obs_space.shape)}")
            if len(obs_space.shape) == 2:
                print(f"[CPUNumpyWrapper.__init__] shape[0]: {obs_space.shape[0]}")

        if hasattr(obs_space, 'shape') and len(obs_space.shape) == 2 and obs_space.shape[0] == 1:
            # Squeeze (1, obs_dim) -> (obs_dim,)
            print(f"[CPUNumpyWrapper.__init__] SQUEEZING obs space from {obs_space.shape}")
            self.observation_space = gym.spaces.Box(
                low=obs_space.low.reshape(-1), high=obs_space.high.reshape(-1), dtype=obs_space.dtype)
            print(f"[CPUNumpyWrapper.__init__] New obs_space.shape: {self.observation_space.shape}")

        # Fix action space bounds if they are inf
        act_space = env.action_space
        if hasattr(act_space, 'low'):
            low = act_space.low.reshape(-1) if hasattr(act_space.low, 'reshape') else act_space.low
            high = act_space.high.reshape(-1) if hasattr(act_space.high, 'reshape') else act_space.high
            if hasattr(low, 'cpu'):
                low, high = low.cpu().numpy(), high.cpu().numpy()
            self.action_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def observation(self, obs):
        print(f"[CPUNumpyWrapper.observation] Input obs type: {type(obs)}, shape: {obs.shape if hasattr(obs, 'shape') else 'N/A'}")
        if hasattr(obs, 'cpu'):
            obs = obs.cpu().numpy()
        if obs.ndim == 2 and obs.shape[0] == 1:
            print(f"[CPUNumpyWrapper.observation] SQUEEZING obs from {obs.shape}")
            obs = obs.squeeze(0)
            print(f"[CPUNumpyWrapper.observation] After squeeze: {obs.shape}")
        result = np.asarray(obs, dtype=np.float32)
        print(f"[CPUNumpyWrapper.observation] Output shape: {result.shape}")
        return result

    def step(self, action):
        obs, rew, terminated, truncated, info = self.env.step(action)
        # Convert all torch tensors to numpy scalars
        if hasattr(rew, 'item'):
            rew = rew.item()
        if hasattr(terminated, 'item'):
            terminated = terminated.item()
        if hasattr(truncated, 'item'):
            truncated = truncated.item()
        return self.observation(obs), float(rew), bool(terminated), bool(truncated), info

class SeqActionWrapper(gym.Wrapper):
    def step(self, action_seq):
        rew_sum = 0
        for action in action_seq:
            obs, rew, terminated, truncated, info = self.env.step(action)
            rew_sum += rew
            if terminated or truncated:
                break
        return obs, rew_sum, terminated, truncated, info

def test_observation_shapes():
    """Test what observation shapes ManiSkill3 actually returns"""
    print("="*60)
    print("Testing ManiSkill3 Observation Shapes")
    print("="*60)

    # Create base environment
    env = gym.make('PegInsertionSide-v1',
                   reward_mode='sparse',
                   obs_mode='state',
                   control_mode='pd_ee_delta_pose',
                   render_mode=None)

    print(f"\n1. Base env observation_space: {env.observation_space}")
    print(f"   Shape: {env.observation_space.shape if hasattr(env.observation_space, 'shape') else 'N/A'}")

    # Reset and check actual observation
    obs, info = env.reset(seed=42)
    print(f"\n2. Actual reset() observation:")
    print(f"   Type: {type(obs)}")
    print(f"   Shape: {obs.shape if hasattr(obs, 'shape') else 'N/A'}")

    # Take a step
    action = env.action_space.sample()
    if hasattr(action, 'cpu'):
        action = action.cpu().numpy()
    obs, rew, terminated, truncated, info = env.step(action)
    print(f"\n3. Actual step() observation:")
    print(f"   Type: {type(obs)}")
    print(f"   Shape: {obs.shape if hasattr(obs, 'shape') else 'N/A'}")

    env.close()

    print("\n" + "="*60)
    print("Testing with CPUNumpyWrapper")
    print("="*60)

    # Create wrapped environment
    env = gym.make('PegInsertionSide-v1',
                   reward_mode='sparse',
                   obs_mode='state',
                   control_mode='pd_ee_delta_pose',
                   render_mode=None)
    env = CPUNumpyWrapper(env)

    print(f"\n4. Wrapped env observation_space: {env.observation_space}")
    print(f"   Shape: {env.observation_space.shape}")

    obs, info = env.reset(seed=42)
    print(f"\n5. Wrapped reset() observation shape: {obs.shape}")

    action = env.action_space.sample()
    obs, rew, terminated, truncated, info = env.step(action)
    print(f"\n6. Wrapped step() observation shape: {obs.shape}")

    env.close()

    print("\n" + "="*60)
    print("Testing with FrameStack")
    print("="*60)

    env = gym.make('PegInsertionSide-v1',
                   reward_mode='sparse',
                   obs_mode='state',
                   control_mode='pd_ee_delta_pose',
                   render_mode=None)
    env = CPUNumpyWrapper(env)
    env = gym.wrappers.FrameStack(env, 2)

    print(f"\n7. FrameStacked env observation_space: {env.observation_space}")
    print(f"   Shape: {env.observation_space.shape}")

    obs, info = env.reset(seed=42)
    print(f"\n8. FrameStacked reset() observation:")
    print(f"   Type: {type(obs)}")
    print(f"   Shape: {obs.shape if hasattr(obs, 'shape') else len(obs)}")
    if hasattr(obs, '__len__') and not hasattr(obs, 'shape'):
        print(f"   LazyFrames length: {len(obs)}")
        print(f"   First frame shape: {np.array(obs[0]).shape}")
        obs_array = np.array(obs)
        print(f"   As numpy array: {obs_array.shape}")

    env.close()

if __name__ == "__main__":
    test_observation_shapes()
