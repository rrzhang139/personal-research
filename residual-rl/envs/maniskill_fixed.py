# ManiSkill3 environment patches
# NOTE: The original ManiSkill2 custom envs (PegInsertionSideEnv_fixed,
# TurnFaucetEnv_COTPC, PushChairEnv_COTPC) subclassed MS2 internal classes
# that no longer exist in MS3. The bugs they fixed may be resolved in MS3.
#
# For now, we use the standard MS3 environments directly.
# If custom patches are needed, they should subclass from:
#   from mani_skill.envs.tasks.tabletop import *
#
# Original MS2 env IDs and their MS3 equivalents:
#   PegInsertionSide-v2 → PegInsertionSide-v1
#   StackCube-v0        → StackCube-v1
#   TurnFaucet-v2       → TurnFaucet-v1 (if available in MS3)
#   PushChair-v2        → (may not exist in MS3, was MS1 legacy env)

import mani_skill.envs  # registers all built-in MS3 environments
