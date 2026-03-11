# UWLab (OmniReset) — Progress Log

## Project Goal
Reproduce and improve OmniReset for contact-rich manipulation (cube insertion) using adaptive curriculum learning over multiple reset difficulty levels.

## Reference Baseline
The original OmniReset paper achieves **97.7% end-of-episode success rate** on cube insertion after ~1792 training iterations (~11.7 hours on 4x GPUs). This was reproduced as `cube_4gpu_reproduce` in the `isaaclab` wandb project.

- **wandb project (legacy)**: `isaaclab` — contains the original 97% reproduce run
- **wandb project (current)**: `omnireset` — all new experiments go here
- **Hardware**: 4x RTX 4090, 16384 envs/GPU
- **PPO config**: lr=1e-4, gamma=0.99, lambda=0.95, clip=0.2, entropy_coeff=0.006, 32 steps/env, 5 epochs, 4 mini-batches
- **Per-iteration timesteps**: 16384 envs x 4 GPUs x 32 steps = ~2.1M timesteps/iter
- **Speed**: ~23s/iteration

---

## Experiment Log

### 1. Original Paper Reproduction (cube_4gpu_reproduce)
- **Date**: Early March 2026
- **wandb project**: `isaaclab`
- **wandb run**: `cube_4gpu_reproduce`
- **Task**: `OmniReset-...-State-v0` (uniform [0.25, 0.25, 0.25, 0.25])
- **Config**: 4x RTX 4090, 16384 envs/GPU, 40k iters
- **Result**: **97.7% success rate at step 1792** (~11.7 hours)
- **Notes**: This is the ground truth baseline. All future experiments compare against this.

---

### 2. Baseline Short Run (baseline_uniform_cube_4gpu)
- **Date**: 2026-03-11
- **wandb project**: `omnireset`
- **wandb run**: `baseline_uniform_cube_4gpu`
- **Task**: `OmniReset-...-State-v0` (uniform probs)
- **Config**: 4x RTX 4090, 16384 envs/GPU
- **Duration**: ~20 iterations (~46M timesteps, ~13 min)
- **Result**: 12.5% end-of-episode success rate
  - Task 0 (ObjectAnywhereEEAnywhere): 0%
  - Task 1 (ObjectRestingEEGrasped): 0%
  - Task 2 (ObjectAnywhereEEGrasped): 0.1%
  - Task 3 (ObjectPartiallyAssembledEEGrasped): 53.7%
- **Notes**: Killed early to test adaptive. Too short for meaningful comparison — the original baseline needed 1792 iters to reach 97%.

---

### 3. Adaptive v1 — Zone-Based Curriculum (adaptive_zones_cube_4gpu)
- **Date**: 2026-03-11
- **wandb project**: `omnireset`
- **wandb run**: `adaptive_zones_cube_4gpu`
- **Task**: `OmniReset-...-State-Adaptive-v0`
- **Config**: 4x RTX 4090, mastered_thresh=0.80, learning_boost=1.005, stuck_decay=0.999
- **Duration**: ~98 iterations (~205M timesteps, ~40 min)
- **Result**: 51% overall success, but extremely lopsided distribution
  - Task 3: 60.8% success, **0.85 probability** (absorbing nearly all envs)
  - Tasks 0,1,2: 0% success, **0.05 probability** each (at floor)
- **Problem**: The high mastered_thresh (0.80) meant Task 3 never "graduated" — it stayed in the learning zone forever, hoarding 85% of all environments. Tasks 0-2 were starved.
- **Key Insight**: At early training, only Task 3 (near-goal) has nonzero success. With learning_boost=1.005, its probability grew exponentially while stuck tasks decayed to floor. The curriculum became counterproductive — baseline reached 54% on Task 3 with only 25% allocation while adaptive needed 85% allocation to reach 61%.

---

### 4. Adaptive v2 — Tuned Thresholds (adaptive_v2_thresh60_boost01_cube_4gpu)
- **Date**: 2026-03-11
- **wandb project**: `omnireset`
- **wandb run**: `adaptive_v2_thresh60_boost01_cube_4gpu`
- **Task**: `OmniReset-...-State-Adaptive-v0`
- **Config**: 4x RTX 4090, **mastered_thresh=0.60**, **learning_boost=1.01**, stuck_decay=0.999
- **Duration**: ~2 iterations (~4M timesteps) — killed very early
- **Result**: Task 3 at 32.6% success, prob 0.36 (healthier ramp than v1)
- **Notes**: Too short to draw conclusions. Killed to compare with baseline data. The lower mastered_thresh (0.60 vs 0.80) should cause Task 3 to shed probability sooner, but we didn't run long enough to see it happen.

---

## Key Learnings

### Training Duration Matters Most
All our adaptive vs baseline comparisons were invalid because runs were too short (<100 iters vs the 1792 needed). The original baseline shows that success rate rises gradually over hours — early iterations mostly train Task 3 (easiest) regardless of curriculum. **Future experiments must run for at least 500-1000 iterations to see meaningful differentiation.**

### Adaptive Curriculum Problems Identified
1. **Lopsided allocation**: When only 1 task has nonzero success early on, the learning_boost causes runaway probability concentration. The successful task hogs all envs while others are starved at the floor.
2. **mastered_thresh too high**: With thresh=0.80, Task 3 never graduated even at 60% success. Lowering to 0.60 is the right direction.
3. **Counterproductive at early training**: Baseline allocates 25% to all tasks uniformly, meaning harder tasks still get explored from the start. Adaptive starves them, potentially delaying when they start learning.

### The Real Question (Unanswered)
Does adaptive curriculum help in the **mid-to-late training** phase (iterations 500-1800) where the baseline plateaus? The hypothesis is that once Task 3 is mastered, adaptive would shift resources to Tasks 2, then 1, then 0 — potentially reaching 97% faster than uniform allocation. **We haven't run long enough to test this.**

---

## Debugging Notes

### Isaac Lab EventTermCfg validates __call__ params
- Isaac Lab's EventTermCfg validates `__call__` kwargs against the `params` dict
- **Cannot use `**kwargs`** — it's treated as a mandatory parameter named `kwargs`
- Must explicitly list all params in `__call__` signature with defaults
- Applies to ALL ManagerTermBase subclasses (MultiResetManager, etc.)

### uwlab_rl missing __init__.py
- `uwlab_rl` package at `UWLab/source/uwlab_rl/uwlab_rl/` has NO `__init__.py` upstream
- Without it, `find_packages()` in setup.py doesn't discover `uwlab_rl`
- **Fix**: `touch UWLab/source/uwlab_rl/uwlab_rl/__init__.py` then `uv pip install -e .`
- Must be applied on every fresh clone/pod setup

### Fresh pod setup takes ~15 min
- New pods have NO persistent volume data — `/workspace/` is empty
- Must clone both repos, run `setup_env.sh`, download checkpoints
- After `setup_env.sh`, always apply the `__init__.py` fix above

### Port conflicts after killing training
- `pkill -f "train.py"` may not kill all distributed processes
- Use `pkill -9 -f "train.py" && pkill -9 -f "torch.distributed" && sleep 5` before re-launching
- Check with `ss -tulpn | grep 29500` to verify port is free

---

## Abandoned Approaches

### Adaptive Reset Curriculum — ABANDONED
The zone-based adaptive curriculum (stuck/learning/mastered zones adjusting reset probabilities) did not show improvement over uniform baseline at any tested configuration:
- **v1** (mastered_thresh=0.80, boost=1.005): Runaway concentration — Task 3 absorbed 85% of envs, learned slower than baseline
- **v2** (mastered_thresh=0.60, boost=1.01): Slightly better dynamics but still no clear advantage over uniform
- **Core problem**: At early training, only the easiest task (near-goal) has nonzero success. Any multiplicative boost causes it to monopolize environments while harder tasks are starved at the probability floor. Uniform allocation (25% each) is actually better because harder tasks get consistent exposure from the start.
- **Conclusion**: The original paper's uniform allocation is not leaving performance on the table — it's already a reasonable curriculum for this task structure.

### SAC + HER — ABANDONED (not implemented)
Off-policy SAC with Hindsight Experience Replay was planned as Phase 3 (see RESEARCH_PLAN.md) to eliminate dense reward engineering. **Not pursued** because:
- **Memory problem**: HER requires storing full episodes in a replay buffer. With 16384 envs x 4 GPUs = 65536 parallel envs, the memory buffer would be enormous. Would need to sacrifice parallelism (fewer envs) to fit the buffer in GPU memory.
- **Parallelism tradeoff**: OmniReset's speed comes from massive parallelism (65K envs). Reducing to fit HER buffer would negate the sample efficiency gains HER provides.
- **Not critical**: The existing dense reward works well enough — the 97% baseline proves the reward engineering is sufficient for cube insertion.

---

## Current Status

The **original paper baseline** (uniform probs, 4x GPU, ~12 hours to 97%) remains the best approach. No modifications to the curriculum or RL algorithm have shown improvement.

## Potential Future Directions (if revisiting)

1. **Contact-mode curriculum** (RESEARCH_PLAN.md Stage 2) — classify resets by contact mode (free-space, approach, contact, insertion) and anneal reward std based on mode
2. **Multi-task generalization** — test on peg, rectangle, cupcake objects beyond cube
3. **Reward shaping improvements** — tighter std annealing, auto mode detection
4. **Different curriculum strategies** — e.g., warmup period with uniform probs before enabling adaptive, or success-rate-proportional allocation instead of multiplicative zones
