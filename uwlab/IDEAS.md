- **Goal: Speeding up RL training to increase iteration speed**
    - Curriculum, SAPG, more careful reward and termination design, action scale, temporally correlated action spaces, policy architectures with history, smarter choice of std, etc.
## Stagger Environments

The ablation showing that OmniReset needs 65k environments to work well might partially be compensating for the temporal homogeneity problem — with staggering, they might get away with fewer environments.

  Sample Efficiency — Low-Hanging Fruit

  3. Adaptive Reset Probabilities (Automatic Curriculum)
  Right now reset types are fixed at [0.25, 0.25, 0.25, 0.25]. But task_3 is already at 42% success while task_0 is at 0%. The agent wastes 25% of
  its experience on a problem it already solves. Idea: shift probability toward unsolved tasks dynamically.

  Two approaches:
  - Proportional to failure rate: sample more from tasks with lower success rate
  - Proportional to learning signal: sample from tasks where the value function has highest uncertainty (most to learn)

  This is essentially what curriculum learning does. The MultiResetManager already tracks per-task success rates — it just doesn't use them. A
  10-line code change could make probabilities adaptive.

  Relevant paper: Reverse Forward Curriculum Learning (RFCL, ICLR 2024) — starts from near-goal states and progressively widens to harder resets.
  Solves PegInsertion from sparse reward with just 1-10 demos.

  4. Tighter Dense Reward std Schedule
  You asked about std=1.0 — here's a concrete improvement. Start with std=1.0 (smooth everywhere, good for exploration) and anneal to std=0.05 over
   training (sharp near goal, good for precision). This gives you:
  - Early: broad gradients everywhere → learn approach/transport
  - Late: tight gradients near goal → learn precise insertion

  This is reward shaping curriculum — the reward function itself becomes harder over time. Easy to implement: std = max(0.05, 1.0 -
  iteration/20000).

  5. Asymmetric Actor-Critic
  Currently the policy sees observations that may be noisy or partial. Give the critic access to privileged info during training: exact object
  pose, exact contact forces, exact distance to goal. Keep the actor's observations realistic. The critic provides lower-variance value estimates,
  accelerating learning without affecting deployment.

  Widely used in Isaac Lab dexterous tasks. If OmniReset doesn't already do this, it's free sample efficiency.

  6. SAPG (mentioned in your bullet list)
  Sharpness-Aware Policy Gradient — regularizes PPO to find flatter optima in the loss landscape. Showed significant improvements on Isaac Gym
  tasks. Direct drop-in for rsl_rl's PPO implementation.

  Sample Efficiency — Medium Effort

  7. Hindsight Experience Replay (HER) for Multi-Reset
  OmniReset uses on-policy PPO. Switching to off-policy (SAC) with HER could be transformative: every failed insertion attempt gets relabeled as
  "successful transport to wherever the object ended up." This is especially powerful for task_0 (EEAnywhere) where the robot accidentally moves
  the object somewhere useful but doesn't get credit because it wasn't the goal.

  Tradeoff: requires rewriting from PPO to SAC. SERL (Berkeley, ICRA 2024) shows SAC + demos + high replay ratio solves insertion in 25 minutes of
  real robot time.

  8. Privileged Action Curriculum
  During early training, apply virtual "helper forces" that nudge the object toward the goal. Gradually remove them. The policy learns task
  structure under relaxed physics, then transfers to realistic constraints.

  Think of it as training wheels — the robot first learns the motion pattern with assistance, then learns to do it unassisted. Paper: arxiv
  2502.15442 (2025).

  9. World Models (DreamerV3)
  Train a latent dynamics model from collected experience, then do policy optimization entirely in imagination. For each real environment step, you
   can do 15-50 imagined rollouts. Dramatic sample efficiency gains.

## OmniReset’s Implicit Taxonomy

Let’s first look at what OmniReset’s tasks actually are and what structure they share:

**Drawer Assembly:** reach → grasp → flip/reorient → transport → insert
**Screw (table leg):** reach → grasp → transport → align → insert → screw
**Peg Insertion:** reach → grasp → transport → align → insert
**Cube Stacking:** reach → grasp → lift → transport → place
**Cupcake on Plate:** reach → grasp → lift → transport → precise placement
**Block Reorientation on Wall:** reach → push into wall → reorient via wall contact → position

Every single one of these follows a common pattern: the robot must move a single object from some initial configuration to a goal configuration. OmniReset’s four reset categories (reaching, near-object, grasped, near-goal) map directly onto phases of this single-object-to-goal template. That’s not a coincidence — they scoped the system to exactly this task family.

## What’s the Actual Structure?

I think the right way to think about this isn’t a flat taxonomy but a **compositional grammar of manipulation phases**, where each phase is defined by a contact mode transition. Let me build this up:

### Level 1: Contact Modes

The fundamental unit of manipulation structure is the **contact mode** — the qualitative pattern of contacts between the robot, the target object, and the environment. The key modes are:

**Free-space** — no contact between robot and object. Robot is reaching/repositioning.

**Non-prehensile contact** — robot touches object but doesn’t have a secure grasp. Pushing, sliding, pivoting, flipping via surface contact.

**Prehensile contact (grasp)** — robot has a stable grasp. Object moves with end-effector.

**Object-environment contact** — the object is in contact with a fixture. Insertion, screwing, sliding into a slot, pressing against a wall.

**Multi-body contact** — multiple objects in contact with each other. Stacking, nesting, interlocking.

OmniReset’s four categories map almost directly onto these:

- D_R → free-space
- D_NO → non-prehensile contact
- D_G → prehensile contact
- D_NG → object-environment contact (near goal)

### Level 2: Phase Transitions

A manipulation task is a **sequence of contact mode transitions**. The key insight is that transitions between modes are where the hard exploration happens — going from free-space to prehensile contact (grasping) is a discontinuous change that random exploration almost never discovers. This is why you need reset coverage at each mode boundary.

For OmniReset’s tasks:

```
Peg insertion:  free-space → prehensile → obj-env contact
Screw:          free-space → prehensile → obj-env contact → constrained rotation
Drawer:         free-space → non-prehensile(flip) → prehensile → obj-env contact
Block reorient: free-space → non-prehensile → obj-env(wall) → non-prehensile
Cube stack:     free-space → prehensile → multi-body contact
```

### Level 3: Tasks as Graphs, Not Chains

Here’s where it gets more interesting for generalization. OmniReset only handles **linear chains** — single object, one sequence of phases. But real-world manipulation is better described as a **directed graph** of contact mode transitions:

**Multi-step assembly** (like the four-leg table): four sequential chains sharing the same object-environment contact mode (the table), where each chain is an OmniReset-style single-object task.

**Tool use:** free-space → grasp tool → tool-object contact → object-environment contact. This is a chain but with an indirect contact mode — the robot acts on the target through an intermediary.

**Bimanual manipulation:** two parallel chains (one per arm) with synchronization constraints at certain modes (e.g., both arms must be in prehensile contact simultaneously for a handoff).

**Rearrangement:** a set of independent single-object chains with shared workspace constraints (objects can’t collide with each other).

## A Proposed Taxonomy for General Manipulation

Here’s how I’d organize this for scalable reset generation:

### Tier 1: Single-Object, Single-Chain (OmniReset’s scope)

**Structure:** Linear sequence of contact mode transitions
**Reset generation:** One set of resets per contact mode, automatically generated from goal + object + grasp sampler
**Examples:** Pick-and-place, insertion, screwing, stacking one object

This is solved by OmniReset’s approach. An LLM could template this for any new task given an object mesh and goal specification.

### Tier 2: Single-Object, Branching-Chain

**Structure:** The same object may require different mode sequences depending on initial conditions
**Reset generation:** Need resets for each branch, plus resets at the branch points
**Examples:** An object that sometimes needs flipping before grasping (drawer task is actually this — sometimes it’s right-side-up, sometimes not). Block reorientation on wall is also this — the wall contact is a branch that only some initial conditions require.

The challenge here is that the LLM or reset proposer needs to anticipate *when* different branches are needed. OmniReset handles this implicitly by making the reset distribution so broad that all branches are covered, but a more principled approach would enumerate the branches explicitly.

### Tier 3: Multi-Object, Independent Chains

**Structure:** Multiple Tier 1/2 tasks sharing a workspace
**Reset generation:** Cartesian product of per-object resets, with collision filtering
**Examples:** Setting a table (each plate/utensil is an independent placement), bin packing, multi-object rearrangement

Scalable because each object’s resets are generated independently. The main challenge is combinatorial explosion — if you have N objects each with M reset categories, you have M^N possible combined resets. In practice you’d sample rather than enumerate.

### Tier 4: Multi-Object, Coupled Chains

**Structure:** Objects interact during manipulation — the mode of one object constrains possible modes of another
**Reset generation:** Need to generate coupled reset states where multiple objects are in specific relational configurations
**Examples:** Assembly where part A must be held while part B is inserted. Stacking where bottom object must be stable before top object is placed. Tool use where tool and target must be in specific relative configuration.

This is where OmniReset’s approach starts breaking down. You can’t just independently sample resets for each object — you need resets for *relational* configurations (tool touching object, two parts aligned for mating). The grasp sampler analogy would be a “contact configuration sampler” that generates physically feasible multi-object arrangements.

### Tier 5: Deformable/Continuous Contact

**Structure:** Contact modes are not discrete but continuous (cloth, rope, liquids, dough)
**Reset generation:** The state space of the object itself is enormous (mesh deformation, particle positions). You can’t enumerate contact modes — they’re continuous.
**Examples:** Laundry folding, cable routing, food manipulation

This is fundamentally harder because the “key regions” of state space are themselves high-dimensional and can’t be easily categorized into discrete modes.

## Most Scalable Path to General Reset Generation

Given this taxonomy, here’s what I think is the most tractable path:

**For Tiers 1-2 (most industrial/practical tasks):** Fully automatable today. An LLM agent could take a task description + object meshes + goal specification and generate OmniReset-style resets by:

1. Calling a grasp sampler on target objects
1. Sampling near-goal perturbations via physics simulation
1. Generating grasped states throughout the workspace
1. Generating reaching states (trivial)

The “taxonomy” is just the linear contact mode chain, which is essentially universal for single-object-to-goal tasks. You don’t need to discover new categories — the four OmniReset categories cover this tier.

**For Tier 3:** Generate Tier 1-2 resets per object, compose them, filter via collision checking. An LLM planner could sequence the sub-tasks (which object to move first based on obstruction/dependency reasoning).

**For Tier 4 (where it gets research-worthy):** You’d need a **relational reset sampler** that generates multi-object configurations satisfying contact constraints. This is closer to constraint-satisfaction / motion planning than to OmniReset’s random sampling approach. A differentiable physics simulator could help — optimize object configurations to satisfy relational constraints (part A in gripper, part B aligned with hole, parts touching but not interpenetrating).

**For Tier 5:** Honestly nobody has a good answer. The deformable state space is too large for discrete reset categories. This is where learning-based approaches (world models that can imagine deformable configurations) or the π*0.6 approach (just collect real data with human corrections) might be more practical than trying to generate simulation resets.

## Connecting This to Your Framework

Given that you have a diffusion policy + residual RL (SAC) setup, the most natural entry point would be Tier 2 or Tier 3 tasks where:

- Your diffusion policy (trained on demos) gives you an implicit model of the contact mode sequence
- Your SAC residual handles the refinement within each contact mode
- OmniReset-style resets (auto-generated for each contact mode boundary) could seed additional RL training to improve robustness at the mode transitions

The research question would be: can you extract the contact mode structure from your diffusion policy’s learned representations to automatically propose reset distributions, rather than specifying them manually? The diffusion policy has already implicitly learned what “grasped” vs “reaching” vs “inserting” looks like from its training data — can you cluster its latent space into modes and sample resets from each cluster? That would close the loop between learned behavior and reset generation without any human specification of the four categories.​​​​​​​​​​​​​​​​