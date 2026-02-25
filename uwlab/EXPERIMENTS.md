# Experiments: Challenging the OmniReset Near-Reset Curriculum Thesis

## Background

OmniReset's core thesis: contact-rich manipulation (e.g., peg-in-hole insertion) can be solved **without reward engineering or demonstrations** by training RL policies with curriculum-based reset state distributions. The curriculum progresses from near-goal states (ObjectPartiallyAssembled) to far-from-goal states (ObjectAnywhereEEAnywhere), letting the agent learn easy-to-hard.

This document proposes experiments to rigorously test where this thesis holds, where it breaks, and whether alternative or complementary methods outperform it.

---

## Experiment 1: OmniReset vs. DEMO3-Style World Model + Stage Rewards

**Hypothesis:** A world-model-based agent (TD-MPC2) with 5-10 demonstrations and learned stage rewards can match OmniReset's performance without needing the reset-state curriculum infrastructure.

**Setup:**
- **OmniReset baseline:** Standard 4-reset curriculum (PartiallyAssembled → Resting → AnywhereGrasped → AnywhereAnywhere), PPO, 4096 envs, 40K iterations (~4 hrs on RTX 4090)
- **DEMO3 variant:** TD-MPC2 backbone, 5-10 teleoperated demonstrations, stage discriminators for (approach → grasp → align → insert), single environment, 500K steps
- **Task:** Cube insertion (OmniReset's default task)
- **Metrics:** Success rate vs. wall-clock time, success rate vs. env steps, final inference success across difficulty tiers (Easy/Medium/Hard initial states from OmniReset's distribution)

**What this tests:** Whether OmniReset's curriculum advantage is simply compensating for the lack of demonstrations and dense rewards. If DEMO3 matches performance with only 5 demos, the curriculum infrastructure may be unnecessary overhead when demonstrations are available.

**Scaling question:** Run with {5, 10, 50, 100} demonstrations. At what demo count does the world model approach dominate? At what count does OmniReset (zero demos) still win?

---

## Experiment 2: Reverse Curriculum (Automated) vs. OmniReset (Hand-Defined Resets)

**Hypothesis:** OmniReset's 4 hand-defined reset types are a coarse approximation of what an automated reverse curriculum would discover. An automated approach may find better intermediate distributions.

**Setup:**
- **OmniReset baseline:** 4 fixed reset types with success-rate-driven promotion
- **Reverse Curriculum Generation (RCG):** Start from goal states, iteratively expand the initial state distribution by taking random actions backward from successful states (Florensa et al., 2017). No hand-defined tiers.
- **Adaptive success-rate scheduler (Paper 1 style):** Apply the IMCOM 2026 paper's success-rate-adaptive scheduling on top of OmniReset's tiers (dynamic reweighting rather than fixed promotion thresholds)
- **Task:** Cube insertion + peg insertion (to test generalization of curriculum design)

**Metrics:** Success rate convergence, env steps to 80% success, success on held-out hardest-tier initial states

**What this tests:** Whether human-designed reset types are optimal or leave performance on the table. If RCG produces smoother curriculum boundaries, it suggests OmniReset's approach doesn't scale — every new task would need manual reset type design.

---

## Experiment 3: Residual RL (Policy Decorator) on Top of OmniReset

**Hypothesis:** OmniReset policies trained purely in simulation will have a performance gap when transferred to perturbed or out-of-distribution settings. A residual RL layer can close this gap with minimal additional online interaction.

**Setup:**
- **Phase 1:** Train OmniReset policy to convergence in simulation (cube insertion, 40K iterations)
- **Phase 2a — Residual RL (sim):** Freeze OmniReset policy as base, train residual policy via SAC/TD3 with progressive exploration budget (Policy Decorator method). Test on perturbed environments: varied clearances (0.5mm, 1mm, 2mm), varied friction coefficients, varied object masses
- **Phase 2b — Residual RL (domain-randomized sim):** Same as 2a but the online refinement environment has heavy domain randomization (visual, dynamics, geometry)
- **Phase 2c — No refinement baseline:** Just deploy OmniReset policy directly on perturbed environments

**Metrics:** Success rate on nominal vs. perturbed environments, sample efficiency of residual refinement (env steps to recover performance)

**What this tests:** Whether OmniReset policies are brittle to distribution shift and whether residual RL is an efficient way to fix it. This directly validates whether Policy Decorator is the right online refinement layer for sim-trained policies.

---

## Experiment 4: Does the Curriculum Actually Matter? Ablation Study

**Hypothesis:** OmniReset's performance comes primarily from massive sim parallelism (4096 envs), not the curriculum structure itself. With enough parallel environments, even uniform random initialization might converge.

**Setup (all use PPO, 4096 envs, same hyperparameters):**
- **A — Full OmniReset:** 4-tier curriculum with progressive promotion
- **B — Uniform random:** Sample initial states uniformly from the full state space (ObjectAnywhereEEAnywhere only)
- **C — Near-goal only:** Only ObjectPartiallyAssembled and ObjectRestingEEGrasped (never see hard states)
- **D — Hard-only:** Only ObjectAnywhereEEAnywhere (skip the curriculum entirely, start from hardest)
- **E — Reverse order:** Start from hard, progressively add easier states (anti-curriculum)

**Metrics:** Learning curves (success rate vs. iterations), final success rate per tier, training stability (variance across 3 seeds)

**What this tests:** The necessity and ordering of the curriculum. If B (uniform random) converges to similar performance given enough env steps, the curriculum is just a sample efficiency trick, not a fundamental enabler. If E (reverse order) works, the near-to-far ordering isn't special.

---

## Experiment 5: Generalization Across Objects

**Hypothesis:** OmniReset's curriculum is object-specific (reset states are defined per geometry). A single policy trained with the curriculum on one object won't generalize to novel objects without retraining the curriculum.

**Setup:**
- **Single-object OmniReset:** Train on cube insertion only, evaluate zero-shot on peg, rectangle, cupcake, fbleg, fbdrawerbottom
- **Multi-object OmniReset:** Train with all 6 objects in the same curriculum (randomize object per episode)
- **Multi-object + visual encoder:** Same as above but use image observations instead of state (test whether visual representations generalize better)

**Metrics:** Zero-shot success rate per object, fine-tuning sample efficiency (env steps to reach 80% on novel object)

**What this tests:** Whether the curriculum approach produces generalizable skills or task-specific controllers. If zero-shot transfer fails completely, it suggests OmniReset solves individual tasks well but doesn't address the generalization problem — which is the real bottleneck for embodied AGI.

---

## Experiment 6: World Model as Learned Simulator for Curriculum Generation

**Hypothesis:** Instead of hand-defining reset types, a world model trained on OmniReset rollouts can generate synthetic starting states for curriculum learning — replacing the need for explicit reset-state datasets.

**Setup:**
- **Phase 1:** Train OmniReset normally, collect all rollout data (state, action, next_state, reward)
- **Phase 2:** Train a world model (TD-MPC2 or Dreamer-V3) on this data
- **Phase 3:** Use the world model to generate starting states by: (a) sampling goal states, (b) rolling backward through the model with random actions, (c) filtering states by predicted difficulty
- **Phase 4:** Train a new policy from scratch using only world-model-generated curriculum states (no hand-defined reset types)

**Metrics:** Compare final success rate of world-model-curriculum policy vs. OmniReset-curriculum policy, quality of generated initial states (coverage of state space, difficulty distribution)

**What this tests:** Whether world models can automate curriculum design. If successful, this removes the main bottleneck of OmniReset (manual reset-state engineering) and makes the approach scalable to arbitrary tasks.

---

## Priority Order

| Priority | Experiment | Rationale |
|----------|-----------|-----------|
| 1 | **Exp 4** (Curriculum ablation) | Cheapest to run, answers the most fundamental question first: does the curriculum actually matter? |
| 2 | **Exp 5** (Object generalization) | Tests whether the whole approach addresses the real problem (generalization) |
| 3 | **Exp 3** (Residual RL on OmniReset) | Directly extends your existing Policy Decorator work to OmniReset |
| 4 | **Exp 2** (Reverse curriculum vs. hand-defined) | Tests whether curriculum design can be automated |
| 5 | **Exp 1** (World model comparison) | Requires implementing TD-MPC2 integration — higher effort |
| 6 | **Exp 6** (World model for curriculum gen) | Most ambitious, do last, builds on results of 1-5 |

---

## Expected Outcomes and What They'd Mean

**If Exp 4 shows uniform random converges similarly:** OmniReset's curriculum is a sample efficiency trick, not a fundamental contribution. The real value is the simulation infrastructure (Isaac Sim parallelism). Pivot toward making the sim data useful for world model training.

**If Exp 5 shows zero-shot generalization fails:** OmniReset produces specialist policies. The path forward is to use OmniReset as a data generation engine — train many specialist policies, collect their rollouts, and use that data to train a generalist (VLA or world model).

**If Exp 3 shows residual RL helps significantly:** There's a real sim-to-deployment gap that online refinement addresses. This validates combining OmniReset (sim curriculum) + Policy Decorator (online refinement) as a pipeline.

**If Exp 1 shows DEMO3 matches with 5 demos:** The curriculum approach is obsolete when demonstrations are available. Pivot toward demonstration-efficient world model methods. OmniReset's remaining value would be generating those 5 demonstrations automatically via RL (since it doesn't need demos itself).

**If Exp 6 works:** World models can replace hand-designed curricula entirely. This is the most exciting outcome — it means the sim → world model → policy pipeline is self-improving.
