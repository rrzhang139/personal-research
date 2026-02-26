# Research Plan: Contact-Mode Curriculum Learning for Generalizable Manipulation

## Vision

Learn new manipulation tasks faster by exploiting the universal structure of contact-mode transitions. Instead of task-specific curriculum schedules (RFCL) or brute-force exploration (OmniReset), automatically decompose any manipulation task into contact modes and use them as a natural, physics-grounded curriculum.

**End goal**: Given a new manipulation task (object mesh + goal spec), automatically discover the contact-mode graph, generate reset states for each mode, and train an adaptive curriculum that generalizes across objects and task structures — without demos, without reward engineering, without per-task tuning.

---

## Part 1: The Problem with Current Approaches

### OmniReset: Robust but Slow

OmniReset defines 4 reset categories for single-object-to-goal tasks and samples uniformly:

| Reset Type | Contact Mode | Prob | Early Success |
|-----------|-------------|------|---------------|
| ObjectPartiallyAssembledEEGrasped | Object-environment + prehensile | 25% | ~42% at 6 min |
| ObjectRestingEEGrasped | Prehensile only | 25% | ~0% for hours |
| ObjectAnywhereEEGrasped | Prehensile only | 25% | ~0% for hours |
| ObjectAnywhereEEAnywhere | Free-space | 25% | 0% for hours |

**Problems:**
1. Fixed probabilities waste 25% of samples on solved tasks (task_3) and 25% on currently-impossible tasks (task_0)
2. The 4 categories are hand-designed for pick-and-place/insertion — they don't generalize
3. PPO discards all data after each update (on-policy)
4. Dense rewards (`exp(-dist/1.0)`) are required despite claiming "no reward engineering"

**What's good:** Zero demos, task-agnostic hyperparams across 6 objects, no forgetting.

### RFCL: Fast but Brittle

RFCL walks backward along a demo trajectory, training from near-goal states first:

```
Step 1: Start at demo timestep T-1 (last state before goal)
Step 2: When 3/3 consecutive successes → step back by δ=4 timesteps
Step 3: Repeat until start_step = 0 (full demo solved)
Step 4: Forward curriculum (PLR) to generalize to full initial distribution
```

**Problems:**
1. **Requires demos** (1-10 per task) — someone must teleoperate or plan
2. **Per-task tuning**: step size δ (2 vs 4 vs 8), horizon ratio φ (1 vs 1.25 vs 3), discount (0.9 vs 0.95 vs 0.99) all vary by task
3. **Assumes temporal = difficulty ordering.** Walking backward from the demo's end assumes later states are easier. This breaks on tasks where difficulty is non-monotonic:

```
Coffee making:  pick cup (easy) → place under spout (medium) → press button (HARD)
                → wait (trivial) → pick cup again (easy) → transport to table (easy)

RFCL backward: transport → pick cup → wait → press button → place → pick cup
               ↑ easy first, makes sense    ↑ now "press button" but cup isn't placed yet!
```

The reverse curriculum hits "press button" before the agent has learned "place cup under spout" — the precondition isn't met. RFCL would need to somehow skip around in the demo, but its mechanism is strictly temporal.

4. **Single-chain assumption.** RFCL follows ONE demo trajectory backward. If the task has multiple valid strategies (the drawer that sometimes needs flipping), the demo shows only one. The curriculum trains toward that specific strategy.

**What's good:** 10-50x sample efficiency, sparse reward only, principled curriculum.

### The Structural Insight Neither Exploits

Both approaches miss that **manipulation tasks have inherent structure that doesn't depend on demos or hand-designed categories**: the contact-mode graph.

---

## Part 2: Contact Modes as Universal Curriculum

### What Are Contact Modes?

A contact mode is the qualitative pattern of contacts between bodies. The fundamental modes for manipulation:

| Mode | Description | Example |
|------|-------------|---------|
| **Free-space** | No contact between robot and object | Arm reaching toward object |
| **Non-prehensile contact** | Robot touches object without stable grasp | Pushing, sliding, flipping against surface |
| **Prehensile contact** | Stable grasp, object moves with end-effector | Carrying, transporting |
| **Object-fixture contact** | Object contacts the goal/environment fixture | Partially inserted, resting in slot |
| **Multi-body contact** | Multiple objects in contact | Stacking, nesting |
| **Tool-mediated contact** | Robot acts on target through intermediary | Using wrench, spatula |
| **Deformable contact** | Continuous contact manifold | Cloth folding, cable routing |

OmniReset's 4 categories are just specific combinations of these basic modes. The question is: **can we define them more generally?**

### Are 4 Modes Enough?

**For OmniReset's current tasks (single-object, pick-place-insert): yes, basically.**

The 4 categories decompose as:
```
EEAnywhere + ObjectAnywhere    = free-space (no useful contacts)
EEGrasped + ObjectAnywhere     = prehensile (grasping, but far from goal)
EEGrasped + ObjectResting      = prehensile (grasping, object near start)
EEGrasped + ObjectPartial      = prehensile + object-fixture (near goal)
```

But even within OmniReset's scope, there's a missing mode: **non-prehensile contact**. The drawer task requires a flip before grasping — that's a non-prehensile phase that doesn't fit cleanly into any of the 4 categories. OmniReset handles it implicitly because the random EEAnywhere resets sometimes start near the object, but it's not explicitly covered.

**For more complex tasks: no, 4 is nowhere near enough.**

### The Most General Decomposition

The most basic, physics-grounded decomposition is:

**A contact mode is defined by which pairs of bodies are in contact.**

For a system with bodies {robot, object, fixture/goal, table, tool}, a contact mode is a subset of possible contact pairs:

```
Mode = {(robot, object), (object, fixture)}   → grasping + partially inserted
Mode = {(robot, object)}                       → grasping, in free space
Mode = {(object, table)}                       → object resting, robot not touching
Mode = {}                                      → everything in free space
```

For N bodies, there are 2^(N choose 2) possible modes — combinatorial explosion. But in practice, most are physically infeasible or irrelevant. The actual modes for a task form a sparse graph.

### Contact Mode Graph (Not Chain)

**Key insight: tasks are graphs of contact-mode transitions, not linear chains.**

For simple insertion (chain):
```
free-space → prehensile → prehensile + object-fixture → done
    ↓           ↓                    ↓
  (reach)    (transport)         (insert)
```

For coffee making (graph):
```
                    ┌─────────────┐
                    │  free-space  │
                    └──────┬──────┘
                           │ reach cup
                    ┌──────▼──────┐
                    │  grasp cup  │
                    └──────┬──────┘
                           │ transport
                    ┌──────▼──────┐
                    │ place under │──── object-fixture contact
                    │   spout     │
                    └──────┬──────┘
                           │ release + reach button
                    ┌──────▼──────┐
                    │ press button│──── robot-fixture contact (DIFFERENT mode)
                    └──────┬──────┘
                           │ reach cup again
                    ┌──────▼──────┐
                    │ re-grasp    │──── back to prehensile
                    │   cup       │
                    └──────┬──────┘
                           │ transport
                    ┌──────▼──────┐
                    │ place on    │──── object-table contact
                    │   table     │
                    └─────────────┘
```

RFCL can only walk backward along this chain. But the hardest transition ("press button" requires releasing cup, reaching the button, pressing with correct force) isn't at either end — it's in the middle. A reverse curriculum would approach it from the wrong direction.

**A contact-mode curriculum would handle this naturally:** Generate reset states for EACH mode. Train on easier modes first (grasping, transport). When those are solved, the "press button" mode becomes learnable because the agent can already get to and from it.

### Why Contact Modes Are the Right Curriculum Levels

1. **Physics-grounded difficulty ordering.** Transitions between contact modes are where the hard exploration happens — going from free-space to prehensile (grasping) is a discontinuous change that random exploration rarely discovers. Within a mode, the dynamics are smooth and learnable.

2. **Task-agnostic.** Contact modes are defined by physics, not by the specific object. Cube insertion and peg insertion have the SAME mode graph (free-space → prehensile → object-fixture). No per-task tuning needed.

3. **Automatically discoverable.** Given a physics simulator, you can detect contact modes from contact reports. No demos, no hand-labeling. Run the simulator with random states, check which bodies are in contact, cluster into modes.

4. **Non-temporal.** Unlike RFCL's demo timesteps, contact modes have no assumed ordering. The curriculum can focus on any mode at any time, and the difficulty ordering emerges from the physics (modes that require more contact transitions to reach the goal are harder).

---

## Part 3: Research Progression (Easy → Hard)

### Stage 1: Validate on OmniReset's Existing Tasks (Cube Insertion)

**Goal:** Show that adaptive contact-mode sampling beats uniform sampling in OmniReset's own testbed. No new infrastructure needed.

#### Experiment 1.1: Adaptive Reset Probabilities (~10 lines of code)

Change `MultiResetManager` to shift probability from solved → unsolved modes:

```python
# Current: fixed [0.25, 0.25, 0.25, 0.25]
# Proposed: adaptive based on per-mode success rate

failure_rates = 1.0 - per_task_success_rates
temperature = 0.5
adaptive_probs = softmax(failure_rates / temperature)
adaptive_probs = clamp(adaptive_probs, min=0.05)  # floor prevents forgetting
adaptive_probs /= adaptive_probs.sum()
```

**Expected result:** 1.5-3x speedup in wall-clock time to 90% success rate.

**Metrics:**
- `wall_clock_time_to_90_pct` vs baseline OmniReset
- `task_0_time_to_first_nonzero` (hardest task starts learning earlier?)
- Per-task success curves (task_3 shouldn't regress)

**What this tells us:** Whether the uniform sampling is actually a bottleneck. If adaptive helps significantly, the contact-mode curriculum idea has legs.

#### Experiment 1.2: Dense Reward Annealing

Anneal `dense_success_reward` std from 1.0 → 0.05 over training:
```python
std = max(0.05, 1.0 * (1.0 - iteration / 20000))
```

**Expected result:** 1.2-1.5x speedup, especially in the precision phase (75% → 90%).

**What this tells us:** Whether the dense reward is compensating for poor curriculum (std=1.0 gives signal everywhere because the curriculum doesn't focus samples where they're needed).

#### Experiment 1.3: Sparse Reward Only (Ablation)

Remove all dense rewards. Keep only sparse success_reward.

**Expected result:** Fails or very slow convergence.

**What this tells us:** Quantifies how much dense rewards carry the system. This is the gap that HER would close later.

#### Experiment 1.4: RFCL Baseline in Isaac Lab

Port RFCL's algorithm into Isaac Lab on the same cube insertion task. Use the pretrained expert to generate 5 demos automatically (no human teleop).

RFCL uses SAC, not PPO. Key implementation work:
- SAC with replay buffer in Isaac Lab
- Demo state extraction from pretrained expert rollouts
- Reverse curriculum with per-demo advancement (100% success over last 3 episodes, step back by δ=4)
- Forward curriculum with PLR

**Expected result:** Faster than OmniReset baseline (RFCL is designed for efficiency). But requires demos and per-task tuning.

**What this tells us:** The efficiency ceiling for curriculum-based approaches on this task. Our hybrid should aim to match this without demos.

### Stage 2: Contact-Mode Adaptive Curriculum (The Core Contribution)

**Goal:** Replace hand-designed reset categories with automatically discovered contact modes. Replace fixed probabilities with physics-grounded adaptive curriculum.

#### Experiment 2.1: Automatic Contact Mode Detection

Instead of 4 hand-coded reset types, detect contact modes from simulator contact reports:

```python
def detect_contact_mode(env):
    contacts = env.get_contact_reports()
    mode = frozenset()
    if (robot, object) in contacts:
        mode |= {"prehensile"}
    if (object, fixture) in contacts:
        mode |= {"object-fixture"}
    if (object, table) in contacts:
        mode |= {"object-resting"}
    # ... etc
    return mode
```

Run the pretrained expert for 1000 episodes, record the contact mode at each timestep. Cluster into distinct modes. This gives you the task's contact-mode chain automatically — no hand-design.

**Expected result:** Recovers OmniReset's 4 modes (or something close) automatically.

**What this tells us:** Whether contact modes are detectable and clusterable from simulation data.

#### Experiment 2.2: Contact-Mode-Aware Curriculum

Use the detected modes as curriculum levels. Train with adaptive sampling:

```python
# Difficulty ordering: modes further from goal (in contact transitions) are harder
# Generate reset states for each mode (via sampling + simulation)
# Adaptive probabilities: shift from solved modes to unsolved modes

for mode in contact_modes:
    if success_rate[mode] > 0.8:
        # Redistribute probability to harder modes
        redistribute(probs, from=mode, to=hardest_unsolved)
```

**Expected result:** Matches or beats RFCL efficiency without demos. Matches OmniReset robustness.

**Key question:** Does the automatic mode detection + adaptive curriculum work as well as hand-designed categories? If the clustering is noisy, modes might not be cleanly separable.

#### Experiment 2.3: Test Across All 6 OmniReset Objects

Run the contact-mode curriculum on cube, peg, rectangle, cupcake, fbleg, fbdrawerbottom.

**Same hyperparameters for all objects.** If the contact modes and adaptive schedule work across all objects without tuning, we've demonstrated task-agnostic curriculum learning.

**Expected result:** Uniform speedup across all objects (1.5-3x vs OmniReset baseline).

**What to watch for:** Does the mode graph differ per object? (Drawer might have a non-prehensile flip mode that others don't.) Does the adaptive schedule need different temperatures per object?

### Stage 3: Off-Policy + HER (Eliminating Dense Rewards)

**Goal:** Replace PPO with SAC + HER to eliminate the need for dense reward engineering entirely.

#### Experiment 3.1: SAC Integration in Isaac Lab

Replace PPO with SAC. Start with single-GPU, 1024 envs for debugging.

Key decisions:
- UTD ratio: start with 10 (following RFCL's sample-efficient config)
- LayerNorm on critic (critical for high UTD per SERL)
- Replay buffer size: 500K transitions
- No HER yet — just SAC + adaptive multi-reset + dense rewards

**Expected result:** 2-3x speedup over PPO baselines from data reuse alone.

**Risk:** SAC with massively parallel envs in Isaac Lab is non-standard. May need significant engineering. Buffer memory management with 16384 envs.

**Fallback:** If SAC doesn't scale, stay with PPO and focus on curriculum contribution only.

#### Experiment 3.2: HER (Eliminate Dense Rewards)

Add HER on top of SAC:
- K=4 relabeled goals per transition (standard)
- Strategy: "future" (random achieved state from later in same episode)
- Remove dense_success_reward and ee_asset_distance
- Keep only sparse binary success_reward

**Expected result:** Matches dense-reward performance with sparse reward only.

**The critical experiment:** Does HER + sparse reward match or beat hand-designed dense rewards?
```
Experiment A: SAC + dense rewards (baseline)
Experiment B: SAC + HER + sparse only
Experiment C: SAC + HER + sparse + tight dense for insertion (std=0.01, within 2cm)
```

If B works → dense rewards are completely unnecessary. Paper claim: "truly reward-free."
If only C works → dense rewards needed for precision phase only. Still a big reduction.

**Experimentation surface:**
- Does HER need more demos to work? (K=4 relabeling might not cover the insertion precision)
- Does goal representation matter? (position-only vs full 6DOF pose)
- Does the HER strategy matter for contact-rich tasks? (future vs final vs random)

#### Experiment 3.3: Demo-Seeded Replay Buffer

Add 1-5 expert demos to the replay buffer permanently:
- 50/50 demo/online batch ratio initially, anneal to 5/95
- Demos provide "expert guidance" for modes the agent hasn't explored yet

**Key question: Does this lose task-agnosticism?**
- If demos come from the same pretrained expert for all objects → still task-agnostic
- If different objects need different demo strategies → task-specific
- Test by using OmniReset's own pretrained expert to generate demos for all 6 objects

### Stage 4: Generalization Beyond OmniReset's Task Family

**Goal:** Show the contact-mode curriculum works on tasks OmniReset can't handle.

#### Experiment 4.1: Multi-Step Task (Tier 2 from Taxonomy)

Choose a task with non-monotonic difficulty, e.g.:
- Pick object → flip/reorient → insert (drawer assembly)
- Or design a simple two-phase task: grasp → place in slot → press button to lock

**Why this matters:** RFCL's reverse curriculum would struggle here (difficulty not monotonic in time). OmniReset's 4 categories wouldn't cover the "press button" mode. Our contact-mode curriculum should handle it naturally — it discovers modes from physics, not from demos or hand-design.

**Expected result:** Contact-mode curriculum solves the task. RFCL struggles or needs careful demo design. OmniReset would need new hand-designed reset categories.

#### Experiment 4.2: Multi-Object Task (Tier 3 from Taxonomy)

Two-object assembly: insert peg, then place cube on top.

Contact modes: {free, prehensile-peg, peg-inserted, free-after-peg, prehensile-cube, cube-placed}

The mode graph has a dependency: cube-placed requires peg-inserted. The curriculum should discover this dependency and train peg insertion first.

**Expected result:** Contact-mode curriculum handles the dependency graph. OmniReset and RFCL are not designed for multi-object tasks.

#### Experiment 4.3: Transfer to New Object (Zero-Shot Generalization)

Train the contact-mode curriculum on 5 objects. Test on 6th object (held out) with zero demos.

The contact-mode graph for a new object should be automatically detected (same physics, same contact mode types). The question is whether the curriculum schedule transfers.

**Expected result:** If the adaptive schedule is truly task-agnostic, it should work on the new object with no additional tuning. Success here would be the strongest generalization claim.

---

## Part 4: How HER Manufactures Dense Signal

### The Problem

With sparse reward (1 if inserted, 0 otherwise), 99%+ of trajectories give zero reward. No gradient, no learning.

Dense rewards (`exp(-dist/std)`) give gradient everywhere but require engineering — choosing std, distance metric, weights. OmniReset uses std=1.0 which saturates (gives 0.74 reward even at 30cm from goal).

### How HER Solves This

After each episode, HER creates K=4 copies of each transition with **relabeled goals**:

```
Original trajectory: goal = "cube in hole", reward = 0 (failed)

Relabeled copy 1: goal = "cube at position where it ended up" → reward = 1
Relabeled copy 2: goal = "cube at position from step 50"      → reward = 1
Relabeled copy 3: goal = "cube at position from step 75"      → reward = 1
Relabeled copy 4: goal = "cube at position from step 90"      → reward = 1
```

Every failure becomes 4 successes (for different goals). The policy learns: "this action sequence moves the cube from A to B reliably."

### Why This Is Dense Signal Without Engineering

- **No distance metric needed** — the "density" comes from relabeling diversity, not reward shape
- **No std to tune** — reward is always binary (sparse), HER adds volume
- **Automatically scales** — early training (random flailing) produces diverse achieved positions → diverse relabeled goals → rich learning signal. Late training (near-goal attempts) produces goals clustered near the hole → precise insertion learning.
- **No per-task adjustment** — K=4 and "future" strategy work across tasks

### Evidence from Literature

| Paper | Task | HER Impact |
|-------|------|------------|
| Andrychowicz et al. 2017 (original) | Multi-goal reaching, pushing | Enabled learning that was impossible without dense reward |
| OpenAI Dactyl 2018 | In-hand cube rotation (24-DOF hand) | Critical for sparse reward in high-dimensional space |
| SERL 2024 (Berkeley) | PCB insertion, cable routing (real robot) | 25-50 min training with demo buffer + SAC |
| MRHER 2024 | Sequential object manipulation | Model-based extension generates virtual goals beyond reached states |

### Open Question: Precision Phase

HER excels at approach/transport (many diverse achieved positions). But for the final insertion (5mm, 0.025 rad), relabeled goals cluster around "near the hole" and might not distinguish "1mm off" from "3mm off."

**Experiments needed:**
- HER-only with sparse reward → does it reach 90%?
- HER + tight dense reward (std=0.01, only within 2cm of goal) → insurance for precision
- HER + more demos of the insertion phase → does observing expert insertions help?

---

## Part 5: Experimentation Surface & Open Questions

### High Priority (could change the approach)

**Q1: Does adaptive sampling actually help significantly?**
If Experiment 1.1 shows <1.2x speedup, the curriculum idea is weaker than expected. The bottleneck might be PPO, not sampling. This determines whether to invest in Stages 2-4 vs jumping straight to SAC.

**Q2: Can contact modes be detected automatically?**
If Experiment 2.1 fails (modes aren't cleanly separable from contact reports), the "automatic discovery" claim falls apart. Fallback: hand-define modes per task family (still better than per-task, but less general).

**Q3: Does SAC scale in Isaac Lab with parallel envs?**
If Experiment 3.1 fails (memory, stability), the off-policy + HER contribution is blocked. Fallback: stay on-policy (PPO) and focus purely on the curriculum contribution.

### Medium Priority (affects efficiency claims)

**Q4: How many demos does HER need to match dense rewards?**
- 0 demos: HER generates goals from its own exploration only
- 1 demo: adds expert-quality goals to the buffer
- 5 demos: more coverage of the mode transitions
If 0 works → fully demo-free. If 5 needed → similar to RFCL's requirements.

**Q5: Does std annealing compose well with adaptive curriculum?**
Hypothesis: yes (adaptive focuses samples, annealing sharpens reward where samples are focused). But could interact negatively if annealing is too aggressive on modes that aren't ready.

**Q6: Is the approach truly task-agnostic across all 6 objects?**
Track which hyperparameters (temperature, min_prob, annealing schedule) need to change per object. Any per-object tuning weakens the generalization claim.

### Research-Level Questions (Stage 4)

**Q7: How to generate reset states for automatically discovered modes?**
OmniReset hand-designs reset generators (grasp sampler, perturbation from goal, etc.). For a new mode discovered automatically, how do you generate valid states? Options:
- Save states visited during training that match the mode
- Use physics perturbation from known states in that mode
- Learn a generative model of states per mode

**Q8: How to order modes by difficulty without demos?**
In OmniReset, the ordering is implicit (PartiallyAssembled < Resting < AnywhereGrasped < AnywhereAnywhere). For an automatically discovered mode graph, how do you determine which modes are easier?
- Distance to goal in mode-transition-space (fewer transitions = easier)
- Success rate during initial random exploration (empirical difficulty)
- Information-theoretic: modes with higher value function variance are at the learning frontier

**Q9: Does the contact-mode graph transfer across similar tasks?**
If cube insertion and peg insertion have the same mode graph (they should — same physics), can a curriculum trained on cube transfer to peg with no additional training? This would be the strongest generalization result.

---

## Part 6: Implementation Timeline

### Phase 1: OmniReset Baselines + Quick Wins (Week 1-2)

| Experiment | Effort | Expected Impact | Depends On |
|-----------|--------|-----------------|------------|
| 1.0: Baseline OmniReset (current run) | Done | Reference point | — |
| 1.1: Adaptive reset probabilities | ~10 lines | 1.5-3x speedup | 1.0 |
| 1.2: Std annealing | ~5 lines | 1.2-1.5x additional | 1.0 |
| 1.3: Sparse reward ablation | Config change | Quantifies dense reward dependency | 1.0 |

**Run all on same 4x4090 pod.** These are config/code changes to existing OmniReset.

**Decision gate:** If 1.1 shows >1.5x speedup → proceed to Stage 2 (contact-mode curriculum is worth pursuing). If <1.2x → the bottleneck is elsewhere, consider jumping to Stage 3 (SAC/HER).

### Phase 2: Contact-Mode Curriculum (Week 2-4)

| Experiment | Effort | Expected Impact | Depends On |
|-----------|--------|-----------------|------------|
| 2.1: Automatic mode detection | Medium (contact API) | Validates auto-discovery | 1.0 |
| 2.2: Mode-aware adaptive curriculum | Medium | Matches RFCL without demos? | 2.1 |
| 1.4: RFCL baseline in Isaac Lab | Large (SAC port) | Efficiency ceiling | — |
| 2.3: Test across 6 objects | Run time only | Task-agnostic claim | 2.2 |

**Decision gate:** If 2.2 matches RFCL on cube → strong result, proceed to multi-task. If not → need SAC/HER to close the gap (Stage 3).

### Phase 3: Off-Policy + HER (Week 4-6)

| Experiment | Effort | Expected Impact | Depends On |
|-----------|--------|-----------------|------------|
| 3.1: SAC in Isaac Lab | Large | 2-3x from data reuse | — |
| 3.2: HER integration | Medium | Eliminate dense rewards | 3.1 |
| 3.3: Demo-seeded buffer | Small | Bootstrap exploration | 3.1 |

**Decision gate:** If 3.2 (HER sparse-only) matches dense-reward performance → "reward-free" claim holds. If not → keep minimal dense reward for precision.

### Phase 4: Generalization (Week 6-8)

| Experiment | Effort | Expected Impact | Depends On |
|-----------|--------|-----------------|------------|
| 4.1: Multi-step task | Large (new env) | Beyond OmniReset scope | 2.2 |
| 4.2: Multi-object task | Large (new env) | Tier 3 taxonomy | 2.2 |
| 4.3: Zero-shot new object | Run time only | Generalization claim | 2.3 |

### Compute Budget

| Phase | Estimated Cost | GPU Time |
|-------|---------------|----------|
| Phase 1 | ~$30 | ~20 hours @ $1.36/hr |
| Phase 2 | ~$50 | ~35 hours |
| Phase 3 | ~$60 | ~45 hours |
| Phase 4 | ~$60 | ~45 hours |
| **Total** | **~$200** | **~145 hours** |

---

## Part 7: What We're Building Toward

### The Dream System

```
Input:  Object mesh + Goal specification
Output: Trained policy that achieves the goal

Pipeline:
1. Auto-detect contact modes by running simulator with random states
2. Build contact-mode graph (which modes transition to which)
3. Generate reset states for each mode (grasp sampling, physics perturbation)
4. Train with adaptive contact-mode curriculum:
   - Start from near-goal modes
   - Adaptively shift samples to learning frontier
   - No demos, no reward engineering, no per-task tuning
5. (Optional) Off-policy + HER for maximum efficiency
```

### Paper Contribution (If Results Hold)

**Title:** "Contact-Mode Curriculum Learning for Generalizable Manipulation Without Demonstrations"

**Story:**
1. Manipulation tasks share universal structure: contact-mode transitions
2. These modes define a natural curriculum (near-goal modes are easier)
3. Adaptive sampling across modes gives OmniReset's robustness with RFCL's efficiency
4. No demos, no reward engineering, no per-task tuning
5. Generalizes beyond single-object tasks to multi-step and multi-object manipulation

**Contribution ladder:**
- [x] Contact modes as universal curriculum levels (conceptual)
- [ ] Adaptive multi-reset matching RFCL efficiency without demos (Stage 2)
- [ ] HER eliminating dense reward engineering (Stage 3)
- [ ] Generalization to multi-step/multi-object tasks (Stage 4)
- [ ] Zero-shot transfer to new objects (Stage 4)

---

## Appendix A: RFCL Technical Details

(From source code analysis at github.com/StoneT2000/rfcl)

**Algorithm:**
- Uses SAC (not PPO) with UTD=0.5 (wall-time) or UTD=10 (sample-efficient)
- 50/50 demo/online batch ratio
- Reverse curriculum: per-demo frontier, geometric start-step sampling
- Advancement: 100% success over last 3 episodes at frontier step → step back by δ
- Forward curriculum: PLR with success_once scoring (score 3 = learning frontier, highest priority)
- Stage 1 → Stage 2 transition when >90% of demos solved
- Anti-forgetting: Stage 1 replay buffer becomes Stage 2 offline buffer

**Per-task tuning required:**
- reverse_step_size δ: 2 (PlugCharger) vs 4 (default) vs 8 (sample-efficient)
- demo_horizon_ratio φ: 1 (MetaWorld) vs 1.25 (PlugCharger) vs 3 (default)
- discount: 0.9 (default) vs 0.95 (PlugCharger) vs 0.99 (PegInsertion, MetaWorld, Adroit)

**Performance:**
- PegInsertionSide: ~80% from 5 demos, <60 min on RTX 4090
- StackCube: ~100% from 5 demos, ~20 min
- Solves tasks where all baselines (RLPD, JSRL, Cal-QL, DAPG) fail at 0%

## Appendix B: OmniReset's Reward Structure Reference

| Reward | Weight | Formula | Purpose |
|--------|--------|---------|---------|
| progress_context | 0.1 | Always returns 0 (state tracker only) | Tracks insertion progress for other rewards |
| success_reward | 1.0 | Binary: 1 if pos<5mm AND rot<0.025rad | Sparse goal signal |
| dense_success_reward | 0.1 | exp(-distance/std), std=1.0 | Smooth gradient toward goal |
| ee_asset_distance | varies | Based on gripper-to-object distance | Reaching/approach signal |
| abnormal_robot | varies | Penalty for joint limits, self-collision | Safety |

**Key issue:** dense_success_reward with std=1.0 gives exp(-0.30)=0.74 reward even 30cm from goal. This saturates early and provides weak gradient. The reward claims to be "engineering-free" but these terms and their weights are carefully designed.

---

## Appendix C: Staggered Environment Initialization (Low Priority)

### The Temporal Homogeneity Problem

OmniReset's ablation shows 65K environments are needed for good performance. Part of this requirement may stem from **temporal homogeneity** — all environments start simultaneously and tend to be at similar phases of execution, reducing data diversity within each PPO batch.

### Staggering Solution

Initialize environments at different episode offsets so that at any training step, the batch contains a mix of early/mid/late episode states:

```python
# At env creation, give each env a random starting offset
for i, env in enumerate(envs):
    env.step_count = random.randint(0, max_episode_length)
```

**Expected benefits:**
- More diverse batches → better gradient estimates per PPO update
- May allow **fewer environments** (e.g., 16K → 8K) with similar performance
- Reduces correlation between environments in the same batch

**Priority: Low.** This is a pure engineering optimization. It doesn't change the fundamental approach, but could reduce compute cost by 2x if OmniReset's env count requirement is partly due to temporal homogeneity.

**Implementation:** ~20 lines. Modify the environment reset logic to initialize at random offsets on the first episode only. Subsequent episodes start normally from reset states.

**Experiment:** Compare 16K staggered vs 16K synchronized vs 8K staggered on cube insertion. If 8K staggered matches 16K synchronized, the hypothesis holds.

---

## Appendix D: LLM-Designed Reward Functions — An Alternative Path

### Overview

Instead of hand-designing rewards or using our contact-mode taxonomy, could an LLM **generate reward functions automatically** from task descriptions? This is an active research area with several key papers.

### Key Papers

#### Eureka (NVIDIA, ICLR 2024)

**Pipeline:** GPT-4 generates candidate reward functions as Python code. Evolutionary optimization: generate K reward candidates per iteration → train RL agents on each → evaluate → feed best + training curves back to GPT-4 → generate improved candidates. Repeat for ~5 iterations.

**Tasks:** 29 Isaac Gym tasks including dexterous manipulation (pen spinning with Shadow Hand), locomotion, and object manipulation. Achieved human-level or better reward design on 83% of tasks.

**What's remarkable:** The pen spinning result — Eureka generated rewards that trained a policy to spin a pen continuously, which no human-designed reward had achieved. This suggests LLMs can discover non-obvious reward structures.

**Limitations:**
- Requires full environment source code as context (the LLM reads the sim code)
- Each iteration requires full RL training (~hours per candidate × K candidates × 5 iterations)
- No physics reasoning — the LLM treats rewards as code optimization, not physics understanding
- Reward hacking still occurs; the evolutionary loop catches some but not all cases
- Compute cost: hundreds of GPU-hours per task for the full evolutionary search

#### Text2Reward (NeurIPS 2023)

**Pipeline:** Three stages — (1) GPT-4 generates reward code from natural language goal + Pythonic environment abstraction, (2) execution-based error correction, (3) human feedback on rollout videos for refinement.

**Tasks:** 17 MetaWorld/ManiSkill2 manipulation tasks + 6 MuJoCo locomotion tasks. 13/17 manipulation tasks matched or exceeded expert rewards.

**Key difference from Eureka:** Uses structured environment representations (class hierarchies with type annotations) rather than raw source code. Requires human-in-the-loop feedback for refinement.

**Limitations:**
- Initial generation quality determines success — poor zero-shot generation can't recover through iteration
- Doesn't handle contact-rich manipulation; tested on standard pick-place tasks only
- Human feedback still needed for complex tasks (not fully autonomous)

#### Language-to-Reward (L2R, Google, 2023)

**Pipeline:** Two-stage — Reward Translator converts natural language → parameterized reward code using pre-defined reward primitives. Motion Controller (MuJoCo MPC, not RL) optimizes actions.

**Key difference:** Uses MPC instead of RL, and uses pre-defined reward templates (the LLM fills in parameters, not arbitrary code). Much more constrained = more reliable but less expressive.

**Tasks:** 17 tasks across quadruped and dexterous manipulator.

**Limitations:** Requires manual template design per robot morphology. Can't express novel reward structures — only parameterize existing ones.

#### VLM-Based Rewards (RoboCLIP, RL-VLM-F, 2023-2024)

**Alternative approach:** Instead of generating reward code, use vision-language models to compute reward from visual observations. RoboCLIP computes cosine similarity between trajectory video embedding and task description embedding.

**Advantage:** No code generation, no environment API needed. Works from pixels.

**Limitation:** Sparse (episode-level) rewards. Current VLMs can't reliably judge sub-millimeter precision needed for insertion tasks.

#### Recent Frameworks (2024-2025)

- **CARD:** LLM Coder + Evaluator with Trajectory Preference Evaluation. Avoids RL training at every iteration by using trajectory preferences. Tested on MetaWorld + ManiSkill2.
- **LEARN-Opt:** Fully autonomous, model-agnostic. Key finding: **smaller LLMs (GPT-4.1-nano) can match larger models** for reward generation, suggesting the bottleneck isn't LLM capability but the reward design problem structure itself.
- **ARCHIE:** GPT-4 generates rewards from natural language with reward formalization constraints that stabilize learning.

### Fundamental Limitations of LLM Reward Design

**1. No physics reasoning.** LLMs generate reward code by pattern-matching from training data and source code context. They don't understand contact mechanics, friction, or force balance. For our contact-rich tasks (5mm insertion tolerance, 0.025 rad alignment), the LLM can write `exp(-distance/std)` but can't reason about why std=1.0 saturates or why contact-mode transitions need different reward structures.

**2. Compute cost is prohibitive for iteration.** Eureka's evolutionary loop requires training a full RL agent per reward candidate per iteration. For OmniReset tasks (16K envs, 6-8 hours per run), evaluating even 5 candidates × 5 iterations = 125-200 GPU-hours per task. Our adaptive curriculum experiments cost ~$30 total.

**3. Reward hacking scales with LLM capability.** Better LLMs write more sophisticated code that exploits simulator quirks more effectively. The evolutionary feedback loop catches obvious hacking but not subtle cases (e.g., a reward that achieves high return by exploiting contact penetration artifacts).

**4. Doesn't solve the hard part.** For contact-rich manipulation, the hard part isn't writing the reward function — it's exploration. Even perfect rewards don't help if the agent never discovers the contact-mode transitions. Eureka's pen spinning works because Shadow Hand has enough random exploration surface; for precise insertion, even perfect rewards combined with PPO + uniform resets would take hours to discover the insertion mode.

**5. Sample efficiency is orthogonal.** LLM rewards don't address sample efficiency at all — they just automate reward engineering. The agent still trains with PPO, still discards data, still needs dense rewards or curriculum for exploration.

### Does Scaling LLMs Solve These Limitations?

**Partially, for limitations 1-3. Not at all for 4-5.**

LEARN-Opt's finding that smaller LLMs match larger ones is telling: the bottleneck isn't the LLM's reasoning ability, it's the **feedback loop structure**. An LLM (of any size) can't predict how a reward function will interact with RL optimization dynamics without actually running the training. This is fundamentally a simulation/execution problem, not a reasoning problem.

**What next-gen models (Opus 4.6+) could realistically do:**
- Better zero-shot reward code from task descriptions (fewer iterations needed)
- Better interpretation of training curves for reward refinement
- Better detection of reward hacking from trajectory analysis
- **Not:** Replace the need for curriculum, exploration strategy, or sample-efficient algorithms

**What would actually help:** A foundation model trained on (task, reward, training curve, final performance) tuples across thousands of RL tasks. This would learn the mapping from task description → effective reward function directly, bypassing the generate-evaluate loop. No such model exists yet.

### Branching Path: Taxonomy vs LLM Rewards

Our research has two potential paths that could converge:

```
Path A: Contact-Mode Taxonomy (our main path)
├── Physics-grounded, interpretable, no LLM needed
├── Works: curriculum structure, adaptive sampling, HER
├── Weakness: requires contact detection API, manual mode definitions for new task families
└── Endgame: auto-discover modes from simulation → auto-generate curriculum

Path B: LLM-Designed Rewards
├── Flexible, task-agnostic input (natural language), no physics knowledge needed
├── Works: reward code generation, iterative refinement
├── Weakness: no exploration/curriculum, high compute, reward hacking
└── Endgame: LLM generates both rewards AND curriculum structure

Path C: Convergence (most interesting)
├── Use LLM to propose contact-mode decomposition from task description
├── Use taxonomy framework to auto-generate resets + curriculum from modes
├── Use LLM to generate per-mode reward functions (simpler than full-task rewards)
├── Use adaptive sampling + HER for exploration (our core contribution)
└── Endgame: fully automated pipeline from task description → trained policy
```

**Path C is the most promising because it uses each component where it's strongest:**
- LLMs are good at semantic decomposition ("this task has a grasp phase, a transport phase, and an insertion phase")
- Our taxonomy framework is good at turning decompositions into curricula
- LLMs generating per-mode rewards is MUCH easier than full-task rewards (each mode has simpler dynamics)
- Our adaptive sampling + HER handles exploration and sample efficiency (the hard part)

**Priority:** Path A is our main research contribution (Stages 1-4). Path C is a future direction for the paper's "discussion" section or a follow-up project. Path B alone is incremental over Eureka.

### North Star: Real2Sim + Contact-Mode Curriculum (Path C+)

**Thesis: Use simulation for data generation to speed up real policy learning.**

The most successful workflow isn't any single paper's approach — it's a pipeline that combines the best of real2sim reconstruction, LLM semantic understanding, and physics-grounded curriculum learning. Here's the north star.

#### The Pipeline (Smartphone Scan → Working Real-World Policy in ~5 Hours)

```
┌─────────────────────────────────────────────────────────────────────┐
│ STEP 1: CAPTURE (5 min)                                             │
│                                                                     │
│   Smartphone scan of real workspace + target objects                │
│   (NVIDIA NuRec / Polaris-style 2DGS / iPhone LiDAR)              │
│   Output: multi-view images + depth                                 │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────────┐
│ STEP 2: RECONSTRUCT (10 min)                                        │
│                                                                     │
│   3D Gaussian Splatting → mesh extraction → Isaac Lab scene         │
│   Objects registered with estimated physics properties              │
│   (mass from volume, friction defaults, domain randomization)       │
│   Robot URDF placed in reconstructed workspace                      │
│                                                                     │
│   Key papers: RialTo (digital twin), RL-GSBridge (GS+physics),     │
│   NVIDIA NuRec (smartphone→USDZ→Isaac Sim), GRS (single RGBD)     │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────────┐
│ STEP 3: DECOMPOSE (1 min)                                           │
│                                                                     │
│   LLM/VLM analyzes the task from language + scene image:           │
│   "Pick up the cup from the table and place it on the shelf"       │
│                                                                     │
│   Output: Contact-mode graph                                        │
│   ┌────────────┐    ┌────────────┐    ┌─────────────┐             │
│   │ free-space │───►│ prehensile │───►│ obj-fixture  │             │
│   │ (reach)    │    │ (grasp+    │    │ (place on    │             │
│   │            │    │  transport) │    │  shelf)      │             │
│   └────────────┘    └────────────┘    └─────────────┘             │
│                                                                     │
│   This is WHERE LLMs excel — semantic task decomposition,          │
│   not reward code generation.                                       │
│   Key insight: contact modes are ROBUST to sim imperfections.      │
│   A cup in a gripper is "prehensile" regardless of mesh quality.   │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────────┐
│ STEP 4: GENERATE RESETS (5 min)                                     │
│                                                                     │
│   For each contact mode, sample physically valid states             │
│   in the reconstructed sim:                                         │
│                                                                     │
│   free-space:   random arm configs (trivial)                       │
│   prehensile:   grasp sampler on object mesh                       │
│   transport:    grasped object at random workspace positions        │
│   near-goal:    object near shelf with small perturbations          │
│                                                                     │
│   Uses the SAME reset generation as OmniReset, but with            │
│   auto-discovered modes instead of hand-designed categories.        │
│   Collision checking filters physically infeasible states.          │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────────┐
│ STEP 5: TRAIN (1-4 hours)                                           │
│                                                                     │
│   Adaptive contact-mode curriculum in Isaac Lab:                    │
│   - 16K+ parallel envs on 4x GPUs                                  │
│   - Start heavy on near-goal (easy) modes                          │
│   - Shift samples to harder modes as they're solved                │
│   - SAC + HER eliminates reward engineering                        │
│   - Sparse reward ONLY (binary: cup on shelf? yes/no)              │
│   - Optional: 1-5 real demos in replay buffer for bootstrap        │
│                                                                     │
│   No dense rewards, no reward tuning, no per-task hyperparams.     │
│   The curriculum IS the reward shaping.                             │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────────┐
│ STEP 6: TRANSFER (~0 cost)                                          │
│                                                                     │
│   Policy trained on reconstructed scene deploys to real robot.     │
│   Minimal sim2real gap because the sim IS the real scene.          │
│                                                                     │
│   Optional: co-train with small real data (SaRC-style)             │
│   for robustness to reconstruction artifacts.                       │
│   RialTo showed +67% robustness with this approach.                │
│   RLinf-Co showed +24% with RL sim-real co-training on VLAs.       │
└─────────────────────────────────────────────────────────────────────┘
```

#### Why This Beats Existing Approaches

| Approach | Setup Time | Human Effort | Reward Eng. | Per-Task Tuning | Demos Needed |
|----------|-----------|-------------|-------------|-----------------|-------------|
| **OmniReset** | Days | Hand-design env + 4 reset types + dense rewards | Heavy | None (task-agnostic) | 0 |
| **RFCL** | Hours | Teleop demos + tune δ, φ, γ per task | None (sparse) | Heavy | 1-10 |
| **Eureka** | Days | Write env source code for LLM | Automated (high compute) | None | 0 |
| **Diffusion policy (Pi0)** | Days | 50-200 teleop demos | N/A (IL) | None | 50-200 |
| **North Star (ours)** | **~5 hours** | **Smartphone scan + language description** | **None (sparse + HER)** | **None** | **0-5 optional** |

#### Why Contact Modes Are the Right Abstraction for Real2Sim

IKER (2025) uses VLM-generated keypoint rewards in reconstructed sims. But keypoints are fragile — they depend on visual features that may not survive reconstruction artifacts. Contact modes are robust because they're physics-grounded:

- A cup in a gripper is "prehensile contact" regardless of cup color, mesh quality, or texture
- A peg partially inserted is "object-fixture contact" regardless of reconstruction fidelity
- Contact modes are detectable from **contact reports** in the simulator, not from vision
- The curriculum structure (which modes, how to sample) transfers across objects with the same task structure

This means the curriculum transfers even when:
- The digital twin isn't perfect (SaRC showed "digital cousins" work — same task semantics, different visual/physical details)
- Domain randomization is applied (modes survive physics perturbation)
- New objects are introduced (any insertion task has the same mode graph)

#### Existing Work That Validates Each Step

| Pipeline Step | Validated By | Key Result |
|--------------|-------------|------------|
| Smartphone → 3D reconstruction | NVIDIA NuRec, Polaris | Production-ready, 5-min scan |
| 3DGS → physics sim | RL-GSBridge (ICRA 2025) | Zero-shot sim2real for grasping |
| Digital twin RL training | RialTo (2024) | +67% policy robustness |
| LLM task decomposition | GRS (CVPR 2025), Gen2Sim (ICRA 2024) | VLM generates tasks from single RGBD |
| VLM-generated rewards in real2sim | IKER (2025) | Multi-step manip with SE(3) control |
| Sim-real co-training | SaRC (RSS 2025) | +38% with just 10 real demos |
| RL fine-tuning for VLAs | RLinf-Co (2025) | +24% on OpenVLA, +20% on Pi0.5 |
| Adaptive curriculum | OmniReset (ICLR 2026) | Task-agnostic across 6 objects |
| Physics-informed deformable twins | PhysTwin (ICCV 2025) | Real-time interactive sim of deformables |

**Every step has been demonstrated independently. Nobody has combined them into a single pipeline.** That's the research opportunity.

#### What's Missing / Hard

**1. Physics property estimation.** The scan gives geometry but not mass, friction, stiffness. Current solutions: default values + domain randomization (works for rigid objects). PhysTwin addresses this for deformables but isn't integrated with RL pipelines.

**2. Grasp sampler needs clean collision geometry.** 3DGS gives appearance but mesh extraction is lossy. RL-GSBridge solves this with "soft mesh binding constraints" — using the mesh for physics and GS for rendering separately.

**3. Articulated objects.** Drawers, hinges, levers need joint parameters (type, limits, damping) that can't be extracted from a passive scan. RialTo uses a manual GUI for adding articulations. GRS matches objects to existing sim-ready asset libraries with known articulation.

**4. Deformable objects.** Cloth, cable, food — the contact-mode framework doesn't apply cleanly because modes are continuous, not discrete. PhysTwin reconstructs deformable physics from video but hasn't been connected to RL training.

**5. Scale validation.** Each individual step works, but the full pipeline hasn't been demonstrated end-to-end. Integration challenges are real — different papers use different sim platforms (MuJoCo vs Isaac Sim), different RL algorithms (PPO vs SAC), and different observation spaces.

#### How This Shapes Our Research

The north star reframes our contribution. Instead of just "faster curriculum for OmniReset," our work becomes:

**"The curriculum and exploration module for the real2sim manipulation pipeline."**

Specifically, Steps 3-5 are our contribution:
- Step 3 (decompose): Contact-mode taxonomy provides the universal structure LLMs decompose into
- Step 4 (generate resets): Auto-generated from contact modes + physics simulation
- Step 5 (train): Adaptive curriculum + SAC + HER with zero reward engineering

Steps 1-2 (capture + reconstruct) are provided by NVIDIA's toolchain and the real2sim community. Step 6 (transfer) is validated by RialTo/SaRC/RLinf-Co.

This framing is more compelling for a paper because:
1. It's not incremental over OmniReset (just "faster") — it's a module in a larger system
2. The contact-mode abstraction is justified by real2sim: modes are robust to reconstruction artifacts
3. The sparse-reward-only claim (HER) becomes essential: you can't hand-tune dense rewards for every new reconstructed scene
4. The task-agnostic claim becomes essential: the pipeline must work on any task the LLM can decompose

### General Framework: Prompt Design for RL Reward Functions

If using an LLM for reward generation (Path B or C), the prompt should include:

**Required context:**
1. **Task specification** — What the robot should achieve (natural language + formal goal if available)
2. **Environment API** — Available observations, actions, contact reports, object poses (structured, not raw source code — Text2Reward's Pythonic abstraction is better than Eureka's raw code dump)
3. **Physics constraints** — Degrees of freedom, joint limits, workspace bounds, gripper type
4. **Success criteria** — Exact thresholds (position tolerance, orientation tolerance)

**For contact-rich tasks specifically, also include:**
5. **Contact mode structure** — Which bodies can contact which (robot-object, object-fixture, etc.)
6. **Phase decomposition** — Break the task into contact-mode phases and request per-phase rewards
7. **Precision requirements** — Where sub-mm precision is needed vs where coarse is fine
8. **Known failure modes** — Common RL failure modes for this task class (e.g., "the agent tends to push the object off the table instead of grasping it")

**Prompt template for per-mode reward generation (Path C):**

```
Task: {task_description}
Current contact mode: {mode_name}
Mode definition: Robot is in {contact_state} with {bodies_in_contact}
Next mode transition: Achieve {next_mode} by {transition_description}
Success for this mode: {mode_success_criteria}

Available observations:
{observation_api}

Generate a reward function for this specific mode transition.
The reward should:
- Be dense (provide gradient from any state within this mode)
- Peak when the agent achieves the next mode transition
- Not reward-hack by exploiting simulator artifacts
- Use simple geometric terms (distances, angles, contact forces)

Return executable Python code.
```

**Why per-mode prompts are better than full-task prompts:**
- Each mode has simpler dynamics → LLM generates simpler, more correct rewards
- Mode transitions are well-defined → less ambiguity in success criteria
- Reward hacking is easier to detect per-mode → evolutionary feedback is cheaper
- Composes naturally with our adaptive curriculum → each mode gets its own reward + sampling probability

---

## Appendix E: Conversation Summary (Feb 17-19, 2026)

### What Happened

1. **Pod setup & training launch:** Pushed code to GitHub. Created new 4x4090 RunPod pod (1wlyxqrt37safq). Set up environment, launched 4-GPU OmniReset cube training with 16384 envs + all 4 resets. Training confirmed running at 22.5s/iter, task_3 at 42% success early.

2. **Training metrics deep dive:** Analyzed every wandb metric — reward terms, per-task success, termination types, alignment errors. Key insight: dense rewards (not sparse) do all the heavy lifting. std=1.0 provides weak gradient everywhere (0.74 reward at 30cm from goal).

3. **Research direction development:** Evolved from "speed up OmniReset" → "contact-mode curriculum as universal manipulation curriculum." Created this research plan document.

4. **Curriculum vs multi-reset analysis:** Compared RFCL (curriculum, demo-based, per-task tuning) with OmniReset (multi-reset, uniform sampling, task-agnostic). Proposed adaptive multi-reset as hybrid.

5. **RFCL deep dive:** Confirmed RFCL requires significant per-task tuning (δ, φ, discount). Scaling compute partially but not fully removes this.

6. **Contact-mode curriculum design:** For tasks with non-monotonic difficulty (hardest part in the middle), designed adaptive probability balancing that shifts samples to unsolved modes. Provided code for contact-mode segmentation from demos and adaptive sampling.

7. **LLM reward design research:** Analyzed Eureka, Text2Reward, L2R, CARD, LEARN-Opt, RoboCLIP. Identified fundamental limitations (no physics reasoning, compute cost, reward hacking). Proposed convergence path (Path C) where LLMs provide semantic decomposition and our framework provides curriculum + exploration.

### Key Decisions

- **Main contribution:** Contact-mode adaptive curriculum (not LLM rewards, not just "faster OmniReset")
- **Testbed:** Isaac Lab (OmniReset's framework), cube insertion as primary benchmark
- **Baseline comparisons:** OmniReset (uniform sampling), RFCL (demo-based curriculum)
- **Implementation order:** Adaptive probabilities → std annealing → auto mode detection → SAC/HER → multi-task generalization
- **Demo stance:** Value demos for safety/predictability, but research contribution should work without them

### Open Items

- [ ] Check training run results (pod likely stopped after ~48h, need to check wandb)
- [ ] Implement Experiment 1.1 (adaptive reset probabilities) — ~10 lines of code
- [ ] Run sparse reward ablation (Experiment 1.3) to quantify dense reward dependency
- [ ] Port RFCL to Isaac Lab for baseline comparison (Experiment 1.4)
- [ ] Investigate staggered env initialization (Appendix C) as compute optimization

---

## Appendix F: Related Paper Analysis — Real2Sim + RL for Manipulation

Six papers most relevant to our North Star pipeline, analyzed in depth (Feb 20, 2026).

### How They Map to Our Pipeline

```
Step 1: Scene Capture     → GRS (RGBD→sim), RL-GSBridge (3DGS), RialTo (phone scan + GUI)
Step 2: Contact Modes     → OUR CONTRIBUTION (no existing paper does this automatically)
Step 3: Reward/Reset Gen  → IKER (VLM keypoint rewards), OUR CONTRIBUTION (mode-specific)
Step 4: RL Training       → RialTo (PPO), RLinf-Co (PPO/ReinFlow + VLA), OmniReset (baseline)
Step 5: Sim2Real Transfer → RL-GSBridge (3DGS rendering), RialTo (teacher-student), SaRC (co-training)
Step 6: Real Deployment   → All papers validate to varying degrees
```

**The unique gap across all 6 papers is Step 2: automatically identifying contact modes from the reconstructed scene and using them to structure the RL curriculum.**

### Paper 1: RialTo — "Reconciling Reality through Simulation" (ICRA 2024)

**Pipeline:** Phone scan → digital twin GUI (~15 min) → RL in Isaac Sim 2022.2.1 + Orbit (4096 parallel envs, PPO) → teacher-student distillation (state teacher → point cloud student) → deploy on Franka Panda.

**Key contribution — Inverse Distillation:** Run a BC policy (trained on 15 real demos) in the digital twin to collect privileged state-action data. This bridges perception→state gap without manual correspondence. Then train RL teacher from that state-based initialization.

**Results:** 91% vs 25% (BC) with pose randomization; 75% under physical disturbances where BC gets 5%. ~2 days wall-clock end-to-end. Tasks: book on shelf, plate on rack, mug on shelf, open drawer/cabinet/toaster, in-the-wild kitchen tasks.

**Why it matters for us:**
- Closest existing system to our North Star pipeline
- Validates that phone-scanned digital twins + RL in Isaac Sim dramatically outperform pure BC
- Sparse rewards work when you have good initialization (inverse distillation)
- Teacher-student pipeline is the standard sim2real bridge

**Gap we fill:** RialTo has no contact modes or curriculum. Flat reset distribution + sparse rewards. Tasks are mostly quasistatic (pick-and-place, door opening) — would likely fail on contact-rich insertion. Adding contact-mode curriculum to RialTo's pipeline = our Path C+.

### Paper 2: IKER — "VLM-Generated Iterative Keypoint Rewards" (2025)

**Pipeline:** 4 RealSense cameras → BundleSDF mesh reconstruction → GPT-4o generates keypoint-based rewards → PPO in IsaacGym (128 envs, ~5 min/subtask) → deploy on XArm7. VLM replans after each execution (iterative replanning, not reward refinement).

**Key contribution — Keypoint rewards:** VLM outputs target keypoint positions; reward is templated as `f = α_dist·r_dist + α_dir·r_dir + α_align·r_align + α_bonus·r_bonus + α_penalty·r_penalty`. VLM specifies *what* keypoints should achieve, not the reward formula itself. Grasping handled by AnyGrasp (real) / heuristic (sim).

**Results:** 70-85% real-world success on shoe placement/pushing. Beats VoxPoser on multi-step (4/10 vs 0/10 on step 3). DR critical (0.2→0.7 real with DR on pushing).

**Why it matters for us:**
- Best existing example of LLM-generated rewards for real2sim manipulation
- Keypoint representation gives VLM spatial grounding (better than Eureka's text-only)
- Iterative replanning sidesteps the feedback loop problem (no reward refinement during training)
- Fast training (~5 min/subtask) makes the replan loop practical

**Gap we fill:** IKER decomposes tasks heuristically. No contact-mode notion. Our taxonomy gives IKER's VLM a principled decomposition: instead of "what should robot do next?", ask "generate reward for transitioning from prehensile contact to object-fixture contact." This is Path B/C convergence.

### Paper 3: RL-GSBridge — "3DGS for Real2Sim2Real" (2024)

**Pipeline:** 1-2 min phone video → COLMAP + SAM-track + openMVS → 3DGS with soft mesh binding → PyBullet physics + photorealistic GS rendering → SAC with baseline controller (SACwB) → zero-shot transfer to KUKA iiwa.

**Key contribution — Soft mesh binding:** Gaussians float along mesh face normals (α_n ∈ [-1,1]) instead of being rigidly bound to mesh surfaces. Compensates for mesh reconstruction inaccuracy. Banana: +9dB PSNR improvement over hard binding.

**Results:** Average 6.6% sim-to-real drop on grasping (vs 80% drop with mesh rendering). 93-100% real grasping success. Zero-shot — no fine-tuning.

**Why it matters for us:**
- Solves the visual domain gap — GS rendering eliminates need for visual domain randomization
- Consumer-grade capture (phone video) → photorealistic sim
- Proves that visual fidelity in sim can essentially close the perception gap

**Gap we fill:** Only tests simple grasping/pick-and-place on PyBullet. Physics insufficient for contact-rich tasks. Combining their 3DGS rendering with Isaac Sim GPU physics + our contact-mode curriculum would be powerful.

### Paper 4: SaRC — "Sim-and-Real Co-Training" (2025, NVIDIA/UT Austin)

**Pipeline:** Pure IL (diffusion policy, behavioral cloning). Mix 10K sim demos (MimicGen) with 20-50 real demos. Loss: `L = 0.99·L_sim + 0.01·L_real`. Uses "digital cousins" — imperfect sims sharing task structure but not exact geometry/physics.

**Key contribution — Digital cousins:** Sim environments need only preserve: (1) same robot/action space, (2) same task goal, (3) same object categories, (4) same fixture categories. Everything else can differ. Even fisheye-vs-pinhole camera mismatch works.

**Results:** 45.3% → 83.2% average (+38pp) across 6 tasks on Franka Panda + GR-1 humanoid. CupPnP unseen objects: 10% → 80%.

**Why it matters for us:**
- Proves digital twins don't need to be perfect — lowers the bar for real2sim scene creation
- Task-level fidelity > visual fidelity for policy transfer
- MimicGen for trajectory augmentation is a useful data generation tool

**Gap we fill:** SaRC is pure IL — no RL. Works for pick-and-place but fails on contact-rich insertion (their stated limitation). Our RL + contact-mode curriculum handles exactly the cases where SaRC's IL breaks down. Natural combination: SaRC-style co-training for easy phases (reaching, grasping), our RL curriculum for hard phases (insertion, assembly).

### Paper 5: RLinf-Co — "RL-Based Sim-Real Co-Training for VLAs" (2026)

**Pipeline:** Two-stage: (1) SFT warm-start on mixed sim+real data, (2) RL fine-tuning in ManiSkill with real-data regularization (`L = L_RL + β·L_SFT(θ; D_real)`). Tests on OpenVLA and Pi0.5.

**Key contribution — Real-regularized RL:** The β·L_SFT term on real data during RL prevents catastrophic forgetting. Without it: 81.4% → 40.3% on Pick and Place. RL without SFT init: near-trivial after 3M steps. Both stages required.

**Results:** +24% over SFT co-training for OpenVLA, +20% for Pi0.5 (averaged across 4 tasks). Open Drawer: 0% (SFT) → 65% (RL-Co) for Pi0.5. 10x data efficiency — 20 demos matches 200-demo baselines.

**Why it matters for us:**
- Directly demonstrates RL in sim beats pure imitation for VLAs, especially on contact-rich tasks
- Two-stage design (SFT init → RL fine-tune) + real-data regularization = right recipe for our pipeline
- Non-photorealistic sim still transfers — task-level fidelity matters more than visual fidelity

**Gap we fill:** Uses flat RL (no curriculum, no contact-mode decomposition). Open Drawer only reaches 65% with Pi0.5. Adding contact-mode curriculum (near-goal resets for contact phase, wider resets for approach) could push significantly higher.

### Paper 6: GRS — "Generating Robotic Simulation Tasks from Real-World Images" (2024)

**Pipeline:** Single RGBD (ZED 2, 1080p) → SAM2 segmentation → GPT-4o object matching (F1=0.89) → LLM generates CLIPort/PyBullet task code → iterative validation with oracle policy + router → outputs runnable sim task.

**Key contribution — Closed-loop validation:** LLM router decides whether to fix sim code or test code. Oracle policy validates solvability. 0.71 average reward (vs 0.47 LLM-only). Scales to 150K Objaverse assets.

**Results:** Oracle-validated sim tasks from 10 real tabletop scenes (~15 objects each). GPT-4o + image+text matching F1=0.89 (vs CLIP F1=0.76). No RL training or sim2real transfer demonstrated.

**Why it matters for us:**
- Automates Step 1 of our North Star pipeline (real image → sim environment)
- VLM-based object correspondence is directly applicable
- Iterative validation loop ensures generated sims are solvable

**Gap we fill:** GRS only generates tasks, not policies. CLIPort/PyBullet (not Isaac Sim), no physics parameter estimation, no contact-rich tasks, no sim2real. GRS is Step 1; our work is Steps 2-5.

### Summary Table

| Paper | Sim Platform | RL? | Contact-Rich? | Real2Sim Method | # Real Demos | Our Advantage |
|-------|-------------|-----|---------------|-----------------|-------------|---------------|
| RialTo | Isaac Sim + Orbit | PPO (4096 envs) | No (quasistatic) | Phone scan + GUI | 15 | Contact-mode curriculum |
| IKER | IsaacGym | PPO (128 envs) | Pushing only | BundleSDF + FoundationPose | 0 | Principled task decomposition |
| RL-GSBridge | PyBullet | SAC | No (grasping) | 3DGS + soft mesh binding | 0 | Better physics (Isaac Sim) + curriculum |
| SaRC | RoboCasa/MuJoCo | No (pure IL) | No (pick-place) | Digital cousins (imperfect OK) | 20-50 | RL for hard contact phases |
| RLinf-Co | ManiSkill | PPO/ReinFlow | Drawer only | Pre-built envs | 20-50 | Contact-mode curriculum for harder tasks |
| GRS | CLIPort/PyBullet | No (oracle only) | No (tabletop) | SAM2 + GPT-4o | 0 | Full training pipeline, not just scene gen |
