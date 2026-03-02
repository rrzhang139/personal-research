# CASHER vs Contact-Mode Curriculum: Analysis

## What CASHER Actually Makes Trivial (and What It Doesn't)

### Scene Creation: Nearly Trivial
- **Effort:** 3-5 min smartphone scan (Polycam/ARCode) + ~13 min GUI articulation in Isaac Sim
- **Who:** Non-experts worldwide. Crowdsourced. No robotics knowledge needed.
- **Result:** 56 kitchen scenes for bowl-in-sink, 36 for box-in-shelf
- **Verdict: Scene creation is essentially solved.** The bottleneck is entirely gone. Anyone with a phone can contribute a training environment.

### Task Definition: NOT Trivial — Hard-Coded Per Task Family
CASHER's tasks use hand-written sparse reward conditions:

```
Bowl in sink:    ||sink_site - object_site||₂ < 0.25  AND  object_upright  AND  gripper_open
Box in cabinet:  cabinet_z > object_z               AND  object_upright
Open cabinet:    cabinet_joint > 0.65                AND  gripper_open
```

These are **manually designed per task type**. Every new task family (insertion, stacking, tool use) would need a new reward function. CASHER has exactly 3 task types across the whole paper.

### Reward Engineering: Minimal but Rigid
14 discretized actions (±3cm position, ±0.02rad orientation, gripper open/close). Sparse binary reward. Simple — but this simplicity comes from the tasks being simple (pick-and-place into a container). For contact-rich tasks (5mm insertion tolerance, 0.025rad alignment), these coarse actions and simple rewards wouldn't work.

### Behavior Generation: Initially Manual, Then Semi-Automated
- First K=10 environments: 10 human teleoperated demos each (100 demos total)
- Subsequent batches: generalist policy generates demos autonomously
- Fallback: when policy success < threshold, humans provide demos again
- **Not fully automated.** The human is always in the loop as a fallback.

---

## Comparison to Your Contact-Mode Research Plan

| Dimension | CASHER | Your Contact-Mode Approach |
|-----------|--------|---------------------------|
| **Scene creation** | Crowdsourced phone scans (solved) | Not addressed (orthogonal — could use CASHER's pipeline) |
| **Task specification** | 3 hand-coded reward functions | Auto-discovered contact-mode graph from physics |
| **Reward engineering** | Sparse but task-specific conditions | Sparse + HER (truly task-agnostic) |
| **Curriculum** | None — flat RL from demos | Adaptive contact-mode curriculum (core contribution) |
| **Demos required** | 10 per env (100+ total) | 0 (or 1-5 optional for bootstrap) |
| **Exploration** | Demo-bootstrapped PPO | Contact-mode resets + adaptive sampling |
| **Generality across tasks** | 3 task types tested | Aims for any manipulation task via mode graph |
| **Generality across scenes** | 56 scenes, same task | Not tested yet (but contact modes are scene-invariant) |
| **Contact-rich tasks** | No (pick-place only, coarse actions) | Yes (designed for 5mm insertion, drawer assembly) |
| **Scaling axis** | More scenes (horizontal) | More task complexity (vertical) |

**The key insight:** CASHER and your work are solving **orthogonal problems**.

- **CASHER scales horizontally** — same task across many environments. The flywheel is: more scenes → better generalist → fewer demos needed per new scene.
- **Your work scales vertically** — harder tasks without demos or reward engineering. The flywheel is: auto-discover modes → adaptive curriculum → sparse reward + HER → no per-task tuning.

They're complementary, not competing. CASHER's scene pipeline (Step 1-2) feeds into your curriculum pipeline (Step 3-5).

---

## The Real Question: Can AI Agents Directly Write Sim Code?

This is the Gen2Sim / GRS approach — skip the structured abstractions, have an LLM generate the entire simulation (scene, task, reward, reset) as code.

### What Exists Today

**GRS (CVPR 2025):** Single RGBD image → GPT-4o identifies objects → matches to Objaverse assets → generates CLIPort/PyBullet task code. F1=0.89 on object matching. But: no RL training, no sim2real, no contact-rich tasks. Oracle policy validates solvability only.

**Gen2Sim (ICRA 2024):** GPT-4 generates articulated object descriptions + task code for RLBench. Can create novel tasks (e.g., "close the laptop"). But: uses pre-existing robot + asset library, generated tasks are still pick-place level.

**Eureka (ICLR 2024):** GPT-4 reads environment source code → generates reward functions as Python code → evolutionary optimization over RL training runs. Achieved pen spinning with Shadow Hand (human experts couldn't design this reward). But: requires full env source code as context, hundreds of GPU-hours per task for the evolutionary search.

### Can Agents Reach Full Generality?

**Short answer: Not yet, and the bottleneck isn't the LLM — it's the feedback loop.**

Here's why:

**1. The "sim code" an agent would need to write is massive and coupled.**

A complete Isaac Lab task definition includes:
- Scene graph (USD assets, positions, materials, physics properties)
- Robot configuration (URDF, joint limits, controller gains)
- Reset state generators (grasp sampling, collision-free poses)
- Observation space (which quantities, normalization, history)
- Action space (OSC params, action scaling, clipping)
- Reward function (dense terms, weights, annealing schedules)
- Termination conditions (success, timeout, safety violations)
- Domain randomization (which params, distributions)
- Curriculum logic (when to advance, how to sample)

That's ~2000 lines of tightly coupled code. An LLM can generate each piece, but the **interactions** between pieces are what matter. A reward that works with one action space fails with another. Reset states that are valid for one controller are infeasible for another.

**2. Verification requires RL training — the slowest possible feedback loop.**

When an LLM generates a reward function, the only way to know if it works is to train an RL agent (~hours). Unlike code compilation (seconds) or unit tests (minutes), RL training gives delayed, noisy feedback. Eureka handles this with evolutionary search, but that's 5 iterations × 5 candidates × hours per run = days per task.

**LEARN-Opt's key finding:** Smaller LLMs (GPT-4.1-nano) match larger models for reward generation. The bottleneck isn't reasoning capability — it's the inability to predict how reward → RL dynamics → behavior without actually running the simulation.

**3. Physics reasoning is fundamentally different from code reasoning.**

LLMs can generate syntactically correct reward code, but they can't reason about:
- Why `exp(-dist/1.0)` saturates at 30cm (gives 0.74 reward regardless of goal proximity)
- Why a 14-action discretization is too coarse for 5mm insertion
- Why contact-mode transitions create exploration barriers that no reward can overcome
- Why temporal homogeneity in parallel envs reduces gradient quality

These require understanding continuous dynamics, not discrete code patterns.

**4. The abstraction gap between "task description" and "sim code" is too large.**

```
"Pick up the cup and place it on the shelf"
        ↓ (what the LLM must bridge)
2000 lines of Isaac Lab Python with specific:
  - grasp approach vectors for this cup geometry
  - shelf collision margins for this shelf depth
  - controller gains tuned for this mass
  - reset distributions that cover the reachable workspace
```

This is like asking an LLM to write a full compiler from a one-sentence spec. It can get the structure right, but the details require domain expertise that comes from running experiments, not from training data.

### So What's the Alternative?

**Structured abstractions (your contact-mode approach) are the right middle ground.**

Instead of LLM → raw sim code (too hard) or human → raw sim code (too slow), use:

```
LLM → contact-mode decomposition → structured curriculum API → sim code
```

The LLM does what it's good at (semantic decomposition: "this task has a grasp phase and an insertion phase"). The structured framework does what physics requires (generate reset states, adaptive sampling, HER). Neither alone is sufficient.

**This is exactly your Path C from the research plan:**
- LLMs propose contact-mode decomposition from task description
- Your taxonomy framework generates resets + curriculum from modes
- LLMs generate per-mode reward functions (much simpler than full-task rewards)
- Adaptive sampling + HER handles exploration (the hard part)

### Why Contact Modes Are Better Than Raw Code Generation

| Property | Raw Code Gen (Eureka/GRS) | Contact-Mode Abstraction |
|----------|--------------------------|--------------------------|
| **Composability** | Each task is a monolithic code block | Modes compose: any task is a graph of modes |
| **Verifiability** | Must train full RL to verify | Each mode can be verified independently |
| **Transfer** | Nothing transfers between tasks | Mode graphs transfer (insertion = insertion, regardless of object) |
| **Failure diagnosis** | "Training failed" — which part? | "Mode 2→3 transition has 0% success" — targeted fix |
| **LLM compatibility** | LLM must reason about full dynamics | LLM only reasons about "what contacts matter" |
| **Compute cost** | Hours per evaluation iteration | Structured curriculum + SAC is single-run |

### When Would Raw Code Gen Win?

If/when we get:
1. **Foundation models trained on (task, reward, training_curve, performance) data** — learning the mapping directly instead of reasoning about it
2. **Differentiable simulators** that let the LLM backpropagate through sim dynamics
3. **Orders of magnitude faster RL** that makes the evolutionary feedback loop cheap

None of these exist today. (1) requires a massive dataset of RL experiments that nobody has collected. (2) exists for simple systems (DiffTaichi) but not for contact-rich manipulation. (3) is what your contact-mode curriculum is trying to achieve.

---

## Bottom Line

**CASHER makes scene creation trivial but leaves task/reward/curriculum untouched.** It scales horizontally (more scenes for the same task) but can't handle harder tasks. Your contact-mode approach is the missing vertical axis — it makes task decomposition, curriculum design, and reward engineering automatic for increasingly complex manipulation.

**AI agents writing raw sim code is the dream but not the near-term answer.** The feedback loop is too slow (hours per evaluation), and the abstraction gap between language and physics is too large. Structured abstractions like contact modes are the right bridge — they constrain the LLM's output space to something verifiable and composable, while preserving the generality that hand-designed categories (OmniReset's 4 reset types) lack.

**The most exciting path is combining them:** CASHER's crowdsourced scenes + your contact-mode curriculum + LLM semantic decomposition. Nobody has built this pipeline. That's the research opportunity.
