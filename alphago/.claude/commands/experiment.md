Run an AlphaZero experiment based on the user's description: $ARGUMENTS

## Instructions

If you are using GPUs, refer to claude.md in /Users/richardzhang/workspace/personal-research

You are running an experiment for the AlphaZero research testbed. Follow these steps exactly:

### 1. Create experiment directory
Create a timestamped directory under `experiments/`:
```
experiments/YYYYMMDD_HHMMSS_<short-name>/
├── config.json      # exact parameters used
├── data/            # training metrics
├── figures/         # plots
└── report.md        # main artifact
```

### 2. Write and run the experiment
Based on the user's description, determine what config changes to make from baseline. Write a self-contained experiment script that:
- Uses `alpha_go` imports
- Configures the AlphaZeroConfig with the appropriate parameter changes
- Runs the training pipeline (or a subset if the experiment is about a specific component)
- Saves per-iteration metrics to `data/metrics.json`
- Generates matplotlib plots to `figures/` (loss curves, win rate progression, etc.)
- All outputs go to the experiment directory

### 3. Generate report.md
After the experiment completes, write a `report.md` in the experiment directory with:

```markdown
# Experiment: <title>

## Hypothesis
What we're testing and why.

## Setup
- **Baseline**: default config values
- **Changes**: what was modified and to what values
- **Config diff**: show only the changed parameters

## Results
Key metrics with references to figures:
- Final vs-random win rate
- Loss curves (see figures/loss.png)
- Arena win rates over iterations (see figures/arena.png)

## Analysis
What the results mean. Was the hypothesis supported?

## Next Steps
Suggested follow-up experiments based on what we learned.
```

### 4. Report to user
After everything completes, show the user:
- Path to the experiment directory
- Key results summary (1-2 lines)
- Pointer to the full report.md

### Important notes
- Always `cd alphago` before running anything
- Activate the venv: `source .venv/bin/activate`
- Run scripts from the alphago directory
- If the experiment would take more than 10 minutes, suggest reducing iterations/games first
- Save ALL plots with tight_layout and reasonable DPI (150)
