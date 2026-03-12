# Personal Research — Shared Infrastructure

Shared GPU/SSH/RunPod infrastructure for research projects. Each project lives in its own repo.

## Projects

| Project | Repo | Description |
|---------|------|-------------|
| AlphaZero | [rrzhang139/alphago](https://github.com/rrzhang139/alphago) | AlphaZero from scratch |
| Quake3 WM | [rrzhang139/quake3-worldmodel](https://github.com/rrzhang139/quake3-worldmodel) | World model for Q3 Arena |
| Residual RL | [rrzhang139/residual-rl](https://github.com/rrzhang139/residual-rl) | Policy Decorator |
| UWLab | [rrzhang139/uwlab-omnireset](https://github.com/rrzhang139/uwlab-omnireset) | OmniReset for UWLab |

## What's Here

- `providers/` — Provider docs (RunPod pricing, setup, SSH patterns)
- `runpod/` — Pod lifecycle scripts (setup.sh, restart.sh, offboard.sh)
- `CLAUDE.md` — GPU cost philosophy, SSH conventions, W&B artifact mandate
- `.claude/commands/` — Slash commands for pod management
