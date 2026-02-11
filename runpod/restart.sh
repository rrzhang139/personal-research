#!/bin/bash
# restart.sh — Run EVERY TIME after pod stop→start
# /workspace survived (venv, packages, models, code all intact)
# Container disk was wiped (git config, claude binary, uv binary gone)
set -e

echo "=== Pod Restart Recovery ==="

# ---- Restore everything from /workspace ----
source /workspace/.bashrc_pod

# ---- Reinstall uv (binary was on container disk) ----
if ! command -v uv &> /dev/null; then
    echo "--- Reinstalling uv ---"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# ---- Reinstall Claude Code (binary was on container disk) ----
if ! command -v claude &> /dev/null; then
    echo "--- Reinstalling Claude Code ---"
    curl -fsSL https://claude.ai/install.sh | bash
fi

# ---- Reinstall tmux if needed ----
if ! command -v tmux &> /dev/null; then
    apt-get update -qq && apt-get install -qq -y tmux > /dev/null 2>&1
fi

echo ""
echo "=== Ready ==="
echo "Venv active. Auth restored. ANTHROPIC_API_KEY set."
echo "All packages still installed (from /workspace/venv)."
echo ""
echo "Run:"
echo "  tmux new -s work"
echo "  claude"
