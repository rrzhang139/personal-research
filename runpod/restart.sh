#!/bin/bash
# restart.sh — Run EVERY TIME after pod stop→start
# /workspace survived, but container disk (git config, claude binary) was wiped
set -e

echo "=== Pod Restart Recovery ==="

# Source everything from /workspace
source /workspace/.bashrc_pod

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
echo "Venv active, auth restored, Claude Code installed."
echo ""
echo "Run:"
echo "  tmux new -s work"
echo "  claude"
