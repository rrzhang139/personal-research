#!/usr/bin/env python3
"""Patch MultiResetManager in events.py to support adaptive reset probabilities.

Experiment 1.1 from RESEARCH_PLAN.md:
  Instead of fixed [0.25, 0.25, 0.25, 0.25], shift sampling toward harder tasks.

Control via environment variables (default = OFF, identical to original):
  OMNIRESET_ADAPTIVE=1          Enable adaptive sampling
  OMNIRESET_TEMPERATURE=0.5     Softmax temperature (lower = more aggressive)
  OMNIRESET_MIN_PROB=0.05       Floor probability per task (prevents forgetting)

Usage:
  python scripts/patch_adaptive.py          # Apply patch
  python scripts/patch_adaptive.py --revert # Revert to original
"""
import sys
import shutil
from pathlib import Path

EVENTS_FILE = Path(
    "/workspace/code/personal-research/uwlab/UWLab/source/uwlab_tasks/"
    "uwlab_tasks/manager_based/manipulation/reset_states/mdp/events.py"
)
BACKUP_FILE = EVENTS_FILE.with_suffix(".py.bak_adaptive")
PATCH_MARKER = "# --- ADAPTIVE PATCH (Experiment 1.1) ---"


def patch():
    if not EVENTS_FILE.exists():
        print(f"ERROR: {EVENTS_FILE} not found")
        sys.exit(1)

    content = EVENTS_FILE.read_text()

    if PATCH_MARKER in content:
        print("Patch already applied. Use --revert to undo.")
        return

    # Ensure os is imported (should already be, but safety check)
    if "\nimport os\n" not in content and "\nimport os " not in content:
        # Add after the last 'import' line in the file header
        lines = content.split("\n")
        last_import_idx = 0
        for i, line in enumerate(lines):
            if line.startswith("import ") or line.startswith("from "):
                last_import_idx = i
        lines.insert(last_import_idx + 1, "import os")
        content = "\n".join(lines)
        print("Added 'import os'")

    # Find the target sampling line in MultiResetManager.__call__
    target = "dataset_indices = torch.multinomial(self.probs, len(env_ids), replacement=True)"

    if target not in content:
        print(f"ERROR: Could not find target line:")
        print(f"  {target}")
        print("The UWLab code may have changed. Manual patching required.")
        sys.exit(1)

    # Detect indentation of the target line
    indent = ""
    for line in content.split("\n"):
        stripped = line.lstrip()
        if stripped.startswith("dataset_indices = torch.multinomial(self.probs"):
            indent = line[: len(line) - len(stripped)]
            break

    if not indent:
        print("ERROR: Could not detect indentation")
        sys.exit(1)

    # Build replacement block
    replacement = f"""{indent}{PATCH_MARKER}
{indent}# Shift sampling probability toward harder (less successful) tasks.
{indent}# When OMNIRESET_ADAPTIVE=0 (default), this block is skipped entirely.
{indent}_adaptive_on = os.environ.get("OMNIRESET_ADAPTIVE", "0") == "1"
{indent}if _adaptive_on and hasattr(self, 'success_monitor') and self.success_monitor.success_size.sum() > 0:
{indent}    _temp = float(os.environ.get("OMNIRESET_TEMPERATURE", "0.5"))
{indent}    _min_p = float(os.environ.get("OMNIRESET_MIN_PROB", "0.05"))
{indent}    _sr = self.success_monitor.get_success_rate()
{indent}    _fr = (1.0 - _sr).clamp(min=1e-6)
{indent}    _aprobs = torch.softmax(_fr / _temp, dim=0).clamp(min=_min_p)
{indent}    _aprobs = _aprobs / _aprobs.sum()
{indent}    self.probs = _aprobs
{indent}dataset_indices = torch.multinomial(self.probs, len(env_ids), replacement=True)
{indent}# --- END ADAPTIVE PATCH ---"""

    # Backup original
    shutil.copy2(EVENTS_FILE, BACKUP_FILE)
    print(f"Backup saved: {BACKUP_FILE}")

    # Apply patch (replace only the first occurrence)
    full_target = f"{indent}{target}"
    patched = content.replace(full_target, replacement, 1)

    if patched == content:
        print("ERROR: Replacement had no effect (indentation mismatch?)")
        print(f"  Expected indent: {repr(indent)}")
        sys.exit(1)

    EVENTS_FILE.write_text(patched)

    # Verify
    verify = EVENTS_FILE.read_text()
    if PATCH_MARKER in verify and "OMNIRESET_ADAPTIVE" in verify:
        print("SUCCESS: Patch applied!")
        print(f"  File: {EVENTS_FILE}")
        print("  Enable:  export OMNIRESET_ADAPTIVE=1")
        print("  Disable: export OMNIRESET_ADAPTIVE=0 (default)")
    else:
        shutil.copy2(BACKUP_FILE, EVENTS_FILE)
        print("FAILED: Verification failed. Restored original.")
        sys.exit(1)


def revert():
    if not BACKUP_FILE.exists():
        print(f"ERROR: No backup found at {BACKUP_FILE}")
        sys.exit(1)

    shutil.copy2(BACKUP_FILE, EVENTS_FILE)
    print(f"Reverted to original: {EVENTS_FILE}")


if __name__ == "__main__":
    if "--revert" in sys.argv:
        revert()
    else:
        patch()
