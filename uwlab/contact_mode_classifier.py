"""LLM-based contact mode classifier for OmniReset curriculum learning.

Uses Claude to classify which contact modes a manipulation task requires
and in what difficulty order, then outputs initial curriculum probabilities.
"""

import json

import anthropic

MODES = ["free-space", "prehensile-far", "prehensile-near", "object-fixture"]

SYSTEM_PROMPT = """You are a robotics manipulation expert. Given a task description,
identify which contact modes the robot must pass through to complete the task.

Available modes (these are the ONLY modes you can use):
- free-space: Robot reaching toward object, no contact with target object yet
- prehensile-far: Robot grasping object, object is far from goal (random position in workspace)
- prehensile-near: Robot grasping object, object is near its resting/start position on table
- object-fixture: Object partially inserted into goal fixture, robot still grasping

For assembly/insertion tasks, the typical progression is:
1. free-space (easiest — just reaching)
2. prehensile-near (grasping object on table)
3. prehensile-far (grasping object at random positions — harder because object could be anywhere)
4. object-fixture (hardest — requires precise alignment for insertion)

Output ONLY valid JSON (no markdown, no explanation outside JSON):
{
  "modes": [
    {"mode": "<name>", "difficulty": <1-4, 1=easiest>, "description": "<what happens in this phase>"}
  ],
  "initial_probs": [<float>, ...],
  "reasoning": "<brief explanation of why this ordering>"
}

Rules:
- All 4 modes must be included for assembly/insertion tasks
- Order modes from easiest to hardest in the "modes" array
- initial_probs must sum to 1.0 and be in the same order as modes
- Give more probability weight to easier modes (curriculum starts easy)
- Mode names must exactly match the available list above"""


def classify_task(task_description: str) -> dict:
    """Classify contact modes for a manipulation task using Claude.

    Args:
        task_description: Natural language description of the manipulation task.

    Returns:
        Dict with keys: modes, initial_probs, reasoning
    """
    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": task_description}],
    )
    text = response.content[0].text

    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.startswith("```")]
        text = "\n".join(lines)

    result = json.loads(text)

    # Validate
    mode_names = {m["mode"] for m in result["modes"]}
    for mode in mode_names:
        if mode not in MODES:
            raise ValueError(f"Unknown mode '{mode}'. Must be one of {MODES}")

    probs = result["initial_probs"]
    if abs(sum(probs) - 1.0) > 0.01:
        raise ValueError(f"Probabilities sum to {sum(probs)}, expected 1.0")

    if len(probs) != len(result["modes"]):
        raise ValueError(
            f"Got {len(probs)} probs but {len(result['modes'])} modes"
        )

    return result
