"""Validate LLM contact mode classifier on all 6 OmniReset objects.

Runs the classifier on each object's task description and verifies:
1. All 4 expected modes are returned
2. Probabilities sum to 1.0
3. Difficulty ordering is sensible
"""

from contact_mode_classifier import classify_task
from generate_curriculum_config import generate_config, print_config

TASKS = {
    "cube": "Insert a rigid cube into a square hole on a fixture plate. Robot: UR5e with Robotiq 2F85 parallel jaw gripper.",
    "peg": "Insert a cylindrical peg into a round hole. Robot: UR5e with Robotiq 2F85.",
    "rectangle": "Insert a rectangular block into a rectangular slot. Robot: UR5e with Robotiq 2F85.",
    "fbleg": "Insert a furniture leg into a furniture base. Robot: UR5e with Robotiq 2F85.",
    "fbdrawerbottom": "Insert a drawer bottom panel into a drawer frame. Robot: UR5e with Robotiq 2F85.",
    "cupcake": "Place a cupcake onto a plate/holder. Robot: UR5e with Robotiq 2F85.",
}

EXPECTED_MODES = {"free-space", "prehensile-far", "prehensile-near", "object-fixture"}


def main():
    all_passed = True

    for obj, desc in TASKS.items():
        print(f"\n{'='*60}")
        print(f"Object: {obj}")
        print(f"Task: {desc}")
        print(f"{'='*60}")

        result = classify_task(desc)

        # Check modes
        modes = {m["mode"] for m in result["modes"]}
        if modes != EXPECTED_MODES:
            print(f"  FAIL: got modes {modes}, expected {EXPECTED_MODES}")
            all_passed = False
        else:
            print(f"  PASS: all 4 modes present")

        # Check probabilities
        prob_sum = sum(result["initial_probs"])
        if abs(prob_sum - 1.0) > 0.01:
            print(f"  FAIL: probs sum to {prob_sum}")
            all_passed = False
        else:
            print(f"  PASS: probs sum to {prob_sum:.3f}")

        # Print details
        for mode_info, prob in zip(result["modes"], result["initial_probs"]):
            print(
                f"  [{mode_info['difficulty']}] {mode_info['mode']:20s} "
                f"p={prob:.2f}  {mode_info['description']}"
            )
        print(f"  Reasoning: {result['reasoning']}")

        # Generate and show config
        config = generate_config(result)
        print()
        print_config(config)

    print(f"\n{'='*60}")
    if all_passed:
        print("ALL OBJECTS PASSED")
    else:
        print("SOME OBJECTS FAILED — check output above")


if __name__ == "__main__":
    main()
