"""Generate MultiResetManager config from LLM contact mode classification.

Maps LLM output (mode names + probabilities) to the S3 dataset paths
used by OmniReset's MultiResetManager.
"""

from uwlab_assets import UWLAB_CLOUD_ASSETS_DIR

MODE_TO_DATASET = {
    "free-space": "ObjectAnywhereEEAnywhere",
    "prehensile-far": "ObjectAnywhereEEGrasped",
    "prehensile-near": "ObjectRestingEEGrasped",
    "object-fixture": "ObjectPartiallyAssembledEEGrasped",
}


def generate_config(llm_output: dict) -> dict:
    """Convert LLM mode classification to MultiResetManager config.

    Args:
        llm_output: Output from classify_task() with keys: modes, initial_probs, reasoning

    Returns:
        Dict with base_paths, probs, mode_labels, difficulty_order ready
        for use in EventTermCfg.params.
    """
    base_paths = []
    mode_labels = []
    difficulty_order = []

    for mode_info in llm_output["modes"]:
        mode_name = mode_info["mode"]
        dataset_name = MODE_TO_DATASET[mode_name]
        base_paths.append(
            f"{UWLAB_CLOUD_ASSETS_DIR}/Datasets/Resets/ObjectPairs/{dataset_name}"
        )
        mode_labels.append(mode_name)
        difficulty_order.append(mode_info["difficulty"])

    return {
        "base_paths": base_paths,
        "probs": llm_output["initial_probs"],
        "mode_labels": mode_labels,
        "difficulty_order": difficulty_order,
    }


def print_config(config: dict) -> None:
    """Pretty-print a generated config for inspection."""
    print("MultiResetManager config:")
    print(f"  Modes: {config['mode_labels']}")
    print(f"  Difficulty: {config['difficulty_order']}")
    print(f"  Probs: {config['probs']}")
    print("  Paths:")
    for path in config["base_paths"]:
        print(f"    {path}")
