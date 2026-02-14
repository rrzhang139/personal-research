"""Upload eval video and metadata to wandb."""

import argparse
import os

import wandb


def main():
    parser = argparse.ArgumentParser(description="Upload eval results to wandb")
    parser.add_argument("--video", required=True, help="Path to eval video (.mp4)")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path used for eval")
    parser.add_argument("--run_name", default="eval", help="Name for this eval run")
    parser.add_argument("--project", default=None, help="wandb project (default: $WANDB_PROJECT or omnireset)")
    args = parser.parse_args()

    project = args.project or os.environ.get("WANDB_PROJECT", "omnireset")

    # Extract checkpoint info
    ckpt_basename = os.path.basename(args.checkpoint)
    ckpt_dir = os.path.dirname(args.checkpoint)

    # Try to figure out training run name from checkpoint path
    # e.g., logs/rsl_rl/ur5e_robotiq_2f85_reset_states_agent/2026-02-14_04-14-46/model_400.pt
    training_run = os.path.basename(ckpt_dir) if ckpt_dir else "unknown"

    run = wandb.init(
        project=project,
        name=f"eval-{args.run_name}",
        job_type="eval",
        config={
            "checkpoint": args.checkpoint,
            "checkpoint_name": ckpt_basename,
            "training_run": training_run,
            "task": "OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Play-v0",
            "object": "cube",
            "num_envs": 1,
            "video_length": 200,
        },
    )

    # Log the video
    wandb.log({"eval/rollout": wandb.Video(args.video, fps=30, format="mp4")})

    print(f"Uploaded video to wandb: {run.url}")

    wandb.finish()


if __name__ == "__main__":
    main()
