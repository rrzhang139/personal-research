import wandb
import os

# Set API key from env
wandb_key = os.getenv("WANDB_API_KEY")
if not wandb_key:
    with open("/workspace/.env") as f:
        for line in f:
            if line.startswith("WANDB_API_KEY="):
                wandb_key = line.strip().split("=", 1)[1].strip('"').strip("'")
                os.environ["WANDB_API_KEY"] = wandb_key
                break

# Initialize run
run = wandb.init(project="residual-rl", name="upload_pose_demos", job_type="dataset")

# Create artifact
artifact = wandb.Artifact(
    name="PegInsertionSide-v1-pose-demos",
    type="dataset",
    description="ManiSkill3 PegInsertionSide-v1 motion planning demos with pd_ee_delta_pose control"
)

# Add the pose demo file
demo_path = "/workspace/datasets/maniskill_demos/PegInsertionSide-v1/motionplanning/trajectory.state.pd_ee_delta_pose.physx_cpu.h5"
if os.path.exists(demo_path):
    artifact.add_file(demo_path, name="trajectory.state.pd_ee_delta_pose.physx_cpu.h5")
    print(f"Added {demo_path}")
    print(f"File size: {os.path.getsize(demo_path) / 1024 / 1024:.1f} MB")
else:
    print(f"ERROR: Demo file not found at {demo_path}")

# Log artifact
run.log_artifact(artifact)
print("Artifact logged successfully")

# Finish run
run.finish()
print("Done!")
