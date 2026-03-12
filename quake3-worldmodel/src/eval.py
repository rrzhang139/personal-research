"""Evaluate a trained world model by generating rollout videos.

Usage:
    python src/eval.py --checkpoint experiments/run/best.pt --data data/episodes --output experiments/run/eval
"""

import argparse
from pathlib import Path

import torch
import numpy as np
from PIL import Image

from model import make_denoiser
from episode import Episode


def load_model(checkpoint_path, device, **kwargs):
    """Load a trained denoiser from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = make_denoiser(**kwargs).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded checkpoint: epoch={ckpt['epoch']+1}, loss={ckpt['loss']:.4f}")
    return model


def autoregressive_rollout(model, seed_episode, start_t, num_steps, device, num_denoise=3):
    """Generate frames autoregressively from a seed episode.

    Uses frames [start_t - L, start_t) as initial context, then generates
    num_steps frames using the actions from the seed episode.
    """
    L = model.num_context

    # Initialize context buffer from real frames
    context = seed_episode.obs[start_t - L : start_t].unsqueeze(0).to(device)  # (1, L, C, H, W)

    real_frames = []
    pred_frames = []

    for i in range(num_steps):
        t = start_t + i
        if t >= len(seed_episode):
            break

        action = seed_episode.act[t].unsqueeze(0).to(device)  # (1,)

        # Generate next frame
        with torch.no_grad():
            pred = model.sample(context, action, num_steps=num_denoise)  # (1, C, H, W)

        # Store
        real_frame = seed_episode.obs[t + 1]  # (C, H, W)
        real_frames.append(((real_frame + 1) / 2 * 255).clamp(0, 255).byte())
        pred_frames.append(((pred[0].cpu() + 1) / 2 * 255).clamp(0, 255).byte())

        # Update context: drop oldest, append prediction
        context = torch.cat([context[:, 1:], pred.unsqueeze(1)], dim=1)

    return real_frames, pred_frames


def compute_psnr(real: torch.Tensor, pred: torch.Tensor) -> float:
    """PSNR between two uint8 images."""
    mse = ((real.float() - pred.float()) ** 2).mean()
    if mse == 0:
        return float("inf")
    return (10 * torch.log10(255**2 / mse)).item()


def make_comparison_video(real_frames, pred_frames, output_path, fps=10):
    """Create side-by-side comparison video."""
    try:
        import imageio
    except ImportError:
        print("imageio not installed, skipping video")
        return

    frames = []
    for r, p in zip(real_frames, pred_frames):
        # (C, H, W) -> (H, W, C)
        r_img = r.permute(1, 2, 0).numpy()
        p_img = p.permute(1, 2, 0).numpy()
        # Side by side with separator
        sep = np.ones((r_img.shape[0], 2, 3), dtype=np.uint8) * 128
        combined = np.concatenate([r_img, sep, p_img], axis=1)
        frames.append(combined)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(str(output_path), frames, fps=fps, codec="libx264")
    print(f"Saved video: {output_path} ({len(frames)} frames)")


def make_comparison_strip(real_frames, pred_frames, output_path, every_n=1):
    """Create image strip showing real vs predicted frames."""
    indices = list(range(0, len(real_frames), every_n))[:16]

    strips = []
    for idx in indices:
        r = real_frames[idx].permute(1, 2, 0).numpy()
        p = pred_frames[idx].permute(1, 2, 0).numpy()
        sep = np.ones((r.shape[0], 1, 3), dtype=np.uint8) * 200
        strips.append(np.concatenate([r, sep, p], axis=1))

    # Stack vertically with label space
    row_sep = np.ones((2, strips[0].shape[1], 3), dtype=np.uint8) * 255
    full = []
    for s in strips:
        full.append(s)
        full.append(row_sep)
    full = np.concatenate(full[:-1], axis=0)

    Image.fromarray(full).save(output_path)
    print(f"Saved strip: {output_path} ({len(indices)} frames)")


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(
        args.checkpoint, device,
        num_actions=args.num_actions,
        img_size=args.res,
        num_context_frames=args.num_context,
        model_size=args.model_size,
    )

    # Load a few episodes for evaluation
    episode_dir = Path(args.data)
    episode_files = sorted(episode_dir.glob("episode_*.pt"))
    if not episode_files:
        print(f"No episodes found in {args.data}")
        return

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_psnr = []

    for ep_idx in range(min(args.num_episodes, len(episode_files))):
        ep = Episode.load(episode_files[ep_idx])
        if len(ep) < args.num_context + args.rollout_length:
            continue

        # Start from a few steps in (after context)
        start_t = args.num_context + 10
        real_frames, pred_frames = autoregressive_rollout(
            model, ep, start_t, args.rollout_length, device,
            num_denoise=args.num_denoise_steps,
        )

        # Compute PSNR per frame
        psnrs = [compute_psnr(r, p) for r, p in zip(real_frames, pred_frames)]
        avg_psnr = np.mean(psnrs)
        all_psnr.extend(psnrs)
        print(f"Episode {ep_idx}: avg PSNR={avg_psnr:.2f} dB "
              f"(frame 1: {psnrs[0]:.1f}, frame {len(psnrs)}: {psnrs[-1]:.1f})")

        # Save outputs
        make_comparison_strip(
            real_frames, pred_frames,
            out_dir / f"strip_ep{ep_idx}.png",
            every_n=max(1, len(real_frames) // 16),
        )
        make_comparison_video(
            real_frames, pred_frames,
            out_dir / f"rollout_ep{ep_idx}.mp4",
        )

    print(f"\nOverall: avg PSNR={np.mean(all_psnr):.2f} dB over {len(all_psnr)} frames")

    if args.wandb:
        import wandb
        run = wandb.init(project="quake3-worldmodel", entity="rzhang139", job_type="eval")
        run.log({"eval/psnr_mean": np.mean(all_psnr)})
        for f in out_dir.glob("*.mp4"):
            run.log({"eval/video": wandb.Video(str(f))})
        run.finish()


def main():
    parser = argparse.ArgumentParser(description="Evaluate world model")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data", type=str, default="data/episodes")
    parser.add_argument("--output", type=str, default="experiments/eval")
    parser.add_argument("--num_actions", type=int, default=10)
    parser.add_argument("--res", type=int, default=84)
    parser.add_argument("--num_context", type=int, default=4)
    parser.add_argument("--model_size", type=str, default="small")
    parser.add_argument("--rollout_length", type=int, default=32)
    parser.add_argument("--num_denoise_steps", type=int, default=3)
    parser.add_argument("--num_episodes", type=int, default=5)
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
