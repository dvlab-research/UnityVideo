import argparse
import json
import os
import time
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download, snapshot_download
from safetensors.torch import load_file

from diffsynth.pipelines.wan_video import ModelConfig, WanVideoPipeline
from diffsynth.utils.data import save_video

from .data import (
    MODALITY_PROMPTS,
    base_model_paths,
    encode_prompt,
    encode_video,
    read_video,
    validate_base_model,
)
from .model import MODALITIES, UnifiedWanModel


MODEL_REPO = "KlingTeam/UnityVideo"
BASE_REPO = "Wan-AI/Wan2.2-TI2V-5B"
DEFAULT_CHECKPOINT = "checkpoints/unityvideo_wan22_ti2v_5b_step15000_ema.safetensors"
TASKS = ("text2all", "video2flow", "flow2video")


def resolve_model_root(value: str | None) -> Path:
    path = Path(value).expanduser() if value else Path(snapshot_download(BASE_REPO))
    validate_base_model(path)
    return path


def resolve_checkpoint(value: str | None) -> Path:
    if value:
        path = Path(value).expanduser()
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    return Path(hf_hub_download(MODEL_REPO, DEFAULT_CHECKPOINT))


def load_pipeline(model_root: Path, checkpoint: Path):
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[ModelConfig(path=path) for path in base_model_paths(model_root)],
        tokenizer_config=ModelConfig(path=str(model_root / "google/umt5-xxl")),
    )
    model = UnifiedWanModel(pipe.dit).to("cuda", torch.bfloat16)
    pipe.dit = None
    missing, unexpected = model.load_state_dict(load_file(str(checkpoint)), strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint mismatch: {len(missing)} missing, {len(unexpected)} unexpected keys"
        )
    model.eval()
    return pipe, model


@torch.no_grad()
def denoise(
    model,
    rgb_latent: torch.Tensor,
    condition_latent: torch.Tensor,
    context: torch.Tensor,
    condition_context: torch.Tensor,
    modality: str,
    steps: int,
    update_rgb: bool,
    update_condition: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    timesteps = torch.linspace(1.0, 0.0, steps + 1, device=rgb_latent.device)
    for index in range(steps):
        timestep = timesteps[index]
        delta = timesteps[index + 1] - timestep
        velocity_rgb, velocity_condition = model(
            rgb_latent,
            condition_latent,
            timestep.repeat(rgb_latent.shape[0]) * 1000.0,
            context,
            context_flow=condition_context,
            modality=modality,
        )
        if update_rgb:
            rgb_latent = rgb_latent + delta * velocity_rgb
        if update_condition:
            condition_latent = condition_latent + delta * velocity_condition
    return rgb_latent, condition_latent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="UnityVideo inference")
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--modality", choices=MODALITIES, default="depth")
    parser.add_argument("--prompt", default="")
    parser.add_argument("--rgb-video")
    parser.add_argument("--condition-video")
    parser.add_argument("--output", required=True)
    parser.add_argument("--condition-output")
    parser.add_argument("--checkpoint")
    parser.add_argument("--model-root", default=os.environ.get("WAN22_TI2V5B_DIR"))
    parser.add_argument("--num-frames", type=int, default=33)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("A CUDA device is required")
    if args.height % 16 or args.width % 16:
        raise SystemExit("--height and --width must be divisible by 16")
    if args.num_frames < 1 or (args.num_frames - 1) % 4:
        raise SystemExit("--num-frames must be 4k+1")
    if args.steps < 1:
        raise SystemExit("--steps must be positive")
    if args.task == "video2flow" and not args.rgb_video:
        raise SystemExit("--rgb-video is required for video2flow")
    if args.task == "flow2video" and not args.condition_video:
        raise SystemExit("--condition-video is required for flow2video")

    started = time.time()
    checkpoint = resolve_checkpoint(args.checkpoint)
    model_root = resolve_model_root(args.model_root)
    pipe, model = load_pipeline(model_root, checkpoint)
    context = encode_prompt(pipe, args.prompt)
    condition_context = encode_prompt(pipe, MODALITY_PROMPTS[args.modality])

    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    if args.task == "video2flow":
        rgb_frames, _ = read_video(args.rgb_video, args.num_frames, args.height, args.width)
        rgb_latent = encode_video(pipe, rgb_frames)
        condition_latent = torch.randn(
            rgb_latent.shape, generator=generator, device="cuda", dtype=torch.bfloat16
        )
        update_rgb, update_condition = False, True
    elif args.task == "flow2video":
        condition_frames, _ = read_video(
            args.condition_video, args.num_frames, args.height, args.width
        )
        condition_latent = encode_video(pipe, condition_frames)
        rgb_latent = torch.randn(
            condition_latent.shape, generator=generator, device="cuda", dtype=torch.bfloat16
        )
        update_rgb, update_condition = True, False
    else:
        latent_shape = (
            1,
            model.in_dim,
            (args.num_frames - 1) // 4 + 1,
            args.height // 8,
            args.width // 8,
        )
        rgb_latent = torch.randn(
            latent_shape, generator=generator, device="cuda", dtype=torch.bfloat16
        )
        condition_latent = torch.randn(
            latent_shape, generator=generator, device="cuda", dtype=torch.bfloat16
        )
        update_rgb, update_condition = True, True

    pipe.text_encoder = None
    torch.cuda.empty_cache()
    rgb_latent, condition_latent = denoise(
        model,
        rgb_latent,
        condition_latent,
        context,
        condition_context,
        args.modality,
        args.steps,
        update_rgb,
        update_condition,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    selected = condition_latent if args.task == "video2flow" else rgb_latent
    frames = pipe.vae_output_to_video(pipe.vae.decode(selected, device="cuda"))
    save_video(frames, str(output), fps=args.fps, quality=5)

    condition_output = None
    if args.task == "text2all":
        condition_output = (
            Path(args.condition_output)
            if args.condition_output
            else output.with_name(f"{output.stem}_{args.modality}{output.suffix}")
        )
        condition_output.parent.mkdir(parents=True, exist_ok=True)
        condition_frames = pipe.vae_output_to_video(
            pipe.vae.decode(condition_latent, device="cuda")
        )
        save_video(condition_frames, str(condition_output), fps=args.fps, quality=5)

    metadata = {
        "task": args.task,
        "modality": args.modality,
        "prompt": args.prompt,
        "checkpoint": str(checkpoint),
        "output": str(output),
        "condition_output": str(condition_output) if condition_output else None,
        "num_frames": args.num_frames,
        "height": args.height,
        "width": args.width,
        "steps": args.steps,
        "seed": args.seed,
        "elapsed_seconds": round(time.time() - started, 2),
    }
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
