import csv
from pathlib import Path
from typing import Iterable, Optional, Sequence

import imageio.v2 as imageio
import torch
from PIL import Image
from safetensors.torch import save_file


MODALITY_PROMPTS = {
    "depth": "depth map visualization, scene depth from camera, near in warm colors and far in cool colors.",
    "densepose": "densepose visualization with body part UV coordinates.",
    "raft": "optical flow visualization showing motion direction with color and magnitude with intensity.",
    "segmentation": "segmentation mask of foreground objects against background.",
    "skeleton": "human skeleton pose visualization with body keypoints and limbs.",
}


def base_model_paths(root: Path) -> list:
    return [
        [
            str(root / "diffusion_pytorch_model-00001-of-00003.safetensors"),
            str(root / "diffusion_pytorch_model-00002-of-00003.safetensors"),
            str(root / "diffusion_pytorch_model-00003-of-00003.safetensors"),
        ],
        str(root / "models_t5_umt5-xxl-enc-bf16.pth"),
        str(root / "Wan2.2_VAE.pth"),
    ]


def validate_base_model(root: Path) -> None:
    paths = base_model_paths(root)
    required = [
        *paths[0],
        paths[1],
        paths[2],
        str(root / "google/umt5-xxl/spiece.model"),
    ]
    missing = [path for path in required if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError("Missing Wan2.2 base files:\n" + "\n".join(missing))


def _crop_resize(frame: Image.Image, width: int, height: int) -> Image.Image:
    source_width, source_height = frame.size
    scale = max(width / source_width, height / source_height)
    frame = frame.resize((round(source_width * scale), round(source_height * scale)))
    source_width, source_height = frame.size
    left = (source_width - width) // 2
    top = (source_height - height) // 2
    return frame.crop((left, top, left + width, top + height))


def read_video(
    path: str | Path,
    num_frames: int,
    height: int,
    width: int,
    frame_indices: Optional[Sequence[int]] = None,
    max_read: int = 400,
) -> tuple[list[Image.Image], list[int]]:
    reader = imageio.get_reader(str(path))
    raw_frames = []
    try:
        for frame in reader:
            raw_frames.append(frame)
            if len(raw_frames) >= max_read:
                break
    except (StopIteration, IndexError, RuntimeError, OSError):
        pass
    finally:
        reader.close()

    if not raw_frames:
        raise RuntimeError(f"No readable frames: {path}")
    if frame_indices is None:
        last = len(raw_frames) - 1
        divisor = max(num_frames - 1, 1)
        frame_indices = [round(index * last / divisor) for index in range(num_frames)]
    indices = [min(index, len(raw_frames) - 1) for index in frame_indices]
    frames = [
        _crop_resize(Image.fromarray(raw_frames[index]).convert("RGB"), width, height)
        for index in indices
    ]
    return frames, indices


class PairedVideoDataset:
    def __init__(
        self,
        metadata_csv: str | Path,
        modality: str,
        num_frames: int,
        height: int,
        width: int,
    ) -> None:
        self.metadata_path = Path(metadata_csv).resolve()
        with self.metadata_path.open(encoding="utf-8", newline="") as handle:
            self.rows = list(csv.DictReader(handle))
        required = {"video", "prompt", modality}
        columns = set(self.rows[0]) if self.rows else set()
        missing = sorted(required - columns)
        if missing:
            raise ValueError(f"Metadata is missing columns: {', '.join(missing)}")
        self.modality = modality
        self.num_frames = num_frames
        self.height = height
        self.width = width

    def __len__(self) -> int:
        return len(self.rows)

    def _resolve(self, value: str) -> Path:
        path = Path(value).expanduser()
        return path if path.is_absolute() else self.metadata_path.parent / path

    def get(self, index: int) -> tuple[list[Image.Image], list[Image.Image], str]:
        row = self.rows[index]
        rgb, indices = read_video(
            self._resolve(row["video"]), self.num_frames, self.height, self.width
        )
        condition, _ = read_video(
            self._resolve(row[self.modality]),
            self.num_frames,
            self.height,
            self.width,
            frame_indices=indices,
        )
        return rgb, condition, row["prompt"]


@torch.no_grad()
def encode_prompt(pipe, prompt: str) -> torch.Tensor:
    ids, mask = pipe.tokenizer([prompt], return_mask=True, add_special_tokens=True)
    ids = ids.to(pipe.device)
    mask = mask.to(pipe.device)
    embedding = pipe.text_encoder(ids, mask)
    lengths = mask.gt(0).sum(dim=1).long()
    for index, length in enumerate(lengths):
        embedding[index, length:] = 0
    return embedding.to(pipe.torch_dtype)


@torch.no_grad()
def encode_video(pipe, frames: Iterable[Image.Image]) -> torch.Tensor:
    video = pipe.preprocess_video(list(frames))
    latent = pipe.vae.encode(video, device=pipe.device)
    if isinstance(latent, (list, tuple)):
        latent = latent[0].unsqueeze(0) if latent[0].dim() == 4 else latent[0]
    return latent.to(dtype=pipe.torch_dtype, device=pipe.device)


def save_checkpoint(model, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        key: value.detach().to(torch.bfloat16).cpu().contiguous()
        for key, value in model.state_dict().items()
    }
    save_file(state, str(path))
