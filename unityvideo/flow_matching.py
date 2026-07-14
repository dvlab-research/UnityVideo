"""Flow-matching utilities for UnityVideo's three training tasks."""

from dataclasses import dataclass

import torch

TASKS = ("text2all", "video2flow", "flow2video")


def make_zt_u(x: torch.Tensor, noise: torch.Tensor, t: torch.Tensor, eps: float):
    """Return (z_t, u) for a per-sample timestep t broadcast over x's shape."""
    while t.dim() < x.dim():
        t = t.unsqueeze(-1)
    z_t = (1 - t) * x + (eps + (1 - eps) * t) * noise
    u = (1 - eps) * noise - x
    return z_t.to(x.dtype), u.to(x.dtype)


@dataclass
class TaskStreams:
    """Noised latents, targets, and which streams are supervised, for one step."""

    z_rgb: torch.Tensor
    z_flow: torch.Tensor
    u_rgb: torch.Tensor
    u_flow: torch.Tensor
    shared_t: torch.Tensor
    rgb_supervised: bool
    flow_supervised: bool


def build_task_streams(
    task: str, rgb_latent: torch.Tensor, flow_latent: torch.Tensor, t: torch.Tensor, eps: float
) -> TaskStreams:
    """Apply the dynamic-noising scheme for `task`.

    `t` is the sampled timestep (b,) in [0,1]. The noisy stream(s) use `t`; a
    clean stream uses t=0. The model is driven by `shared_t` (shareT), which is
    `t` because the noisy stream's timestep is always `t`.
    """
    if task not in TASKS:
        raise ValueError(f"unknown task: {task}")
    noise_rgb = torch.randn_like(rgb_latent)
    noise_flow = torch.randn_like(flow_latent)
    zero = torch.zeros_like(t)

    t_rgb = t if task != "video2flow" else zero
    t_flow = t if task != "flow2video" else zero

    z_rgb, u_rgb = make_zt_u(rgb_latent, noise_rgb, t_rgb, eps)
    z_flow, u_flow = make_zt_u(flow_latent, noise_flow, t_flow, eps)

    return TaskStreams(
        z_rgb=z_rgb,
        z_flow=z_flow,
        u_rgb=u_rgb,
        u_flow=u_flow,
        shared_t=t,
        rgb_supervised=task in ("text2all", "flow2video"),
        flow_supervised=task in ("text2all", "video2flow"),
    )


def sample_timestep(
    batch_size: int, device, shift: float = 5.0, distribution: str = "logit_normal"
) -> torch.Tensor:
    """Sample a timestep and apply Wan's shift toward higher noise."""
    if distribution == "logit_normal":
        t = torch.sigmoid(torch.randn(batch_size, device=device))
    else:
        t = torch.rand(batch_size, device=device)
    if shift and shift != 1.0:
        t = shift * t / (1 + (shift - 1) * t)
    return t


def stream_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean-squared flow-matching loss for one stream."""
    return ((pred.float() - target.float()) ** 2).mean()
