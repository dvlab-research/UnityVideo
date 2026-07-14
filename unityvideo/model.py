"""Two-stream RGB and auxiliary-modality extension of the Wan2.2 DiT."""

import copy
from typing import Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange

from diffsynth.models.wan_video_dit import (
    WanModel,
    sinusoidal_embedding_1d,
)

MODALITIES = ("depth", "raft", "segmentation", "skeleton", "densepose")


def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """y = x * (1 + scale) + shift, broadcast over the token dim."""
    return x * (1 + scale) + shift


def _checkpoint(fn, use_ckpt: bool, *args):
    if use_ckpt:
        return torch.utils.checkpoint.checkpoint(fn, *args, use_reentrant=False)
    return fn(*args)


class UnifiedWanModel(nn.Module):
    """Wraps a pretrained `WanModel` and adds a second (modality) stream."""

    def __init__(self, base: WanModel):
        super().__init__()
        self.base = base
        self.dim = base.dim
        self.in_dim = base.in_dim
        self.out_dim = base.head.head.weight.shape[0] // (
            base.patch_size[0] * base.patch_size[1] * base.patch_size[2]
        )
        self.patch_size = base.patch_size
        self.num_heads = base.blocks[0].self_attn.num_heads

        self.flow_patch_embedding = nn.Conv3d(
            self.in_dim, self.dim, kernel_size=self.patch_size, stride=self.patch_size
        )
        self.flow_patch_embedding.load_state_dict(base.patch_embedding.state_dict())

        self.flow_in = nn.Linear(self.dim, self.dim)
        nn.init.zeros_(self.flow_in.weight)
        nn.init.zeros_(self.flow_in.bias)

        self.modality_embeddings = nn.ParameterDict(
            {m: nn.Parameter(torch.zeros(self.dim)) for m in MODALITIES}
        )

        self.flow_head = copy.deepcopy(base.head)

    def _patchify(self, latent: torch.Tensor, patch_embed: nn.Conv3d):
        """latent (b,c,F,H,W) -> grid (b,f,h,w,dim) and (f,h,w)."""
        x = patch_embed(latent)  # b dim f h w
        f, h, w = x.shape[2], x.shape[3], x.shape[4]
        x = rearrange(x, "b c f h w -> b f h w c")
        return x, (f, h, w)

    def _build_freqs(self, f: int, h: int, w: int, device) -> torch.Tensor:
        """3D RoPE frequency table for an (f, h, w) token grid."""
        freqs = self.base.freqs
        out = torch.cat(
            [
                freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1,
        ).reshape(f * h * w, 1, -1)
        return out.to(device)

    def _block_forward(self, block, x, ctx_rgb, ctx_flow, t_mod, freqs, f, h, w):
        """Wan DiTBlock forward with two cross-attn contexts.

        x is the joint sequence of (f * h * 2w) tokens; the first w columns per
        row are RGB tokens and the next w are modality tokens. Self-attn runs on
        the joint sequence (cross-modal). Cross-attn is computed twice with the
        same weights but two different contexts; the outputs are merged back.
        FFN runs on the joint sequence.
        """
        mod = block.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(6, dim=1)

        input_x = _modulate(block.norm1(x), shift_msa, scale_msa)
        x = block.gate(x, gate_msa, block.self_attn(input_x, freqs))

        w2 = 2 * w
        x_grid = rearrange(x, "b (f h w2) c -> b f h w2 c", f=f, h=h, w2=w2)
        rgb_x = rearrange(x_grid[:, :, :, :w, :], "b f h w c -> b (f h w) c")
        flow_x = rearrange(x_grid[:, :, :, w:, :], "b f h w c -> b (f h w) c")
        rgb_x = rgb_x + block.cross_attn(block.norm3(rgb_x), ctx_rgb)
        flow_x = flow_x + block.cross_attn(block.norm3(flow_x), ctx_flow)
        rgb_grid = rearrange(rgb_x, "b (f h w) c -> b f h w c", f=f, h=h, w=w)
        flow_grid = rearrange(flow_x, "b (f h w) c -> b f h w c", f=f, h=h, w=w)
        x_grid = torch.cat([rgb_grid, flow_grid], dim=3)
        x = rearrange(x_grid, "b f h w2 c -> b (f h w2) c")

        input_x = _modulate(block.norm2(x), shift_mlp, scale_mlp)
        x = block.gate(x, gate_mlp, block.ffn(input_x))
        return x

    def forward(
        self,
        rgb_latent: torch.Tensor,
        flow_latent: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_flow: Optional[torch.Tensor] = None,
        modality: Optional[str] = None,
        use_gradient_checkpointing: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict RGB and modality flow-matching velocities.

        rgb_latent / flow_latent: (b, in_dim, F, H, W) noisy latents.
        timestep: (b,) shared timestep (shareT) for modulation.
        context: (b, L, text_dim) text embeddings -- the RGB stream's prompt.
        context_flow: optional (b, L, text_dim) text embeddings for the modality
            stream's prompt. Defaults to `context` (no split cross-attn).
        modality: which modality the second stream represents. If given, the
            corresponding `modality_embeddings[modality]` is added to every
            modality token (per-modality identity).
        Returns (v_rgb, v_flow), each (b, in_dim, F, H, W).
        """
        base = self.base
        dtype = rgb_latent.dtype

        t = base.time_embedding(sinusoidal_embedding_1d(base.freq_dim, timestep).to(dtype))
        t_mod = base.time_projection(t).unflatten(1, (6, self.dim))

        ctx_rgb = base.text_embedding(context)
        ctx_flow = base.text_embedding(context_flow) if context_flow is not None else ctx_rgb

        rgb_grid, (f, h, w) = self._patchify(rgb_latent, base.patch_embedding)
        flow_grid, _ = self._patchify(flow_latent, self.flow_patch_embedding)

        flow_grid = self.flow_in(flow_grid)

        if modality is not None and modality in self.modality_embeddings:
            flow_grid = flow_grid + self.modality_embeddings[modality].view(1, 1, 1, 1, -1)

        joint = torch.cat([rgb_grid, flow_grid], dim=3)
        x = rearrange(joint, "b f h w c -> b (f h w) c")

        freqs = self._build_freqs(f, h, 2 * w, x.device)

        for block in base.blocks:
            x = _checkpoint(
                self._block_forward,
                use_gradient_checkpointing and self.training,
                block,
                x,
                ctx_rgb,
                ctx_flow,
                t_mod,
                freqs,
                f,
                h,
                w,
            )

        x = rearrange(x, "b (f h w) c -> b f h w c", f=f, h=h, w=2 * w)
        rgb_x = rearrange(x[:, :, :, :w, :], "b f h w c -> b (f h w) c")
        flow_x = rearrange(x[:, :, :, w:, :], "b f h w c -> b (f h w) c")

        rgb_out = base.head(rgb_x, t)
        flow_out = self.flow_head(flow_x, t)

        v_rgb = base.unpatchify(rgb_out, (f, h, w))
        v_flow = base.unpatchify(flow_out, (f, h, w))
        return v_rgb, v_flow

    def trainable_parameter_groups(self):
        """Return (added_params, base_params) for optimizer lr groups."""
        added = (
            list(self.flow_patch_embedding.parameters())
            + list(self.flow_in.parameters())
            + list(self.flow_head.parameters())
            + list(self.modality_embeddings.parameters())
        )
        added_ids = {id(p) for p in added}
        base_params = [p for p in self.base.parameters() if id(p) not in added_ids]
        return added, base_params
