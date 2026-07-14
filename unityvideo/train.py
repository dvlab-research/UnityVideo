#!/usr/bin/env python
"""Distributed UnityVideo training on Wan2.2-TI2V-5B."""

import argparse
import json
import math
import os
import queue
import random
import threading
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import bitsandbytes as bnb
from torch.nn.parallel import DistributedDataParallel as DDP

from diffsynth.pipelines.wan_video import ModelConfig, WanVideoPipeline

from .data import (
    MODALITY_PROMPTS,
    PairedVideoDataset,
    base_model_paths,
    encode_prompt,
    encode_video,
    save_checkpoint,
    validate_base_model,
)
from .flow_matching import TASKS, build_task_streams, sample_timestep, stream_loss
from .model import MODALITIES, UnifiedWanModel


def cosine_lr_factor(step: int, total_steps: int, warmup: int, min_ratio: float) -> float:
    """Linear warmup followed by cosine decay."""
    if step < warmup:
        return (step + 1) / max(warmup, 1)
    progress = (step - warmup) / max(total_steps - warmup, 1)
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_ratio + (1.0 - min_ratio) * cosine


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", default="")
    parser.add_argument("--modality", default="depth")
    parser.add_argument(
        "--modalities", default="", help="comma list e.g. 'depth,raft'; overrides --modality"
    )
    parser.add_argument(
        "--metadatas", default="", help="comma list of metadata csv, aligned with --modalities"
    )
    parser.add_argument(
        "--modality-weights",
        default="",
        help="comma list of sampling weights, aligned with --modalities",
    )
    parser.add_argument(
        "--preenc-cap",
        type=int,
        default=0,
        help="cap per-modality pre-encoded prompts (0=auto by steps); "
        "limits CPU prompt cache to avoid cgroup OOM",
    )
    parser.add_argument("--model-root", default=os.environ.get("WAN22_TI2V5B_DIR"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-frames", type=int, default=17)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--flow-lr-mult", type=float, default=10.0)
    parser.add_argument(
        "--warmup", type=int, default=200, help="linear warmup steps before cosine decay"
    )
    parser.add_argument(
        "--lr-min-ratio", type=float, default=0.1, help="final lr as a fraction of max lr"
    )
    parser.add_argument(
        "--prefetch-queue", type=int, default=4, help="async sample prefetch queue depth"
    )
    parser.add_argument(
        "--precomputed-latents-dir",
        default="",
        help="if set, load precomputed (rgb,flow,prompt_emb) "
        ".safetensors from this dir instead of running "
        "VAE+T5 online (huge memory + speed win).",
    )
    parser.add_argument(
        "--resume",
        default="",
        help="load this checkpoint into the model (strict=False) "
        "before training; new params (e.g. modality_embeddings) "
        "stay at their init.",
    )
    parser.add_argument(
        "--flow-prompt-prefix",
        default=None,
        help="modality-stream prompt; auto-derived from --modality "
        "if unset (paper's in-context cross-attn context).",
    )
    parser.add_argument("--eps", type=float, default=0.0)
    parser.add_argument("--timestep-shift", type=float, default=5.0)
    parser.add_argument(
        "--task-weights",
        default="0.5,0.25,0.25",
        help="sampling weights for text2all,video2flow,flow2video",
    )
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--max-grad-norm", type=float, default=1.5)
    parser.add_argument(
        "--timestep-dist",
        default="logit_normal",
        choices=["logit_normal", "uniform"],
        help="flow-matching timestep distribution",
    )
    parser.add_argument(
        "--ema-decay", type=float, default=0.9999, help="EMA decay; zero disables EMA checkpoints"
    )
    parser.add_argument(
        "--step-per-ema", type=int, default=1, help="update EMA every N optimizer steps"
    )
    parser.add_argument(
        "--prompt-dropout",
        type=float,
        default=0.1,
        help="probability of replacing text with the empty prompt",
    )
    args = parser.parse_args()
    if not args.model_root:
        parser.error("--model-root or WAN22_TI2V5B_DIR is required")
    if args.height % 16 or args.width % 16:
        parser.error("--height and --width must be divisible by 16")
    if args.num_frames < 1 or (args.num_frames - 1) % 4:
        parser.error("--num-frames must be 4k+1")

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        "nccl",
        timeout=timedelta(minutes=30),
        device_id=torch.device(f"cuda:{local_rank}"),
    )
    is_main = rank == 0
    device = f"cuda:{local_rank}"

    def log(*a):
        if is_main:
            print(*a, flush=True)

    torch.manual_seed(args.seed + rank)
    out_dir = Path(args.output)
    if is_main:
        out_dir.mkdir(parents=True, exist_ok=True)
    task_weights = [float(x) for x in args.task_weights.split(",")]
    if (
        len(task_weights) != len(TASKS)
        or any(weight < 0 for weight in task_weights)
        or sum(task_weights) <= 0
    ):
        parser.error("--task-weights must contain three non-negative values with a positive sum")
    if args.modalities:
        mm_mods = [m.strip() for m in args.modalities.split(",") if m.strip()]
        mm_metas = [m.strip() for m in args.metadatas.split(",") if m.strip()]
        assert len(mm_mods) == len(mm_metas), "--modalities/--metadatas length mismatch"
        if args.modality_weights:
            mm_w = [float(x) for x in args.modality_weights.split(",")]
        else:
            mm_w = [1.0] * len(mm_mods)
        assert len(mm_w) == len(mm_mods), "--modality-weights length mismatch"
    else:
        assert args.metadata, "need --metadata (single) or --modalities+--metadatas (multi)"
        mm_mods, mm_metas, mm_w = [args.modality], [args.metadata], [1.0]
    unknown_modalities = sorted(set(mm_mods) - set(MODALITIES))
    if unknown_modalities:
        parser.error(f"unsupported modalities: {', '.join(unknown_modalities)}")
    if any(weight < 0 for weight in mm_w) or sum(mm_w) <= 0:
        parser.error("--modality-weights must be non-negative with a positive sum")
    log(f"modality plan: {list(zip(mm_mods, mm_w))}")

    LOAD_CONC = int(os.environ.get("LOAD_CONCURRENCY", "3"))
    local_world = int(os.environ.get("LOCAL_WORLD_SIZE", str(world)))
    n_load_batches = (local_world + LOAD_CONC - 1) // LOAD_CONC
    my_load_batch = local_rank // LOAD_CONC
    pipe = None
    model = None
    for _b in range(n_load_batches):
        if _b == my_load_batch:
            log(f"loading pipeline (load batch {_b + 1}/{n_load_batches}, concurrency {LOAD_CONC})")
            root = Path(args.model_root)
            validate_base_model(root)
            pipe = WanVideoPipeline.from_pretrained(
                torch_dtype=torch.bfloat16,
                device=device,
                model_configs=[ModelConfig(path=p) for p in base_model_paths(root)],
                tokenizer_config=ModelConfig(path=str(root / "google/umt5-xxl")),
            )
            model = UnifiedWanModel(pipe.dit).to(device, torch.bfloat16)
            pipe.dit = None
            if args.resume:
                from safetensors.torch import load_file

                sd = load_file(args.resume)
                missing, unexpected = model.load_state_dict(sd, strict=False)
                del sd
                import gc as _gc

                _gc.collect()
                log(
                    f"resumed from {args.resume} | missing={len(missing)} "
                    f"unexpected={len(unexpected)}"
                )
                if missing:
                    log(f"  missing (first 5): {missing[:5]}")
        dist.barrier()
    model.train()
    for p in model.parameters():
        p.requires_grad_(True)
    for m in (pipe.vae, pipe.text_encoder):
        m.requires_grad_(False)
        m.eval()

    PRECOMP = bool(args.precomputed_latents_dir)
    if PRECOMP:
        pipe.vae = None
        pipe.text_encoder = None
        torch.cuda.empty_cache()
        log(f"precomputed mode -> dropped VAE+T5; latents dir: {args.precomputed_latents_dir}")

    if PRECOMP and len(mm_mods) > 1:
        raise SystemExit("precomputed-latents mode supports a single modality only")

    datasets, my_indices_map, prompt_caches, flow_ctxs, orders = {}, {}, {}, {}, {}
    w_sum = sum(mm_w)
    for mi, mod in enumerate(mm_mods):
        ds = PairedVideoDataset(mm_metas[mi], mod, args.num_frames, args.height, args.width)
        datasets[mod] = ds
        order_m = list(range(len(ds)))
        random.Random(args.seed + mi).shuffle(order_m)
        orders[mod] = order_m
        shard_m = order_m[rank::world]
        share = max(1, int(args.steps * (mm_w[mi] / w_sum) * 1.3) + 50)
        if args.preenc_cap > 0:
            share = min(share, args.preenc_cap)
        my_idx = shard_m[: min(len(shard_m), share)]
        if not my_idx:
            raise RuntimeError(
                f"rank {rank} has no samples for {mod}; add data or reduce WORLD_SIZE"
            )
        my_indices_map[mod] = my_idx
        log(f"[{mod}] rows={len(ds)} shard={len(shard_m)} preenc={len(my_idx)}")
        if not PRECOMP:
            flow_prefix = args.flow_prompt_prefix or MODALITY_PROMPTS.get(
                mod, f"{mod} visualization."
            )
            pc = {}
            for i in my_idx:
                pc[i] = encode_prompt(pipe, ds.rows[i]["prompt"]).cpu()
            prompt_caches[mod] = pc
            flow_ctxs[mod] = encode_prompt(pipe, flow_prefix).cpu()
    empty_ctx = None
    if not PRECOMP:
        empty_ctx = encode_prompt(pipe, "").cpu()
        pipe.text_encoder = None
        torch.cuda.empty_cache()
    else:
        mod = mm_mods[0]
        from safetensors.torch import load_file as _safe_load

        lat_dir = Path(args.precomputed_latents_dir)
        my_indices_map[mod] = [
            i for i in my_indices_map[mod] if (lat_dir / f"{i:07d}.safetensors").is_file()
        ]
        log(f"precomputed: {len(my_indices_map[mod])} latent files for {mod}")
        flow_path = lat_dir / "flow_ctx_fixed.safetensors"
        if flow_path.is_file():
            flow_ctxs[mod] = _safe_load(str(flow_path))["emb"].unsqueeze(0)
            log(f"loaded fixed flow context from {flow_path}")
        else:
            log(f"WARNING: {flow_path} missing; flow stream uses zeros")
            flow_ctxs[mod] = torch.zeros(1, 512, 4096, dtype=torch.bfloat16)

    added, base_params = model.trainable_parameter_groups()

    def make_master(params):
        return [p.detach().clone().float().requires_grad_(True) for p in params]

    master_base = make_master(base_params)
    master_added = make_master(added)
    param_pairs = list(zip(base_params + added, master_base + master_added))
    optimizer = bnb.optim.AdamW8bit(
        [
            {"params": master_base, "lr": args.lr},
            {"params": master_added, "lr": args.lr * args.flow_lr_mult},
        ]
    )

    ddp = DDP(
        model,
        device_ids=[local_rank],
        find_unused_parameters=True,
        broadcast_buffers=False,
        gradient_as_bucket_view=True,
    )
    log(f"DDP world={world} | base lr={args.lr:.1e} flow lr={args.lr * args.flow_lr_mult:.1e}")

    masters_all = master_base + master_added  # same order as param_pairs
    ema_params = None
    if is_main and args.ema_decay > 0:
        ema_params = [m.detach().clone().cpu() for m in masters_all]
        log(f"EMA enabled: decay={args.ema_decay} over {len(ema_params)} tensors (rank0 fp32)")

    def save_ema(path):
        """Write the EMA weights as a parallel checkpoint by temporarily swapping
        them into the live model, saving, then restoring the trained weights."""
        if ema_params is None:
            return
        backup = [p.detach().clone() for p, _ in param_pairs]
        with torch.no_grad():
            for (p, _), e in zip(param_pairs, ema_params):
                p.data.copy_(e.to(device=p.device, dtype=p.dtype))
        save_checkpoint(model, path)
        with torch.no_grad():
            for (p, _), b in zip(param_pairs, backup):
                p.data.copy_(b)
        del backup

    skipped = [0]
    stop_evt = threading.Event()
    prefetch_qs = {m: queue.Queue(maxsize=args.prefetch_queue) for m in mm_mods}
    cursors = {m: rank for m in mm_mods}
    my_sets = {m: set(my_indices_map[m]) for m in mm_mods}
    lat_dir_path = Path(args.precomputed_latents_dir) if PRECOMP else None
    from safetensors.torch import load_file as _safe_load_pref

    def _fetch_one(mod, idx):
        if PRECOMP:
            d = _safe_load_pref(str(lat_dir_path / f"{idx:07d}.safetensors"))
            return d["rgb"], d["flow"], d["prompt_emb"]
        rgb_frames, flow_frames, _ = datasets[mod].get(idx)
        return rgb_frames, flow_frames, None

    def prefetcher(mod):
        order_m, myset, q = orders[mod], my_sets[mod], prefetch_qs[mod]
        while not stop_evt.is_set():
            for _ in range(64):
                if stop_evt.is_set():
                    return
                idx = order_m[cursors[mod] % len(order_m)]
                cursors[mod] += world
                if idx not in myset:
                    continue
                try:
                    item = _fetch_one(mod, idx)
                except Exception as exc:  # noqa: BLE001
                    skipped[0] += 1
                    print(
                        f"[rank{rank}] {mod} prefetch skip {idx}: {type(exc).__name__}: {exc}",
                        flush=True,
                    )
                    continue
                while not stop_evt.is_set():
                    try:
                        q.put((idx,) + item, timeout=30)
                        break
                    except queue.Full:
                        continue
                break

    pf_threads = [threading.Thread(target=prefetcher, args=(m,), daemon=True) for m in mm_mods]
    for th in pf_threads:
        th.start()

    init_lrs = [g["lr"] for g in optimizer.param_groups]

    mon = (out_dir / "train_unified.jsonl").open("w") if is_main else None
    started = time.time()

    for step in range(args.steps):
        if is_main:
            picked = random.choices(range(len(TASKS)), weights=task_weights, k=1)[0]
            picked_mod = random.choices(range(len(mm_mods)), weights=mm_w, k=1)[0]
            sel = torch.tensor([picked, picked_mod], device=device)
        else:
            sel = torch.zeros(2, dtype=torch.long, device=device)
        dist.broadcast(sel, src=0)
        task = TASKS[int(sel[0])]
        cur_mod = mm_mods[int(sel[1])]

        lr_factor = cosine_lr_factor(step, args.steps, args.warmup, args.lr_min_ratio)
        for g, base in zip(optimizer.param_groups, init_lrs):
            g["lr"] = base * lr_factor

        idx, rgb_data, flow_data, emb = prefetch_qs[cur_mod].get(timeout=600)
        if PRECOMP:
            rgb_latent = rgb_data.unsqueeze(0).to(device, dtype=torch.bfloat16, non_blocking=True)
            flow_latent = flow_data.unsqueeze(0).to(device, dtype=torch.bfloat16, non_blocking=True)
            context = emb.unsqueeze(0).to(device, dtype=torch.bfloat16, non_blocking=True)
        else:
            rgb_latent = encode_video(pipe, rgb_data)
            flow_latent = encode_video(pipe, flow_data)
            context = prompt_caches[cur_mod][idx].to(device)
        if empty_ctx is not None and random.random() < args.prompt_dropout:
            context = empty_ctx.to(device)

        t = sample_timestep(
            rgb_latent.shape[0], rgb_latent.device, args.timestep_shift, args.timestep_dist
        )
        s = build_task_streams(task, rgb_latent, flow_latent, t, args.eps)

        context_flow = flow_ctxs[cur_mod].to(device, non_blocking=True)
        v_rgb, v_flow = ddp(
            s.z_rgb,
            s.z_flow,
            s.shared_t * 1000.0,
            context,
            context_flow=context_flow,
            modality=cur_mod,
            use_gradient_checkpointing=True,
        )

        rgb_loss = (
            stream_loss(v_rgb, s.u_rgb) if s.rgb_supervised else torch.zeros((), device=device)
        )
        flow_loss = (
            stream_loss(v_flow, s.u_flow) if s.flow_supervised else torch.zeros((), device=device)
        )
        loss = rgb_loss + flow_loss

        model.zero_grad(set_to_none=True)
        loss.backward()
        for p, m in param_pairs:
            m.grad = p.grad.float() if p.grad is not None else None
        grad_norm = torch.nn.utils.clip_grad_norm_([m for _, m in param_pairs], args.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            for p, m in param_pairs:
                p.data.copy_(m.data)
            if ema_params is not None and (step + 1) % args.step_per_ema == 0:
                d = args.ema_decay**args.step_per_ema
                for e, m in zip(ema_params, masters_all):
                    e.mul_(d).add_(m.data.to("cpu"), alpha=1.0 - d)

        stats = torch.stack((rgb_loss.detach(), flow_loss.detach())).float()
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        stats /= world
        rgb_m, flow_m = stats[0].item(), stats[1].item()

        if is_main and step % args.log_every == 0:
            rec = {
                "step": step,
                "task": task,
                "modality": cur_mod,
                "t": round(float(t.mean()), 4),
                "rgb_loss": round(rgb_m, 6),
                "flow_loss": round(flow_m, 6),
                "loss": round(rgb_m + flow_m, 6),
                "grad_norm": round(float(grad_norm), 4),
                "lr_factor": round(lr_factor, 4),
                "elapsed": round(time.time() - started, 1),
            }
            mon.write(json.dumps(rec) + "\n")
            mon.flush()
            print(
                f"step {step:5d} | {task:10s} {cur_mod:11s} t={rec['t']:.3f} | "
                f"rgb_loss={rgb_m:.4f} flow_loss={flow_m:.4f} | "
                f"grad={rec['grad_norm']:.2f} | {rec['elapsed']:.1f}s",
                flush=True,
            )

        if args.save_every and (step + 1) % args.save_every == 0:
            if is_main:
                save_checkpoint(model, out_dir / f"step-{step + 1}.safetensors")
                save_ema(out_dir / f"step-{step + 1}.ema.safetensors")
            dist.barrier()

    stop_evt.set()
    for q in prefetch_qs.values():
        try:
            while True:
                q.get_nowait()
        except queue.Empty:
            pass
    for th in pf_threads:
        th.join(timeout=10)

    if is_main:
        save_checkpoint(model, out_dir / "final.safetensors")
        save_ema(out_dir / "final.ema.safetensors")
        mon.close()
        print(f"done. skipped={skipped[0]} monitor={out_dir / 'train_unified.jsonl'}", flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
