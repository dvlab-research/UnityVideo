---
license: apache-2.0
library_name: unityvideo
pipeline_tag: text-to-video
base_model: Wan-AI/Wan2.2-TI2V-5B
tags:
  - video-generation
  - multi-modal
  - depth-estimation
  - optical-flow
  - controllable-generation
---

# UnityVideo Wan2.2-TI2V-5B

This repository contains the five-modality, three-task UnityVideo checkpoint
built on Wan2.2-TI2V-5B. Source code and usage instructions are available at
[JIA-Lab-research/UnityVideo](https://github.com/JIA-Lab-research/UnityVideo).

## Capabilities

- Modalities: depth, DensePose, optical flow (RAFT), segmentation, skeleton.
- Tasks: text-to-RGB+modality, video-to-modality, modality-to-video.
- Architecture: joint RGB/modality self-attention, split text cross-attention,
  modality identity embeddings, and separate RGB/modality output heads.

## Files

| File | Size | SHA-256 | Use |
| --- | ---: | --- | --- |
| `checkpoints/unityvideo_wan22_ti2v_5b_step15000_ema.safetensors` | 10,020,954,352 bytes | `0df3909e312526c46f68097958afa055868f73354fe4276d693f7ebc398e6a39` | Inference |
| `checkpoints/unityvideo_wan22_ti2v_5b_step15000.safetensors` | 10,020,954,352 bytes | `4ee83a43ffdbaee90e7ff6a25fe108d8d65d255cd0803e3972bbbfb5b01db48c` | Fine-tuning |

The checkpoints contain the full two-stream DiT. Download the VAE, text encoder,
tokenizer, and base configuration from
[Wan-AI/Wan2.2-TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B).

## Quick start

```bash
git clone https://github.com/JIA-Lab-research/UnityVideo.git
cd UnityVideo
pip install -e .

unityvideo-infer \
  --task video2flow \
  --modality depth \
  --rgb-video input.mp4 \
  --output depth.mp4
```

The CLI downloads this checkpoint and the Wan2.2 base model on first use. See
the GitHub README for all inference modes, local download commands, metadata
format, and distributed training.

## Training recipe

- Base: Wan2.2-TI2V-5B.
- Resolution: 256 x 256, 33 frames.
- Modalities: depth, DensePose, RAFT, segmentation, skeleton.
- Modality sampling: 0.2, 0.2, 0.2, 0.4, 0.4.
- Task sampling (`text2all`, `video2flow`, `flow2video`): 0.5, 0.25, 0.25.
- EMA decay: 0.9999.
- Effective released step: 15,000.

## Limitations

The released model was evaluated at 256 x 256 with 33 frames. Higher
resolutions, longer videos, unseen modality encodings, and safety-critical uses
require independent validation. The historical A14B dual-expert experiments
used an RGB+depth-only recipe and are not represented by this checkpoint.

## Citation

```bibtex
@article{huang2025unityvideo,
  title={UnityVideo: Unified Multi-Modal Multi-Task Learning for Enhancing World-Aware Video Generation},
  author={Huang, Jiehui and Zhang, Yuechen and He, Xu and Gao, Yuan and Cen, Zhi and Xia, Bin and Zhou, Yan and Tao, Xin and Wan, Pengfei and Jia, Jiaya},
  journal={arXiv preprint arXiv:2512.07831},
  year={2025}
}
```
