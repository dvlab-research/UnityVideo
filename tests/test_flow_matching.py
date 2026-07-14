import torch

from unityvideo.flow_matching import TASKS, build_task_streams, make_zt_u


def test_flow_matching_endpoints():
    value = torch.ones(1, 2, 1, 2, 2)
    noise = torch.full_like(value, 3.0)
    clean, target = make_zt_u(value, noise, torch.tensor([0.0]), eps=0.0)
    noisy, _ = make_zt_u(value, noise, torch.tensor([1.0]), eps=0.0)
    assert torch.equal(clean, value)
    assert torch.equal(noisy, noise)
    assert torch.equal(target, noise - value)


def test_three_task_supervision():
    rgb = torch.ones(1, 2, 1, 2, 2)
    condition = torch.ones_like(rgb)
    timestep = torch.tensor([0.5])
    streams = {task: build_task_streams(task, rgb, condition, timestep, eps=0.0) for task in TASKS}
    assert streams["text2all"].rgb_supervised
    assert streams["text2all"].flow_supervised
    assert not streams["video2flow"].rgb_supervised
    assert streams["video2flow"].flow_supervised
    assert streams["flow2video"].rgb_supervised
    assert not streams["flow2video"].flow_supervised
    assert torch.equal(streams["video2flow"].z_rgb, rgb)
    assert torch.equal(streams["flow2video"].z_flow, condition)
