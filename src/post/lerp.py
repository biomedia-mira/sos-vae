import torch


def lerp(a, b, t):
    step = b - a
    return a + step * t


def slerp(a, b, t):
    a_norm = a / torch.norm(a, dim=1, keepdim=True)
    b_norm = b / torch.norm(b, dim=1, keepdim=True)
    omega = torch.acos((a_norm * b_norm).sum(1))
    so = torch.sin(omega)
    res = (torch.sin((1.0 - t) * omega) / so).unsqueeze(1) * a + (
        torch.sin(t * omega) / so
    ).unsqueeze(1) * b
    return res
