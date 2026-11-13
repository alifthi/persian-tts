"""Monotonic Alignment Search (Glow-TTS / VITS), numba-accelerated, run on CPU."""
import numba
import numpy as np
import torch


@numba.njit(parallel=True, cache=True)
def _mas(paths, values, t_ys, t_xs):
    b = values.shape[0]
    for i in numba.prange(b):
        path, value, t_y, t_x = paths[i], values[i], t_ys[i], t_xs[i]
        max_neg = -1e9
        for y in range(t_y):
            for x in range(max(0, t_x + y - t_y), min(t_x, y + 1)):
                v_cur = max_neg if x == y else value[y - 1, x]
                if x == 0:
                    v_prev = 0.0 if y == 0 else max_neg
                else:
                    v_prev = value[y - 1, x - 1]
                value[y, x] += max(v_prev, v_cur)
        idx = t_x - 1
        for y in range(t_y - 1, -1, -1):
            path[y, idx] = 1
            if idx != 0 and (idx == y or value[y - 1, idx] < value[y - 1, idx - 1]):
                idx -= 1


@torch.no_grad()
def maximum_path(neg_cent: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """neg_cent: [b, t_y, t_x]; mask: [b, t_y, t_x] -> path [b, t_y, t_x]."""
    device, dtype = neg_cent.device, neg_cent.dtype
    values = (neg_cent.float() * mask).cpu().numpy().astype(np.float32)
    paths = np.zeros_like(values, dtype=np.int32)
    t_ys = mask.sum(1)[:, 0].cpu().numpy().astype(np.int32)
    t_xs = mask.sum(2)[:, 0].cpu().numpy().astype(np.int32)
    _mas(paths, values, t_ys, t_xs)
    return torch.from_numpy(paths).to(device=device, dtype=dtype)
