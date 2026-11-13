import math

import torch
import torch.nn.functional as F


def init_weights(m, mean=0.0, std=0.01):
    if "Conv" in m.__class__.__name__:
        m.weight.data.normal_(mean, std)


def get_padding(kernel_size, dilation=1):
    return (kernel_size * dilation - dilation) // 2


def sequence_mask(length: torch.Tensor, max_length=None) -> torch.Tensor:
    if max_length is None:
        max_length = length.max()
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)


def generate_path(duration: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """duration: [b, 1, t_x]; mask: [b, 1, t_y, t_x] -> hard alignment path [b, 1, t_y, t_x]. Export-friendly."""
    b, _, t_y, t_x = mask.shape
    cum = torch.cumsum(duration.reshape(b, t_x), dim=1).reshape(b * t_x)  # 2-D cumsum: layout-safe for onnx2tf
    path = sequence_mask(cum, t_y).to(mask.dtype).view(b, t_x, t_y)
    path = path - F.pad(path, [0, 0, 1, 0, 0, 0])[:, :-1]
    return path.unsqueeze(1).transpose(2, 3) * mask


def rand_slice_segments(x, x_lengths, segment_size):
    b, _, t = x.shape
    ids_max = torch.clamp(x_lengths - segment_size + 1, min=1)
    ids = (torch.rand(b, device=x.device) * ids_max).long()
    idx = ids.unsqueeze(1) + torch.arange(segment_size, device=x.device).unsqueeze(0)
    idx = idx.clamp(max=t - 1)
    return torch.gather(x, 2, idx.unsqueeze(1).expand(-1, x.size(1), -1)), ids


def slice_segments(x, ids_str, segment_size):
    idx = ids_str.unsqueeze(1) + torch.arange(segment_size, device=x.device).unsqueeze(0)
    idx = idx.clamp(max=x.size(2) - 1)
    return torch.gather(x, 2, idx.unsqueeze(1).expand(-1, x.size(1), -1))


def fused_add_tanh_sigmoid_multiply(a, b, n_channels: int):
    in_act = a + b
    return torch.tanh(in_act[:, :n_channels]) * torch.sigmoid(in_act[:, n_channels:])


def kl_loss(z_p, logs_q, m_p, logs_p, z_mask):
    z_p, logs_q, m_p, logs_p, z_mask = (t.float() for t in (z_p, logs_q, m_p, logs_p, z_mask))
    kl = logs_p - logs_q - 0.5
    kl += 0.5 * ((z_p - m_p) ** 2) * torch.exp(-2.0 * logs_p)
    return torch.sum(kl * z_mask) / torch.sum(z_mask)


LRELU_SLOPE = 0.1
LOG_2PI = math.log(2 * math.pi)
