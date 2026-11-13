"""Building blocks: relative-position transformer, WaveNet, coupling flows, HiFi-GAN resblocks."""
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.parametrizations import weight_norm
from torch.nn.utils.parametrize import remove_parametrizations

from .commons import LRELU_SLOPE, fused_add_tanh_sigmoid_multiply, get_padding, init_weights


def _remove_wn(m):
    for mod in m.modules():
        if hasattr(mod, "parametrizations") and "weight" in mod.parametrizations:
            remove_parametrizations(mod, "weight")


class LayerNorm(nn.Module):
    """Channel-first LayerNorm."""

    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))
        self.eps = eps
        self.channels = channels

    def forward(self, x):
        x = x.transpose(1, -1)
        x = F.layer_norm(x, (self.channels,), self.gamma, self.beta, self.eps)  # static shape for export
        return x.transpose(1, -1)


class MultiHeadAttention(nn.Module):
    """Self-attention with windowed relative position embeddings (Shaw et al.)."""

    def __init__(self, channels, n_heads, p_dropout=0.0, window_size=4):
        super().__init__()
        assert channels % n_heads == 0
        self.n_heads, self.k_channels, self.window_size = n_heads, channels // n_heads, window_size
        self.conv_q = nn.Conv1d(channels, channels, 1)
        self.conv_k = nn.Conv1d(channels, channels, 1)
        self.conv_v = nn.Conv1d(channels, channels, 1)
        self.conv_o = nn.Conv1d(channels, channels, 1)
        self.drop = nn.Dropout(p_dropout)
        rel_std = self.k_channels ** -0.5
        self.emb_rel_k = nn.Parameter(torch.randn(1, window_size * 2 + 1, self.k_channels) * rel_std)
        self.emb_rel_v = nn.Parameter(torch.randn(1, window_size * 2 + 1, self.k_channels) * rel_std)
        for c in (self.conv_q, self.conv_k, self.conv_v):
            nn.init.xavier_uniform_(c.weight)

    def forward(self, x, attn_mask):
        q, k, v = self.conv_q(x), self.conv_k(x), self.conv_v(x)
        b, d, t = k.shape
        q = q.view(b, self.n_heads, self.k_channels, t).transpose(2, 3)
        k = k.view(b, self.n_heads, self.k_channels, t).transpose(2, 3)
        v = v.view(b, self.n_heads, self.k_channels, t).transpose(2, 3)
        q = q / math.sqrt(self.k_channels)
        scores = torch.matmul(q, k.transpose(-2, -1))
        rel_k = self._get_relative_embeddings(self.emb_rel_k, t)
        rel_logits = torch.matmul(q, rel_k.unsqueeze(0).transpose(-2, -1))
        scores = scores + self._relative_to_absolute(rel_logits)
        scores = scores.masked_fill(attn_mask == 0, -1e4)
        p = self.drop(F.softmax(scores, dim=-1))
        out = torch.matmul(p, v)
        rel_w = self._absolute_to_relative(p)
        out = out + torch.matmul(rel_w, self._get_relative_embeddings(self.emb_rel_v, t).unsqueeze(0))
        out = out.transpose(2, 3).contiguous().view(b, d, t)
        return self.conv_o(out)

    def _get_relative_embeddings(self, emb, length):
        """Embeddings for relative positions -(L-1)..(L-1); zero outside the window.

        Implemented as a clamped gather so the graph is shape-dynamic and free of
        data-dependent Python control flow (exports cleanly to ONNX / TensorFlow).
        """
        rel = torch.arange(2 * length - 1, device=emb.device) - (length - 1)
        valid = (rel.abs() <= self.window_size).to(emb.dtype).unsqueeze(-1)
        idx = torch.clamp(rel + self.window_size, 0, 2 * self.window_size)
        return emb[0].index_select(0, idx) * valid

    @staticmethod
    def _relative_to_absolute(x):
        b, h, l, _ = x.shape
        x = F.pad(x, [0, 1])
        x = x.reshape(b, h, l * 2 * l)
        x = F.pad(x, [0, l - 1])
        return x.reshape(b, h, l + 1, 2 * l - 1)[:, :, :l, l - 1:]

    @staticmethod
    def _absolute_to_relative(x):
        b, h, l, _ = x.shape
        x = F.pad(x, [0, l - 1])
        x = x.reshape(b, h, l * l + l * (l - 1))
        x = F.pad(x, [l, 0])
        return x.reshape(b, h, l, 2 * l)[:, :, :, 1:]


class FFN(nn.Module):
    def __init__(self, in_ch, out_ch, filter_ch, kernel_size, p_dropout=0.0):
        super().__init__()
        self.pad = kernel_size // 2
        self.conv_1 = nn.Conv1d(in_ch, filter_ch, kernel_size)
        self.conv_2 = nn.Conv1d(filter_ch, out_ch, kernel_size)
        self.drop = nn.Dropout(p_dropout)

    def forward(self, x, x_mask):
        x = self.conv_1(F.pad(x * x_mask, [self.pad, self.pad]))
        x = self.drop(torch.relu(x))
        x = self.conv_2(F.pad(x * x_mask, [self.pad, self.pad]))
        return x * x_mask


class Encoder(nn.Module):
    def __init__(self, hidden, filter_ch, n_heads, n_layers, kernel_size=1, p_dropout=0.0, window_size=4):
        super().__init__()
        self.drop = nn.Dropout(p_dropout)
        # attribute names match the reference VITS / Piper checkpoints
        self.attn_layers = nn.ModuleList(MultiHeadAttention(hidden, n_heads, p_dropout, window_size) for _ in range(n_layers))
        self.norm_layers_1 = nn.ModuleList(LayerNorm(hidden) for _ in range(n_layers))
        self.ffn_layers = nn.ModuleList(FFN(hidden, hidden, filter_ch, kernel_size, p_dropout) for _ in range(n_layers))
        self.norm_layers_2 = nn.ModuleList(LayerNorm(hidden) for _ in range(n_layers))

    def forward(self, x, x_mask):
        attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
        x = x * x_mask
        for a, n1, f, n2 in zip(self.attn_layers, self.norm_layers_1, self.ffn_layers, self.norm_layers_2):
            x = n1(x + self.drop(a(x, attn_mask)))
            x = n2(x + self.drop(f(x, x_mask)))
        return x * x_mask


class WN(nn.Module):
    def __init__(self, hidden, kernel_size, dilation_rate, n_layers, gin_channels=0, p_dropout=0.0):
        super().__init__()
        self.hidden, self.n_layers = hidden, n_layers
        self.in_layers, self.res_skip_layers = nn.ModuleList(), nn.ModuleList()
        self.drop = nn.Dropout(p_dropout)
        self.cond_layer = weight_norm(nn.Conv1d(gin_channels, 2 * hidden * n_layers, 1)) if gin_channels else None
        for i in range(n_layers):
            d = dilation_rate ** i
            self.in_layers.append(weight_norm(nn.Conv1d(hidden, 2 * hidden, kernel_size, dilation=d,
                                                        padding=(kernel_size * d - d) // 2)))
            out = 2 * hidden if i < n_layers - 1 else hidden
            self.res_skip_layers.append(weight_norm(nn.Conv1d(hidden, out, 1)))

    def forward(self, x, x_mask, g=None):
        output = torch.zeros_like(x)
        if g is not None and self.cond_layer is not None:
            g = self.cond_layer(g)
        for i in range(self.n_layers):
            x_in = self.in_layers[i](x)
            g_l = g[:, i * 2 * self.hidden:(i + 1) * 2 * self.hidden] if g is not None else torch.zeros_like(x_in)
            acts = self.drop(fused_add_tanh_sigmoid_multiply(x_in, g_l, self.hidden))
            rs = self.res_skip_layers[i](acts)
            if i < self.n_layers - 1:
                x = (x + rs[:, :self.hidden]) * x_mask
                output = output + rs[:, self.hidden:]
            else:
                output = output + rs
        return output * x_mask


class ResidualCouplingLayer(nn.Module):
    def __init__(self, channels, hidden, kernel_size, dilation_rate, n_layers, gin_channels=0):
        super().__init__()
        self.half = channels // 2
        self.pre = nn.Conv1d(self.half, hidden, 1)
        self.enc = WN(hidden, kernel_size, dilation_rate, n_layers, gin_channels)
        self.post = nn.Conv1d(hidden, self.half, 1)  # mean-only coupling
        self.post.weight.data.zero_()
        self.post.bias.data.zero_()

    def forward(self, x, x_mask, g=None, reverse=False):
        x0, x1 = x[:, :self.half], x[:, self.half:]
        m = self.post(self.enc(self.pre(x0) * x_mask, x_mask, g)) * x_mask
        x1 = (m + x1 * x_mask) if not reverse else (x1 - m) * x_mask
        return torch.cat([x0, x1], 1)


class ResidualCouplingBlock(nn.Module):
    def __init__(self, channels, hidden, kernel_size, dilation_rate, n_layers, n_flows=4, gin_channels=0):
        super().__init__()
        self.flows = nn.ModuleList(
            ResidualCouplingLayer(channels, hidden, kernel_size, dilation_rate, n_layers, gin_channels)
            for _ in range(n_flows))

    def forward(self, x, x_mask, g=None, reverse=False):
        flows = self.flows if not reverse else reversed(self.flows)
        for f in flows:
            if not reverse:
                x = torch.flip(f(x, x_mask, g=g), [1])
            else:
                x = f(torch.flip(x, [1]), x_mask, g=g, reverse=True)
        return x


class ResBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super().__init__()
        self.convs1 = nn.ModuleList(weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=d,
                                                          padding=get_padding(kernel_size, d))) for d in dilation)
        self.convs2 = nn.ModuleList(weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=1,
                                                          padding=get_padding(kernel_size, 1))) for _ in dilation)
        self.convs1.apply(init_weights)
        self.convs2.apply(init_weights)

    def forward(self, x, x_mask=None):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt if x_mask is None else xt * x_mask)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt if x_mask is None else xt * x_mask)
            x = xt + x
        return x if x_mask is None else x * x_mask


class ResBlock2(nn.Module):
    """HiFi-GAN V2/V3 residual block (one conv per dilation)."""

    def __init__(self, channels, kernel_size=3, dilation=(1, 3)):
        super().__init__()
        self.convs = nn.ModuleList(weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=d,
                                                         padding=get_padding(kernel_size, d))) for d in dilation)
        self.convs.apply(init_weights)

    def forward(self, x, x_mask=None):
        for c in self.convs:
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c(xt if x_mask is None else xt * x_mask)
            x = xt + x
        return x if x_mask is None else x * x_mask
