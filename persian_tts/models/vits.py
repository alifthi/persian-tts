"""VITS synthesizer (Kim et al., 2021) with a deterministic duration predictor.

Inference uses only ops supported by ONNX opset>=17 and onnx2tf (TensorFlow / TFLite).
"""
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.parametrizations import spectral_norm, weight_norm

from . import commons, modules
from .commons import LRELU_SLOPE, get_padding, init_weights
from .monotonic_align import maximum_path


def _frac(x):
    return x - torch.floor(x)  # export-friendly frac for x >= 0


def _hash_gaussian(ref: torch.Tensor, seed: torch.Tensor) -> torch.Tensor:
    """Pseudo-random N(0,1) noise shaped like `ref` using only elementwise math (sin-hash + Box-Muller).

    Lets the graph run where random ops are unsupported (onnx2tf / TFLite). `seed` varies the noise.
    """
    c = torch.arange(ref.shape[1], device=ref.device, dtype=ref.dtype)
    t = torch.arange(ref.shape[2], device=ref.device, dtype=ref.dtype)
    idx = (c.unsqueeze(1) * 1000.0 + t.unsqueeze(0)).unsqueeze(0) + seed * 7.123  # [1, C, T], broadcasts over batch
    idx = idx + torch.zeros_like(ref)
    u1 = _frac(torch.abs(torch.sin(idx * 12.9898) * 43758.5453)).clamp(1e-6, 1.0)
    u2 = _frac(torch.abs(torch.sin(idx * 78.233) * 24634.6345))
    return torch.sqrt(-2.0 * torch.log(u1)) * torch.cos(6.283185307 * u2)


class TextEncoder(nn.Module):
    def __init__(self, n_vocab, out_channels, hidden, filter_ch, n_heads, n_layers, kernel_size, p_dropout, window_size):
        super().__init__()
        self.hidden = hidden
        self.emb = nn.Embedding(n_vocab, hidden)
        nn.init.normal_(self.emb.weight, 0.0, hidden ** -0.5)
        self.encoder = modules.Encoder(hidden, filter_ch, n_heads, n_layers, kernel_size, p_dropout, window_size)
        self.proj = nn.Conv1d(hidden, out_channels * 2, 1)
        self.out_channels = out_channels

    def forward(self, x, x_lengths):
        x = self.emb(x) * math.sqrt(self.hidden)
        x = x.transpose(1, 2)
        x_mask = commons.sequence_mask(x_lengths, x.size(2)).unsqueeze(1).to(x.dtype)
        x = self.encoder(x, x_mask)
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        return x, m, logs, x_mask


class PosteriorEncoder(nn.Module):
    def __init__(self, in_ch, out_ch, hidden, kernel_size, dilation_rate, n_layers, gin_channels=0):
        super().__init__()
        self.out_ch = out_ch
        self.pre = nn.Conv1d(in_ch, hidden, 1)
        self.enc = modules.WN(hidden, kernel_size, dilation_rate, n_layers, gin_channels)
        self.proj = nn.Conv1d(hidden, out_ch * 2, 1)

    def forward(self, x, x_lengths, g=None):
        x_mask = commons.sequence_mask(x_lengths, x.size(2)).unsqueeze(1).to(x.dtype)
        x = self.enc(self.pre(x) * x_mask, x_mask, g=g)
        m, logs = torch.split(self.proj(x) * x_mask, self.out_ch, dim=1)
        z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask
        return z, m, logs, x_mask


class DurationPredictor(nn.Module):
    def __init__(self, in_ch, filter_ch, kernel_size, p_dropout, gin_channels=0):
        super().__init__()
        self.drop = nn.Dropout(p_dropout)
        self.conv_1 = nn.Conv1d(in_ch, filter_ch, kernel_size, padding=kernel_size // 2)
        self.norm_1 = modules.LayerNorm(filter_ch)
        self.conv_2 = nn.Conv1d(filter_ch, filter_ch, kernel_size, padding=kernel_size // 2)
        self.norm_2 = modules.LayerNorm(filter_ch)
        self.proj = nn.Conv1d(filter_ch, 1, 1)
        self.cond = nn.Conv1d(gin_channels, in_ch, 1) if gin_channels else None

    def forward(self, x, x_mask, g=None):
        if g is not None and self.cond is not None:
            x = x + self.cond(g)
        x = self.drop(self.norm_1(torch.relu(self.conv_1(x * x_mask))))
        x = self.drop(self.norm_2(torch.relu(self.conv_2(x * x_mask))))
        return self.proj(x * x_mask) * x_mask  # log-duration


class Generator(nn.Module):
    """HiFi-GAN V1 generator."""

    def __init__(self, initial_ch, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates,
                 upsample_initial_channel, upsample_kernel_sizes, gin_channels=0, resblock="2"):
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.conv_pre = nn.Conv1d(initial_ch, upsample_initial_channel, 7, 1, padding=3)
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(weight_norm(nn.ConvTranspose1d(upsample_initial_channel // (2 ** i),
                                                           upsample_initial_channel // (2 ** (i + 1)),
                                                           k, u, padding=(k - u) // 2)))
        self.resblocks = nn.ModuleList()
        ch = upsample_initial_channel
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for k, d in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                self.resblocks.append((modules.ResBlock if str(resblock) == "1" else modules.ResBlock2)(ch, k, d))
        self.conv_post = nn.Conv1d(ch, 1, 7, 1, padding=3, bias=False)
        self.ups.apply(init_weights)
        self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1) if gin_channels else None

    def forward(self, x, g=None):
        x = self.conv_pre(x)
        if g is not None and self.cond is not None:
            x = x + self.cond(g)
        for i, up in enumerate(self.ups):
            x = up(F.leaky_relu(x, LRELU_SLOPE))
            xs = self.resblocks[i * self.num_kernels](x)
            for j in range(1, self.num_kernels):
                xs = xs + self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = self.conv_post(F.leaky_relu(x))
        return torch.tanh(x)


class SynthesizerTrn(nn.Module):
    def __init__(self, n_vocab: int, spec_channels: int, segment_size: int, n_speakers: int, cfg):
        super().__init__()
        self.segment_size = segment_size  # in frames
        self.n_speakers = n_speakers
        # speaker conditioning is always on (even for 1 speaker) so pretrained conditioning layers are reused
        gin = cfg.gin_channels
        self.enc_p = TextEncoder(n_vocab, cfg.inter_channels, cfg.hidden_channels, cfg.filter_channels,
                                 cfg.n_heads, cfg.n_layers, cfg.kernel_size, cfg.p_dropout, cfg.window_size)
        self.dec = Generator(cfg.inter_channels, cfg.resblock_kernel_sizes, cfg.resblock_dilation_sizes,
                             cfg.upsample_rates, cfg.upsample_initial_channel, cfg.upsample_kernel_sizes, gin,
                             cfg.resblock)
        self.enc_q = PosteriorEncoder(spec_channels, cfg.inter_channels, cfg.hidden_channels, 5, 1, 16, gin)
        self.flow = modules.ResidualCouplingBlock(cfg.inter_channels, cfg.hidden_channels, 5, 1, 4,
                                                  cfg.n_flows, gin)
        self.dp = DurationPredictor(cfg.hidden_channels, cfg.dp_filter_channels, 3, 0.5, gin)
        self.emb_g = nn.Embedding(max(1, n_speakers), gin)
        self.portable_noise = False  # True: RNG-free Gaussian noise (TensorFlow/TFLite export)
        self.static_max_frames = None  # int: fixed output frames (static shapes for TensorFlow/TFLite)

    def _g(self, sid, batch_ref=None):
        if sid is None:
            ref = batch_ref if batch_ref is not None else self.emb_g.weight
            sid = torch.zeros(ref.shape[0] if batch_ref is not None else 1, dtype=torch.long, device=ref.device)
        return self.emb_g(sid).unsqueeze(-1)

    def forward(self, x, x_lengths, y, y_lengths, sid=None):
        g = self._g(sid, x)
        x, m_p, logs_p, x_mask = self.enc_p(x, x_lengths)
        z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=g)
        z_p = self.flow(z, y_mask, g=g)

        with torch.no_grad():  # monotonic alignment search on the prior likelihood
            m_p_f, logs_p_f, z_p_f = m_p.float(), logs_p.float(), z_p.float()
            s_p_sq_r = torch.exp(-2 * logs_p_f)
            neg_cent1 = torch.sum(-0.5 * math.log(2 * math.pi) - logs_p_f, [1], keepdim=True)
            neg_cent2 = torch.matmul(-0.5 * (z_p_f ** 2).transpose(1, 2), s_p_sq_r)
            neg_cent3 = torch.matmul(z_p_f.transpose(1, 2), m_p_f * s_p_sq_r)
            neg_cent4 = torch.sum(-0.5 * (m_p_f ** 2) * s_p_sq_r, [1], keepdim=True)
            neg_cent = neg_cent1 + neg_cent2 + neg_cent3 + neg_cent4
            attn_mask = x_mask.unsqueeze(2) * y_mask.unsqueeze(-1)
            attn = maximum_path(neg_cent, attn_mask.squeeze(1)).unsqueeze(1).detach()

        w = attn.sum(2)  # [b, 1, t_x] frames per token
        logw_target = torch.log(w + 1e-6) * x_mask
        logw = self.dp(x.detach(), x_mask, g=g.detach() if g is not None else None)
        l_dur = torch.sum((logw.float() - logw_target.float()) ** 2) / torch.sum(x_mask)

        m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2)
        logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(1, 2)

        z_slice, ids_slice = commons.rand_slice_segments(z, y_lengths, self.segment_size)
        o = self.dec(z_slice, g=g)
        return o, l_dur, attn, ids_slice, x_mask, y_mask, (z, z_p, m_p, logs_p, m_q, logs_q)

    def infer(self, x, x_lengths, sid=None, noise_scale: float = 0.667, length_scale: float = 1.0,
              noise_scale_w: float = 0.8):
        scales = torch.tensor([noise_scale, length_scale, noise_scale_w], device=x.device)
        return self.infer_onnx(x, x_lengths, scales, sid)

    def infer_onnx(self, x, x_lengths, scales, sid=None):
        """Export entry point. scales = [noise_scale, length_scale, noise_scale_w] (noise_scale_w unused:
        deterministic duration predictor; kept for Piper-compatible I/O)."""
        noise_scale, length_scale = scales[0], scales[1]
        g = self._g(sid, x)
        x, m_p, logs_p, x_mask = self.enc_p(x, x_lengths)
        logw = self.dp(x, x_mask, g=g)
        w = torch.exp(logw) * x_mask * length_scale
        w_ceil = torch.ceil(w)
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        if self.static_max_frames:
            y_lengths = torch.clamp_max(y_lengths, self.static_max_frames)
        y_mask = commons.sequence_mask(y_lengths, self.static_max_frames).unsqueeze(1).to(x_mask.dtype)
        attn_mask = x_mask.unsqueeze(2) * y_mask.unsqueeze(-1)
        attn = commons.generate_path(w_ceil, attn_mask)
        m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2)
        logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(1, 2)
        noise = _hash_gaussian(m_p, scales[2]) if self.portable_noise else torch.randn_like(m_p)
        z_p = m_p + noise * torch.exp(logs_p) * noise_scale
        z = self.flow(z_p, y_mask, g=g, reverse=True)
        o = self.dec(z * y_mask, g=g)
        return o, y_lengths

    def remove_weight_norm(self):
        modules._remove_wn(self)


# --------------------------------------------------------------------------- discriminators
class DiscriminatorP(nn.Module):
    def __init__(self, period, kernel_size=5, stride=3):
        super().__init__()
        self.period = period
        chs = [1, 32, 128, 512, 1024]
        self.convs = nn.ModuleList(
            weight_norm(nn.Conv2d(chs[i], chs[i + 1], (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size), 0)))
            for i in range(4))
        self.convs.append(weight_norm(nn.Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(get_padding(kernel_size), 0))))
        self.conv_post = weight_norm(nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []
        b, c, t = x.shape
        if t % self.period:
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)
        for l in self.convs:
            x = F.leaky_relu(l(x), LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        return torch.flatten(x, 1, -1), fmap


class DiscriminatorS(nn.Module):
    def __init__(self, use_spectral_norm=False):
        super().__init__()
        norm = spectral_norm if use_spectral_norm else weight_norm
        self.convs = nn.ModuleList([
            norm(nn.Conv1d(1, 16, 15, 1, padding=7)),
            norm(nn.Conv1d(16, 64, 41, 4, groups=4, padding=20)),
            norm(nn.Conv1d(64, 256, 41, 4, groups=16, padding=20)),
            norm(nn.Conv1d(256, 1024, 41, 4, groups=64, padding=20)),
            norm(nn.Conv1d(1024, 1024, 41, 4, groups=256, padding=20)),
            norm(nn.Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        self.conv_post = norm(nn.Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []
        for l in self.convs:
            x = F.leaky_relu(l(x), LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        return torch.flatten(x, 1, -1), fmap


class MultiPeriodDiscriminator(nn.Module):
    def __init__(self, periods=(2, 3, 5, 7, 11), use_spectral_norm=False):
        super().__init__()
        self.discriminators = nn.ModuleList([DiscriminatorS(use_spectral_norm)] +
                                            [DiscriminatorP(p) for p in periods])

    def forward(self, y, y_hat):
        y_d_rs, y_d_gs, fmap_rs, fmap_gs = [], [], [], []
        for d in self.discriminators:
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs
