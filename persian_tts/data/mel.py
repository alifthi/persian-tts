"""STFT / mel utilities (training only; inference does not need them)."""
import functools

import torch


@functools.lru_cache(maxsize=8)
def _mel_basis(sr, n_fft, n_mels, fmin, fmax, device):
    import librosa
    m = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)
    return torch.from_numpy(m).float().to(device)


@functools.lru_cache(maxsize=8)
def _window(win, device):
    return torch.hann_window(win).to(device)


def spectrogram(y: torch.Tensor, n_fft: int, hop: int, win: int) -> torch.Tensor:
    """y: [B, T] -> magnitude [B, n_fft//2+1, frames]."""
    pad = (n_fft - hop) // 2
    y = torch.nn.functional.pad(y.unsqueeze(1), (pad, pad), mode="reflect").squeeze(1)
    spec = torch.stft(y.float(), n_fft, hop_length=hop, win_length=win, window=_window(win, str(y.device)),
                      center=False, return_complex=True)
    return torch.sqrt(spec.real ** 2 + spec.imag ** 2 + 1e-6)


def spec_to_mel(spec, sr, n_fft, n_mels, fmin, fmax):
    mel = torch.matmul(_mel_basis(sr, n_fft, n_mels, fmin, fmax, str(spec.device)), spec)
    return torch.log(torch.clamp(mel, min=1e-5))


def mel_spectrogram(y, sr, n_fft, hop, win, n_mels, fmin, fmax):
    return spec_to_mel(spectrogram(y, n_fft, hop, win), sr, n_fft, n_mels, fmin, fmax)
