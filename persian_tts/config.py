"""Typed experiment configuration (YAML <-> dataclasses), with dotted CLI overrides."""
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import List, Optional

import yaml


@dataclass
class AudioConfig:
    sample_rate: int = 22050
    n_fft: int = 1024
    hop_length: int = 256
    win_length: int = 1024
    n_mels: int = 80
    fmin: float = 0.0
    fmax: Optional[float] = None


@dataclass
class ModelConfig:
    hidden_channels: int = 192
    filter_channels: int = 768
    n_heads: int = 2
    n_layers: int = 6
    kernel_size: int = 3
    p_dropout: float = 0.1
    inter_channels: int = 192
    # Decoder defaults match the Piper `_base_model` checkpoint we fine-tune from.
    resblock: str = "2"
    resblock_kernel_sizes: List[int] = field(default_factory=lambda: [3, 5, 7])
    resblock_dilation_sizes: List[List[int]] = field(default_factory=lambda: [[1, 2], [2, 6], [3, 12]])
    upsample_rates: List[int] = field(default_factory=lambda: [8, 8, 4])
    upsample_initial_channel: int = 256
    upsample_kernel_sizes: List[int] = field(default_factory=lambda: [16, 16, 8])
    gin_channels: int = 512  # speaker embedding size (multi-speaker only)
    n_flows: int = 4
    window_size: int = 4  # relative attention window
    dp_filter_channels: int = 256
    use_sdp: bool = False  # stochastic duration predictor (not TF-friendly); deterministic by default


@dataclass
class DataConfig:
    root: str = "data/processed"
    train_manifest: str = "data/processed/train.jsonl"
    val_manifest: str = "data/processed/val.jsonl"
    speakers: str = "data/processed/speakers.json"
    add_blank: bool = True  # chars frontend only (phonemes always use Piper's pad-interleaved layout)
    text_frontend: str = "chars"  # chars | phonemes (espeak-ng IPA, Piper phoneme table; much better quality)
    max_frames_per_batch: int = 20000
    max_batch_size: int = 32
    eval_batch_size: int = 8  # eval decodes full utterances (no segment slicing); keep small on small GPUs
    num_workers: int = 8


@dataclass
class TrainConfig:
    seed: int = 1234
    max_steps: int = 500_000
    learning_rate: float = 2e-4
    learning_rate_d: float = 1e-4
    betas: List[float] = field(default_factory=lambda: [0.8, 0.99])
    betas_d: List[float] = field(default_factory=lambda: [0.5, 0.9])
    freeze_decoder_steps: int = 0  # keep pretrained vocoder fixed while the new text encoder adapts
    eps: float = 1e-9
    lr_decay: float = 0.999875  # per epoch-equivalent; applied per step as decay**(1/steps_per_epoch)
    warmup_steps: int = 1000
    # Cosine anneal to a fraction of the base LR, on top of warmup/lr_decay. Disabled when
    # anneal_end_step == 0. Use to converge a run whose losses have gone flat at the base LR.
    anneal_start_step: int = 0
    anneal_end_step: int = 0
    anneal_final_scale: float = 0.05  # LR multiplier reached at anneal_end_step
    grad_clip: float = 100.0
    precision: str = "bf16"  # bf16 | fp16 | fp32
    segment_size: int = 8192  # waveform samples decoded per item (random slice)
    c_mel: float = 45.0
    c_kl: float = 1.0
    log_every: int = 50
    eval_every: int = 2000
    save_every: int = 5000
    keep_last: int = 5
    ema_decay: float = 0.9995
    init_from: Optional[str] = None  # warm-start checkpoint (e.g. from a larger run); must be commercially licensed
    compile: bool = False


@dataclass
class ExperimentConfig:
    name: str = "persian-vits"
    out_dir: str = "runs"
    audio: AudioConfig = field(default_factory=AudioConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


def _build(cls, d):
    kw = {}
    for f in fields(cls):
        if f.name in d:
            v = d[f.name]
            kw[f.name] = _build(f.type if isinstance(f.type, type) else globals()[f.type], v) \
                if is_dataclass(f.default_factory() if callable(f.default_factory) else None) else v
    unknown = set(d) - {f.name for f in fields(cls)}
    if unknown:
        raise KeyError(f"unknown config keys for {cls.__name__}: {unknown}")
    return cls(**kw)


def load_config(path: Optional[str] = None, overrides: Optional[List[str]] = None) -> ExperimentConfig:
    d = yaml.safe_load(open(path, encoding="utf-8")) if path else {}
    d = d or {}
    for o in overrides or []:
        key, val = o.split("=", 1)
        cur = d
        *parents, last = key.split(".")
        for p in parents:
            cur = cur.setdefault(p, {})
        v = yaml.safe_load(val)
        if isinstance(v, str):
            try:
                v = float(v)  # YAML 1.1 reads '1e-4' as a string
            except ValueError:
                pass
        cur[last] = v
    return _build(ExperimentConfig, d)


def to_dict(cfg) -> dict:
    return asdict(cfg)
