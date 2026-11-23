"""LoRA fine-tuning of Orpheus-3B on Persian.

Full fine-tuning a 3B model needs ~48 GB of optimizer state and is not on the
table for this project's hardware. LoRA on the attention and MLP projections
trains ~0.1-1% of the parameters and, for adapting a multilingual-pretrained
TTS LM to one language and one voice, is the standard and sufficient choice.

The embedding layer is the exception worth noting: Orpheus's audio-code rows
(the 28672 tokens from AUDIO_TOKEN_BASE up) are already trained, because the
base checkpoint is a *pretrained TTS* model rather than a text LLM with new
tokens bolted on. That is exactly why the pretrained variant is the right
starting point and why the embedding table does not need to be unfrozen.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import List, Optional

import yaml

#: Llama attention + MLP projections. Targeting the MLP as well as attention
#: matters for TTS: prosody and timbre live in the feed-forward stack, and
#: attention-only LoRA underfits noticeably on speech.
DEFAULT_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


@dataclass
class OrpheusTrainConfig:
    """Config for `scripts/train_orpheus.py`, mirrored in configs/orpheus_*.yaml."""

    # --- model ---
    base_model: str = "unsloth/orpheus-3b-0.1-pretrained"
    snac_model: str = "hubertsiuzdak/snac_24khz"
    #: 4-bit NF4 base weights. Required to fit 3B on a 4 GB card; costs some
    #: quality versus bf16 and should be off whenever VRAM allows.
    load_in_4bit: bool = True
    attn_implementation: str = "sdpa"

    # --- LoRA ---
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: list(DEFAULT_TARGET_MODULES))

    # --- data ---
    train_manifest: str = "data/processed/orpheus_train.jsonl"
    val_manifest: str = "data/processed/orpheus_val.jsonl"
    speakers: Optional[List[str]] = None
    max_length: int = 2048
    #: Drop utterances with more than this many audio tokens (0 = keep all).
    #: Transformer cost is superlinear in sequence length, so on CPU a few long
    #: clips dominate the epoch. 512 tokens ~= 6.2s and keeps 76% of `mana`.
    max_audio_tokens: int = 0

    # --- optimisation ---
    learning_rate: float = 2e-4
    lr_scheduler: str = "cosine"
    warmup_steps: int = 100
    max_steps: int = 4000
    batch_size: int = 1
    grad_accum: int = 16
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    #: bf16 on Ampere+, fp16 on older cards, fp32 on CPU. Resolved at runtime.
    precision: str = "bf16"
    gradient_checkpointing: bool = True
    seed: int = 1234

    # --- bookkeeping ---
    output_dir: str = "runs/orpheus-fa"
    log_every: int = 10
    eval_every: int = 250
    save_every: int = 250
    keep_last: int = 3

    def effective_batch(self) -> int:
        return self.batch_size * self.grad_accum


def load_orpheus_config(
    path: Optional[str] = None, overrides: Optional[List[str]] = None
) -> OrpheusTrainConfig:
    """Load YAML + dotted CLI overrides, rejecting unknown keys.

    Same contract as ``persian_tts.config.load_config``: a typo in a config key
    must be an error, not a silently-ignored setting that makes a multi-hour run
    train with the wrong hyperparameters.
    """
    d = yaml.safe_load(open(path, encoding="utf-8")) if path else {}
    d = d or {}
    for override in overrides or []:
        key, val = override.split("=", 1)
        parsed = yaml.safe_load(val)
        if isinstance(parsed, str):
            try:
                parsed = float(parsed)  # YAML 1.1 reads '2e-4' as a string
            except ValueError:
                pass
        d[key] = parsed

    known = {f.name for f in fields(OrpheusTrainConfig)}
    unknown = set(d) - known
    if unknown:
        raise KeyError(
            f"unknown config keys for OrpheusTrainConfig: {sorted(unknown)}; "
            f"known keys: {sorted(known)}"
        )
    return OrpheusTrainConfig(**d)


def orpheus_config_to_dict(cfg: OrpheusTrainConfig) -> dict:
    return asdict(cfg)
