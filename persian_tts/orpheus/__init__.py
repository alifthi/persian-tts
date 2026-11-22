"""Orpheus-3B Persian fine-tune: a transformers/Llama + SNAC codec TTS stack.

This is a second, independent model line alongside the VITS stack in
``persian_tts.models``. Where VITS predicts a mel spectrogram and vocodes it,
Orpheus is a 3B decoder-only LLM that autoregressively predicts *discrete SNAC
audio codes* from text, which a neural codec decodes to 24 kHz waveform.

The practical consequences for this repo:

* Prosody comes from the LM, not from a duration predictor, which is what the
  VITS run cannot fix (``use_sdp: false`` gives flat averaged durations).
* Waveform quality is the SNAC decoder's, not a 256-channel HiFi-GAN's, which
  removes the other structural ceiling documented in the README.
* Both the base weights (Apache-2.0, ``unsloth/orpheus-3b-0.1-pretrained``) and
  SNAC (MIT) are commercially licensable, matching this project's existing
  licence gate. Higher-MOS alternatives (F5-TTS, MaskGCT, OuteTTS, XTTS-v2) are
  all non-commercial and were rejected for that reason, not for quality.
"""

from persian_tts.orpheus.codec import (
    AUDIO_TOKEN_BASE,
    CODES_PER_FRAME,
    SNAC_CODEBOOK_SIZE,
    SNAC_SAMPLE_RATE,
    SnacTokenizer,
    flat_tokens_to_snac_codes,
    is_audio_token,
    snac_codes_to_flat_tokens,
)

__all__ = [
    "AUDIO_TOKEN_BASE",
    "CODES_PER_FRAME",
    "SNAC_CODEBOOK_SIZE",
    "SNAC_SAMPLE_RATE",
    "SnacTokenizer",
    "flat_tokens_to_snac_codes",
    "is_audio_token",
    "snac_codes_to_flat_tokens",
]
