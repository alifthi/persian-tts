"""SNAC audio codes <-> Orpheus flat token ids.

SNAC (`hubertsiuzdak/snac_24khz`) is a *multi-scale* residual codec. For a given
waveform it returns three code tensors at three different rates::

    level 0:  T    codes   (~12.5 Hz)
    level 1:  2T   codes   (~25   Hz)
    level 2:  4T   codes   (~50   Hz)

Orpheus flattens one such "frame" (1 + 2 + 4 = 7 codes covering the same slice of
time) into a strictly increasing run of 7 token ids, so a plain causal LM can
model audio as text. Each of the 7 slots gets its own 4096-wide band of the
vocabulary, which means the LM never has to disambiguate *which* codebook a code
came from -- the position in the band already says so.

    token = AUDIO_TOKEN_BASE + slot * SNAC_CODEBOOK_SIZE + code

The interleave order below is Orpheus's and is not arbitrary: it is causal in
time. Within a frame, slots are emitted so that every coarse code precedes the
finer codes that refine the same half-frame::

    slot 0: L0[t]        coarse, whole frame
    slot 1: L1[2t]       first half
    slot 2: L2[4t]       first quarter
    slot 3: L2[4t+1]     second quarter
    slot 4: L1[2t+1]     second half
    slot 5: L2[4t+2]     third quarter
    slot 6: L2[4t+3]     fourth quarter

Getting this order wrong produces a model that trains to a perfectly normal loss
and decodes to noise, so :func:`snac_codes_to_flat_tokens` and
:func:`flat_tokens_to_snac_codes` are exact inverses and are tested as such.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import torch

#: First token id of the audio-code region of the Orpheus vocabulary.
#: ``<custom_token_10>`` == 128266 in the Llama-3 tokenizer Orpheus inherits.
AUDIO_TOKEN_BASE = 128266

#: Codes per SNAC codebook (and therefore the stride between slot bands).
SNAC_CODEBOOK_SIZE = 4096

#: Flat tokens emitted per SNAC frame (1 coarse + 2 mid + 4 fine).
CODES_PER_FRAME = 7

#: Audio sample rate SNAC-24khz operates at. Input audio must be resampled.
SNAC_SAMPLE_RATE = 24000

#: Measured SNAC frame rate on this corpus (nominal 12.5; the encoder's strides
#: do not divide 24 kHz exactly). Used to convert seconds <-> token budgets.
FRAMES_PER_SECOND = 11.77

#: (level, index-within-frame) that each flat slot draws from.
#: Index is relative to the start of the frame at that level.
SLOT_LAYOUT = (
    (0, 0),
    (1, 0),
    (2, 0),
    (2, 1),
    (1, 1),
    (2, 2),
    (2, 3),
)

#: How many codes each SNAC level contributes per frame.
LEVEL_WIDTHS = (1, 2, 4)


def snac_codes_to_flat_tokens(codes: Sequence[torch.Tensor]) -> List[int]:
    """Flatten a batch-of-one SNAC code triple into Orpheus audio token ids.

    Args:
        codes: the three tensors returned by ``SNAC.encode``, each shaped
            ``(1, T * LEVEL_WIDTHS[level])``.

    Returns:
        A list of ``7 * T`` token ids, already offset into the audio region.
    """
    if len(codes) != len(LEVEL_WIDTHS):
        raise ValueError(f"expected {len(LEVEL_WIDTHS)} SNAC levels, got {len(codes)}")

    flat_levels = [c.reshape(-1).tolist() for c in codes]
    n_frames = len(flat_levels[0])
    for level, width in enumerate(LEVEL_WIDTHS):
        expected = n_frames * width
        if len(flat_levels[level]) != expected:
            raise ValueError(
                f"SNAC level {level} has {len(flat_levels[level])} codes, "
                f"expected {expected} for {n_frames} frames"
            )

    tokens: List[int] = []
    for t in range(n_frames):
        for slot, (level, offset) in enumerate(SLOT_LAYOUT):
            code = flat_levels[level][t * LEVEL_WIDTHS[level] + offset]
            if not 0 <= code < SNAC_CODEBOOK_SIZE:
                raise ValueError(f"SNAC code {code} out of range at frame {t} slot {slot}")
            tokens.append(AUDIO_TOKEN_BASE + slot * SNAC_CODEBOOK_SIZE + code)
    return tokens


def flat_tokens_to_snac_codes(
    tokens: Sequence[int], device: Optional[torch.device] = None
) -> List[torch.Tensor]:
    """Inverse of :func:`snac_codes_to_flat_tokens`.

    Trailing tokens that do not complete a 7-slot frame are dropped: a generated
    sequence is truncated wherever the LM happened to stop, and a partial frame
    cannot be decoded.

    Raises:
        ValueError: if a token falls outside the audio region or lands in a band
            that does not match its position in the frame. Both indicate the
            sequence was not produced by this layout, which must fail loudly
            rather than decode to noise.
    """
    n_frames = len(tokens) // CODES_PER_FRAME
    levels: List[List[int]] = [[] for _ in LEVEL_WIDTHS]

    for t in range(n_frames):
        for slot, (level, _offset) in enumerate(SLOT_LAYOUT):
            token = int(tokens[t * CODES_PER_FRAME + slot])
            code = token - AUDIO_TOKEN_BASE - slot * SNAC_CODEBOOK_SIZE
            if not 0 <= code < SNAC_CODEBOOK_SIZE:
                raise ValueError(
                    f"token {token} at frame {t} slot {slot} is not in that slot's band "
                    f"[{AUDIO_TOKEN_BASE + slot * SNAC_CODEBOOK_SIZE}, "
                    f"{AUDIO_TOKEN_BASE + (slot + 1) * SNAC_CODEBOOK_SIZE}); "
                    "the sequence is misaligned or not Orpheus audio tokens"
                )
            levels[level].append(code)

    return [
        torch.tensor(level, dtype=torch.long, device=device).unsqueeze(0)
        for level in levels
    ]


def is_audio_token(token: int) -> bool:
    """True if ``token`` lies anywhere in the 7-band audio-code region."""
    return (
        AUDIO_TOKEN_BASE
        <= token
        < AUDIO_TOKEN_BASE + CODES_PER_FRAME * SNAC_CODEBOOK_SIZE
    )


class SnacTokenizer:
    """Lazily-loaded SNAC encoder/decoder that speaks Orpheus flat tokens.

    Held separate from the dataset and the trainer so preprocessing (CPU, run
    once over the corpus) and synthesis (loads the LM too) share one
    implementation of the code<->token mapping.
    """

    def __init__(
        self,
        repo_id: str = "hubertsiuzdak/snac_24khz",
        device: str = "cpu",
    ) -> None:
        self.repo_id = repo_id
        self.device = device
        self._model = None

    @property
    def model(self):
        """The underlying ``SNAC`` module, loaded on first use."""
        if self._model is None:
            from snac import SNAC

            self._model = SNAC.from_pretrained(self.repo_id).eval().to(self.device)
        return self._model

    def encode(self, wav: torch.Tensor) -> List[int]:
        """Encode a mono 24 kHz waveform to Orpheus audio token ids.

        Args:
            wav: 1-D float tensor of samples at :data:`SNAC_SAMPLE_RATE`.
        """
        if wav.ndim != 1:
            raise ValueError(f"expected a 1-D mono waveform, got shape {tuple(wav.shape)}")
        x = wav.to(self.device)[None, None]
        with torch.inference_mode():
            codes = self.model.encode(x)
        return snac_codes_to_flat_tokens([c.cpu() for c in codes])

    def decode(self, tokens: Sequence[int]) -> torch.Tensor:
        """Decode Orpheus audio token ids back to a 1-D 24 kHz waveform."""
        codes = flat_tokens_to_snac_codes(tokens, device=torch.device(self.device))
        with torch.inference_mode():
            wav = self.model.decode(codes)
        return wav[0, 0].detach().cpu()
