"""Round-trip tests for the Orpheus flat-token <-> SNAC code mapping.

These run without network or weights: the mapping is pure arithmetic, and it is
exactly the piece whose failure mode is silent (a wrong interleave trains to a
normal loss and decodes to noise), so it is worth pinning precisely.
"""

import pytest
import torch

from persian_tts.orpheus.codec import (
    AUDIO_TOKEN_BASE,
    CODES_PER_FRAME,
    LEVEL_WIDTHS,
    SNAC_CODEBOOK_SIZE,
    flat_tokens_to_snac_codes,
    is_audio_token,
    snac_codes_to_flat_tokens,
)


def _fake_codes(n_frames: int, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    return [
        torch.randint(
            0, SNAC_CODEBOOK_SIZE, (1, n_frames * width), generator=g, dtype=torch.long
        )
        for width in LEVEL_WIDTHS
    ]


@pytest.mark.parametrize("n_frames", [1, 2, 17])
def test_round_trip_is_identity(n_frames):
    codes = _fake_codes(n_frames)
    tokens = snac_codes_to_flat_tokens(codes)
    assert len(tokens) == n_frames * CODES_PER_FRAME
    back = flat_tokens_to_snac_codes(tokens)
    for original, restored in zip(codes, back):
        assert torch.equal(original, restored)


def test_every_slot_lands_in_its_own_band():
    """Slot i must occupy [base + i*4096, base + (i+1)*4096) and nothing else."""
    tokens = snac_codes_to_flat_tokens(_fake_codes(8, seed=1))
    for i, token in enumerate(tokens):
        slot = i % CODES_PER_FRAME
        low = AUDIO_TOKEN_BASE + slot * SNAC_CODEBOOK_SIZE
        assert low <= token < low + SNAC_CODEBOOK_SIZE
        assert is_audio_token(token)


def test_partial_trailing_frame_is_dropped():
    """Generation stops wherever the LM stops; a partial frame cannot decode."""
    tokens = snac_codes_to_flat_tokens(_fake_codes(3))
    truncated = tokens[: 2 * CODES_PER_FRAME + 3]
    back = flat_tokens_to_snac_codes(truncated)
    for level, width in enumerate(LEVEL_WIDTHS):
        assert back[level].shape == (1, 2 * width)


def test_misaligned_sequence_raises_instead_of_decoding_noise():
    tokens = snac_codes_to_flat_tokens(_fake_codes(4))
    with pytest.raises(ValueError, match="not in that slot's band"):
        flat_tokens_to_snac_codes(tokens[1:])


def test_non_audio_token_raises():
    tokens = snac_codes_to_flat_tokens(_fake_codes(2))
    tokens[0] = 42  # a text token
    with pytest.raises(ValueError, match="not in that slot's band"):
        flat_tokens_to_snac_codes(tokens)


def test_level_width_mismatch_raises():
    codes = _fake_codes(4)
    codes[1] = codes[1][:, :-1]
    with pytest.raises(ValueError, match="SNAC level 1"):
        snac_codes_to_flat_tokens(codes)
