"""Tests for the Orpheus sequence format and batch collation.

No weights or network: a stub tokenizer stands in for the Llama tokenizer so
these pin the *format*, which is what silently breaks a fine-tune.
"""

import json

import pytest
import torch

from persian_tts.orpheus.codec import CODES_PER_FRAME, snac_codes_to_flat_tokens
from persian_tts.orpheus.dataset import OrpheusDataset, collate
from persian_tts.orpheus.prompt import (
    BOS,
    END_OF_AI,
    END_OF_HUMAN,
    END_OF_SPEECH,
    EOS,
    IGNORE_INDEX,
    PAD_TOKEN,
    START_OF_AI,
    START_OF_HUMAN,
    START_OF_SPEECH,
    build_example,
    build_generation_prompt,
)
from tests.test_orpheus_codec import _fake_codes


class StubTokenizer:
    """Maps each character to its ordinal; enough to exercise the format."""

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) % 1000 for c in text]}


def _audio_tokens(n_frames=3):
    return snac_codes_to_flat_tokens(_fake_codes(n_frames))


def test_prompt_region_is_fully_loss_masked():
    audio = _audio_tokens()
    ex = build_example(text_ids=[1, 2, 3], audio_tokens=audio)
    n_masked = sum(1 for lab in ex.labels if lab == IGNORE_INDEX)
    # BOS, SOH, 3 text, EOT, EOH
    assert n_masked == 7
    assert ex.labels[:7] == [IGNORE_INDEX] * 7
    assert IGNORE_INDEX not in ex.labels[7:]


def test_supervised_region_is_exactly_the_audio_response():
    audio = _audio_tokens(2)
    ex = build_example(text_ids=[1, 2], audio_tokens=audio)
    supervised = [lab for lab in ex.labels if lab != IGNORE_INDEX]
    assert supervised == [START_OF_AI, START_OF_SPEECH, *audio, END_OF_SPEECH, END_OF_AI, EOS]


def test_eos_is_supervised_so_the_model_learns_to_stop():
    ex = build_example(text_ids=[1], audio_tokens=_audio_tokens(1))
    assert ex.labels[-1] == EOS


def test_generation_prompt_matches_the_training_prefix():
    """Inference must start from the exact distribution training ended on."""
    text_ids, audio = [5, 6, 7], _audio_tokens(2)
    ex = build_example(text_ids=text_ids, audio_tokens=audio)
    prompt = build_generation_prompt(text_ids=text_ids)
    assert ex.input_ids[: len(prompt)] == prompt
    # the very next token the model must produce is the first audio code
    assert ex.input_ids[len(prompt)] == audio[0]


def test_speaker_is_prepended_inside_the_human_turn():
    audio = _audio_tokens(1)
    plain = build_example(text_ids=[9], audio_tokens=audio)
    voiced = build_example(
        text_ids=[9], audio_tokens=audio, speaker="mana", speaker_ids=[70, 71]
    )
    assert voiced.input_ids[:2] == [BOS, START_OF_HUMAN]
    assert voiced.input_ids[2:4] == [70, 71]
    assert len(voiced) == len(plain) + 2


def test_mismatched_lengths_rejected():
    from persian_tts.orpheus.prompt import OrpheusExample

    with pytest.raises(ValueError, match="same length"):
        OrpheusExample(input_ids=[1, 2, 3], labels=[1, 2])


def test_collate_pads_right_and_masks_padding(tmp_path):
    batch = [
        {"input_ids": [1, 2, 3], "labels": [1, 2, 3]},
        {"input_ids": [4, 5], "labels": [IGNORE_INDEX, 5]},
    ]
    out = collate(batch)
    assert out["input_ids"].shape == (2, 3)
    assert out["input_ids"][1].tolist() == [4, 5, PAD_TOKEN]
    assert out["attention_mask"][1].tolist() == [1, 1, 0]
    # padding must not contribute loss
    assert out["labels"][1].tolist() == [IGNORE_INDEX, 5, IGNORE_INDEX]


def _write_manifest(path, rows):
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8"
    )
    return str(path)


def test_dataset_round_trips_a_manifest(tmp_path):
    audio = _audio_tokens(4)
    manifest = _write_manifest(
        tmp_path / "m.jsonl",
        [{"id": "a", "text": "سلام", "speaker": "mana", "audio_tokens": audio}],
    )
    ds = OrpheusDataset(manifest, StubTokenizer(), max_length=4096)
    assert len(ds) == 1
    item = ds[0]
    assert len(item["input_ids"]) == len(item["labels"])
    assert item["input_ids"][-1] == EOS
    # the audio run must survive intact
    assert audio[0] in item["input_ids"]


def test_dataset_rejects_a_manifest_with_a_broken_frame_count(tmp_path):
    """A stale manifest must fail loudly, not train to a normal loss on noise."""
    bad = _audio_tokens(2)[:-1]
    manifest = _write_manifest(
        tmp_path / "bad.jsonl", [{"id": "a", "text": "x", "speaker": "m", "audio_tokens": bad}]
    )
    with pytest.raises(ValueError, match="7-slot Orpheus frames"):
        OrpheusDataset(manifest, StubTokenizer())


def test_dataset_speaker_allow_list(tmp_path):
    audio = _audio_tokens(1)
    manifest = _write_manifest(
        tmp_path / "m.jsonl",
        [
            {"id": "a", "text": "x", "speaker": "mana", "audio_tokens": audio},
            {"id": "b", "text": "y", "speaker": "psd_1", "audio_tokens": audio},
        ],
    )
    ds = OrpheusDataset(manifest, StubTokenizer(), speakers=["mana"])
    assert len(ds) == 1 and ds.rows[0]["speaker"] == "mana"


def test_dataset_raises_rather_than_truncating_audio(tmp_path):
    manifest = _write_manifest(
        tmp_path / "m.jsonl",
        [{"id": "long", "text": "x", "speaker": "m", "audio_tokens": _audio_tokens(50)}],
    )
    ds = OrpheusDataset(manifest, StubTokenizer(), max_length=32)
    with pytest.raises(ValueError, match="prepare_orpheus"):
        ds[0]
