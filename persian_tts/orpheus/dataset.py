"""Dataset and collator for Orpheus fine-tuning.

Audio is *not* encoded here. SNAC encoding costs seconds of CPU per utterance
(measured on this box), which would dominate training step time and re-encode
the same clip every epoch. ``scripts/prepare_orpheus.py`` runs it once over the
corpus and writes a token manifest; this module only reads that manifest.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
from torch.utils.data import Dataset

from ..text.normalizer import normalize
from .codec import CODES_PER_FRAME, is_audio_token
from .prompt import IGNORE_INDEX, PAD_TOKEN, build_example


class OrpheusDataset(Dataset):
    """Reads the token manifest produced by ``scripts/prepare_orpheus.py``.

    Each line is ``{"text": str, "speaker": str, "audio_tokens": [int, ...]}``.
    Text is tokenized lazily (cheap) while audio tokens are read as-is (the
    expensive part, already done offline).
    """

    def __init__(
        self,
        manifest: str,
        tokenizer,
        max_length: int = 2048,
        speakers: Optional[Sequence[str]] = None,
        max_audio_tokens: int = 0,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self._speaker_cache: Dict[str, List[int]] = {}

        allowed = set(speakers) if speakers else None
        self.rows: List[dict] = []
        skipped_speaker = skipped_long = skipped_bad = 0

        for line in Path(manifest).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if allowed is not None and row["speaker"] not in allowed:
                skipped_speaker += 1
                continue
            tokens = row["audio_tokens"]
            # A manifest written by an older/incompatible layout must not reach
            # the trainer: it would train to a normal loss and decode to noise.
            if not tokens or len(tokens) % CODES_PER_FRAME or not is_audio_token(tokens[0]):
                skipped_bad += 1
                continue
            # Transformer cost grows superlinearly in sequence length, so on CPU
            # a handful of long utterances dominate the epoch. Capping the audio
            # length trades corpus coverage for step rate; 0 disables.
            if max_audio_tokens and len(tokens) > max_audio_tokens:
                skipped_long += 1
                continue
            self.rows.append(row)

        if skipped_bad:
            raise ValueError(
                f"{manifest}: {skipped_bad} rows have audio_tokens that are not a whole "
                "number of 7-slot Orpheus frames. Rebuild with scripts/prepare_orpheus.py."
            )
        if skipped_speaker:
            print(f"{manifest}: skipped {skipped_speaker} rows (speaker not in allow-list)")
        if skipped_long:
            print(
                f"{manifest}: skipped {skipped_long} rows longer than "
                f"{max_audio_tokens} audio tokens"
            )

        # Built once so the trainer can report, and so the length sampler can
        # bucket without re-tokenizing.
        self.lengths = [len(r["audio_tokens"]) for r in self.rows]
        self._skipped_long = skipped_long

    def _speaker_ids(self, speaker: str) -> List[int]:
        if speaker not in self._speaker_cache:
            self._speaker_cache[speaker] = self.tokenizer(
                f"{speaker}: ", add_special_tokens=False
            )["input_ids"]
        return self._speaker_cache[speaker]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        text_ids = self.tokenizer(
            normalize(row["text"]), add_special_tokens=False
        )["input_ids"]
        example = build_example(
            text_ids=text_ids,
            audio_tokens=row["audio_tokens"],
            speaker=row.get("speaker"),
            speaker_ids=self._speaker_ids(row["speaker"]) if row.get("speaker") else None,
        )
        # Truncating would cut the audio mid-utterance and teach the model to
        # stop early, so over-long examples are dropped at prepare time instead;
        # this is a last-resort guard against a stale manifest.
        if len(example) > self.max_length:
            raise ValueError(
                f"example {row.get('id', idx)} is {len(example)} tokens > max_length "
                f"{self.max_length}; re-run scripts/prepare_orpheus.py with a matching "
                "--max-length so it is filtered offline"
            )
        return {"input_ids": example.input_ids, "labels": example.labels}


def collate(batch: List[dict]) -> Dict[str, torch.Tensor]:
    """Right-pad a batch to its longest example.

    Padding is right-side with :data:`PAD_TOKEN` and ``attention_mask`` 0, and
    the pad positions are ``IGNORE_INDEX`` in labels so they contribute no loss.
    """
    width = max(len(b["input_ids"]) for b in batch)
    input_ids, labels, mask = [], [], []
    for b in batch:
        pad = width - len(b["input_ids"])
        input_ids.append(b["input_ids"] + [PAD_TOKEN] * pad)
        labels.append(b["labels"] + [IGNORE_INDEX] * pad)
        mask.append([1] * len(b["input_ids"]) + [0] * pad)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(mask, dtype=torch.long),
    }
