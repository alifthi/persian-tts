"""Dataset, collate function and length-bucketed (distributed) batch sampler."""
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset, Sampler

from ..text import get_frontend, text_to_ids
from .mel import spectrogram


class TTSDataset(Dataset):
    def __init__(self, manifest: str, root: str, speakers: List[str], cfg_audio, add_blank=True,
                 max_text_len: int = 1000, frontend: str = "chars"):
        self.root = Path(root)
        self.cfg = cfg_audio
        self.add_blank = add_blank
        self.spk2id: Dict[str, int] = {s: i for i, s in enumerate(speakers)}
        rows = [json.loads(l) for l in Path(manifest).read_text(encoding="utf-8").splitlines() if l.strip()]
        _, encode = get_frontend(frontend)
        self.items = []
        for r in rows:
            if r["speaker"] not in self.spk2id:
                continue
            ids = (text_to_ids(r["text"], add_blank=add_blank, already_normalized=True) if frontend == "chars"
                   else encode(r["text"], already_normalized=True))
            if 3 < len(ids) <= max_text_len:
                self.items.append((r["audio"], ids, self.spk2id[r["speaker"]], r["duration"]))
        if len(self.items) < len(rows):
            print(f"{manifest}: kept {len(self.items)}/{len(rows)} utterances (unknown speaker or text length)")
        # frame lengths used by the bucket sampler
        self.lengths = [int(d * cfg_audio.sample_rate / cfg_audio.hop_length) for *_, d in self.items]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, ids, spk, _ = self.items[i]
        wav, sr = sf.read(self.root / path, dtype="float32")
        assert sr == self.cfg.sample_rate, f"{path}: sr {sr} != {self.cfg.sample_rate}; re-run prepare_data"
        wav = torch.from_numpy(wav)
        spec = spectrogram(wav.unsqueeze(0), self.cfg.n_fft, self.cfg.hop_length, self.cfg.win_length).squeeze(0)
        n = spec.shape[-1]
        wav = wav[: n * self.cfg.hop_length]
        return torch.LongTensor(ids), spec, wav, spk


def collate(batch):
    batch = sorted(batch, key=lambda x: x[1].shape[-1], reverse=True)
    B = len(batch)
    tl = torch.LongTensor([b[0].numel() for b in batch])
    sl = torch.LongTensor([b[1].shape[-1] for b in batch])
    wl = torch.LongTensor([b[2].numel() for b in batch])
    text = torch.zeros(B, tl.max(), dtype=torch.long)
    spec = torch.zeros(B, batch[0][1].shape[0], sl.max())
    wav = torch.zeros(B, 1, wl.max())
    for i, (t, s, w, _) in enumerate(batch):
        text[i, : t.numel()] = t
        spec[i, :, : s.shape[-1]] = s
        wav[i, 0, : w.numel()] = w
    sid = torch.LongTensor([b[3] for b in batch])
    return dict(text=text, text_lengths=tl, spec=spec, spec_lengths=sl, wav=wav, wav_lengths=wl, sid=sid)


class BucketBatchSampler(Sampler):
    """Groups utterances of similar length; batch size is in *frames* to keep memory flat.

    Works for single GPU and DDP (each rank gets a disjoint, equally sized slice of batches).
    """

    def __init__(self, lengths, max_frames_per_batch: int, max_batch_size: int = 64,
                 num_replicas: int = 1, rank: int = 0, shuffle=True, seed: int = 0, bucket_size_mult: int = 100):
        self.lengths = np.asarray(lengths)
        self.max_frames = max_frames_per_batch
        self.max_bs = max_batch_size
        self.num_replicas, self.rank = num_replicas, rank
        self.shuffle, self.seed, self.epoch = shuffle, seed, 0
        self.mult = bucket_size_mult
        self._batches = self._make_batches()

    def set_epoch(self, epoch: int):
        self.epoch = epoch
        self._batches = self._make_batches()

    def _make_batches(self):
        rng = random.Random(self.seed + self.epoch)
        idx = list(range(len(self.lengths)))
        if self.shuffle:
            rng.shuffle(idx)
        chunk = self.max_bs * self.mult
        batches = []
        for c in range(0, len(idx), chunk):
            part = sorted(idx[c:c + chunk], key=lambda i: self.lengths[i])
            cur, cur_max = [], 0
            for i in part:
                m = max(cur_max, self.lengths[i])
                if cur and (m * (len(cur) + 1) > self.max_frames or len(cur) >= self.max_bs):
                    batches.append(cur)
                    cur, m = [], self.lengths[i]
                cur.append(i)
                cur_max = m
            if cur:
                batches.append(cur)
        if self.shuffle:
            rng.shuffle(batches)
        n = math.ceil(len(batches) / self.num_replicas) * self.num_replicas
        batches += batches[: n - len(batches)]  # pad so every rank has the same count
        return batches[self.rank::self.num_replicas]

    def __iter__(self):
        return iter(self._batches)

    def __len__(self):
        return len(self._batches)
