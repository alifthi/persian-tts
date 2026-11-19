"""Registry of open Persian speech corpora that permit commercial use.

Every source yields `Utterance` records with raw audio (float32 numpy, any sample rate).
Licenses were checked on the dataset cards (Nov 2026); re-verify before a commercial release
and keep the attribution in the model card (see scripts/push_to_hub.py).
"""
from __future__ import annotations

import csv
import io
import os
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterator, Optional

import numpy as np
import soundfile as sf


@dataclass
class Utterance:
    uid: str
    text: str
    speaker: str
    audio: np.ndarray  # float32 mono
    sr: int


@dataclass
class Source:
    name: str
    license: str
    url: str
    description: str
    loader: Callable[..., Iterator[Utterance]]
    commercial_ok: bool = True
    requires_local_path: bool = False


def _to_mono(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x.mean(axis=1) if x.ndim == 2 else x


def _decode(data: bytes) -> tuple[np.ndarray, int]:
    try:
        a, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
        return _to_mono(a), sr
    except Exception:  # mp3 etc. without libsndfile mp3 support
        import librosa
        a, sr = librosa.load(io.BytesIO(data), sr=None, mono=True)
        return a.astype(np.float32), sr


def _hf_parquet_rows(repo: str, columns=None) -> Iterator[dict]:
    """Stream rows from a dataset repo's parquet files, without needing loading scripts."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download

    files = sorted(f for f in HfApi().list_repo_files(repo, repo_type="dataset") if f.endswith(".parquet"))
    for f in files:
        local = hf_hub_download(repo, f, repo_type="dataset")
        pf = pq.ParquetFile(local)
        for batch in pf.iter_batches(batch_size=64, columns=columns):
            yield from batch.to_pylist()


# ---------------------------------------------------------------- Mana-TTS (CC0, ~100h, 1 speaker)
def load_mana_tts(max_cer: float = 0.1, min_quality: str = "MIDDLE", **_) -> Iterator[Utterance]:
    order = {"LOW": 0, "MIDDLE": 1, "HIGH": 2}
    for row in _hf_parquet_rows("MahtaFetrat/Mana-TTS",
                                ["file name", "transcript", "CER", "match quality", "audio", "samplerate"]):
        if row["CER"] is not None and row["CER"] > max_cer:
            continue
        if order.get(str(row["match quality"]).upper(), 0) < order[min_quality]:
            continue
        yield Utterance(f"mana_{Path(row['file name']).stem}", row["transcript"], "mana",
                        np.asarray(row["audio"], dtype=np.float32), int(row["samplerate"]))


# ---------------------------------------------------------------- GPTInformal-Persian (CC0, informal speech)
def load_gptinformal(**_) -> Iterator[Utterance]:
    for row in _hf_parquet_rows("MahtaFetrat/GPTInformal-Persian",
                                ["file name", "transcript", "audio", "samplerate"]):
        yield Utterance(f"gptinf_{Path(row['file name']).stem}", row["transcript"], "gptinformal",
                        np.asarray(row["audio"], dtype=np.float32), int(row["samplerate"]))


# ---------------------------------------------------------------- ParsiGoo (CC0, multi-speaker)
def load_parsigoo(cache_dir: str = "data/raw/parsigoo", **_) -> Iterator[Utterance]:
    from huggingface_hub import hf_hub_download

    zpath = hf_hub_download("Kamtera/ParsiGoo", "datasets.zip", repo_type="dataset")
    with zipfile.ZipFile(zpath) as zf:
        names = set(zf.namelist())
        for meta in sorted(n for n in names if n.endswith("metadata.csv")):
            spk_dir = os.path.dirname(meta)
            speaker = os.path.basename(spk_dir)
            for line in zf.read(meta).decode("utf-8").splitlines():
                cols = line.split("|")
                if len(cols) < 2:
                    continue
                text, wav = cols[0].strip(), cols[1].strip()
                wpath = f"{spk_dir}/wavs/{wav}"
                if wpath not in names:
                    continue
                a, sr = _decode(zf.read(wpath))
                yield Utterance(f"parsigoo_{speaker}_{Path(wav).stem}", text, f"parsigoo_{speaker}", a, sr)


# ---------------------------------------------------------------- Persian-Speech-Dataset (MIT, multi-speaker, emotional)
def load_persian_speech(emotions: Optional[tuple] = ("neutral", "Neutral", "N", None), **_) -> Iterator[Utterance]:
    for row in _hf_parquet_rows("SeyedAli/Persian-Speech-Dataset",
                                ["audio_id", "audio", "speaker_id", "emotion", "transcript"]):
        if emotions is not None and row["emotion"] not in emotions:
            continue
        a, sr = _decode(row["audio"]["bytes"])
        yield Utterance(f"psd_{row['audio_id']}", row["transcript"], f"psd_{row['speaker_id']}", a, sr)


# ---------------------------------------------------------------- FLEURS fa_ir (CC-BY-4.0, read speech, many speakers)
def load_fleurs(splits=("train", "dev"), **_) -> Iterator[Utterance]:
    from huggingface_hub import hf_hub_download

    for split in splits:
        tsv = hf_hub_download("google/fleurs", f"data/fa_ir/{split}.tsv", repo_type="dataset")
        meta = {}
        with open(tsv, encoding="utf-8") as f:
            for cols in csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
                if len(cols) >= 3:
                    meta[cols[1]] = (cols[0], cols[2])
        tgz = hf_hub_download("google/fleurs", f"data/fa_ir/audio/{split}.tar.gz", repo_type="dataset")
        with tarfile.open(tgz, "r:gz") as tf:
            for m in tf:
                base = os.path.basename(m.name)
                if not m.isfile() or base not in meta:
                    continue
                a, sr = _decode(tf.extractfile(m).read())
                # FLEURS does not publish speaker ids; one pseudo-speaker per corpus
                yield Utterance(f"fleurs_{Path(base).stem}", meta[base][1], "fleurs", a, sr)


# ---------------------------------------------------------------- Common Voice fa (CC0, crowd-sourced; local download)
def load_common_voice(path: Optional[str] = None, min_clips_per_speaker: int = 300, min_upvotes: int = 2, **_):
    """`path` = extracted Common Voice 'fa' folder (validated.tsv + clips/), from Mozilla Data Collective."""
    if not path:
        raise ValueError("common_voice requires `path` to the extracted CV 'fa' directory")
    root = Path(path)
    with open(root / "validated.tsv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE))
    rows = [r for r in rows if int(r.get("up_votes") or 0) >= min_upvotes and int(r.get("down_votes") or 0) == 0]
    counts: Dict[str, int] = {}
    for r in rows:
        counts[r["client_id"]] = counts.get(r["client_id"], 0) + 1
    keep = {c for c, n in counts.items() if n >= min_clips_per_speaker}  # TTS needs many clips per voice
    for r in rows:
        if r["client_id"] not in keep:
            continue
        a, sr = _decode((root / "clips" / r["path"]).read_bytes())
        yield Utterance(f"cv_{Path(r['path']).stem}", r["sentence"], f"cv_{r['client_id'][:12]}", a, sr)


SOURCES: Dict[str, Source] = {s.name: s for s in [
    Source("mana_tts", "CC0-1.0", "https://huggingface.co/datasets/MahtaFetrat/Mana-TTS",
           "~114h single-speaker read speech (aligned audiobooks/magazine)", load_mana_tts),
    Source("gptinformal", "CC0-1.0", "https://huggingface.co/datasets/MahtaFetrat/GPTInformal-Persian",
           "~9h single-speaker informal/colloquial Persian", load_gptinformal),
    # Excluded by default: speakers are synthetic output of commercial TTS engines (e.g. Microsoft Edge
    # fa-IR-DilaraNeural/FaridNeural). Vendor terms restrict training speech models on that output,
    # so the CC0 label does not make it safe for commercial use.
    Source("parsigoo", "CC0-1.0 (synthetic, vendor ToS risk)", "https://huggingface.co/datasets/Kamtera/ParsiGoo",
           "multi-speaker corpus generated with commercial TTS engines", load_parsigoo, commercial_ok=False),
    Source("persian_speech", "MIT", "https://huggingface.co/datasets/SeyedAli/Persian-Speech-Dataset",
           "multi-speaker emotional speech (neutral subset used by default)", load_persian_speech),
    Source("fleurs", "CC-BY-4.0", "https://huggingface.co/datasets/google/fleurs",
           "FLEURS fa_ir read speech; attribution required", load_fleurs),
    Source("common_voice", "CC0-1.0", "https://datacollective.mozillafoundation.org",
           "Mozilla Common Voice fa; download manually, pass path", load_common_voice,
           requires_local_path=True),
]}
