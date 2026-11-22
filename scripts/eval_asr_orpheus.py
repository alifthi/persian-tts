#!/usr/bin/env python
"""ASR intelligibility eval for Orpheus adapters, comparable to eval_asr.py.

Same metric and the same normalization as the VITS evaluation, so an Orpheus
CER sits on the one scale this project already uses::

    ground truth (ceiling)   CER 0.179 / WER 0.573
    VITS phonemes + swept    CER 0.217 / WER 0.672

Usage::

    python scripts/eval_asr_orpheus.py \
        --adapter base=none fa=runs/orpheus-fa/step_00004000 --n 24

Always include the base model. The absolute numbers carry Whisper's own Persian
error rate, so only the difference between entries in one run is meaningful --
and without the base column there is nothing to attribute a change to.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Reuse the VITS eval's scoring verbatim; two implementations of CER would
# silently drift and make the two model lines incomparable, which is the whole
# point of running this.
from scripts.eval_asr import score, transcribe  # noqa: E402


def synth_orpheus(texts, adapter, base_model, speaker, temperature, top_p, seed):
    """Synthesize via the CLI so eval exercises the exact shipped path.

    A separate process per utterance is slow but keeps a 3B model from sitting
    resident next to a training job -- this repo has lost runs to the OOM killer
    picking the trainer when side work inflated total RSS.
    """
    wavs = []
    with tempfile.TemporaryDirectory() as td:
        for i, text in enumerate(texts):
            out = Path(td) / f"{i}.wav"
            cmd = [
                sys.executable,
                str(Path(__file__).resolve().parent / "synthesize_orpheus.py"),
                "--text", text,
                "--out", str(out),
                "--base-model", base_model,
                "--speaker", speaker,
                "--temperature", str(temperature),
                "--top-p", str(top_p),
            ]
            if seed is not None:
                cmd += ["--seed", str(seed)]
            if adapter and adapter != "none":
                cmd += ["--adapter", adapter]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode or not out.exists():
                print(f"  utt {i} failed: {r.stderr.strip()[-200:]}", flush=True)
                wavs.append(np.zeros(16000, dtype=np.float32))  # scores as total loss
                continue
            wavs.append(sf.read(out, dtype="float32")[0])
            print(f"  synthesized {i+1}/{len(texts)}", flush=True)
    return wavs, 24000


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--adapter",
        nargs="+",
        required=True,
        help="label=path entries; use 'none' as the path for the base model",
    )
    ap.add_argument("--base-model", default="unsloth/orpheus-3b-0.1-pretrained")
    ap.add_argument("--manifest", default="data/processed/val.jsonl")
    ap.add_argument("--speaker", default="mana")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--asr", default="openai/whisper-small")
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=1234, help="fixed so reruns are comparable")
    ap.add_argument("--out", default="exports/asr_eval_orpheus.json")
    a = ap.parse_args()

    rows = [
        json.loads(l)
        for l in Path(a.manifest).read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    rows = [r for r in rows if 20 <= len(r["text"]) <= 160][: a.n]
    texts = [r["text"] for r in rows]
    print(f"{len(texts)} sentences, speaker={a.speaker}, asr={a.asr}", flush=True)

    root = Path(a.manifest).parent
    gt = [sf.read(root / r["audio"], dtype="float32")[0] for r in rows]
    gt_sr = sf.read(root / rows[0]["audio"])[1]
    sets = [("ground_truth", gt, gt_sr)]

    for spec in a.adapter:
        label, _, path = spec.rpartition("=")
        label = label or Path(path).name
        print(f"synthesizing with {label} ({path}) ...", flush=True)
        wavs, sr = synth_orpheus(
            texts, path, a.base_model, a.speaker, a.temperature, a.top_p, a.seed
        )
        sets.append((label, wavs, sr))

    results = {}
    for label, wavs, sr in sets:
        print(f"ASR on {label} ...", flush=True)
        results[label] = score(texts, transcribe(wavs, sr, a.asr))
        print(f"  {label}: CER {results[label]['cer']:.3f} WER {results[label]['wer']:.3f}", flush=True)

    print("\n{:<22s} {:>8s} {:>8s}".format("model", "CER", "WER"))
    for k, v in results.items():
        print("{:<22s} {:>8.3f} {:>8.3f}".format(k, v["cer"], v["wer"]))

    # Quantified diff against the base, which is what actually answers
    # "did the fine-tune help".
    if "base" in results:
        b = results["base"]
        for k, v in results.items():
            if k in ("base", "ground_truth"):
                continue
            d = b["cer"] - v["cer"]
            print(
                f"\n{k} vs base: CER {b['cer']:.3f} -> {v['cer']:.3f} "
                f"({d:+.3f}, {d/b['cer']*100:+.1f}% relative)"
            )

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(
        json.dumps(
            dict(asr=a.asr, speaker=a.speaker, n=len(texts),
                 temperature=a.temperature, top_p=a.top_p, results=results),
            indent=2, ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
