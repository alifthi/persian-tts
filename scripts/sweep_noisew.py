#!/usr/bin/env python
"""Sweep noise_w (duration-predictor noise) at the tuned noise_scale/length_scale.

noise_w governs per-phoneme duration jitter, i.e. rhythm. It was never swept
independently: the earlier grid only varied it alongside noise_scale, where the
much larger noise_scale effect masked it entirely.
"""
import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.eval_asr import score, transcribe        # noqa: E402
from scripts.sweep_inference import synth_scales      # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--manifest", default="data/processed/val.jsonl")
    ap.add_argument("--speaker", default="mana")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--noise-scale", type=float, default=0.0)
    ap.add_argument("--length-scale", type=float, default=1.05)
    ap.add_argument("--asr", default="openai/whisper-small")
    ap.add_argument("--out", default="exports/sweep_noisew.json")
    a = ap.parse_args()

    texts = [json.loads(l)["text"] for l in open(a.manifest, encoding="utf-8")][: a.n]
    results = []
    for nw in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        wavs, sr = synth_scales(a.onnx, texts, a.speaker, a.noise_scale, a.length_scale, nw)
        s = score(texts, transcribe(wavs, sr, a.asr))
        dur = sum(len(w) for w in wavs) / sr
        s.update(noise_w=nw, total_audio_s=round(dur, 2))
        results.append(s)
        print(f"noise_w={nw:.1f}  CER {s['cer']:.4f}  WER {s['wer']:.4f}  audio {dur:.1f}s", flush=True)

    results.sort(key=lambda r: r["cer"])
    Path(a.out).write_text(json.dumps(results, indent=1), encoding="utf-8")
    print("\n==== RANKED ====")
    for r in results:
        print(f"noise_w={r['noise_w']:.1f}  CER {r['cer']:.4f}  WER {r['wer']:.4f}")
