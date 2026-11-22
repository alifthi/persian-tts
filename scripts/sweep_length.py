#!/usr/bin/env python
"""Fine sweep of length_scale at noise_scale=0 (the tuned optimum).

length_scale is the only remaining live inference knob once noise_scale=0 makes the
latent deterministic and noise_w is confirmed inert. Speaking rate strongly affects
intelligibility, and 1.05 was the best of a COARSE grid (1.0/1.05/1.10/1.15) -- the
true optimum may sit between those points.
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
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--asr", default="openai/whisper-small")
    ap.add_argument("--out", default="exports/sweep_length.json")
    a = ap.parse_args()

    texts = [json.loads(l)["text"] for l in open(a.manifest, encoding="utf-8")][: a.n]
    results = []
    for ls in [0.95, 1.00, 1.025, 1.05, 1.075, 1.10, 1.125, 1.15, 1.20, 1.30]:
        wavs, sr = synth_scales(a.onnx, texts, a.speaker, 0.0, ls, 0.8)
        s = score(texts, transcribe(wavs, sr, a.asr))
        s.update(length_scale=ls, total_audio_s=round(sum(len(w) for w in wavs) / sr, 1))
        results.append(s)
        print(f"ls={ls:.3f}  CER {s['cer']:.4f}  WER {s['wer']:.4f}  audio {s['total_audio_s']}s", flush=True)

    results.sort(key=lambda r: r["cer"])
    Path(a.out).write_text(json.dumps(results, indent=1), encoding="utf-8")
    print("\n==== RANKED (best CER first) ====")
    for r in results:
        print(f"ls={r['length_scale']:.3f}  CER {r['cer']:.4f}  WER {r['wer']:.4f}")
