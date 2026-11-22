#!/usr/bin/env python
"""Repeat-measure the top inference configs to separate real gains from sampling noise.

VITS inference is stochastic, so a single CER number is a sample, not a property.
This runs each config REPEATS times and reports mean/std, which is what you need
before baking a default into a release.
"""
import argparse, json, statistics, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.eval_asr import score, transcribe        # noqa: E402
from scripts.sweep_inference import synth_scales      # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--manifest", default="data/processed/val.jsonl")
    ap.add_argument("--speaker", default="mana")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--asr", default="openai/whisper-small")
    ap.add_argument("--out", default="exports/verify_scales.json")
    a = ap.parse_args()

    texts = [json.loads(l)["text"] for l in open(a.manifest, encoding="utf-8")][: a.n]
    CANDIDATES = [
        (0.667, 1.00, 0.8),   # old default
        (0.150, 1.05, 0.8),
        (0.000, 1.05, 0.8),
        (0.333, 1.10, 0.8),
        (0.250, 1.05, 0.8),
    ]
    out = []
    for ns, ls, nw in CANDIDATES:
        cers, wers = [], []
        for r in range(a.repeats):
            wavs, sr = synth_scales(a.onnx, texts, a.speaker, ns, ls, nw)
            s = score(texts, transcribe(wavs, sr, a.asr))
            cers.append(s["cer"]); wers.append(s["wer"])
            print(f"  ns={ns} ls={ls} rep{r+1}: CER {s['cer']:.4f}", flush=True)
        rec = dict(noise_scale=ns, length_scale=ls, noise_w=nw,
                   cer_mean=statistics.mean(cers), wer_mean=statistics.mean(wers),
                   cer_std=statistics.pstdev(cers) if len(cers) > 1 else 0.0,
                   cer_runs=cers)
        out.append(rec)
        print(f"ns={ns} ls={ls} -> CER {rec['cer_mean']:.4f} +/- {rec['cer_std']:.4f}", flush=True)

    out.sort(key=lambda r: r["cer_mean"])
    Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("\n==== MEAN OF", a.repeats, "RUNS (best first) ====")
    for r in out:
        print(f"ns={r['noise_scale']:.3f} ls={r['length_scale']:.2f}  "
              f"CER {r['cer_mean']:.4f} +/- {r['cer_std']:.4f}   WER {r['wer_mean']:.4f}")
    base = next(r for r in out if r["noise_scale"] == 0.667)
    best = out[0]
    print(f"\nbest vs default: {base['cer_mean']:.4f} -> {best['cer_mean']:.4f} "
          f"({(base['cer_mean'] - best['cer_mean']) / base['cer_mean'] * 100:+.2f}% relative)")
    print(f"gap is {(base['cer_mean'] - best['cer_mean']) / max(best['cer_std'], 1e-9):.1f}x the run-to-run std")
