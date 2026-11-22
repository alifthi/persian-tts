#!/usr/bin/env python
"""Grid-search ONNX inference scales (noise_scale, length_scale, noise_w) against ASR CER/WER.

Training is converged; these knobs are free quality. Piper defaults (0.667, 1.0, 0.8) were
never tuned for this voice. Synthesis is done once per config, transcription once per config
with the SAME Whisper model, so the numbers are directly comparable.

    python scripts/sweep_inference.py --onnx exports/onnx_final/model.onnx --n 16
"""
import argparse, json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from persian_tts.text import get_frontend          # noqa: E402
from scripts.eval_asr import score, transcribe     # noqa: E402


def synth_scales(onnx_path, texts, speaker, ns, ls, nw):
    import onnxruntime as ort
    cfg = json.loads(Path(onnx_path + ".json").read_text(encoding="utf-8"))
    _, encode = get_frontend(cfg.get("text_frontend", "chars"))
    sid = np.array([cfg["speaker_id_map"][speaker]], dtype=np.int64)
    scales = np.array([ns, ls, nw], dtype=np.float32)
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    out = []
    for t in texts:
        ids = np.array([encode(t)], dtype=np.int64)
        a = sess.run(None, {"input": ids, "input_lengths": np.array([ids.shape[1]], dtype=np.int64),
                            "scales": scales, "sid": sid})[0]
        out.append(np.asarray(a, dtype=np.float32).reshape(-1))
    return out, cfg["audio"]["sample_rate"]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--manifest", default="data/processed/val.jsonl")
    ap.add_argument("--speaker", default="mana")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--asr", default="openai/whisper-small")
    ap.add_argument("--out", default="exports/sweep_inference.json")
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.manifest, encoding="utf-8")][: a.n]
    texts = [r["text"] for r in rows]

    # (noise_scale, length_scale, noise_w); first entry is the Piper default = baseline
    GRID = [
        (0.667, 1.00, 0.8),   # Piper default = baseline, keep first
        (0.000, 1.05, 0.8),   # fully deterministic latent
        (0.150, 1.05, 0.8),
        (0.250, 1.00, 0.8),
        (0.250, 1.05, 0.8),
        (0.250, 1.10, 0.8),
        (0.333, 1.05, 0.8),   # previous winner
        (0.333, 1.10, 0.8),
        (0.333, 1.15, 0.8),
        (0.400, 1.10, 0.8),
        (0.400, 1.15, 0.8),
        (0.500, 1.10, 0.8),
    ]
    results = []
    for ns, ls, nw in GRID:
        tag = f"ns={ns} ls={ls} nw={nw}"
        print(f"\n=== {tag} ===", flush=True)
        wavs, sr = synth_scales(a.onnx, texts, a.speaker, ns, ls, nw)
        hyps = transcribe(wavs, sr, a.asr)
        s = score(texts, hyps)
        s.update(noise_scale=ns, length_scale=ls, noise_w=nw)
        results.append(s)
        print(f"{tag}  CER {s['cer']:.4f}  WER {s['wer']:.4f}", flush=True)

    results.sort(key=lambda r: r["cer"])
    Path(a.out).write_text(json.dumps(results, indent=1), encoding="utf-8")
    base = next(r for r in results if (r["noise_scale"], r["length_scale"], r["noise_w"]) == GRID[0])
    best = results[0]
    print("\n==== RANKED (best CER first) ====")
    for r in results:
        mark = "  <- DEFAULT" if r is base else ("  <- BEST" if r is best else "")
        print(f"ns={r['noise_scale']:.3f} ls={r['length_scale']:.2f} nw={r['noise_w']:.1f}"
              f"  CER {r['cer']:.4f}  WER {r['wer']:.4f}{mark}")
    d = base["cer"] - best["cer"]
    print(f"\nbest vs default: CER {base['cer']:.4f} -> {best['cer']:.4f} "
          f"({d / base['cer'] * 100:+.2f}% relative)")
