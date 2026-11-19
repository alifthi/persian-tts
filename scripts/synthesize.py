#!/usr/bin/env python
"""Synthesize Persian speech with a PyTorch checkpoint or an exported ONNX model.

python scripts/synthesize.py --ckpt runs/.../best.pt --text "سلام دنیا" --out hello.wav [--speaker mana]
python scripts/synthesize.py --onnx exports/onnx/model.onnx --text "سلام دنیا" --out hello.wav
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from persian_tts.text import get_frontend  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--ckpt")
    src.add_argument("--onnx")
    ap.add_argument("--text", required=True)
    ap.add_argument("--out", default="out.wav")
    ap.add_argument("--speaker", default="mana", help="speaker name (see speakers.json); mana is the clean 98 h voice")
    ap.add_argument("--noise-scale", type=float, default=0.667)
    ap.add_argument("--length-scale", type=float, default=1.0, help=">1 slower, <1 faster")
    a = ap.parse_args()

    if a.onnx:
        cfg = json.loads(Path(a.onnx + ".json").read_text(encoding="utf-8"))
        _, encode = get_frontend(cfg.get("text_frontend", "chars"))
    else:
        from persian_tts.export.onnx_export import load_generator
        g, ck, acfg = load_generator(a.ckpt)
        encode = ck["encode"]
    ids = np.array([encode(a.text)], dtype=np.int64)
    scales = np.array([a.noise_scale, a.length_scale, 0.8], dtype=np.float32)
    if a.onnx:
        import onnxruntime as ort
        sid = cfg["speaker_id_map"][a.speaker]
        sess = ort.InferenceSession(a.onnx, providers=ort.get_available_providers())
        audio = sess.run(None, {"input": ids, "input_lengths": np.array([ids.shape[1]], dtype=np.int64),
                                "scales": scales, "sid": np.array([sid], dtype=np.int64)})[0]
        sr = cfg["audio"]["sample_rate"]
    else:
        import torch
        sid = ck["speakers"].index(a.speaker)
        with torch.no_grad():
            audio, _ = g.infer_onnx(torch.from_numpy(ids), torch.LongTensor([ids.shape[1]]),
                                    torch.from_numpy(scales), torch.LongTensor([sid]))
        audio, sr = audio.numpy(), acfg.sample_rate
    sf.write(a.out, audio[0, 0], sr)
    print(f"wrote {a.out} ({audio.shape[-1] / sr:.2f}s)")
