#!/usr/bin/env python
"""python scripts/export_onnx.py --ckpt runs/persian-vits-finetune/checkpoints/best.pt --out exports/onnx
For TensorFlow: add --portable --out exports/onnx_tf, then run scripts/export_tf.py on that model."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from persian_tts.export.onnx_export import export  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="exports/onnx")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--portable", action="store_true",
                    help="RNG-free noise + static output length; required as input for scripts/export_tf.py")
    ap.add_argument("--max-frames", type=int, default=2600, help="static output frames (2600 = ~30s @22.05kHz)")
    a = ap.parse_args()
    print("exported", export(a.ckpt, a.out, a.opset, portable=a.portable, max_frames=a.max_frames))
