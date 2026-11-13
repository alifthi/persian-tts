#!/usr/bin/env python
"""python scripts/export_onnx.py --ckpt runs/persian-vits-finetune/checkpoints/best.pt --out exports/onnx"""
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
    a = ap.parse_args()
    print("exported", export(a.ckpt, a.out, a.opset))
