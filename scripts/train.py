#!/usr/bin/env python
"""Train Persian VITS.

Single GPU:  python scripts/train.py -c configs/base.yaml
Multi GPU:   torchrun --nproc_per_node=8 scripts/train.py -c configs/base.yaml
Overrides:   python scripts/train.py -c configs/base.yaml train.learning_rate=1e-4 data.max_batch_size=16
Training resumes automatically from <out_dir>/<name>/checkpoints/latest.pt (use --no-resume to start fresh).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from persian_tts.config import load_config  # noqa: E402
from persian_tts.training.trainer import Trainer  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("overrides", nargs="*", help="dotted key=value overrides")
    args = ap.parse_args()
    Trainer(load_config(args.config, args.overrides), resume=not args.no_resume).fit()
