#!/usr/bin/env python
"""Print a one-screen training progress report for the CPU phoneme run.

Used by the Telegram cron job. Reads only checkpoint metadata + the log tail,
so it is safe to run while training holds the files.
"""
import argparse
import re
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def human(seconds: float) -> str:
    seconds = max(0, int(seconds))
    d, rem = divmod(seconds, 86400)
    h, rem = divmod(rem, 3600)
    m = rem // 60
    if d:
        return f"{d}d {h}h"
    if h:
        return f"{h}h {m}m"
    return f"{m}m"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/persian-vits-laptop-phonemes")
    ap.add_argument("--log", default="runs/cpu_train.log")
    ap.add_argument("--max-steps", type=int, default=150000)
    ap.add_argument(
        "--start",
        default="",
        help="ISO time or epoch seconds when the CURRENT run began (anchors the "
        "rate calculation; defaults to the log's own oldest-line timestamp).",
    )
    a = ap.parse_args()

    run = ROOT / a.run
    log = ROOT / a.log
    ck = run / "checkpoints" / "latest.pt"

    lines = []
    if not ck.exists():
        print(f"Persian TTS: no checkpoint yet at {ck}")
        return

    import torch

    meta = torch.load(ck, map_location="cpu", weights_only=False)
    step = int(meta.get("step", 0))
    best = meta.get("best_val")
    epoch = meta.get("epoch")
    age = time.time() - ck.stat().st_mtime

    pct = 100.0 * step / a.max_steps if a.max_steps else 0.0
    lines.append("Persian TTS - CPU fine-tune (phonemes)")
    lines.append(f"step {step:,} / {a.max_steps:,}  ({pct:.1f}%)   epoch {epoch}")
    if isinstance(best, float):
        lines.append(f"best val mel-L1: {best:.4f}")
    lines.append(f"last checkpoint: {human(age)} ago")

    # Liveness comes from the process table, not checkpoint age: checkpoints are
    # only written every save_every steps (~2.7 h on CPU), so a fresh-but-slow run
    # would otherwise be misreported as stalled.
    import subprocess

    alive = subprocess.run(
        ["pgrep", "-f", "scripts/train.py"], capture_output=True
    ).returncode == 0
    lines.append(f"status: {'running' if alive else 'NOT RUNNING'}")

    # Anchor for the rate calculation: an explicit --start wins; otherwise fall
    # back to the log's creation time. Parsed once, before it is used below.
    start_ts = 0.0
    if a.start:
        try:
            start_ts = float(a.start)
        except ValueError:
            from datetime import datetime

            try:
                start_ts = datetime.fromisoformat(a.start).timestamp()
            except ValueError:
                start_ts = 0.0
    elif log.exists():
        start_ts = log.stat().st_ctime

    done = max(0, a.max_steps - step)

    if log.exists():
        txt = log.read_text(errors="ignore").splitlines()[-400:]
        # Progress beyond the last checkpoint: the newest step the log has printed.
        logged = [int(m.group(1)) for ln in txt if (m := re.search(r"step[ =](\d+)", ln))]
        if logged:
            live = max(logged)
            if live > step:
                lines.append(f"live step (log): {live:,}")
                done = max(0, a.max_steps - live)
            # Measure the REAL step rate from the gap between the first and last
            # step lines, using the log's mtime as the timestamp of the newest one.
            # Do NOT use st_ctime as the start: rotating/renaming the log resets it
            # and yields an absurdly optimistic rate. And never use a cold benchmark
            # constant -- the sampler is frame-bucketed, so later (longer) buckets
            # cost far more per step than the short batches a fresh iterator serves.
            advanced = live - min(logged)
            span = log.stat().st_mtime - start_ts if start_ts else 0.0
            if advanced > 0 and span > 0:
                sec_per_step = span / advanced
                rate_txt = (
                    f"{sec_per_step:.1f} s/step"
                    if sec_per_step < 120
                    else f"{sec_per_step / 60:.1f} min/step"
                )
                lines.append(f"measured rate: {rate_txt}")
                lines.append(
                    f"remaining: {done:,} steps (~{human(done * sec_per_step)} at this rate)"
                )
            else:
                lines.append(f"remaining: {done:,} steps (rate unknown yet)")
        tail = [ln for ln in txt if ln.strip() and "Warning" not in ln][-3:]
        if tail:
            lines.append("recent log:")
            lines.extend("  " + ln[:160] for ln in tail)

    print("\n".join(lines))


if __name__ == "__main__":
    main()
