#!/usr/bin/env python
"""One-time SNAC encoding of the corpus into an Orpheus token manifest.

SNAC encoding is the expensive, deterministic part of Orpheus training: on this
project's CPU it runs at a small multiple of real time, so doing it inside the
dataloader would both dominate step time and redo identical work every epoch.
This script runs it once and writes ``audio_tokens`` next to the text.

The output manifest is append-only and resumable by ``id``: a corpus of 82k
utterances takes hours, and the job must survive being interrupted.

Usage::

    python scripts/prepare_orpheus.py \
        --manifest data/processed/train.jsonl \
        --root data/processed \
        --out data/processed/orpheus_train.jsonl \
        --workers 12
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from persian_tts.orpheus.codec import (  # noqa: E402
    CODES_PER_FRAME,
    SNAC_SAMPLE_RATE,
    SnacTokenizer,
)
from persian_tts.orpheus.prompt import (  # noqa: E402
    BOS,
    END_OF_AI,
    END_OF_HUMAN,
    END_OF_SPEECH,
    END_OF_TEXT,
    EOS,
    START_OF_AI,
    START_OF_HUMAN,
    START_OF_SPEECH,
)

#: Fixed overhead of the prompt/response scaffolding, so --max-length can be
#: enforced here (offline) rather than blowing up mid-training.
SCAFFOLD_TOKENS = len(
    [
        BOS,
        START_OF_HUMAN,
        END_OF_TEXT,
        END_OF_HUMAN,
        START_OF_AI,
        START_OF_SPEECH,
        END_OF_SPEECH,
        END_OF_AI,
        EOS,
    ]
)


def load_done_ids(out_path: Path) -> set:
    """Ids already present in a partially-written output manifest."""
    if not out_path.exists():
        return set()
    done = set()
    with out_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["id"])
            except (json.JSONDecodeError, KeyError):
                # A crash mid-write can leave one torn final line; drop it and
                # let this run redo that utterance.
                continue
    return done


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True, help="source jsonl (text + audio path)")
    ap.add_argument("--root", required=True, help="dir that manifest 'audio' paths are relative to")
    ap.add_argument("--out", required=True, help="destination token jsonl (resumable)")
    ap.add_argument("--snac", default="hubertsiuzdak/snac_24khz")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument(
        "--max-length",
        type=int,
        default=2048,
        help="drop utterances whose full example would exceed this many tokens",
    )
    ap.add_argument(
        "--max-text-tokens",
        type=int,
        default=256,
        help="budget reserved for the text prompt when enforcing --max-length",
    )
    ap.add_argument("--speaker", default=None, help="only this speaker")
    ap.add_argument("--limit", type=int, default=0, help="stop after N new rows (0 = all)")
    args = ap.parse_args()

    torch.set_num_threads(args.workers)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = load_done_ids(out_path)
    print(f"{len(done)} utterances already encoded in {out_path}", flush=True)

    rows = [
        json.loads(line)
        for line in Path(args.manifest).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.speaker:
        rows = [r for r in rows if r["speaker"] == args.speaker]
    todo = [r for r in rows if r["id"] not in done]
    print(f"{len(todo)} of {len(rows)} utterances left to encode", flush=True)

    # Longest audio we can afford, in SNAC frames, after reserving the text
    # budget. ~12.5 frames/s, 7 tokens/frame.
    max_audio_tokens = args.max_length - args.max_text_tokens - SCAFFOLD_TOKENS
    max_frames = max_audio_tokens // CODES_PER_FRAME
    print(
        f"audio budget: {max_audio_tokens} tokens = {max_frames} frames "
        f"~= {max_frames / 12.5:.1f}s",
        flush=True,
    )

    import librosa  # imported late: pulls in numba, slow to import

    snac = SnacTokenizer(repo_id=args.snac, device="cpu")
    root = Path(args.root)
    written = skipped_long = failed = 0
    t0 = time.time()

    with out_path.open("a", encoding="utf-8") as fh:
        for i, row in enumerate(todo):
            if args.limit and written >= args.limit:
                break
            if row.get("duration", 0) > max_frames / 12.5:
                skipped_long += 1
                continue
            try:
                wav, _ = librosa.load(root / row["audio"], sr=SNAC_SAMPLE_RATE, mono=True)
                peak = float(np.abs(wav).max())
                if peak < 1e-4:
                    failed += 1
                    continue
                tokens = snac.encode(torch.from_numpy(wav))
            except Exception as e:  # noqa: BLE001 - one bad file must not stop a 12h job
                failed += 1
                print(f"  FAIL {row['id']}: {type(e).__name__} {e}", flush=True)
                continue

            if len(tokens) > max_audio_tokens:
                skipped_long += 1
                continue

            fh.write(
                json.dumps(
                    {
                        "id": row["id"],
                        "text": row["text"],
                        "speaker": row["speaker"],
                        "duration": row.get("duration"),
                        "audio_tokens": tokens,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            written += 1
            if written % 50 == 0:
                fh.flush()
                rate = written / (time.time() - t0)
                remaining = (len(todo) - i - 1) / rate if rate else 0
                print(
                    f"  {written} encoded | {rate:.2f} utt/s | "
                    f"~{remaining/3600:.1f}h left | long={skipped_long} fail={failed}",
                    flush=True,
                )

    print(
        f"done: wrote {written}, skipped_long {skipped_long}, failed {failed} "
        f"in {(time.time()-t0)/60:.1f} min",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
