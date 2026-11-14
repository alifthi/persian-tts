#!/usr/bin/env python
"""Download, clean and resample open Persian corpora into a unified TTS manifest.

Example:
    python scripts/prepare_data.py --sources mana_tts gptinformal parsigoo persian_speech fleurs \
        --out data/processed --sr 22050
    # Common Voice (manual download):
    python scripts/prepare_data.py --sources common_voice --common-voice-path /data/cv-fa --out data/processed

Outputs: <out>/wavs/<source>/<uid>.wav, <out>/train.jsonl, <out>/val.jsonl, <out>/speakers.json, <out>/stats.json
Re-running is incremental: existing wavs are reused.
"""
import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, FIRST_COMPLETED, wait
from pathlib import Path

import soundfile as sf
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from persian_tts.data.audio import preprocess_audio  # noqa: E402
from persian_tts.data.sources import SOURCES  # noqa: E402
from persian_tts.text import normalize, unknown_chars  # noqa: E402


def _process(job):
    uid, text, speaker, audio, sr, out_path, target_sr = job
    try:
        if Path(out_path).exists():
            info = sf.info(out_path)
            return uid, text, speaker, out_path, info.frames / info.samplerate, None
        y = preprocess_audio(audio, sr, target_sr)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        sf.write(out_path, y, target_sr, subtype="PCM_16")
        return uid, text, speaker, out_path, len(y) / target_sr, None
    except Exception as e:  # corrupted audio etc.
        return uid, text, speaker, out_path, 0.0, repr(e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+", default=["mana_tts", "gptinformal", "persian_speech", "fleurs"],
                    choices=list(SOURCES))
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--sr", type=int, default=22050)
    ap.add_argument("--min-dur", type=float, default=0.8)
    ap.add_argument("--max-dur", type=float, default=15.0)
    ap.add_argument("--max-unknown-ratio", type=float, default=0.02)
    ap.add_argument("--min-cps", type=float, default=4.0, help="min normalized chars/sec (detect misalignment)")
    ap.add_argument("--max-cps", type=float, default=25.0)
    ap.add_argument("--val-per-speaker", type=int, default=20)
    ap.add_argument("--min-speaker-utts", type=int, default=50, help="drop speakers with fewer utterances")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="max utterances per source (debug)")
    ap.add_argument("--common-voice-path", default=None)
    ap.add_argument("--allow-noncommercial", action="store_true")
    ap.add_argument("--exclude-speakers", nargs="*", default=[], help="speaker names or prefixes to drop")
    ap.add_argument("--rebuild-only", action="store_true",
                    help="re-filter/re-split existing manifests (no download/decoding); keeps only --sources")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    records, rejected = [], Counter()

    if args.rebuild_only:
        for split in ("train", "val"):
            p = out / f"{split}.jsonl"
            if p.exists():
                for l in p.read_text(encoding="utf-8").splitlines():
                    if l.strip():
                        r = json.loads(l)
                        r["text"] = normalize(r["text"])
                        records.append(r)
        args.sources = [n for n in args.sources if SOURCES[n].commercial_ok or args.allow_noncommercial]
        records = [r for r in records if r["source"] in args.sources]

    for name in ([] if args.rebuild_only else args.sources):
        src = SOURCES[name]
        if not src.commercial_ok and not args.allow_noncommercial:
            print(f"skip {name}: license {src.license} is not commercial-friendly")
            continue
        print(f"==> {name} ({src.license}) {src.url}")
        kwargs = {"path": args.common_voice_path} if name == "common_voice" else {}
        pending, n = set(), 0
        with ProcessPoolExecutor(args.workers) as ex:
            def drain(block_until_one):
                nonlocal pending
                if not pending:
                    return
                done, pending = wait(pending, return_when=FIRST_COMPLETED) if block_until_one else (
                    {f for f in pending if f.done()}, {f for f in pending if not f.done()})
                for f in done:
                    uid, text, spk, path, dur, err = f.result()
                    if err:
                        rejected["decode_error"] += 1
                        continue
                    records.append(dict(id=uid, audio=str(Path(path).relative_to(out)), text=text,
                                        speaker=spk, source=name, duration=round(dur, 3)))

            for utt in tqdm(src.loader(**kwargs), desc=name, unit="utt"):
                if args.limit and n >= args.limit:
                    break
                text = normalize(utt.text or "")
                if not text:
                    rejected["empty_text"] += 1
                    continue
                raw_dur = len(utt.audio) / utt.sr
                if raw_dur < args.min_dur or raw_dur > args.max_dur + 2.0:
                    rejected["duration_raw"] += 1
                    continue
                if len(unknown_chars(text)) and sum(c in unknown_chars(text) for c in text.lower()) / len(text) > args.max_unknown_ratio:
                    rejected["unknown_chars"] += 1
                    continue
                n += 1
                pending.add(ex.submit(_process, (utt.uid, text, utt.speaker, utt.audio, utt.sr,
                                                 str(out / "wavs" / name / f"{utt.uid}.wav"), args.sr)))
                while len(pending) >= args.workers * 4:
                    drain(True)
            while pending:
                drain(True)

    # post-filters on processed (trimmed) audio
    kept = []
    for r in records:
        if any(r["speaker"] == x or r["speaker"].startswith(x) for x in args.exclude_speakers):
            rejected["excluded_speaker"] += 1
            continue
        if any(c in unknown_chars(r["text"]) for c in r["text"].lower()) and \
                sum(c in unknown_chars(r["text"]) for c in r["text"].lower()) / len(r["text"]) > args.max_unknown_ratio:
            rejected["unknown_chars"] += 1
            continue
        if not (args.min_dur <= r["duration"] <= args.max_dur):
            rejected["duration"] += 1
            continue
        cps = len(r["text"]) / r["duration"]
        if not (args.min_cps <= cps <= args.max_cps):
            rejected["speaking_rate"] += 1
            continue
        kept.append(r)
    # dedupe ids
    kept = list({r["id"]: r for r in kept}.values())

    # merge with existing manifest from previous runs of other sources
    old = []
    for split in ([] if args.rebuild_only else ("train", "val")):
        p = out / f"{split}.jsonl"
        if p.exists():
            old += [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    new_sources = {r["source"] for r in kept} | set(args.sources)
    kept = [r for r in old if r["source"] not in new_sources
            and (SOURCES[r["source"]].commercial_ok or args.allow_noncommercial)] + kept

    by_spk = defaultdict(list)
    for r in kept:
        by_spk[r["speaker"]].append(r)
    by_spk = {s: v for s, v in by_spk.items() if len(v) >= args.min_speaker_utts}
    speakers = sorted(by_spk)

    rng = random.Random(args.seed)
    train, val = [], []
    for s in speakers:
        utts = sorted(by_spk[s], key=lambda r: r["id"])
        rng.shuffle(utts)
        k = min(args.val_per_speaker, max(1, len(utts) // 50))
        val += utts[:k]
        train += utts[k:]
    for split, rows in (("train", train), ("val", val)):
        with open(out / f"{split}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (out / "speakers.json").write_text(json.dumps(speakers, ensure_ascii=False, indent=1), encoding="utf-8")
    hours = defaultdict(float)
    for r in train + val:
        hours[r["source"]] += r["duration"] / 3600
    stats = dict(sample_rate=args.sr, n_train=len(train), n_val=len(val), n_speakers=len(speakers),
                 hours_per_source={k: round(v, 2) for k, v in hours.items()}, rejected=dict(rejected),
                 licenses={k: SOURCES[k].license for k in hours})
    (out / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
