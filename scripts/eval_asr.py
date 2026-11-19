"""Objective intelligibility eval: synthesize val sentences, transcribe with Whisper, score CER/WER.

Mel L1 cannot compare two models with different text frontends (different inputs, different
alignments), and it says nothing about whether the speech is *understandable*. This does:
it feeds each model's own text back through an ASR model and measures how much of it survives.

The absolute numbers include Whisper's own Persian error rate; only the DIFFERENCE between
models evaluated in the same run is meaningful.

    python scripts/eval_asr.py --onnx exports/onnx_phonemes/model.onnx --n 24
    python scripts/eval_asr.py --onnx chars=exports/onnx/model.onnx phonemes=exports/onnx_phonemes/model.onnx
"""
import argparse
import json
import sys
import unicodedata
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from persian_tts.text import get_frontend  # noqa: E402
from persian_tts.text.normalizer import normalize  # noqa: E402

ZWNJ = "\u200c"
_PUNCT = {c for c in ".,،؛:!؟?\"'()-" }


def norm_for_scoring(text: str) -> str:
    """Normalization shared by reference and hypothesis: no punctuation, no ZWNJ, single spaces."""
    text = unicodedata.normalize("NFC", normalize(text)).replace(ZWNJ, " ")
    text = "".join(" " if c in _PUNCT else c for c in text)
    text = "".join(c for c in text if not unicodedata.combining(c))  # drop optional Arabic diacritics
    return " ".join(text.split())


def edit_distance(a, b) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def synth(onnx_path: str, texts, speaker: str, noise_scale=0.667):
    """-> list of float32 waveforms at the model's sample rate."""
    import onnxruntime as ort
    cfg = json.loads(Path(onnx_path + ".json").read_text(encoding="utf-8"))
    _, encode = get_frontend(cfg.get("text_frontend", "chars"))
    sid = np.array([cfg["speaker_id_map"][speaker]], dtype=np.int64)
    scales = np.array([noise_scale, 1.0, 0.8], dtype=np.float32)
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    out = []
    for t in texts:
        ids = np.array([encode(t)], dtype=np.int64)
        audio = sess.run(None, {"input": ids, "input_lengths": np.array([ids.shape[1]], dtype=np.int64),
                                "scales": scales, "sid": sid})[0]
        out.append(np.asarray(audio, dtype=np.float32).reshape(-1))
    return out, cfg["audio"]["sample_rate"]


def transcribe(wavs, sr, model_id: str, batch_size: int = 4):
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    proc = WhisperProcessor.from_pretrained(model_id)
    model = WhisperForConditionalGeneration.from_pretrained(model_id).eval()
    if sr != 16000:  # Whisper is fixed at 16 kHz; linear resample is enough for ASR scoring
        wavs = [np.interp(np.arange(0, len(w), sr / 16000), np.arange(len(w)), w).astype(np.float32) for w in wavs]
    forced = proc.get_decoder_prompt_ids(language="fa", task="transcribe")
    hyps = []
    for i in range(0, len(wavs), batch_size):
        feats = proc(wavs[i: i + batch_size], sampling_rate=16000, return_tensors="pt").input_features
        with torch.no_grad():
            ids = model.generate(feats, forced_decoder_ids=forced, max_new_tokens=200)
        hyps += proc.batch_decode(ids, skip_special_tokens=True)
        print(f"  transcribed {min(i + batch_size, len(wavs))}/{len(wavs)}", flush=True)
    return hyps


def score(refs, hyps):
    cer_n = cer_d = wer_n = wer_d = 0
    for r, h in zip(refs, hyps):
        r, h = norm_for_scoring(r), norm_for_scoring(h)
        cer_n += edit_distance(r, h); cer_d += len(r)
        rw, hw = r.split(), h.split()
        wer_n += edit_distance(rw, hw); wer_d += len(rw)
    return dict(cer=cer_n / max(cer_d, 1), wer=wer_n / max(wer_d, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", nargs="+", required=True, help="model.onnx paths, optionally label=path")
    ap.add_argument("--manifest", default="data/processed/val.jsonl")
    ap.add_argument("--speaker", default="mana")
    ap.add_argument("--n", type=int, default=24, help="number of val sentences")
    ap.add_argument("--asr", default="openai/whisper-small")
    ap.add_argument("--noise-scale", type=float, default=0.667)
    ap.add_argument("--save-wavs", default=None, help="directory to keep the synthesized audio")
    ap.add_argument("--out", default="exports/asr_eval.json")
    a = ap.parse_args()

    rows = [json.loads(l) for l in Path(a.manifest).read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = [r for r in rows if 20 <= len(r["text"]) <= 160][: a.n]
    texts = [r["text"] for r in rows]
    print(f"{len(texts)} sentences, speaker={a.speaker}, asr={a.asr}")

    # ground-truth recordings give the ASR model's own floor on this data
    import soundfile as sf
    root = Path(a.manifest).parent
    gt = [sf.read(root / r["audio"], dtype="float32")[0] for r in rows]
    gt_sr = sf.read(root / rows[0]["audio"])[1]

    results, all_hyps = {}, {}
    sets = [("ground_truth", gt, gt_sr)]
    for spec in a.onnx:
        label, _, path = spec.rpartition("=")
        label = label or Path(path).parent.name
        print(f"synthesizing with {label} ({path}) ...", flush=True)
        wavs, sr = synth(path, texts, a.speaker, a.noise_scale)
        sets.append((label, wavs, sr))
        if a.save_wavs:
            d = Path(a.save_wavs) / label
            d.mkdir(parents=True, exist_ok=True)
            for r, w in zip(rows, wavs):
                sf.write(d / f"{r['id']}.wav", w, sr)

    for label, wavs, sr in sets:
        print(f"ASR on {label} ...", flush=True)
        hyps = transcribe(wavs, sr, a.asr)
        results[label] = score(texts, hyps)
        all_hyps[label] = hyps
        print(f"  {label}: CER {results[label]['cer']:.3f}  WER {results[label]['wer']:.3f}", flush=True)

    print("\n{:<22s} {:>8s} {:>8s}".format("model", "CER", "WER"))
    for k, v in results.items():
        print("{:<22s} {:>8.3f} {:>8.3f}".format(k, v["cer"], v["wer"]))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(dict(asr=a.asr, speaker=a.speaker, n=len(texts), results=results,
                                           texts=texts, hypotheses=all_hyps), ensure_ascii=False, indent=1),
                           encoding="utf-8")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
