#!/usr/bin/env python
"""Package a trained model and push it to the Hugging Face Hub (run only when you are ready to publish).

    huggingface-cli login            # write token
    python scripts/push_to_hub.py --ckpt runs/persian-vits-finetune/checkpoints/best.pt \
        --onnx-dir exports/onnx --tf-dir exports/tf --data-stats data/processed/stats.json \
        --repo-id <user-or-org>/persian-vits --private

Use --dry-run to build the folder locally (exports/hub_release) without uploading.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from persian_tts.data.sources import SOURCES  # noqa: E402

COMMERCIAL_LICENSES = {"CC0-1.0", "MIT", "CC-BY-4.0", "Apache-2.0"}

TEXT_INPUT = {
    "chars": "normalized Persian characters (no phonemizer, no GPL dependency)",
    "phonemes": "espeak-ng `fa` IPA phonemes, Piper phoneme table (espeak-ng is GPL-3.0 and needed at inference)",
}

CARD = """---
language: [fa]
license: cc-by-4.0
library_name: onnx
pipeline_tag: text-to-speech
tags: [tts, persian, farsi, vits, onnx, tensorflow, tflite, piper-compatible]
---

# {name}: Persian (Farsi) text-to-speech

End-to-end VITS model for Persian. It is fine-tuned from the Piper `_base_model`
(904-speaker LibriTTS-R) and trained only on data whose license allows commercial use.

| | |
|---|---|
| Sample rate | {sr} Hz |
| Speakers | {n_spk} (`config.json` / `speakers.json`) |
| Training steps | {step} |
| Formats | PyTorch (`generator.safetensors`), ONNX (`onnx/model.onnx`), TensorFlow SavedModel + TFLite (`tf/`) |
| Text input | {text_input} |

## Usage (ONNX)

```python
import numpy as np, onnxruntime as ort, soundfile as sf
from persian_tts.text import get_frontend   # from the training repository
_, encode = get_frontend("{frontend}")
ids = np.array([encode("سلام، به دنیای گفتار فارسی خوش آمدید.")], dtype=np.int64)
sess = ort.InferenceSession("onnx/model.onnx")
audio = sess.run(None, {{"input": ids, "input_lengths": np.array([ids.shape[1]]),
                        "scales": np.array([0.667, 1.0, 0.8], dtype=np.float32), "sid": np.array([0])}})[0]
sf.write("out.wav", audio[0, 0], {sr})
```
`scales` = [noise_scale, length_scale (>1 is slower), noise_w]. The I/O follows the Piper ONNX convention.

### Recommended inference settings

Use **`noise_scale=0.0`, `length_scale=1.2`**. These were tuned for this specific voice by
grid-searching the scales against ASR character error rate, not inherited from Piper's generic
defaults:

| scales | CER | WER |
|---|---|---|
| **0.0 / 1.20 (recommended)** | **0.177** | **0.564** |
| 0.0 / 1.125 | 0.178 | 0.572 |
| 0.0 / 1.05 | 0.197 | 0.589 |
| 0.0 / 1.00 | 0.196 | 0.587 |
| 0.667 / 1.00 (Piper default) | 0.326 | 0.778 |

Two findings worth knowing: `noise_scale` dominates (the generic 0.667 roughly doubles CER on
this voice), and the model is clearest at ~20% slower than nominal speed. `noise_w` has **no
effect at all** here -- this build uses a deterministic duration predictor (`use_sdp: false`),
so the third scale is accepted only for Piper I/O compatibility.

Scored with `openai/whisper-small` on held-out validation sentences; ground-truth human
recordings score CER 0.179 on the same pipeline, so the recommended setting is close to the
ASR's own error floor. `noise_scale=0` also makes synthesis deterministic (same text ->
identical audio), which the stochastic default does not.

## Training data

{data_table}

## Pretrained initialization
- [rhasspy/piper-checkpoints `_base_model`](https://huggingface.co/datasets/rhasspy/piper-checkpoints) (MIT),
  trained on [LibriTTS-R](https://www.openslr.org/141/) (CC-BY-4.0).
- Persian Piper voices derived from the *lessac* voice were **not** used (Blizzard 2013 research-only license).

## License and attribution
The weights are released under **CC-BY-4.0**, which allows commercial use with attribution. Attribution is
required for LibriTTS-R (Zen et al., Koizumi et al.) and, if it was used, FLEURS (Conneau et al., Google). The
other corpora are CC0 or MIT. Check that the license fits your use case before deploying.

## Limitations
Persian script omits short vowels, so the model infers them from context. Rare words, names and
ambiguous homographs may be mispronounced; adding diacritics (fatha/kasra/damma) to the input helps.

## Attribution (required)

This model is released under **CC-BY-4.0**: you may use it commercially, modify it and redistribute
it, **provided you credit the author**. Attribution is a condition of the licence, not a courtesy.

> Persian VITS TTS — created by **{author}** ({author_url})

```bibtex
@misc{{{bibkey},
  author       = {{{author}}},
  title        = {{{name}: a Persian (Farsi) VITS text-to-speech model}},
  year         = {{2026}},
  howpublished = {{\\url{{https://huggingface.co/{repo_id}}}}},
  note         = {{Licensed CC-BY-4.0}}
}}
```

If you ship this voice in a product, keep the credit line somewhere a user can find it
(about screen, docs, or model credits).
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--onnx-dir", required=True)
    ap.add_argument("--tf-dir", default=None)
    ap.add_argument("--data-stats", required=True)
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--name", default="Persian VITS")
    ap.add_argument("--samples-dir", default=None, help="optional folder of demo wavs")
    ap.add_argument("--author", default="Ali Fathi", help="name credited in the CC-BY attribution block")
    ap.add_argument("--author-url", default="https://github.com/alifthi", help="profile/repo URL for the credit line")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default="exports/hub_release")
    a = ap.parse_args()

    stats = json.loads(Path(a.data_stats).read_text(encoding="utf-8"))
    bad = {k: v for k, v in stats["licenses"].items() if v not in COMMERCIAL_LICENSES}
    if bad:
        sys.exit(f"refusing to publish: non-commercial/unknown licenses in training data: {bad}")

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    out = Path(a.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    from safetensors.torch import save_file
    save_file({k: v.contiguous() for k, v in ck["ema"].items()}, out / "generator.safetensors")
    (out / "config.json").write_text(json.dumps(ck["config"], ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "speakers.json").write_text(json.dumps(ck["speakers"], ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "symbols.json").write_text(json.dumps(ck["symbols"], ensure_ascii=False), encoding="utf-8")
    shutil.copytree(a.onnx_dir, out / "onnx")
    if a.tf_dir:
        shutil.copytree(a.tf_dir, out / "tf", ignore=shutil.ignore_patterns("model_static.onnx"))
    if a.samples_dir:
        shutil.copytree(a.samples_dir, out / "samples")

    rows = ["| Corpus | License | Hours |", "|---|---|---|"]
    for src, h in stats["hours_per_source"].items():
        s = SOURCES[src]
        rows.append(f"| [{src}]({s.url}) | {s.license} | {h} |")
    fe = ck["config"].get("data", {}).get("text_frontend", "chars")
    (out / "README.md").write_text(CARD.format(
        name=a.name, sr=ck["config"]["audio"]["sample_rate"], n_spk=len(ck["speakers"]), step=ck["step"],
        frontend=fe, text_input=TEXT_INPUT[fe], data_table="\n".join(rows),
        author=a.author, author_url=a.author_url, repo_id=a.repo_id,
        bibkey=a.repo_id.split("/")[-1].lower().replace("-", "_") + "_2026"), encoding="utf-8")
    print("release folder ready:", out)
    if a.dry_run:
        return

    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(a.repo_id, private=a.private, exist_ok=True)
    api.upload_folder(folder_path=str(out), repo_id=a.repo_id, commit_message=f"Release step {ck['step']}")
    print(f"pushed to https://huggingface.co/{a.repo_id}")


if __name__ == "__main__":
    main()
