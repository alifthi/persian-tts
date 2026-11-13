# Persian TTS: commercially licensable VITS fine-tune

This repository fine-tunes the **Piper `_base_model`** VITS (904 LibriTTS-R speakers) for Persian. It includes
the data pipeline, training, and export to **ONNX**, **TensorFlow SavedModel** and **TFLite**.

## Why this model
| Requirement | Choice |
|---|---|
| Commercial use | Base weights are MIT (code) and CC-BY-4.0 (LibriTTS-R data). Every training corpus is CC0, MIT or CC-BY-4.0. |
| ONNX and TF compatible | VITS with a **deterministic duration predictor** and gather-based relative attention. Input lengths are dynamic in ONNX, and the graph converts with onnx2tf. |
| Quality per compute | End-to-end VITS (no separate vocoder), the architecture behind Piper's on-device voices. |

Models that were **excluded**, and why:
- `facebook/mms-tts-fas`: CC-BY-NC.
- Persian Piper voices amir, ganji and reza_ibrahim: they derive from *lessac*, which is research-only.
- F5-TTS and XTTS: the weights are non-commercial and they don't export cleanly to ONNX or TF.

The text frontend works on normalized Persian characters. It needs no espeak or GPL phonemizer at runtime.
The character embeddings are initialized from the pretrained model's IPA phoneme embeddings
([persian_tts/text/ipa_init.py](persian_tts/text/ipa_init.py)), so pronunciation knowledge carries over.

## Datasets ([persian_tts/data/sources.py](persian_tts/data/sources.py))
| Name | License | Notes |
|---|---|---|
| Mana-TTS | CC0 | ~114 h, single speaker, filtered by ASR CER |
| GPTInformal-Persian | CC0 | ~9 h, informal speech |
| ParsiGoo | CC0 | multi-speaker |
| SeyedAli/Persian-Speech-Dataset | MIT | multi-speaker, neutral-emotion subset |
| FLEURS fa_ir | CC-BY-4.0 | read speech, attribution required |
| Common Voice fa | CC0 | manual download from Mozilla Data Collective (`--common-voice-path`) |

## Workflow
```bash
python -m venv .venv --system-site-packages && .venv/bin/pip install -e ".[dev]"

# 1. data (incremental, re-runnable)
python scripts/prepare_data.py --out data/processed --workers 16

# 2. pretrained init (download rhasspy/piper-checkpoints _base_model first)
python scripts/convert_piper_checkpoint.py \
  --ckpt data/pretrained/piper_base/_base_model/base_model.ckpt \
  --piper-config data/pretrained/piper_base/_base_model/config.json \
  --speakers data/processed/speakers.json --out data/pretrained/persian_init.pt

# 3. fine-tune (auto-resumes; SIGTERM-safe)
torchrun --nproc_per_node=4 scripts/train.py -c configs/base.yaml
tensorboard --logdir runs

# 4. export + verify
python scripts/export_onnx.py --ckpt runs/persian-vits-finetune/checkpoints/best.pt --out exports/onnx
.venv-tf/bin/python scripts/export_tf.py --onnx exports/onnx/model.onnx --out exports/tf   # Python <= 3.12 env

# 5. listen
python scripts/synthesize.py --onnx exports/onnx/model.onnx --text "سلام، حال شما چطور است؟" --out hi.wav

# 6. publish (license-gated; --dry-run first)
python scripts/push_to_hub.py --ckpt .../best.pt --onnx-dir exports/onnx --tf-dir exports/tf \
  --data-stats data/processed/stats.json --repo-id <org>/persian-vits --dry-run
```
`configs/smoke.yaml` runs the whole pipeline on a 4 GB GPU in about a minute.

## Training details
- GAN training: mel L1 ×45, KL, feature matching, LSGAN with MPD+MSD. Generator and discriminator are both warm-started from Piper.
- The duration predictor is trained on Monotonic Alignment Search targets.
- The decoder is frozen for the first 10k steps while the new text embeddings adapt.
- AdamW with warmup and exponential decay, bf16 autocast, gradient clipping, and EMA weights (used for eval and export).
- Batches are bucketed by frame count. DDP via torchrun. Checkpoints are written atomically, the best one is chosen by validation mel L1, and audio samples go to TensorBoard.

## License
Code: Apache-2.0. Released weights: CC-BY-4.0 (see the model card that `push_to_hub.py` generates).
