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

## Text frontend
`data.text_frontend: phonemes` (recommended) converts text to IPA with espeak-ng's `fa` voice and uses
Piper's phoneme table, so the pretrained embeddings load exactly. Persian script leaves out short vowels;
the character frontend (`chars`) has to guess them. espeak-ng is GPL-3.0 and has to be present at
inference time too (`dnf install espeak-ng`), just like with Piper.

Measured with `scripts/eval_asr.py` (24 val sentences, speaker `mana`, `openai/whisper-small`; the
ground-truth row is the ASR model's own error floor on this data, so only the gaps are meaningful):

| model | steps | CER | WER |
| --- | --- | --- | --- |
| ground-truth recordings | — | 0.179 | 0.573 |
| phonemes | 105k | **0.270** | **0.744** |
| chars | 150k | 0.362 | 0.764 |

The phoneme model is ~25 % better on CER with 30 % fewer training steps, which is why it is the default.

## Workflow
```bash
python -m venv .venv --system-site-packages && .venv/bin/pip install -e ".[dev]"

# 1. data (incremental, re-runnable)
python scripts/prepare_data.py --out data/processed --workers 16

# 2. pretrained init (download rhasspy/piper-checkpoints _base_model first)
python scripts/convert_piper_checkpoint.py \
  --ckpt data/pretrained/piper_base/_base_model/base_model.ckpt \
  --piper-config data/pretrained/piper_base/_base_model/config.json \
  --speakers data/processed/speakers.json --frontend phonemes --out data/pretrained/persian_init_phonemes.pt

# 3. fine-tune (auto-resumes; SIGTERM-safe)
torchrun --nproc_per_node=4 scripts/train.py -c configs/base.yaml data.text_frontend=phonemes \
  train.init_from=data/pretrained/persian_init_phonemes.pt
python scripts/train.py -c configs/laptop_phonemes.yaml      # 4 GB laptop GPU
tensorboard --logdir runs

# 4. export + verify
python scripts/export_onnx.py --ckpt runs/persian-vits-finetune/checkpoints/best.pt --out exports/onnx
python scripts/export_onnx.py --ckpt .../best.pt --out exports/onnx_tf --portable   # static shapes, RNG-free noise
.venv-tf/bin/python scripts/export_tf.py --onnx exports/onnx_tf/model.onnx --out exports/tf   # SavedModel + TFLite

# 5. listen
python scripts/synthesize.py --onnx exports/onnx/model.onnx --text "سلام، حال شما چطور است؟" --out hi.wav  # --speaker mana (default)

# 5b. objective intelligibility (synthesize val text -> Whisper -> CER/WER; compares frontends)
python scripts/eval_asr.py --onnx chars=exports/onnx_chars/model.onnx phonemes=exports/onnx_phonemes/model.onnx --n 24

# 6. publish (license-gated; --dry-run first)
python scripts/push_to_hub.py --ckpt .../best.pt --onnx-dir exports/onnx --tf-dir exports/tf \
  --data-stats data/processed/stats.json --repo-id <org>/persian-vits --dry-run
```
`configs/smoke.yaml` runs the whole pipeline on a 4 GB GPU in about a minute.

TensorFlow environment (TF has no Python 3.14 wheels):
`conda create -p .venv-tf python=3.12 && .venv-tf/bin/pip install "tensorflow==2.19.*" tf_keras onnx2tf onnx==1.17.0 onnxruntime onnxsim onnx_graphsurgeon sng4onnx sne4onnx psutil ai_edge_litert`

Measured export accuracy (max absolute error on the waveform): ONNX vs PyTorch ≤ 2e-5,
TensorFlow vs ONNX ≤ 1.2e-5, TFLite fp32 vs ONNX ≤ 8e-6. The TF/TFLite model takes input padded to 400 tokens
and returns a fixed 30 s buffer plus `output_lengths`, the number of valid samples.

## Training details
- GAN training: mel L1 ×45, KL, feature matching, LSGAN with MPD+MSD. Generator and discriminator are both warm-started from Piper.
- The duration predictor is trained on Monotonic Alignment Search targets.
- The decoder is frozen for the first 10k steps while the new text embeddings adapt.
- AdamW with warmup and exponential decay, bf16 autocast, gradient clipping, and EMA weights (used for eval and export).
- Batches are bucketed by frame count. DDP via torchrun. Checkpoints are written atomically, the best one is chosen by validation mel L1, and audio samples go to TensorBoard.

## Orpheus-3B: a second model line (transformers + neural codec)

The VITS model above has two ceilings that no amount of further training removes:
`use_sdp: false` gives flat, averaged prosody (the usual source of "robotic"), and
`upsample_initial_channel: 256` is Piper's *medium* vocoder tier (the usual source of
"buzzy/metallic"). Both are architectural.

`persian_tts/orpheus/` is a separate stack that does not have them. It fine-tunes
[Orpheus-3B](https://huggingface.co/unsloth/orpheus-3b-0.1-pretrained) — a
`LlamaForCausalLM` under 🤗 transformers — to autoregressively predict discrete
[SNAC](https://huggingface.co/hubertsiuzdak/snac_24khz) audio codes from text, which the
codec decodes to 24 kHz waveform. Prosody comes from a 3B language model rather than a
duration predictor, and waveform quality from SNAC rather than a 256-channel HiFi-GAN.

**Why this model and not a better-sounding one.** Orpheus is the strongest TTS model that
clears this project's commercial-licence gate: the base is Apache-2.0 and SNAC is MIT.
F5-TTS, MaskGCT, OuteTTS, XTTS-v2 and openaudio-s1 all score higher on MOS and are all
CC-BY-NC or more restrictive; they were rejected on licence, not quality, for the same
reason ParsiGoo was excluded from the corpus.

### How the audio is tokenized

SNAC is a multi-scale codec: per frame it emits 1 coarse + 2 mid + 4 fine codes at three
different rates. Orpheus flattens those 7 codes into 7 strictly-banded token ids
(slot *i* owns vocabulary band `128266 + i*4096`), so a plain causal LM models audio as
text. The interleave is causal in time, not grouped by level. Getting it wrong yields a
model that trains to a perfectly normal loss and decodes to noise, so the mapping lives in
one module with exact-inverse tests, and was additionally verified to reproduce SNAC's own
codes bit-for-bit on real corpus audio.

Measured on this corpus: **11.77 SNAC frames/s = 82.4 tokens/s** of speech (the published
12.5 frames/s is nominal; the encoder's strides do not divide 24 kHz exactly). A 20-second
utterance is ~1650 audio tokens.

### Workflow

```bash
# 1. SNAC-encode the corpus once (slow, resumable by id, append-only).
#    ~1.1 utt/s on 6 CPU threads => ~21 h for the full 82k-utterance corpus.
python scripts/prepare_orpheus.py --manifest data/processed/train.jsonl \
  --root data/processed --out data/processed/orpheus_train.jsonl --workers 6
python scripts/prepare_orpheus.py --manifest data/processed/val.jsonl \
  --root data/processed --out data/processed/orpheus_val.jsonl --workers 6

# 2. LoRA fine-tune (auto-resumes from the newest checkpoint)
python scripts/train_orpheus.py -c configs/orpheus_fa.yaml        # 24 GB card, bf16
python scripts/train_orpheus.py -c configs/orpheus_fa_4bit.yaml   # 4 GB card, NF4

# 3. listen — omit --adapter for the base model, which is the A/B baseline
python scripts/synthesize_orpheus.py --adapter runs/orpheus-fa/step_00004000 \
  --text "سلام، حال شما چطور است؟" --out out_orpheus.wav

# 4. objective intelligibility, on the same CER scale as the VITS line
python scripts/eval_asr_orpheus.py --adapter base=none fa=runs/orpheus-fa/step_00004000 --n 24
```

### Training notes

- **LoRA, not full fine-tuning.** A 3B model's AdamW state is ~48 GB. LoRA over the
  attention *and* MLP projections trains <1% of the parameters; targeting the MLP matters
  for TTS specifically, since prosody and timbre live in the feed-forward stack.
- **The `pretrained` base, not the `ft` one.** Its audio-code embedding rows are already
  trained (it is a pretrained TTS model, not a text LLM with new tokens bolted on), which
  is what lets the embedding table stay frozen and makes LoRA sufficient.
- **Only the audio region is supervised.** Training the text region would spend 3B
  parameters re-learning Persian text modelling, competing with audio for the same weights.
  The terminating EOS *is* supervised, or the model never learns to stop. Padding uses
  Llama's `finetune_right_pad_id`, never EOS — padding with EOS teaches early stopping.
- **Speaker conditioning is plain text** (`"mana: ..."`) inside the prompt, so an unseen
  speaker degrades to a nearby voice instead of indexing out of bounds.
- **Single speaker by default.** `mana` is 98.2 h of the corpus's 111.7 h; `gptinformal`
  has 6.1 h, `fleurs` 7.0 h and the four `psd_*` speakers under 0.15 h each. Spending
  4000 steps on a 98:1 imbalance is how the `psd_*` voices came out untrained in the VITS
  run.
- **Sampling knobs before compute.** `--temperature` / `--top-p` /
  `--repetition-penalty` are Orpheus's analogue of VITS `noise_scale`, where sweeping
  bought 40% relative CER in minutes against 1.5% from a 17k-step anneal. Sweep them
  against CER before concluding the model needs more training.

Status: data pipeline, training, synthesis and eval are implemented and the codec path is
verified end-to-end on real audio. No Persian adapter has been trained yet — the GPU on
this machine is unusable (the NVIDIA akmod is built only for kernels 7.2.5/7.2.7 while
7.3.3 runs), and a 3B LoRA is not viable on CPU. Numbers will be filled in here when a
run completes.

## License
Code: Apache-2.0. Released weights: CC-BY-4.0 (see the model card that `push_to_hub.py` generates).
