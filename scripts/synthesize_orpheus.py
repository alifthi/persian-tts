#!/usr/bin/env python
"""Synthesize Persian speech with a fine-tuned Orpheus adapter.

Usage::

    python scripts/synthesize_orpheus.py \
        --adapter runs/orpheus-fa/step_00004000 \
        --text "سلام، حال شما چطور است؟" --out out.wav

Omit ``--adapter`` to hear the base model, which is the correct A/B baseline for
judging what the Persian fine-tune actually bought.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from persian_tts.orpheus.codec import (  # noqa: E402
    AUDIO_TOKEN_BASE,
    CODES_PER_FRAME,
    SNAC_CODEBOOK_SIZE,
    SNAC_SAMPLE_RATE,
    SnacTokenizer,
)
from persian_tts.orpheus.prompt import (  # noqa: E402
    END_OF_AI,
    END_OF_SPEECH,
    EOS,
    build_generation_prompt,
)
from persian_tts.text.normalizer import normalize  # noqa: E402

#: Audio tokens per second of speech. SNAC-24khz is nominally 12.5 frames/s but
#: measures 11.77 on this corpus (82.4 tok/s over 36s of audio), because the
#: encoder's strides do not divide the sample rate exactly. Using the nominal
#: rate would under-budget max_new_tokens and clip the end off long utterances.
TOKENS_PER_SECOND = 82.4


def extract_audio_tokens(generated: list[int]) -> list[int]:
    """Keep the leading run of valid audio tokens, stopping at any terminator.

    A generated sequence can end on a stop token, on max_new_tokens, or -- when
    sampling goes wrong -- on a stray text token. Truncating at the first
    non-audio token is what keeps a bad generation quiet rather than noisy.
    """
    audio: list[int] = []
    hi = AUDIO_TOKEN_BASE + CODES_PER_FRAME * SNAC_CODEBOOK_SIZE
    for token in generated:
        if token in (END_OF_SPEECH, END_OF_AI, EOS):
            break
        if not AUDIO_TOKEN_BASE <= token < hi:
            break
        audio.append(token)
    # Only whole frames can be decoded.
    return audio[: len(audio) - len(audio) % CODES_PER_FRAME]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--text", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--adapter", default=None, help="LoRA checkpoint dir; omit for the base model")
    ap.add_argument("--base-model", default="unsloth/orpheus-3b-0.1-pretrained")
    ap.add_argument("--snac", default="hubertsiuzdak/snac_24khz")
    ap.add_argument("--speaker", default="mana")
    # Sampling defaults follow the Orpheus reference. They are the Orpheus
    # equivalent of VITS's noise_scale and, per this repo's experience, are
    # worth sweeping against ASR CER before spending any compute on retraining.
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--repetition-penalty", type=float, default=1.1)
    ap.add_argument("--max-seconds", type=float, default=20.0)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    dtype = torch.float16 if args.device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.base_model, dtype=dtype).to(args.device)

    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter)
        model = model.merge_and_unload()  # fold LoRA in; faster generation
    model.eval()
    model.config.use_cache = True

    text_ids = tokenizer(normalize(args.text), add_special_tokens=False)["input_ids"]
    speaker_ids = (
        tokenizer(f"{args.speaker}: ", add_special_tokens=False)["input_ids"]
        if args.speaker
        else None
    )
    prompt = build_generation_prompt(text_ids, args.speaker, speaker_ids)
    input_ids = torch.tensor([prompt], dtype=torch.long, device=args.device)

    max_new = int(args.max_seconds * TOKENS_PER_SECOND)
    t0 = time.time()
    with torch.inference_mode():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            max_new_tokens=max_new,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            eos_token_id=END_OF_SPEECH,
            pad_token_id=EOS,
        )
    gen_s = time.time() - t0

    generated = out[0, len(prompt):].tolist()
    audio_tokens = extract_audio_tokens(generated)
    if not audio_tokens:
        raise SystemExit(
            "the model produced no valid audio tokens. With an untrained or "
            "mismatched adapter this is expected; check --adapter and --speaker."
        )
    dropped = len(generated) - len(audio_tokens)

    wav = SnacTokenizer(repo_id=args.snac, device="cpu").decode(audio_tokens)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.out, wav.numpy(), SNAC_SAMPLE_RATE)

    duration = len(wav) / SNAC_SAMPLE_RATE
    print(
        f"wrote {args.out} | {duration:.2f}s audio from "
        f"{len(audio_tokens)//CODES_PER_FRAME} frames "
        f"({dropped} trailing tokens dropped) | generated in {gen_s:.1f}s "
        f"({duration/gen_s:.2f}x realtime on {args.device})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
