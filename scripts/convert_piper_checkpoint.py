#!/usr/bin/env python
"""Convert the Piper `_base_model` (VITS, LibriTTS-R, CC-BY-4.0 data / MIT code) into an init
checkpoint for Persian fine-tuning.

    python scripts/convert_piper_checkpoint.py \
        --ckpt data/pretrained/piper_base/_base_model/base_model.ckpt \
        --piper-config data/pretrained/piper_base/_base_model/config.json \
        --speakers data/processed/speakers.json --out data/pretrained/persian_init.pt

* generator + MPD discriminator weights are loaded 1:1 (legacy weight_norm -> parametrizations,
  flows re-indexed to drop Flip modules);
* character embeddings are initialized from the mean of mapped IPA phoneme embeddings;
* each Persian speaker embedding is initialized from a distinct pretrained speaker;
* the stochastic duration predictor and MRD discriminator are not used (deterministic DP is trained).
"""
import argparse
import json
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from persian_tts.config import ModelConfig  # noqa: E402
from persian_tts.models import MultiPeriodDiscriminator, SynthesizerTrn  # noqa: E402
from persian_tts.text import SYMBOLS  # noqa: E402
from persian_tts.text.ipa_init import CHAR_TO_IPA  # noqa: E402


def map_key(k: str):
    for pre in ("model_g.", "model_d."):
        if k.startswith(pre):
            k = k[len(pre):]
            break
    else:
        return None  # model_mrd etc.
    if k.startswith("dp."):
        return None
    if k.startswith("flow.flows."):
        parts = k.split(".")
        idx = int(parts[2])
        if idx % 2:  # Flip modules have no params, but be safe
            return None
        parts[2] = str(idx // 2)
        k = ".".join(parts)
    k = k.replace(".weight_g", ".parametrizations.weight.original0").replace(".weight_v", ".parametrizations.weight.original1")
    return k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--piper-config", required=True)
    ap.add_argument("--speakers", required=True)
    ap.add_argument("--out", default="data/pretrained/persian_init.pt")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    pcfg = json.loads(Path(args.piper_config).read_text())
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    hp, src = ck["hyper_parameters"], ck["state_dict"]
    assert hp["sample_rate"] == 22050 and hp["hop_length"] == 256

    speakers = json.loads(Path(args.speakers).read_text(encoding="utf-8"))
    mcfg = ModelConfig()  # defaults mirror the base model
    for k in ("hidden_channels", "filter_channels", "n_heads", "n_layers", "kernel_size", "inter_channels",
              "resblock_kernel_sizes", "resblock_dilation_sizes", "upsample_rates", "upsample_initial_channel",
              "upsample_kernel_sizes", "gin_channels"):
        assert getattr(mcfg, k) == hp[k], f"config mismatch {k}: {getattr(mcfg, k)} vs {hp[k]}"
    assert str(mcfg.resblock) == str(hp["resblock"])

    net_g = SynthesizerTrn(len(SYMBOLS), 513, 32, len(speakers), mcfg)
    net_d = MultiPeriodDiscriminator(use_spectral_norm=hp["use_spectral_norm"])
    g_sd, d_sd = net_g.state_dict(), net_d.state_dict()
    loaded_g, loaded_d, skipped = 0, 0, []
    for k, v in src.items():
        nk = map_key(k)
        if nk is None:
            continue
        tgt = g_sd if k.startswith("model_g.") else d_sd
        if nk in ("enc_p.emb.weight", "emb_g.weight"):
            continue  # handled below
        if nk not in tgt or tgt[nk].shape != v.shape:
            skipped.append((k, tuple(v.shape)))
            continue
        tgt[nk] = v.clone()
        if tgt is g_sd:
            loaded_g += 1
        else:
            loaded_d += 1

    # ---- character embeddings from IPA phoneme embeddings
    pid = pcfg["phoneme_id_map"]
    src_emb = src["model_g.enc_p.emb.weight"]
    emb = g_sd["enc_p.emb.weight"].clone()
    unmapped = []
    for i, sym in enumerate(SYMBOLS):
        phs = [p for p in CHAR_TO_IPA.get(sym, []) if p in pid]
        if not phs:
            unmapped.append(sym)
            continue
        emb[i] = torch.stack([src_emb[pid[p][0]] for p in phs]).mean(0)
    g_sd["enc_p.emb.weight"] = emb

    # ---- speaker embeddings: distinct pretrained speakers as starting points
    src_spk = src["model_g.emb_g.weight"]
    rng = random.Random(args.seed)
    picks = rng.sample(range(src_spk.shape[0]), len(speakers))
    g_sd["emb_g.weight"] = src_spk[picks].clone()

    net_g.load_state_dict(g_sd)
    net_d.load_state_dict(d_sd)
    n_new = sum(1 for k in g_sd if k.startswith("dp."))
    print(f"generator: loaded {loaded_g} tensors (+emb, emb_g); new: duration predictor ({n_new} tensors)")
    print(f"discriminator: loaded {loaded_d}/{len(d_sd)} tensors")
    if skipped:
        print("skipped (shape/name mismatch):", skipped[:10])
    if unmapped:
        print("symbols without IPA init (random):", unmapped)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(net_g=net_g.state_dict(), net_d=net_d.state_dict(), symbols=SYMBOLS, speakers=speakers,
                    source=dict(repo="rhasspy/piper-checkpoints", path="_base_model/base_model.ckpt",
                                data="LibriTTS-R (CC-BY-4.0)", code_license="MIT",
                                speaker_init=dict(zip(speakers, picks)))), args.out)
    print("saved", args.out)


if __name__ == "__main__":
    main()
