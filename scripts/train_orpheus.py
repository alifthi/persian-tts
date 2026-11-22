#!/usr/bin/env python
"""Fine-tune Orpheus-3B on Persian with LoRA.

Usage::

    python scripts/train_orpheus.py -c configs/orpheus_fa.yaml
    python scripts/train_orpheus.py -c configs/orpheus_fa.yaml learning_rate=1e-4

Resumes automatically from the newest checkpoint in ``output_dir``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from persian_tts.orpheus.config import (  # noqa: E402
    load_orpheus_config,
    orpheus_config_to_dict,
)
from persian_tts.orpheus.dataset import OrpheusDataset, collate  # noqa: E402


def resolve_device_and_dtype(requested: str):
    """Pick the device and a dtype the device can actually do.

    A bf16 config on CPU is not a no-op: CPU bf16 autocast is *slower* than
    fp32, so an unusable GPU silently costs twice. This reports what it chose.
    """
    if torch.cuda.is_available():
        device = "cuda"
        supports_bf16 = torch.cuda.is_bf16_supported()
        if requested == "bf16" and not supports_bf16:
            print("bf16 requested but this GPU cannot do it; falling back to fp16")
            requested = "fp16"
    else:
        device = "cpu"
        if requested != "fp32":
            print(
                f"no CUDA device: {requested} requested but CPU autocast is slower "
                "than fp32; using fp32"
            )
            requested = "fp32"
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[requested]
    return device, dtype, requested


def build_model(cfg, device, dtype):
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM

    quant_config = None
    if cfg.load_in_4bit:
        if device != "cuda":
            raise SystemExit(
                "load_in_4bit requires CUDA (bitsandbytes has no CPU kernels). "
                "Set load_in_4bit=false to run on CPU."
            )
        from transformers import BitsAndBytesConfig

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        dtype=dtype,
        quantization_config=quant_config,
        attn_implementation=cfg.attn_implementation,
        device_map={"": device} if device == "cuda" else None,
    )
    if device != "cuda":
        model = model.to(device)

    if cfg.load_in_4bit:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=cfg.gradient_checkpointing
        )
    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    model.config.use_cache = False  # incompatible with gradient checkpointing

    lora = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    return model


def find_latest_checkpoint(out_dir: Path):
    ckpts = sorted(
        (p for p in out_dir.glob("step_*") if p.is_dir()),
        key=lambda p: int(p.name.split("_")[1]),
    )
    return ckpts[-1] if ckpts else None


def save_checkpoint(model, optimizer, scheduler, step, out_dir: Path, cfg):
    path = out_dir / f"step_{step:08d}"
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)  # LoRA adapter only: tens of MB, not 6 GB
    torch.save(
        {
            "step": step,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        },
        path / "trainer_state.pt",
    )
    (path / "orpheus_config.json").write_text(
        json.dumps(orpheus_config_to_dict(cfg), indent=2), encoding="utf-8"
    )

    # Prune, but never the 'best' link target.
    ckpts = sorted(
        (p for p in out_dir.glob("step_*") if p.is_dir()),
        key=lambda p: int(p.name.split("_")[1]),
    )
    for old in ckpts[: -cfg.keep_last]:
        shutil.rmtree(old, ignore_errors=True)
    return path


@torch.no_grad()
def evaluate(model, loader, device, dtype, max_batches=50):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.autocast(device_type=device, dtype=dtype, enabled=dtype != torch.float32):
            out = model(**batch)
        n = int((batch["labels"] != -100).sum())
        total_loss += float(out.loss) * n
        total_tokens += n
    model.train()
    if not total_tokens:
        return float("nan")
    return total_loss / total_tokens


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-c", "--config", default=None)
    ap.add_argument("overrides", nargs="*", help="dotted key=value overrides")
    args = ap.parse_args()

    cfg = load_orpheus_config(args.config, args.overrides)
    torch.manual_seed(cfg.seed)

    device, dtype, precision = resolve_device_and_dtype(cfg.precision)
    print(f"device={device} dtype={precision} effective_batch={cfg.effective_batch()}")
    if device == "cpu" and cfg.load_in_4bit:
        print("note: forcing load_in_4bit=false on CPU")
        cfg.load_in_4bit = False

    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer, get_scheduler

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)

    train_ds = OrpheusDataset(
        cfg.train_manifest, tokenizer, max_length=cfg.max_length, speakers=cfg.speakers
    )
    val_ds = OrpheusDataset(
        cfg.val_manifest, tokenizer, max_length=cfg.max_length, speakers=cfg.speakers
    )
    print(f"train {len(train_ds)} utts | val {len(val_ds)} utts")

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=2,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate
    )

    model = build_model(cfg, device, dtype)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = get_scheduler(
        cfg.lr_scheduler,
        optimizer,
        num_warmup_steps=cfg.warmup_steps,
        num_training_steps=cfg.max_steps,
    )

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    start_step = 0
    latest = find_latest_checkpoint(out_dir)
    if latest is not None:
        state = torch.load(latest / "trainer_state.pt", map_location="cpu")
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_step = state["step"]
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file

        set_peft_model_state_dict(
            model, load_file(latest / "adapter_model.safetensors")
        )
        print(f"resumed from {latest} at step {start_step}")

    best_val = float("inf")
    best_path = out_dir / "best_val.json"
    if best_path.exists():
        best_val = json.loads(best_path.read_text())["val_loss"]

    scaler = torch.amp.GradScaler(enabled=(precision == "fp16"))
    model.train()
    step = start_step
    accum_loss, accum_n = 0.0, 0
    t_window = time.time()

    print(f"training to step {cfg.max_steps}", flush=True)
    while step < cfg.max_steps:
        for micro, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(
                device_type=device, dtype=dtype, enabled=dtype != torch.float32
            ):
                out = model(**batch)
                loss = out.loss / cfg.grad_accum
            scaler.scale(loss).backward()
            accum_loss += float(out.loss)
            accum_n += 1

            if (micro + 1) % cfg.grad_accum:
                continue

            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            step += 1

            if step % cfg.log_every == 0:
                dt = time.time() - t_window
                mean = accum_loss / max(accum_n, 1)
                mem = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0.0
                print(
                    f"step {step:7d} | loss {mean:.4f} | ppl {math.exp(min(mean, 20)):8.2f} "
                    f"| lr {scheduler.get_last_lr()[0]:.2e} | gnorm {float(grad_norm):7.2f} "
                    f"| {cfg.log_every/dt:.3f} step/s | mem {mem:.2f}GB",
                    flush=True,
                )
                accum_loss, accum_n = 0.0, 0
                t_window = time.time()

            if step % cfg.eval_every == 0:
                if device == "cuda":
                    torch.cuda.empty_cache()
                val = evaluate(model, val_loader, device, dtype)
                flag = ""
                if val < best_val:
                    delta, pct = best_val - val, (best_val - val) / best_val * 100
                    flag = (
                        " NEW BEST"
                        if math.isinf(best_val)
                        else f" NEW BEST (-{delta:.4f}, -{pct:.2f}%)"
                    )
                    best_val = val
                    best_path.write_text(
                        json.dumps({"step": step, "val_loss": val}), encoding="utf-8"
                    )
                    save_checkpoint(model, optimizer, scheduler, step, out_dir, cfg)
                print(f"step {step:7d} | val_loss {val:.4f}{flag}", flush=True)

            if step % cfg.save_every == 0:
                p = save_checkpoint(model, optimizer, scheduler, step, out_dir, cfg)
                print(f"saved {p}", flush=True)

            if step >= cfg.max_steps:
                break

    save_checkpoint(model, optimizer, scheduler, step, out_dir, cfg)
    print(f"finished at step {step}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
