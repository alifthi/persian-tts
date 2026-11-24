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


_STOP = {"requested": False}


def _install_sigterm_handler():
    """Finish the current optimizer step, save, and exit cleanly on SIGTERM/SIGINT.

    An open-ended run is stopped by the user, so the stop path has to be the
    normal path: SIGKILL would lose every step since the last ``save_every``.
    """
    import signal

    def _handler(signum, frame):  # noqa: ARG001
        if _STOP["requested"]:
            return
        _STOP["requested"] = True
        print(
            f"\nreceived signal {signum}: finishing the current step, "
            "saving a checkpoint, then exiting",
            flush=True,
        )

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, _handler)


def _worker_ignore_signals(worker_id: int) -> None:  # noqa: ARG001
    """DataLoader workers must ignore SIGTERM so a clean stop stays clean.

    systemd sends SIGTERM to every process in the unit's cgroup, workers
    included. A worker that dies first makes the parent's very next
    ``next(loader)`` raise "DataLoader worker exited unexpectedly", which
    aborts the shutdown path before it can checkpoint. Ignoring the signal in
    workers lets the parent finish its step, save, and exit; the workers are
    reaped when the parent does.
    """
    import signal

    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def find_latest_checkpoint(out_dir: Path):
    ckpts = sorted(
        (p for p in out_dir.glob("step_*") if p.is_dir()),
        key=lambda p: int(p.name.split("_")[1]),
    )
    return ckpts[-1] if ckpts else None


def prune_checkpoints(out_dir: Path, keep_last: int, best_step=None) -> list:
    """Delete oldest checkpoints beyond ``keep_last``, always sparing the best.

    best_val.json names a step. If pruning removes that step the run's best
    result is unrecoverable and the A/B baseline for any later comparison is
    gone, so the best checkpoint does not count against the budget.

    Returns the directories removed.
    """
    ckpts = sorted(
        (p for p in out_dir.glob("step_*") if p.is_dir()),
        key=lambda p: int(p.name.split("_")[1]),
    )
    best_name = f"step_{best_step:08d}" if best_step is not None else None
    prunable = [p for p in ckpts if p.name != best_name]
    removed = prunable[: max(0, len(prunable) - keep_last)]
    for old in removed:
        shutil.rmtree(old, ignore_errors=True)
    return removed


def save_checkpoint(model, optimizer, scheduler, step, out_dir: Path, cfg, best_step=None):
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
    prune_checkpoints(out_dir, cfg.keep_last, best_step)
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
        cfg.train_manifest, tokenizer, max_length=cfg.max_length,
        speakers=cfg.speakers, max_audio_tokens=cfg.max_audio_tokens,
    )
    val_ds = OrpheusDataset(
        cfg.val_manifest, tokenizer, max_length=cfg.max_length,
        speakers=cfg.speakers, max_audio_tokens=cfg.max_audio_tokens,
    )
    print(f"train {len(train_ds)} utts | val {len(val_ds)} utts")

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=2,
        drop_last=True,
        worker_init_fn=_worker_ignore_signals,
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
        # A resume must not silently inherit the previous run's learning rate.
        # optimizer.load_state_dict() restores the old param_group lr, and
        # LambdaLR.load_state_dict() restores the old base_lrs, so the config's
        # learning_rate is overwritten by whatever the last run used. Re-assert
        # it: the config is the source of truth for the schedule's amplitude,
        # the checkpoint only for its position.
        for group in optimizer.param_groups:
            group["lr"] = cfg.learning_rate
            group["initial_lr"] = cfg.learning_rate
        scheduler.base_lrs = [cfg.learning_rate] * len(scheduler.base_lrs)
        print(f"resumed from {latest} at step {start_step}")
        print(f"learning rate re-asserted from config: {cfg.learning_rate:.2e}")

    best_val, best_step = float("inf"), None
    best_path = out_dir / "best_val.json"
    if best_path.exists():
        saved = json.loads(best_path.read_text())
        best_val, best_step = saved["val_loss"], saved["step"]

    scaler = torch.amp.GradScaler(enabled=(precision == "fp16"))
    _install_sigterm_handler()
    model.train()
    step = start_step
    accum_loss, accum_n = 0.0, 0
    t_window = time.time()

    print(f"training to step {cfg.max_steps}", flush=True)
    while step < cfg.max_steps:
        for micro, batch in enumerate(train_loader):
            if _STOP["requested"] and micro % cfg.grad_accum == 0:
                break  # clean boundary: nothing half-accumulated to throw away
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(
                device_type=device, dtype=dtype, enabled=dtype != torch.float32
            ):
                out = model(**batch)
                loss = out.loss / cfg.grad_accum
            scaler.scale(loss).backward()
            accum_loss += float(out.loss.detach())
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
                    best_val, best_step = val, step
                    best_path.write_text(
                        json.dumps({"step": step, "val_loss": val}), encoding="utf-8"
                    )
                    save_checkpoint(
                        model, optimizer, scheduler, step, out_dir, cfg, best_step
                    )
                print(f"step {step:7d} | val_loss {val:.4f}{flag}", flush=True)

            if step % cfg.save_every == 0:
                p = save_checkpoint(
                    model, optimizer, scheduler, step, out_dir, cfg, best_step
                )
                print(f"saved {p}", flush=True)

            if step >= cfg.max_steps or _STOP["requested"]:
                break
        if _STOP["requested"]:
            break

    save_checkpoint(model, optimizer, scheduler, step, out_dir, cfg, best_step)
    reason = "stopped by signal" if _STOP["requested"] else "finished"
    print(f"{reason} at step {step} | best val_loss {best_val:.4f} at step {best_step}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
