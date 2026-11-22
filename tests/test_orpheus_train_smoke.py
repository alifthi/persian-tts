"""End-to-end smoke test of the Orpheus training loop.

Runs `scripts/train_orpheus.py` against a ~20M-parameter random Llama with
Orpheus's vocabulary, so the loop, LoRA wiring, loss masking, checkpointing,
pruning and resume are all exercised without the 6 GB download or a GPU.

Marked slow: it builds a model and runs two subprocesses (~1 min on CPU).
Skipped automatically when transformers/peft are not installed.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from persian_tts.orpheus.codec import snac_codes_to_flat_tokens

pytest.importorskip("peft")
pytest.importorskip("transformers")

REPO = Path(__file__).resolve().parents[1]
VOCAB_SIZE = 156940  # must match Orpheus, or audio token ids fall outside the embedding


@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory):
    from transformers import LlamaConfig, LlamaForCausalLM

    path = tmp_path_factory.mktemp("tiny_orpheus")
    cfg = LlamaConfig(
        vocab_size=VOCAB_SIZE,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=2048,
    )
    LlamaForCausalLM(cfg).save_pretrained(path)
    return path


def _manifest(path: Path, n: int, frames: int = 12, seed: int = 0) -> str:
    g = torch.Generator().manual_seed(seed)
    rows = []
    for i in range(n):
        codes = [torch.randint(0, 4096, (1, frames * w), generator=g) for w in (1, 2, 4)]
        rows.append(
            {
                "id": f"u{i}",
                "text": "سلام دنیا این یک آزمایش است",
                "speaker": "mana",
                "duration": frames / 11.77,
                "audio_tokens": snac_codes_to_flat_tokens(codes),
            }
        )
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    return str(path)


def _run(tiny_model, out_dir, train, val, max_steps):
    """A stub tokenizer keeps this offline; the trainer only needs text ids."""
    cmd = [
        sys.executable,
        str(REPO / "scripts" / "train_orpheus.py"),
        f"base_model={tiny_model}",
        "load_in_4bit=false",
        "precision=fp32",
        "gradient_checkpointing=false",
        f"train_manifest={train}",
        f"val_manifest={val}",
        "max_length=512",
        "batch_size=2",
        "grad_accum=2",
        "warmup_steps=2",
        "lora_r=8",
        "lora_alpha=16",
        "log_every=2",
        "eval_every=4",
        "save_every=4",
        "keep_last=2",
        f"max_steps={max_steps}",
        f"output_dir={out_dir}",
    ]
    return subprocess.run(cmd, capture_output=True, text=True, cwd=REPO, timeout=900)


@pytest.mark.slow
def test_train_resume_and_best_checkpoint_survives_pruning(tiny_model, tmp_path, monkeypatch):
    """The whole loop, plus the pruning bug that destroys an A/B baseline.

    keep_last prunes oldest-first. If the best checkpoint is older than the
    last `keep_last` saves it gets deleted, leaving best_val.json pointing at
    nothing -- the run's best result becomes unrecoverable. It must be exempt.
    """
    # The tiny model dir has no tokenizer; point the trainer at the real one if
    # it is cached, otherwise skip rather than hit the network.
    try:
        from transformers import AutoTokenizer

        AutoTokenizer.from_pretrained(
            "unsloth/orpheus-3b-0.1-pretrained", local_files_only=True
        ).save_pretrained(tiny_model)
    except Exception:  # noqa: BLE001
        pytest.skip("Orpheus tokenizer not cached locally")

    train = _manifest(tmp_path / "train.jsonl", 16)
    val = _manifest(tmp_path / "val.jsonl", 4, seed=1)
    out = tmp_path / "run"

    r = _run(tiny_model, out, train, val, max_steps=8)
    assert r.returncode == 0, r.stderr[-3000:]
    assert "trainable params" in r.stdout, "LoRA was not applied"
    assert "NEW BEST" in r.stdout
    assert "finished at step 8" in r.stdout

    best = json.loads((out / "best_val.json").read_text())
    ckpts = {p.name for p in out.glob("step_*")}
    assert ckpts, "no checkpoints written"
    assert f"step_{best['step']:08d}" in ckpts, (
        f"best checkpoint (step {best['step']}) was pruned; best_val.json now "
        f"points at nothing. Present: {sorted(ckpts)}"
    )
    assert (out / f"step_{best['step']:08d}" / "adapter_model.safetensors").exists()

    r2 = _run(tiny_model, out, train, val, max_steps=12)
    assert r2.returncode == 0, r2.stderr[-3000:]
    assert "resumed from" in r2.stdout, "did not resume from the latest checkpoint"
    assert "finished at step 12" in r2.stdout
    best2 = json.loads((out / "best_val.json").read_text())
    assert f"step_{best2['step']:08d}" in {p.name for p in out.glob("step_*")}


@pytest.mark.slow
def test_adapter_checkpoint_is_small(tiny_model, tmp_path):
    """A checkpoint must be the LoRA adapter, not a copy of the base model."""
    try:
        from transformers import AutoTokenizer

        AutoTokenizer.from_pretrained(
            "unsloth/orpheus-3b-0.1-pretrained", local_files_only=True
        ).save_pretrained(tiny_model)
    except Exception:  # noqa: BLE001
        pytest.skip("Orpheus tokenizer not cached locally")

    train = _manifest(tmp_path / "train.jsonl", 8)
    out = tmp_path / "run"
    r = _run(tiny_model, out, train, train, max_steps=4)
    assert r.returncode == 0, r.stderr[-3000:]
    ckpt = next(out.glob("step_*"))
    base_size = sum(p.stat().st_size for p in Path(tiny_model).glob("*.safetensors"))
    adapter_size = (ckpt / "adapter_model.safetensors").stat().st_size
    assert adapter_size < base_size / 10, (
        f"adapter is {adapter_size/1e6:.1f} MB vs base {base_size/1e6:.1f} MB; "
        "the base weights are being checkpointed too"
    )
