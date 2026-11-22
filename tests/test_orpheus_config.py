"""Tests for the Orpheus training config loader.

A typo'd key in a YAML config must be an error, not a silently-ignored setting
that makes a multi-hour run train with the wrong hyperparameters.
"""

import pytest
import yaml

from persian_tts.orpheus.config import (
    DEFAULT_TARGET_MODULES,
    OrpheusTrainConfig,
    load_orpheus_config,
    orpheus_config_to_dict,
)

SHIPPED_CONFIGS = ["configs/orpheus_fa.yaml", "configs/orpheus_fa_4bit.yaml"]


def test_defaults_round_trip():
    cfg = OrpheusTrainConfig()
    assert orpheus_config_to_dict(cfg)["lora_r"] == cfg.lora_r
    assert cfg.effective_batch() == cfg.batch_size * cfg.grad_accum


@pytest.mark.parametrize("path", SHIPPED_CONFIGS)
def test_shipped_configs_load(path):
    cfg = load_orpheus_config(path)
    assert cfg.max_steps > 0
    assert cfg.target_modules == DEFAULT_TARGET_MODULES
    assert cfg.speakers == ["mana"]


def test_four_bit_config_fits_a_small_card():
    """Guard the choices that exist purely to fit 4 GB."""
    cfg = load_orpheus_config("configs/orpheus_fa_4bit.yaml")
    assert cfg.load_in_4bit is True
    assert cfg.batch_size == 1
    assert cfg.gradient_checkpointing is True
    # the effective batch must still be meaningful despite batch_size 1
    assert cfg.effective_batch() >= 16


def test_unknown_key_raises_with_the_known_keys_listed(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump({"learnign_rate": 1e-4}), encoding="utf-8")
    with pytest.raises(KeyError, match="learnign_rate"):
        load_orpheus_config(str(p))


def test_override_parses_scientific_notation():
    """YAML 1.1 reads '2e-4' as a string; a string LR would crash AdamW."""
    cfg = load_orpheus_config(None, ["learning_rate=1e-5"])
    assert isinstance(cfg.learning_rate, float)
    assert cfg.learning_rate == pytest.approx(1e-5)


def test_override_beats_file(tmp_path):
    cfg = load_orpheus_config("configs/orpheus_fa.yaml", ["max_steps=7"])
    assert cfg.max_steps == 7


def test_unknown_override_key_also_raises():
    with pytest.raises(KeyError):
        load_orpheus_config(None, ["lora_rank=8"])
