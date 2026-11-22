"""Fast unit tests for checkpoint pruning.

The lesson being pinned: `keep_last` prunes oldest-first, and the best
checkpoint is usually NOT among the newest. Deleting it leaves best_val.json
pointing at nothing -- the run's best result becomes unrecoverable and the A/B
baseline for a later comparison is gone. This repo has already lost a VITS
baseline to exactly that.
"""

import pytest

from scripts.train_orpheus import find_latest_checkpoint, prune_checkpoints


def _make(tmp_path, steps):
    for s in steps:
        d = tmp_path / f"step_{s:08d}"
        d.mkdir()
        (d / "adapter_model.safetensors").write_bytes(b"x")
    return tmp_path


def _present(tmp_path):
    return sorted(int(p.name.split("_")[1]) for p in tmp_path.glob("step_*"))


def test_prunes_oldest_beyond_keep_last(tmp_path):
    _make(tmp_path, [100, 200, 300, 400])
    prune_checkpoints(tmp_path, keep_last=2)
    assert _present(tmp_path) == [300, 400]


def test_best_checkpoint_is_never_pruned(tmp_path):
    """The regression: best at step 100 with keep_last=2 and four saves."""
    _make(tmp_path, [100, 200, 300, 400])
    prune_checkpoints(tmp_path, keep_last=2, best_step=100)
    assert 100 in _present(tmp_path), "best checkpoint was deleted"
    assert _present(tmp_path) == [100, 300, 400]


def test_best_does_not_count_against_the_budget(tmp_path):
    """keep_last=2 means two recent PLUS the best, not two including it."""
    _make(tmp_path, [10, 20, 30, 40, 50])
    prune_checkpoints(tmp_path, keep_last=2, best_step=10)
    assert _present(tmp_path) == [10, 40, 50]


def test_best_already_among_the_newest_still_does_not_count(tmp_path):
    """Even when best IS the newest, it is exempt, so keep_last others survive."""
    _make(tmp_path, [10, 20, 30])
    prune_checkpoints(tmp_path, keep_last=2, best_step=30)
    assert _present(tmp_path) == [10, 20, 30]
    # ...and one more save pushes the oldest non-best out
    _make(tmp_path, [40])
    prune_checkpoints(tmp_path, keep_last=2, best_step=30)
    assert _present(tmp_path) == [20, 30, 40]


def test_nothing_pruned_below_the_budget(tmp_path):
    _make(tmp_path, [10, 20])
    assert prune_checkpoints(tmp_path, keep_last=5) == []
    assert _present(tmp_path) == [10, 20]


def test_sorting_is_numeric_not_lexicographic(tmp_path):
    """step_00000900 vs step_00001000: zero-padding makes these agree, but the
    sort key must be the integer regardless of padding width."""
    _make(tmp_path, [900, 1000, 1100])
    prune_checkpoints(tmp_path, keep_last=1)
    assert _present(tmp_path) == [1100]


def test_find_latest_checkpoint(tmp_path):
    assert find_latest_checkpoint(tmp_path) is None
    _make(tmp_path, [10, 2000, 300])
    assert find_latest_checkpoint(tmp_path).name == "step_00002000"
