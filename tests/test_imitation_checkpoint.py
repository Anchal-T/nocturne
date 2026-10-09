# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Tests for resumable imitation-learning checkpoints."""
import torch

from examples.imitation_learning.model import ImitationAgent
from examples.imitation_learning.train import (
    _atomic_torch_save,
    _epoch_from_path,
    _load_resume_state,
)


def _model():
    return ImitationAgent({
        "n_inputs": 4,
        "hidden_layers": [8],
        "discrete": True,
        "actions_discretizations": [3, 5],
        "actions_bounds": [[-1, 1], [-2, 2]],
        "device": "cpu",
    })


def test_epoch_from_checkpoint_path():
    assert _epoch_from_path("/tmp/model_50.pth") == 50
    assert _epoch_from_path("/tmp/checkpoint_60.pt") == 60
    assert _epoch_from_path("/tmp/latest.pt") == 0


def test_loads_legacy_model_checkpoint(tmp_path):
    source = _model()
    path = tmp_path / "model_50.pth"
    _atomic_torch_save(source, path)

    target = _model()
    epoch, optimizer_state = _load_resume_state(target, path, "cpu")

    assert epoch == 50
    assert optimizer_state is None
    for source_param, target_param in zip(
        source.parameters(), target.parameters()
    ):
        assert torch.equal(source_param, target_param)


def test_loads_resumable_checkpoint(tmp_path):
    source = _model()
    optimizer = torch.optim.Adam(source.parameters(), lr=3e-4)
    path = tmp_path / "checkpoint_20.pt"
    _atomic_torch_save({
        "completed_epoch": 17,
        "model_state_dict": source.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, path)

    target = _model()
    epoch, optimizer_state = _load_resume_state(target, path, "cpu")

    assert epoch == 17
    assert optimizer_state is not None
    assert not (tmp_path / "checkpoint_20.pt.tmp").exists()
