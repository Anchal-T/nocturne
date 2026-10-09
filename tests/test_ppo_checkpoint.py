# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Tests for atomic checkpoint writes and legacy ValueNorm restore."""
import os

import torch
import torch.nn as nn

from algos.ppo.base_runner import _atomic_torch_save, _try_load_state_dict
from algos.ppo.utils.valuenorm import ValueNorm


def test_atomic_save_replaces_destination(tmp_path):
    path = str(tmp_path / "actor.pt")
    _atomic_torch_save({"step": 3}, path)
    assert os.path.exists(path)
    assert not os.path.exists(path + ".tmp")
    loaded = torch.load(path)
    assert loaded["step"] == 3


def test_value_norm_state_dict_contains_buffers():
    value_norm = ValueNorm(1, device=torch.device("cpu"))
    state = value_norm.state_dict()
    assert "running_mean" in state
    assert "running_mean_sq" in state
    assert "debiasing_term" in state


def test_try_load_skips_empty_and_incomplete_state():
    value_norm = ValueNorm(1, device=torch.device("cpu"))
    value_norm.update(torch.ones(8, 1))
    before = value_norm.running_mean.clone()
    _try_load_state_dict(value_norm, {}, "value_normalizer")
    _try_load_state_dict(value_norm, None, "value_normalizer")
    _try_load_state_dict(
        value_norm, {"not_a_buffer": torch.tensor(1.0)}, "value_normalizer"
    )
    assert torch.equal(value_norm.running_mean, before)


def test_value_norm_loads_legacy_partial_state():
    value_norm = ValueNorm(1, device=torch.device("cpu"))
    value_norm.load_state_dict({}, strict=False)
    other = ValueNorm(1, device=torch.device("cpu"))
    other.update(torch.ones(4, 1))
    value_norm.load_state_dict(other.state_dict(), strict=False)
    assert torch.allclose(value_norm.running_mean, other.running_mean)


class _DummyNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("running_mean", torch.zeros(1))


def test_try_load_shape_mismatch_does_not_raise():
    module = _DummyNorm()
    _try_load_state_dict(
        module, {"running_mean": torch.zeros(3)}, "value_normalizer"
    )
    assert tuple(module.running_mean.shape) == (1,)
