# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Trainer-facing distributed helpers that do not require GPUs."""
from nocturne.utils.distributed import shard_sequence


def test_waymo_rank_shards_are_disjoint():
    files = [f"tfrecord-{i:05d}.json" for i in range(10)]
    shards = [shard_sequence(files, rank, 2) for rank in range(2)]
    assert not set(shards[0]).intersection(shards[1])
    assert sorted(shards[0] + shards[1]) == files
