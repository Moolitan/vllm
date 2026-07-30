# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.segmentia_shared_kv import (
    SharedSkillKVBank,
    SharedSkillKVKey,
    SharedSkillKVState,
)

pytestmark = pytest.mark.cpu_test


def make_key(token_hash: str = "skill-a") -> SharedSkillKVKey:
    return SharedSkillKVKey(
        model_fingerprint="qwen3-14b",
        kv_dtype="bfloat16",
        kv_layout="paged-kv-v1",
        block_size=16,
        tp_world_size=1,
        token_hash=token_hash,
        num_tokens=64,
        correction_version="prefix-k-v1",
    )


def test_shared_bank_allocates_one_block_set_for_multiple_requests() -> None:
    block_pool = BlockPool(
        num_gpu_blocks=32, enable_caching=True, hash_block_size=16
    )
    bank = SharedSkillKVBank(block_pool, num_kv_cache_groups=1)
    initial_free = block_pool.get_num_free_blocks()

    owner_lease = bank.acquire(make_key(), "request-0", 4)
    follower_lease = bank.acquire(make_key(), "request-1", 4)

    assert owner_lease is not None and owner_lease.is_load_owner
    assert follower_lease is not None and not follower_lease.is_load_owner
    owner_binding = bank.bind_request_range(owner_lease, 32, 96)
    follower_binding = bank.bind_request_range(follower_lease, 48, 112)
    assert owner_binding.shared_start == 32
    assert follower_binding.shared_start == 48
    entry = bank.get(make_key())
    assert entry is not None
    assert entry.lease_count == 2
    assert len(entry.block_ids[0]) == 4
    assert block_pool.get_num_free_blocks() == initial_free - 4

    bank.mark_loading(owner_lease)
    bank.mark_ready(owner_lease)
    assert entry.state == SharedSkillKVState.READY
    assert bank.release("request-0")
    assert bank.get_request_binding("request-0") is None
    assert bank.release("request-1")
    assert entry.lease_count == 0
    assert block_pool.get_num_free_blocks() == initial_free - 4

    assert bank.evict(make_key())
    assert block_pool.get_num_free_blocks() == initial_free


def test_shared_bank_acquire_is_idempotent_per_request() -> None:
    block_pool = BlockPool(
        num_gpu_blocks=16, enable_caching=False, hash_block_size=16
    )
    bank = SharedSkillKVBank(block_pool, num_kv_cache_groups=1)
    initial_free = block_pool.get_num_free_blocks()

    first = bank.acquire(make_key(), "request-0", 4)
    second = bank.acquire(make_key(), "request-0", 4)

    assert first == second
    entry = bank.get(make_key())
    assert entry is not None and entry.lease_count == 1
    assert block_pool.get_num_free_blocks() == initial_free - 4
    assert bank.release("request-0")
    assert not bank.release("request-0")
    assert bank.get(make_key()) is None
    assert block_pool.get_num_free_blocks() == initial_free


def test_shared_bank_load_owner_release_fails_closed() -> None:
    block_pool = BlockPool(
        num_gpu_blocks=16, enable_caching=False, hash_block_size=16
    )
    bank = SharedSkillKVBank(block_pool, num_kv_cache_groups=1)
    owner = bank.acquire(make_key(), "request-0", 4)
    follower = bank.acquire(make_key(), "request-1", 4)
    assert owner is not None and follower is not None
    bank.mark_loading(owner)

    assert bank.release("request-0")
    entry = bank.get(make_key())
    assert entry is not None
    assert entry.state == SharedSkillKVState.FAILED
    assert bank.acquire(make_key(), "request-2", 4) is None
    assert bank.release("request-1")
    assert bank.get(make_key()) is None


def test_shared_bank_allocation_failure_has_no_partial_state() -> None:
    block_pool = BlockPool(
        num_gpu_blocks=4, enable_caching=False, hash_block_size=16
    )
    bank = SharedSkillKVBank(block_pool, num_kv_cache_groups=1)
    initial_free = block_pool.get_num_free_blocks()

    assert bank.acquire(make_key(), "request-0", 4) is None
    assert bank.get(make_key()) is None
    assert block_pool.get_num_free_blocks() == initial_free


def test_shared_bank_rejects_incompatible_request_range() -> None:
    block_pool = BlockPool(
        num_gpu_blocks=16, enable_caching=False, hash_block_size=16
    )
    bank = SharedSkillKVBank(block_pool, num_kv_cache_groups=1)
    lease = bank.acquire(make_key(), "request-0", 4)
    assert lease is not None

    with pytest.raises(ValueError, match="block-aligned"):
        bank.bind_request_range(lease, 32, 95)
    with pytest.raises(ValueError, match="does not match"):
        bank.bind_request_range(lease, 32, 80)


def test_shared_bank_completion_ack_is_owner_only_and_idempotent() -> None:
    block_pool = BlockPool(
        num_gpu_blocks=16, enable_caching=False, hash_block_size=16
    )
    bank = SharedSkillKVBank(block_pool, num_kv_cache_groups=1)
    owner = bank.acquire(make_key(), "request-0", 4)
    follower = bank.acquire(make_key(), "request-1", 4)
    assert owner is not None and follower is not None
    bank.mark_loading(owner)

    with pytest.raises(ValueError, match="only the load owner"):
        bank.complete_load("request-1", success=True)
    entry = bank.complete_load("request-0", success=True)
    assert entry.state == SharedSkillKVState.READY
    assert bank.complete_load("request-0", success=True) is entry


def test_shared_bank_failed_completion_requires_reason() -> None:
    block_pool = BlockPool(
        num_gpu_blocks=16, enable_caching=False, hash_block_size=16
    )
    bank = SharedSkillKVBank(block_pool, num_kv_cache_groups=1)
    owner = bank.acquire(make_key(), "request-0", 4)
    assert owner is not None
    bank.mark_loading(owner)

    with pytest.raises(ValueError, match="requires a reason"):
        bank.complete_load("request-0", success=False)
    entry = bank.complete_load(
        "request-0", success=False, failure_reason="synthetic H2D failure"
    )
    assert entry.state == SharedSkillKVState.FAILED
    assert entry.failure_reason == "synthetic H2D failure"
