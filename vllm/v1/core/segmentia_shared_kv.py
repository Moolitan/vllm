# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass, field
from enum import Enum

from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import KVCacheBlock


class SharedSkillKVState(str, Enum):
    ALLOCATING = "allocating"
    LOADING = "loading"
    READY = "ready"
    FAILED = "failed"


@dataclass(frozen=True)
class SharedSkillKVKey:
    model_fingerprint: str
    kv_dtype: str
    kv_layout: str
    block_size: int
    tp_world_size: int
    token_hash: str
    num_tokens: int
    correction_version: str


@dataclass(frozen=True)
class SharedSkillKVLease:
    key: SharedSkillKVKey
    request_id: str
    is_load_owner: bool


@dataclass(frozen=True)
class SharedSkillKVRequestBinding:
    key: SharedSkillKVKey
    shared_start: int
    shared_end: int


@dataclass
class SharedSkillKVEntry:
    key: SharedSkillKVKey
    blocks: tuple[tuple[KVCacheBlock, ...], ...]
    state: SharedSkillKVState
    load_owner_request_id: str
    request_ids: set[str] = field(default_factory=set)
    last_access: int = 0
    failure_reason: str | None = None

    @property
    def lease_count(self) -> int:
        return len(self.request_ids)

    @property
    def block_ids(self) -> tuple[tuple[int, ...], ...]:
        return tuple(
            tuple(block.block_id for block in group) for group in self.blocks
        )


class SharedSkillKVBank:
    """Owns one physical KV block set for each resident Skill body."""

    def __init__(self, block_pool: BlockPool, num_kv_cache_groups: int) -> None:
        if num_kv_cache_groups != 1:
            raise ValueError("shared Skill KV currently requires one KV cache group")
        self._block_pool = block_pool
        self._num_kv_cache_groups = num_kv_cache_groups
        self._entries: dict[SharedSkillKVKey, SharedSkillKVEntry] = {}
        self._request_to_key: dict[str, SharedSkillKVKey] = {}
        self._request_bindings: dict[str, SharedSkillKVRequestBinding] = {}
        self._access_clock = 0

    def acquire(
        self,
        key: SharedSkillKVKey,
        request_id: str,
        num_blocks_per_group: int,
    ) -> SharedSkillKVLease | None:
        """Acquire an idempotent request lease, allocating on first use."""
        if not request_id:
            raise ValueError("request_id must not be empty")
        if num_blocks_per_group <= 0:
            raise ValueError("num_blocks_per_group must be positive")
        if key.block_size <= 0:
            raise ValueError("shared Skill block_size must be positive")
        if key.num_tokens <= 0 or key.num_tokens % key.block_size != 0:
            raise ValueError("shared token range must contain whole blocks")
        expected_blocks = key.num_tokens // key.block_size
        if num_blocks_per_group != expected_blocks:
            raise ValueError(
                "num_blocks_per_group does not match the shared token range"
            )

        prior_key = self._request_to_key.get(request_id)
        if prior_key is not None:
            if prior_key != key:
                raise ValueError("request already leases a different shared Skill")
            entry = self._entries[prior_key]
            self._touch(entry)
            return SharedSkillKVLease(
                key=key,
                request_id=request_id,
                is_load_owner=entry.load_owner_request_id == request_id,
            )

        entry = self._entries.get(key)
        if entry is None:
            total_blocks = num_blocks_per_group * self._num_kv_cache_groups
            if total_blocks > self._block_pool.get_num_free_blocks():
                return None
            flat_blocks = self._block_pool.get_new_blocks(total_blocks)
            blocks = tuple(
                tuple(
                    flat_blocks[
                        group_idx
                        * num_blocks_per_group : (group_idx + 1)
                        * num_blocks_per_group
                    ]
                )
                for group_idx in range(self._num_kv_cache_groups)
            )
            entry = SharedSkillKVEntry(
                key=key,
                blocks=blocks,
                state=SharedSkillKVState.ALLOCATING,
                load_owner_request_id=request_id,
            )
            self._entries[key] = entry
        else:
            if len(entry.blocks) != self._num_kv_cache_groups or any(
                len(group) != num_blocks_per_group for group in entry.blocks
            ):
                raise ValueError("resident shared Skill has incompatible geometry")
            if entry.state == SharedSkillKVState.FAILED:
                return None

        entry.request_ids.add(request_id)
        self._request_to_key[request_id] = key
        self._touch(entry)
        return SharedSkillKVLease(
            key=key,
            request_id=request_id,
            is_load_owner=entry.load_owner_request_id == request_id,
        )

    def mark_loading(self, lease: SharedSkillKVLease) -> None:
        entry = self._entry_for_lease(lease)
        if not lease.is_load_owner:
            raise ValueError("only the load owner may start loading")
        if entry.state != SharedSkillKVState.ALLOCATING:
            raise ValueError("shared Skill must be allocating before loading")
        entry.state = SharedSkillKVState.LOADING
        self._touch(entry)

    def mark_ready(self, lease: SharedSkillKVLease) -> None:
        entry = self._entry_for_lease(lease)
        if not lease.is_load_owner:
            raise ValueError("only the load owner may finish loading")
        if entry.state != SharedSkillKVState.LOADING:
            raise ValueError("shared Skill must be loading before becoming ready")
        entry.state = SharedSkillKVState.READY
        self._touch(entry)

    def mark_failed(self, lease: SharedSkillKVLease, reason: str) -> None:
        entry = self._entry_for_lease(lease)
        if not lease.is_load_owner:
            raise ValueError("only the load owner may fail loading")
        if entry.state not in {
            SharedSkillKVState.ALLOCATING,
            SharedSkillKVState.LOADING,
        }:
            raise ValueError("only an unfinished shared Skill may fail")
        entry.state = SharedSkillKVState.FAILED
        entry.failure_reason = reason
        self._touch(entry)

    def bind_request_range(
        self,
        lease: SharedSkillKVLease,
        shared_start: int,
        shared_end: int,
    ) -> SharedSkillKVRequestBinding:
        entry = self._entry_for_lease(lease)
        if shared_start < 0 or shared_end <= shared_start:
            raise ValueError("request shared range must be non-empty")
        if shared_start % lease.key.block_size or shared_end % lease.key.block_size:
            raise ValueError("request shared range must be block-aligned")
        if shared_end - shared_start != lease.key.num_tokens:
            raise ValueError("request shared range does not match the Bank key")
        binding = SharedSkillKVRequestBinding(
            key=lease.key,
            shared_start=shared_start,
            shared_end=shared_end,
        )
        prior = self._request_bindings.get(lease.request_id)
        if prior is not None and prior != binding:
            raise ValueError("request already has a different shared range")
        self._request_bindings[lease.request_id] = binding
        self._touch(entry)
        return binding

    def release(self, request_id: str) -> bool:
        """Release one request lease; repeated releases are harmless."""
        key = self._request_to_key.pop(request_id, None)
        if key is None:
            return False
        self._request_bindings.pop(request_id, None)
        entry = self._entries[key]
        entry.request_ids.remove(request_id)
        if (
            entry.load_owner_request_id == request_id
            and entry.state
            in {SharedSkillKVState.ALLOCATING, SharedSkillKVState.LOADING}
        ):
            entry.state = SharedSkillKVState.FAILED
            entry.failure_reason = "load owner released before the entry was ready"
        self._touch(entry)
        if entry.state == SharedSkillKVState.FAILED and entry.lease_count == 0:
            self.evict(key)
        return True

    def evict(self, key: SharedSkillKVKey) -> bool:
        entry = self._entries.get(key)
        if entry is None:
            return False
        if entry.lease_count:
            raise ValueError("cannot evict a leased shared Skill")
        if entry.state in {
            SharedSkillKVState.ALLOCATING,
            SharedSkillKVState.LOADING,
        }:
            raise ValueError("cannot evict a shared Skill while it is loading")
        del self._entries[key]
        flat_blocks = [block for group in entry.blocks for block in group]
        self._block_pool.free_blocks(reversed(flat_blocks))
        return True

    def get(self, key: SharedSkillKVKey) -> SharedSkillKVEntry | None:
        return self._entries.get(key)

    def get_for_request(self, request_id: str) -> SharedSkillKVEntry | None:
        key = self._request_to_key.get(request_id)
        return self._entries.get(key) if key is not None else None

    def get_request_binding(
        self, request_id: str
    ) -> SharedSkillKVRequestBinding | None:
        return self._request_bindings.get(request_id)

    def complete_load(
        self,
        request_id: str,
        *,
        success: bool,
        failure_reason: str | None = None,
    ) -> SharedSkillKVEntry:
        """Apply one worker acknowledgement to the load-owner entry."""
        entry = self.get_for_request(request_id)
        if entry is None:
            raise ValueError("shared Skill load request has no active lease")
        if entry.load_owner_request_id != request_id:
            raise ValueError("only the load owner may complete a shared Skill load")
        if entry.state == SharedSkillKVState.READY and success:
            self._touch(entry)
            return entry
        if (
            entry.state == SharedSkillKVState.FAILED
            and not success
            and entry.failure_reason == failure_reason
        ):
            self._touch(entry)
            return entry
        if entry.state != SharedSkillKVState.LOADING:
            raise ValueError("shared Skill load acknowledgement is out of order")
        if success:
            if failure_reason is not None:
                raise ValueError("successful shared Skill load cannot have a reason")
            entry.state = SharedSkillKVState.READY
            entry.failure_reason = None
        else:
            if not failure_reason:
                raise ValueError("failed shared Skill load requires a reason")
            entry.state = SharedSkillKVState.FAILED
            entry.failure_reason = failure_reason
        self._touch(entry)
        return entry

    def _entry_for_lease(self, lease: SharedSkillKVLease) -> SharedSkillKVEntry:
        entry = self._entries.get(lease.key)
        if entry is None or lease.request_id not in entry.request_ids:
            raise ValueError("shared Skill lease is no longer active")
        return entry

    def _touch(self, entry: SharedSkillKVEntry) -> None:
        self._access_clock += 1
        entry.last_access = self._access_clock
