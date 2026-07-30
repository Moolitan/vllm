# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch

from vllm.v1.core.sched.output import SegmentiaSharedKVBatchData


@dataclass(frozen=True)
class SegmentiaSharedAttentionLayout:
    request_ids: tuple[str, ...]
    block_table_rows: tuple[tuple[int, ...], ...]
    shared_start_positions: tuple[int, ...]
    shared_token_count: int
    shared_block_count: int
    private_block_counts: tuple[int, ...]


@dataclass(frozen=True)
class SegmentiaSharedAttentionPlan:
    layout: SegmentiaSharedAttentionLayout
    query_request_indices: tuple[int, ...]


@dataclass(frozen=True)
class SegmentiaSharedAttentionTensors:
    block_table: torch.Tensor
    query_request_indices: torch.Tensor
    shared_start_positions: torch.Tensor
    shared_token_count: int
    k_offsets_by_layer: Mapping[str, torch.Tensor]


class SegmentiaSharedAttentionWorkspace:
    """Persistent staging buffers for one homogeneous shared-Skill batch."""

    def __init__(
        self,
        *,
        max_num_reqs: int,
        max_num_tokens: int,
        max_num_blocks_per_req: int,
        layer_shapes: Mapping[str, tuple[int, int]],
        device: torch.device,
        pin_memory: bool,
    ) -> None:
        if min(max_num_reqs, max_num_tokens, max_num_blocks_per_req) <= 0:
            raise ValueError("workspace dimensions must be positive")
        if not layer_shapes:
            raise ValueError("workspace requires at least one attention layer")

        self.max_num_reqs = max_num_reqs
        self.max_num_tokens = max_num_tokens
        self.max_num_blocks_per_req = max_num_blocks_per_req
        self.block_table_cpu = torch.empty(
            (max_num_reqs, max_num_blocks_per_req),
            dtype=torch.int32,
            device="cpu",
            pin_memory=pin_memory,
        )
        self.block_table = torch.empty_like(
            self.block_table_cpu, device=device, pin_memory=False
        )
        self.query_request_indices_cpu = torch.empty(
            max_num_tokens,
            dtype=torch.int64,
            device="cpu",
            pin_memory=pin_memory,
        )
        self.query_request_indices = torch.empty_like(
            self.query_request_indices_cpu, device=device, pin_memory=False
        )
        self.shared_start_positions_cpu = torch.empty(
            max_num_reqs,
            dtype=torch.int64,
            device="cpu",
            pin_memory=pin_memory,
        )
        self.shared_start_positions = torch.empty_like(
            self.shared_start_positions_cpu, device=device, pin_memory=False
        )
        self.k_offsets_by_layer = {
            layer_name: torch.zeros(
                (max_num_reqs, num_kv_heads, head_dim),
                dtype=torch.float32,
                device=device,
            )
            for layer_name, (num_kv_heads, head_dim) in layer_shapes.items()
        }

    def stage(
        self,
        plan: SegmentiaSharedAttentionPlan,
        *,
        null_block_id: int,
    ) -> SegmentiaSharedAttentionTensors:
        layout = plan.layout
        num_reqs = len(layout.request_ids)
        num_tokens = len(plan.query_request_indices)
        if num_reqs > self.max_num_reqs or num_tokens > self.max_num_tokens:
            raise ValueError("shared batch exceeds workspace capacity")
        if any(
            len(row) > self.max_num_blocks_per_req
            for row in layout.block_table_rows
        ):
            raise ValueError("shared attention row exceeds workspace capacity")

        block_table_cpu = self.block_table_cpu[:num_reqs]
        block_table_cpu.fill_(null_block_id)
        for req_idx, row in enumerate(layout.block_table_rows):
            block_table_cpu[req_idx, : len(row)] = torch.tensor(
                row, dtype=torch.int32
            )
        self.query_request_indices_cpu[:num_tokens] = torch.tensor(
            plan.query_request_indices, dtype=torch.int64
        )
        self.shared_start_positions_cpu[:num_reqs] = torch.tensor(
            layout.shared_start_positions, dtype=torch.int64
        )

        self.block_table[:num_reqs].copy_(block_table_cpu, non_blocking=True)
        self.query_request_indices[:num_tokens].copy_(
            self.query_request_indices_cpu[:num_tokens], non_blocking=True
        )
        self.shared_start_positions[:num_reqs].copy_(
            self.shared_start_positions_cpu[:num_reqs], non_blocking=True
        )
        for offsets in self.k_offsets_by_layer.values():
            offsets[:num_reqs].zero_()

        return SegmentiaSharedAttentionTensors(
            block_table=self.block_table[:num_reqs],
            query_request_indices=self.query_request_indices[:num_tokens],
            shared_start_positions=self.shared_start_positions[:num_reqs],
            shared_token_count=layout.shared_token_count,
            k_offsets_by_layer={
                layer_name: offsets[:num_reqs]
                for layer_name, offsets in self.k_offsets_by_layer.items()
            },
        )


def build_segmentia_shared_attention_layout(
    batch: SegmentiaSharedKVBatchData,
    normal_block_ids: Mapping[str, Sequence[int]],
    block_size: int,
    null_block_id: int,
) -> SegmentiaSharedAttentionLayout:
    """Build the read-only attention view for one homogeneous Skill batch."""
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if not batch.bank_key:
        raise ValueError("shared Skill batch must have a bank key")
    if not batch.requests:
        raise ValueError("shared Skill batch must contain requests")
    if (
        batch.shared_token_count <= 0
        or batch.shared_token_count % block_size != 0
    ):
        raise ValueError("shared token count must contain whole blocks")

    shared_block_count = batch.shared_token_count // block_size
    if len(batch.shared_block_ids) != shared_block_count:
        raise ValueError("shared block count does not match shared token count")
    if len(set(batch.shared_block_ids)) != shared_block_count:
        raise ValueError("shared block IDs must be unique")
    if null_block_id in batch.shared_block_ids:
        raise ValueError("shared Bank cannot contain the null block")

    request_ids = tuple(request.req_id for request in batch.requests)
    if len(set(request_ids)) != len(request_ids):
        raise ValueError("shared Skill batch contains duplicate request IDs")
    if set(normal_block_ids) != set(request_ids):
        raise ValueError("shared Skill metadata must cover the whole scheduled batch")

    rows: list[tuple[int, ...]] = []
    starts: list[int] = []
    private_counts: list[int] = []
    for request in batch.requests:
        if request.shared_start < 0 or request.shared_end <= request.shared_start:
            raise ValueError("request shared range must be non-empty")
        if request.shared_start % block_size or request.shared_end % block_size:
            raise ValueError("request shared range must be block-aligned")
        if request.shared_end - request.shared_start != batch.shared_token_count:
            raise ValueError("request shared range has a different token count")

        blocks = tuple(normal_block_ids[request.req_id])
        start_block = request.shared_start // block_size
        end_block = request.shared_end // block_size
        if end_block > len(blocks):
            raise ValueError("request block table does not cover the shared range")
        if any(block != null_block_id for block in blocks[start_block:end_block]):
            raise ValueError("request shared range must contain only null blocks")

        private_blocks = blocks[:start_block] + blocks[end_block:]
        rows.append(batch.shared_block_ids + private_blocks)
        starts.append(request.shared_start)
        private_counts.append(len(private_blocks))

    return SegmentiaSharedAttentionLayout(
        request_ids=request_ids,
        block_table_rows=tuple(rows),
        shared_start_positions=tuple(starts),
        shared_token_count=batch.shared_token_count,
        shared_block_count=shared_block_count,
        private_block_counts=tuple(private_counts),
    )


def build_segmentia_shared_attention_plan(
    batch: SegmentiaSharedKVBatchData,
    normal_block_ids: Mapping[str, Sequence[int]],
    scheduled_tokens: Mapping[str, int],
    block_size: int,
    null_block_id: int,
) -> SegmentiaSharedAttentionPlan:
    """Build layout plus the exact query-token to request mapping."""
    layout = build_segmentia_shared_attention_layout(
        batch, normal_block_ids, block_size, null_block_id
    )
    if tuple(scheduled_tokens) != layout.request_ids:
        raise ValueError("scheduled request order must match shared batch order")

    query_request_indices: list[int] = []
    for req_idx, req_id in enumerate(layout.request_ids):
        num_tokens = scheduled_tokens[req_id]
        if num_tokens <= 0:
            raise ValueError("every shared request must schedule positive tokens")
        query_request_indices.extend([req_idx] * num_tokens)

    return SegmentiaSharedAttentionPlan(
        layout=layout,
        query_request_indices=tuple(query_request_indices),
    )
