# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.config import CUDAGraphMode
from vllm.v1.core.sched.output import (
    SegmentiaSharedKVBatchData,
    SegmentiaSharedRequestData,
)
from vllm.v1.worker.segmentia_shared_attention import (
    SegmentiaSharedAttentionWorkspace,
    build_segmentia_shared_attention_layout,
    build_segmentia_shared_attention_plan,
)
from vllm.v1.worker.kv_connector_model_runner_mixin import (
    _segmentia_shared_load_success_results,
)

pytestmark = pytest.mark.cpu_test


def make_batch() -> SegmentiaSharedKVBatchData:
    return SegmentiaSharedKVBatchData(
        bank_key="skill-a",
        shared_block_ids=(90, 91),
        shared_token_count=8,
        requests=(
            SegmentiaSharedRequestData("request-0", 8, 16),
            SegmentiaSharedRequestData("request-1", 4, 12),
        ),
    )


def test_shared_attention_layout_keeps_write_tables_unchanged() -> None:
    normal = {
        "request-0": [10, 11, 0, 0, 12, 13],
        "request-1": [20, 0, 0, 21, 22],
    }
    original = {request_id: blocks.copy() for request_id, blocks in normal.items()}

    layout = build_segmentia_shared_attention_layout(
        make_batch(), normal, block_size=4, null_block_id=0
    )

    assert layout.request_ids == ("request-0", "request-1")
    assert layout.block_table_rows == (
        (90, 91, 10, 11, 12, 13),
        (90, 91, 20, 21, 22),
    )
    assert layout.shared_start_positions == (8, 4)
    assert layout.shared_token_count == 8
    assert layout.shared_block_count == 2
    assert layout.private_block_counts == (4, 3)
    assert normal == original


@pytest.mark.parametrize(
    ("batch", "normal", "message"),
    [
        (
            make_batch(),
            {
                "request-0": [10, 11, 0, 0, 12],
                "request-1": [20, 0, 0, 21],
                "ordinary": [30],
            },
            "whole scheduled batch",
        ),
        (
            replace(
                make_batch(),
                requests=(
                    SegmentiaSharedRequestData("request-0", 8, 15),
                    SegmentiaSharedRequestData("request-1", 4, 12),
                ),
            ),
            {
                "request-0": [10, 11, 0, 0, 12],
                "request-1": [20, 0, 0, 21],
            },
            "block-aligned",
        ),
        (
            make_batch(),
            {
                "request-0": [10, 11, 0, 42, 12],
                "request-1": [20, 0, 0, 21],
            },
            "only null blocks",
        ),
    ],
)
def test_shared_attention_layout_rejects_unsafe_batches(
    batch: SegmentiaSharedKVBatchData,
    normal: dict[str, list[int]],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        build_segmentia_shared_attention_layout(
            batch, normal, block_size=4, null_block_id=0
        )


def test_shared_attention_plan_maps_queries_in_execution_order() -> None:
    normal = {
        "request-0": [10, 11, 0, 0, 12, 13],
        "request-1": [20, 0, 0, 21, 22],
    }
    plan = build_segmentia_shared_attention_plan(
        make_batch(),
        normal,
        {"request-0": 2, "request-1": 3},
        block_size=4,
        null_block_id=0,
    )

    assert plan.query_request_indices == (0, 0, 1, 1, 1)

    with pytest.raises(ValueError, match="request order"):
        build_segmentia_shared_attention_plan(
            make_batch(),
            normal,
            {"request-1": 3, "request-0": 2},
            block_size=4,
            null_block_id=0,
        )


def test_shared_attention_workspace_stages_without_touching_normal_table() -> None:
    normal = {
        "request-0": [10, 11, 0, 0, 12, 13],
        "request-1": [20, 0, 0, 21, 22],
    }
    original = {request_id: blocks.copy() for request_id, blocks in normal.items()}
    plan = build_segmentia_shared_attention_plan(
        make_batch(),
        normal,
        {"request-0": 2, "request-1": 1},
        block_size=4,
        null_block_id=0,
    )
    workspace = SegmentiaSharedAttentionWorkspace(
        max_num_reqs=4,
        max_num_tokens=8,
        max_num_blocks_per_req=8,
        layer_shapes={"model.layers.0.self_attn.attn": (2, 4)},
        device=torch.device("cpu"),
        pin_memory=False,
    )
    workspace.k_offsets_by_layer["model.layers.0.self_attn.attn"].fill_(1)

    tensors = workspace.stage(plan, null_block_id=0)

    assert tensors.block_table.tolist() == [
        [90, 91, 10, 11, 12, 13, 0, 0],
        [90, 91, 20, 21, 22, 0, 0, 0],
    ]
    assert tensors.query_request_indices.tolist() == [0, 0, 1]
    assert tensors.shared_start_positions.tolist() == [8, 4]
    assert not tensors.k_offsets_by_layer[
        "model.layers.0.self_attn.attn"
    ].any()
    assert normal == original


def test_model_runner_stages_separate_shared_attention_state() -> None:
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    normal_rows = np.array(
        [
            [10, 11, 0, 0, 12, 13, 0, 0],
            [20, 0, 0, 21, 22, 0, 0, 0],
        ],
        dtype=np.int32,
    )
    normal_snapshot = normal_rows.copy()
    normal_table = SimpleNamespace(
        blocks_per_kv_block=1,
        block_size=4,
        max_num_blocks_per_req=8,
        num_blocks_per_row=np.array([6, 5], dtype=np.int32),
        get_numpy_array=lambda: normal_rows,
    )
    workspace = SegmentiaSharedAttentionWorkspace(
        max_num_reqs=2,
        max_num_tokens=4,
        max_num_blocks_per_req=8,
        layer_shapes={"layer-0": (2, 4)},
        device=torch.device("cpu"),
        pin_memory=False,
    )
    runner = SimpleNamespace(
        kv_cache_config=SimpleNamespace(kv_cache_groups=[object()]),
        dcp_world_size=1,
        input_batch=SimpleNamespace(
            req_ids=["request-0", "request-1"],
            block_table=[normal_table],
            num_computed_tokens_cpu=np.array([16, 12], dtype=np.int32),
        ),
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(
                model_type="qwen3",
                rope_parameters={
                    "rope_type": "default",
                    "rope_theta": 1_000_000.0,
                },
            )
        ),
        segmentia_shared_attention_workspace=workspace,
        segmentia_shared_attention_tensors=None,
        segmentia_shared_rope_theta=None,
    )
    scheduler_output = SimpleNamespace(
        segmentia_shared_kv=make_batch(),
        num_scheduled_tokens={"request-0": 2, "request-1": 1},
    )

    tensors = GPUModelRunner._prepare_segmentia_shared_attention(
        runner,
        scheduler_output,
        num_reqs=2,
        use_spec_decode=False,
        should_ubatch=False,
        cudagraph_mode=CUDAGraphMode.NONE,
    )

    assert tensors is runner.segmentia_shared_attention_tensors
    assert tensors is not None
    assert tensors.query_request_indices.tolist() == [0, 0, 1]
    assert tensors.block_table[0, :6].tolist() == [90, 91, 10, 11, 12, 13]
    assert runner.segmentia_shared_rope_theta == 1_000_000.0
    assert np.array_equal(normal_rows, normal_snapshot)


def test_model_runner_rejects_query_before_shared_segment_end() -> None:
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    normal_table = SimpleNamespace(
        blocks_per_kv_block=1,
        block_size=4,
        max_num_blocks_per_req=8,
    )
    runner = SimpleNamespace(
        kv_cache_config=SimpleNamespace(kv_cache_groups=[object()]),
        dcp_world_size=1,
        input_batch=SimpleNamespace(
            req_ids=["request-0", "request-1"],
            block_table=[normal_table],
            num_computed_tokens_cpu=np.array([15, 12], dtype=np.int32),
        ),
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(
                model_type="qwen3",
                rope_parameters={"rope_type": "default"},
            )
        ),
    )
    scheduler_output = SimpleNamespace(segmentia_shared_kv=make_batch())

    with pytest.raises(RuntimeError, match="every query after B1"):
        GPUModelRunner._prepare_segmentia_shared_attention(
            runner,
            scheduler_output,
            num_reqs=2,
            use_spec_decode=False,
            should_ubatch=False,
            cudagraph_mode=CUDAGraphMode.NONE,
        )


def test_worker_acknowledges_only_completed_loading_owner() -> None:
    owner_request = SegmentiaSharedRequestData("owner", 8, 16)
    loading_batch = replace(
        make_batch(),
        requests=(owner_request,),
        bank_state="loading",
        load_owner_request_id="owner",
    )

    results = _segmentia_shared_load_success_results(
        SimpleNamespace(segmentia_shared_kv=loading_batch)
    )

    assert len(results) == 1
    assert results[0].request_id == "owner"
    assert results[0].success
    ready_batch = replace(
        loading_batch,
        bank_state="ready",
        load_owner_request_id=None,
    )
    assert not _segmentia_shared_load_success_results(
        SimpleNamespace(segmentia_shared_kv=ready_batch)
    )
