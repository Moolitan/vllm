# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from types import SimpleNamespace

import pytest
import torch

import vllm.v1.attention.backends.flash_attn as flash_attn_backend
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.attention.backends.flash_attn import (
    cascade_attention,
    merge_attn_states,
    prepare_segmentia_prefix_inputs,
)

try:
    from vllm.vllm_flash_attn import (
        fa_version_unsupported_reason,
        flash_attn_varlen_func,
        is_fa_version_supported,
    )
except ImportError:
    if current_platform.is_rocm():
        pytest.skip(
            "vllm_flash_attn is not supported for vLLM on ROCm.",
            allow_module_level=True,
        )

NUM_HEADS = [(4, 4), (8, 2), (16, 2)]
HEAD_SIZES = [128, 192, 256]
BLOCK_SIZES = [16]
DTYPES = [torch.float16, torch.bfloat16]


@pytest.mark.parametrize("num_tokens", [1, 39, 16912])
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@torch.inference_mode()
def test_merge_kernel(
    num_tokens: int,
    num_heads: tuple[int, int],
    head_size: int,
    dtype: torch.dtype,
):
    torch.set_default_device("cuda")
    set_random_seed(0)
    num_query_heads = num_heads[0]
    num_kv_heads = num_heads[1]
    assert num_query_heads % num_kv_heads == 0

    # Prepare inputs.
    prefix_output = torch.randn(num_tokens, num_query_heads, head_size, dtype=dtype)
    suffix_output = torch.randn(num_tokens, num_query_heads, head_size, dtype=dtype)
    prefix_lse = torch.randn(num_query_heads, num_tokens, dtype=torch.float32)
    suffix_lse = torch.randn(num_query_heads, num_tokens, dtype=torch.float32)

    # Run the kernel.
    output = torch.empty(num_tokens, num_query_heads, head_size, dtype=dtype)
    merge_attn_states(output, prefix_output, prefix_lse, suffix_output, suffix_lse)

    # Reference implementation.
    max_lse = torch.maximum(prefix_lse, suffix_lse)
    p_lse = torch.exp(prefix_lse - max_lse)
    s_lse = torch.exp(suffix_lse - max_lse)
    p_scale = p_lse / (p_lse + s_lse)
    s_scale = s_lse / (p_lse + s_lse)
    p_scale = p_scale.transpose(0, 1).unsqueeze(2)
    s_scale = s_scale.transpose(0, 1).unsqueeze(2)
    ref_output = p_scale * prefix_output + s_scale * suffix_output
    ref_output = ref_output.to(dtype)

    # Compare the results.
    torch.testing.assert_close(output, ref_output, atol=1e-2, rtol=1e-2)


CASES = [
    # Case 1. A general case.
    ([(129, 871), (18, 280), (37, 988), (1023, 2304), (1, 257)], 256),
    # Case 2. Flash-decoding case.
    ([(1, 1023), (1, 879), (1, 778), (1, 1777)] * 100, 512),
]


@pytest.mark.parametrize("seq_lens_and_common_prefix", CASES)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("soft_cap", [None, 50])
@pytest.mark.parametrize("num_blocks", [2048])
@pytest.mark.parametrize("fa_version", [2, 3])
@torch.inference_mode()
def test_cascade(
    seq_lens_and_common_prefix: tuple[list[tuple[int, int]], int],
    num_heads: tuple[int, int],
    head_size: int,
    dtype: torch.dtype,
    block_size: int,
    soft_cap: float | None,
    num_blocks: int,
    fa_version: int,
) -> None:
    torch.set_default_device("cuda")
    if not is_fa_version_supported(fa_version):
        pytest.skip(
            f"Flash attention version {fa_version} not supported due "
            f'to: "{fa_version_unsupported_reason(fa_version)}"'
        )

    set_random_seed(0)

    window_size = (-1, -1)
    scale = head_size**-0.5
    num_query_heads = num_heads[0]
    num_kv_heads = num_heads[1]
    assert num_query_heads % num_kv_heads == 0
    key_cache = torch.randn(
        num_blocks, block_size, num_kv_heads, head_size, dtype=dtype
    )
    value_cache = torch.randn_like(key_cache)

    seq_lens, common_prefix_len = seq_lens_and_common_prefix
    num_seqs = len(seq_lens)
    query_lens = [x[0] for x in seq_lens]
    kv_lens = [x[1] for x in seq_lens]
    max_query_len = max(query_lens)
    max_kv_len = max(kv_lens)

    total_num_query_tokens = sum(query_lens)
    query = torch.randn(total_num_query_tokens, num_query_heads, head_size, dtype=dtype)
    cu_query_lens = torch.tensor([0] + query_lens, dtype=torch.int32).cumsum(
        dim=0, dtype=torch.int32
    )
    kv_lens_tensor = torch.tensor(kv_lens, dtype=torch.int32)
    max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
    block_tables = torch.randint(
        0, num_blocks, (num_seqs, max_num_blocks_per_seq), dtype=torch.int32
    )

    assert common_prefix_len > 0
    assert common_prefix_len % block_size == 0
    num_common_kv_blocks = common_prefix_len // block_size
    # Make sure the first `num_common_kv_blocks` blocks are the same.
    block_tables[:, :num_common_kv_blocks] = block_tables[0, :num_common_kv_blocks]

    # Run the regular attention.
    ref_output = flash_attn_varlen_func(
        q=query,
        k=key_cache,
        v=value_cache,
        cu_seqlens_q=cu_query_lens,
        seqused_k=kv_lens_tensor,
        max_seqlen_q=max_query_len,
        max_seqlen_k=max_kv_len,
        softmax_scale=scale,
        causal=True,
        window_size=window_size,
        block_table=block_tables,
        softcap=soft_cap if soft_cap is not None else 0,
    )

    # Run cascade attention.
    assert all(common_prefix_len < kv_len for kv_len in kv_lens)
    cu_prefix_query_lens = torch.tensor([0, total_num_query_tokens], dtype=torch.int32)
    prefix_kv_lens = torch.tensor([common_prefix_len], dtype=torch.int32)
    suffix_kv_lens = kv_lens_tensor - common_prefix_len
    output = torch.empty_like(query)
    cascade_attention(
        output=output,
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        cu_query_lens=cu_query_lens,
        max_query_len=max_query_len,
        cu_prefix_query_lens=cu_prefix_query_lens,
        prefix_kv_lens=prefix_kv_lens,
        suffix_kv_lens=suffix_kv_lens,
        max_kv_len=max_kv_len,
        softmax_scale=scale,
        alibi_slopes=None,
        sliding_window=window_size,
        logits_soft_cap=soft_cap if soft_cap is not None else 0,
        block_table=block_tables,
        common_prefix_len=common_prefix_len,
        max_num_splits=0,  # no max
        fa_version=fa_version,
    )

    # Compare the results.
    torch.testing.assert_close(output, ref_output, atol=1e-2, rtol=1e-2)


def _rope_delta_neox(tensor: torch.Tensor, delta: int) -> torch.Tensor:
    """Apply a constant Qwen-style RoPE delta to post-RoPE Q or K."""
    head_size = tensor.shape[-1]
    work = tensor.float()
    inv_freq = 1.0 / (
        1_000_000.0
        ** (
            torch.arange(0, head_size, 2, device=tensor.device, dtype=torch.float32)
            / head_size
        )
    )
    angle = delta * inv_freq
    cos = angle.cos().view(1, 1, -1)
    sin = angle.sin().view(1, 1, -1)
    first, second = work.chunk(2, dim=-1)
    rotated = torch.cat(
        (first * cos - second * sin, second * cos + first * sin), dim=-1
    )
    return rotated.to(tensor.dtype)


def test_segmentia_shared_middle_segment_prepares_query_and_lse_bias() -> None:
    query = torch.randn(3, 4, 8, dtype=torch.float32)
    request_indices = torch.tensor([0, 0, 1], dtype=torch.int64)
    shared_starts = torch.tensor([8, 20], dtype=torch.int64)
    k_offset = torch.randn(2, 2, 8, dtype=torch.float32)
    metadata = SimpleNamespace(
        segmentia_query_request_indices=request_indices,
        segmentia_shared_start_positions=shared_starts,
        segmentia_rope_theta=1_000_000.0,
        segmentia_k_offset=k_offset,
    )

    prefix_query, prefix_lse_bias = prepare_segmentia_prefix_inputs(
        query,
        metadata,
        num_query_heads=4,
        num_kv_heads=2,
        softmax_scale=8**-0.5,
    )

    expected_query = torch.cat(
        (_rope_delta_neox(query[:2], -8), _rope_delta_neox(query[2:], -20))
    )
    expanded_offset = k_offset.index_select(0, request_indices).repeat_interleave(
        2, dim=1
    )
    expected_bias = (
        torch.einsum("thd,thd->th", query, expanded_offset) * 8**-0.5
    ).transpose(0, 1)
    torch.testing.assert_close(prefix_query, expected_query)
    torch.testing.assert_close(prefix_lse_bias, expected_bias)


def test_cascade_uses_corrected_query_and_bias_only_for_shared_segment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    query = torch.randn(2, 4, 8)
    prefix_query = torch.randn_like(query)
    prefix_lse_bias = torch.randn(4, 2)
    key_cache = torch.randn(4, 16, 2, 8)
    value_cache = torch.randn_like(key_cache)
    calls: list[dict[str, object]] = []
    merged_prefix_lse: list[torch.Tensor] = []

    def fake_flash_attn_varlen_func(**kwargs):
        calls.append(kwargs)
        current_query = kwargs["q"]
        assert isinstance(current_query, torch.Tensor)
        return current_query.clone(), torch.zeros(4, 2)

    def fake_merge_attn_states(
        output,
        prefix_output,
        prefix_lse,
        suffix_output,
        suffix_lse,
    ):
        merged_prefix_lse.append(prefix_lse.clone())
        output.copy_(prefix_output + suffix_output)

    monkeypatch.setattr(
        flash_attn_backend,
        "flash_attn_varlen_func",
        fake_flash_attn_varlen_func,
        raising=False,
    )
    monkeypatch.setattr(
        flash_attn_backend, "merge_attn_states", fake_merge_attn_states
    )

    output = torch.empty_like(query)
    cascade_attention(
        output=output,
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        cu_query_lens=torch.tensor([0, 1, 2], dtype=torch.int32),
        max_query_len=1,
        cu_prefix_query_lens=torch.tensor([0, 2], dtype=torch.int32),
        prefix_kv_lens=torch.tensor([16], dtype=torch.int32),
        suffix_kv_lens=torch.tensor([16, 16], dtype=torch.int32),
        max_kv_len=32,
        softmax_scale=8**-0.5,
        alibi_slopes=None,
        sliding_window=(-1, -1),
        logits_soft_cap=0,
        block_table=torch.tensor([[0, 1], [0, 2]], dtype=torch.int32),
        common_prefix_len=16,
        max_num_splits=0,
        fa_version=2,
        prefix_query=prefix_query,
        prefix_lse_bias=prefix_lse_bias,
    )

    assert calls[0]["q"] is prefix_query
    assert calls[1]["q"] is query
    torch.testing.assert_close(merged_prefix_lse[0], prefix_lse_bias)
    torch.testing.assert_close(output, prefix_query + query)


@pytest.mark.parametrize("dtype", DTYPES)
@torch.inference_mode()
def test_cascade_shared_middle_segment_with_query_and_lse_correction(
    dtype: torch.dtype,
) -> None:
    """A middle Skill can use one physical KV copy across requests.

    The reference materializes a position-shifted and offset-corrected Skill K
    for every request in logical order ``private_before | Skill | current``.
    The cascade path keeps one source-position Skill copy and views the same
    attention as ``shared Skill | private_before,current``. It moves the RoPE
    position delta to the prefix query and the uniform K offset to prefix LSE.
    """
    torch.set_default_device("cuda")
    if not is_fa_version_supported(2):
        pytest.skip(
            "FlashAttention 2 is unsupported: "
            f'"{fa_version_unsupported_reason(2)}"'
        )
    set_random_seed(23)

    num_reqs = 2
    num_query_heads = 40
    num_kv_heads = 8
    head_size = 128
    block_size = 16
    shared_len = 2 * block_size
    private_before_len = block_size
    query_len = 1
    private_len = private_before_len + query_len
    scale = head_size**-0.5
    position_deltas = (-173, 911)

    # Blocks 0 and 1 are the only physical source-position Skill copy.
    num_blocks = 10
    source_key_cache = torch.zeros(
        num_blocks, block_size, num_kv_heads, head_size, dtype=dtype
    )
    source_value_cache = torch.zeros_like(source_key_cache)
    shared_key = torch.randn(shared_len, num_kv_heads, head_size, dtype=dtype)
    shared_value = torch.randn_like(shared_key)
    source_key_cache[0:2].copy_(shared_key.view(2, block_size, num_kv_heads, head_size))
    source_value_cache[0:2].copy_(
        shared_value.view(2, block_size, num_kv_heads, head_size)
    )

    private_block_ids = (2, 3)
    current_block_ids = (4, 5)
    for private_block, current_block in zip(
        private_block_ids, current_block_ids, strict=True
    ):
        source_key_cache[private_block].normal_()
        source_value_cache[private_block].normal_()
        source_key_cache[current_block].normal_()
        source_value_cache[current_block].normal_()

    query = torch.randn(
        num_reqs * query_len, num_query_heads, head_size, dtype=dtype
    )
    offset = torch.randn(
        num_reqs, num_kv_heads, head_size, dtype=torch.float32
    ) * 0.03

    # Reference cache: request 0 owns blocks 6/7 and request 1 owns 8/9.
    materialized_key_cache = source_key_cache.clone()
    materialized_value_cache = source_value_cache.clone()
    materialized_skill_blocks = ((6, 7), (8, 9))
    reference_block_table = torch.empty((num_reqs, 4), dtype=torch.int32)
    for req_idx, (delta, skill_blocks) in enumerate(
        zip(position_deltas, materialized_skill_blocks, strict=True)
    ):
        corrected_key = _rope_delta_neox(shared_key, delta)
        corrected_key.add_(offset[req_idx].to(dtype).unsqueeze(0))
        skill_block_tensor = torch.tensor(skill_blocks, dtype=torch.long)
        corrected_key_blocks = corrected_key.view(
            2, block_size, num_kv_heads, head_size
        )
        shared_value_blocks = shared_value.view(
            2, block_size, num_kv_heads, head_size
        )
        materialized_key_cache.index_copy_(
            0, skill_block_tensor, corrected_key_blocks
        )
        materialized_value_cache.index_copy_(
            0, skill_block_tensor, shared_value_blocks
        )
        torch.testing.assert_close(
            materialized_key_cache.index_select(0, skill_block_tensor),
            corrected_key_blocks,
        )
        torch.testing.assert_close(
            materialized_value_cache.index_select(0, skill_block_tensor),
            shared_value_blocks,
        )
        reference_block_table[req_idx] = torch.tensor(
            [private_block_ids[req_idx], *skill_blocks, current_block_ids[req_idx]],
            dtype=torch.int32,
        )

    cu_query_lens = torch.arange(num_reqs + 1, dtype=torch.int32)
    reference_kv_lens = torch.full(
        (num_reqs,), private_before_len + shared_len + query_len, dtype=torch.int32
    )
    reference_output = flash_attn_varlen_func(
        q=query,
        k=materialized_key_cache,
        v=materialized_value_cache,
        cu_seqlens_q=cu_query_lens,
        seqused_k=reference_kv_lens,
        max_seqlen_q=query_len,
        max_seqlen_k=int(reference_kv_lens[0]),
        softmax_scale=scale,
        causal=True,
        window_size=(-1, -1),
        block_table=reference_block_table,
        fa_version=2,
    )

    # Prototype cache view: both rows start with the exact same Skill blocks.
    shared_block_table = torch.tensor(
        [
            [0, 1, private_block_ids[0], current_block_ids[0]],
            [0, 1, private_block_ids[1], current_block_ids[1]],
        ],
        dtype=torch.int32,
    )
    assert torch.equal(shared_block_table[0, :2], shared_block_table[1, :2])

    prefix_query, prefix_lse_bias = prepare_segmentia_prefix_inputs(
        query,
        SimpleNamespace(
            segmentia_query_request_indices=torch.arange(
                num_reqs, dtype=torch.int64
            ),
            segmentia_shared_start_positions=torch.tensor(
                position_deltas, dtype=torch.int64
            ),
            segmentia_rope_theta=1_000_000.0,
            segmentia_k_offset=offset,
        ),
        num_query_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        softmax_scale=scale,
    )

    output = torch.empty_like(query)
    cascade_attention(
        output=output,
        query=query,
        key_cache=source_key_cache,
        value_cache=source_value_cache,
        cu_query_lens=cu_query_lens,
        max_query_len=query_len,
        cu_prefix_query_lens=torch.tensor([0, num_reqs], dtype=torch.int32),
        prefix_kv_lens=torch.tensor([shared_len], dtype=torch.int32),
        suffix_kv_lens=torch.full((num_reqs,), private_len, dtype=torch.int32),
        max_kv_len=shared_len + private_len,
        softmax_scale=scale,
        alibi_slopes=None,
        sliding_window=(-1, -1),
        logits_soft_cap=0,
        block_table=shared_block_table,
        common_prefix_len=shared_len,
        max_num_splits=0,
        fa_version=2,
        prefix_query=prefix_query,
        prefix_lse_bias=prefix_lse_bias,
    )

    torch.testing.assert_close(output, reference_output, atol=2e-2, rtol=2e-2)
