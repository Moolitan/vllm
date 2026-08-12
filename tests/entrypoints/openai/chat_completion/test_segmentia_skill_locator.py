from __future__ import annotations

import json

import pytest

from vllm.entrypoints.openai.chat_completion.serving import (
    _locate_segmentia_skill_span,
    _record_skill_locator,
    _record_structured_skill_action,
    _token_ids_sha256,
)
from vllm.entrypoints.openai.engine.protocol import FunctionCall, ToolCall


def locator_config(skill_tokens: list[int]) -> dict[str, object]:
    marker = skill_tokens[:2]
    return {
        "cache_id": "internal-comms",
        "skill_name": "internal-comms",
        "token_count": len(skill_tokens),
        "token_ids_sha256": _token_ids_sha256(skill_tokens),
        "locator": {
            "kind": "context_segment_start_marker_v1",
            "start_marker_token_ids": marker,
            "start_marker_token_count": len(marker),
            "start_marker_token_ids_sha256": _token_ids_sha256(marker),
        },
    }


def test_locator_uses_final_prompt_tokens_and_full_span_hash() -> None:
    skill_tokens = [101, 102, 103, 104]
    config = locator_config(skill_tokens)

    start, end = _locate_segmentia_skill_span(
        [1, 2, *skill_tokens, 9], config
    )

    assert (start, end) == (2, 6)
    assert config["segment_start"] == 2
    assert config["segment_end"] == 6


def test_locator_rejects_tampered_or_repeated_skill_span() -> None:
    skill_tokens = [101, 102, 103, 104]
    with pytest.raises(ValueError, match="exactly one authenticated"):
        _locate_segmentia_skill_span(
            [1, 101, 102, 999, 104, 9], locator_config(skill_tokens)
        )

    repeated = [*skill_tokens, 8, *skill_tokens]
    with pytest.raises(ValueError, match=r"matches=\[0, 5\]"):
        _locate_segmentia_skill_span(repeated, locator_config(skill_tokens))


def test_locator_rejects_stale_marker_metadata() -> None:
    skill_tokens = [101, 102, 103, 104]
    config = locator_config(skill_tokens)
    config["locator"]["start_marker_token_count"] = 3  # type: ignore[index]

    with pytest.raises(ValueError, match="start_marker_token_count is stale"):
        _locate_segmentia_skill_span(skill_tokens, config)


def test_locator_sets_cache_end_after_prefix_location() -> None:
    skill_tokens = [101, 102, 103, 104]
    config = locator_config(skill_tokens)
    config["correction_mode"] = "prefix_k_headwise"

    _locate_segmentia_skill_span([7, *skill_tokens, 8], config)

    assert config["cache_end"] == 5


def test_vllm_timeline_records_monotonic_boundaries(
    tmp_path, monkeypatch
) -> None:
    action_path = tmp_path / "action.jsonl"
    locator_path = tmp_path / "locator.jsonl"
    monkeypatch.setattr(
        "vllm.entrypoints.openai.chat_completion.serving._SKILL_ACTION_TRACE_PATH",
        str(action_path),
    )
    monkeypatch.setattr(
        "vllm.entrypoints.openai.chat_completion.serving._SKILL_LOCATOR_TRACE_PATH",
        str(locator_path),
    )
    tool_call = ToolCall(
        id="tool-call-1",
        function=FunctionCall(name="skill", arguments='{"name":"docx"}'),
    )

    _record_structured_skill_action(
        "chatcmpl-segmentia-window-request-a",
        tool_call,
        unix_ns=10,
        monotonic_ns=100,
    )
    _record_skill_locator(
        "chatcmpl-segmentia-window-request-b",
        20,
        30,
        source_tool_call_id="tool-call-1",
        request_received_monotonic_ns=200,
        tokenization_completed_monotonic_ns=300,
        locator_start_monotonic_ns=310,
        locator_end_monotonic_ns=320,
        status="ok",
        skill_name="docx",
        segment_start=5,
        segment_end=10,
    )

    action = json.loads(action_path.read_text(encoding="utf-8"))
    locator = json.loads(locator_path.read_text(encoding="utf-8"))
    assert action["skill_action_ready_monotonic_ns"] == 100
    assert action["tool_call_id"] == "tool-call-1"
    assert locator["source_tool_call_id"] == "tool-call-1"
    assert locator["request_received_monotonic_ns"] == 200
    assert locator["tokenization_completed_monotonic_ns"] == 300
    assert locator["locator_duration_ms"] == 0.00001
    assert action["boot_id"] == locator["boot_id"]
