from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

from vllm import SamplingParams
from vllm.entrypoints.openai.chat_completion.serving import (
    _parse_structured_skill_action,
    _record_structured_skill_action,
    _trailing_tool_observations,
)
from vllm.entrypoints.openai.engine.protocol import FunctionCall, ToolCall
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.async_llm import AsyncLLM


def test_t0_parser_accepts_only_complete_skill_actions() -> None:
    assert (
        _parse_structured_skill_action(
            ToolCall(
                id="tool-call-1",
                function=FunctionCall(
                    name="skill", arguments='{"name":"internal-comms"}'
                ),
            )
        )
        == "internal-comms"
    )
    assert (
        _parse_structured_skill_action(
            ToolCall(
                id="tool-call-2",
                function=FunctionCall(name="skill", arguments='{"name":'),
            )
        )
        is None
    )


def test_only_trailing_tool_results_are_forwarded_to_cskcache() -> None:
    messages = [
        {"role": "tool", "name": "skill", "tool_call_id": "old", "content": "x"},
        {"role": "assistant", "content": "done"},
        {"role": "tool", "name": "terminal", "tool_call_id": "term", "content": "ok"},
        {
            "role": "tool",
            "name": "skill",
            "tool_call_id": "new",
            "content": [{"type": "text", "text": "skill body"}],
        },
    ]

    assert _trailing_tool_observations(messages) == [
        ("term", "terminal", "ok"),
        ("new", "skill", "skill body"),
    ]
    assert (
        _parse_structured_skill_action(
            ToolCall(
                id="tool-call-3",
                function=FunctionCall(name="terminal", arguments='{"name":"x"}'),
            )
        )
        is None
    )


def test_t0_timeline_records_the_complete_structured_action(
    tmp_path, monkeypatch
) -> None:
    action_path = tmp_path / "action.jsonl"
    monkeypatch.setattr(
        "vllm.entrypoints.openai.chat_completion.serving._SKILL_ACTION_TRACE_PATH",
        str(action_path),
    )
    tool_call = ToolCall(
        id="tool-call-1",
        function=FunctionCall(name="skill", arguments='{"name":"docx"}'),
    )

    _record_structured_skill_action(
        "chatcmpl-cskcache-window-request-a",
        tool_call,
        unix_ns=10,
        monotonic_ns=100,
    )

    action = json.loads(action_path.read_text(encoding="utf-8"))
    assert action["skill_action_ready_monotonic_ns"] == 100
    assert action["tool_call_id"] == "tool-call-1"
    assert action["skill_name"] == "docx"


def make_cskcache_candidate_request(*, n: int = 1) -> EngineCoreRequest:
    sampling_params = SamplingParams(
        max_tokens=1,
        n=n,
        extra_args={
            "kv_transfer_params": {
                "cskcache_candidate": {"ticket": "tool-call-1"},
                "unrelated": "preserved",
            }
        },
    )
    return EngineCoreRequest(
        request_id="chatcmpl-external-a1b2c3d4",
        external_req_id="chatcmpl-external",
        prompt_token_ids=[10, 11, 12],
        mm_features=None,
        sampling_params=sampling_params,
        pooling_params=None,
        arrival_time=0.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
    )


def test_candidate_is_authenticated_with_internal_request_id() -> None:
    request = make_cskcache_candidate_request()
    engine = AsyncLLM.__new__(AsyncLLM)
    verified = {
        "ticket": "tool-call-1",
        "request_id": request.request_id,
        "cache_object_id": "mcp-builder:object",
        "segment_start": 100,
        "segment_end": 2019,
    }
    engine.authenticate_csk_request = AsyncMock(return_value=verified)
    engine.cancel_csk_prefetch = AsyncMock()

    asyncio.run(engine._authenticate_cskcache_candidate(request))

    engine.authenticate_csk_request.assert_awaited_once_with(
        "tool-call-1", "chatcmpl-external-a1b2c3d4", [10, 11, 12]
    )
    engine.cancel_csk_prefetch.assert_not_awaited()
    transfer_params = request.sampling_params.extra_args["kv_transfer_params"]
    assert "cskcache_candidate" not in transfer_params
    assert transfer_params["cskcache_verified"] == verified
    assert transfer_params["unrelated"] == "preserved"


def test_add_request_authenticates_only_after_internal_id_assignment() -> None:
    request = make_cskcache_candidate_request()
    request.request_id = "chatcmpl-external"
    request.external_req_id = None
    engine = AsyncLLM.__new__(AsyncLLM)
    engine.output_handler = None
    engine.log_requests = False
    engine.vllm_config = MagicMock()
    engine.vllm_config.cache_config.kv_sharing_fast_prefill = False
    engine.engine_core = MagicMock()
    engine.engine_core.resources.engine_dead = False
    engine.engine_core.add_request_async = AsyncMock()
    engine.output_processor = MagicMock()
    engine._run_output_handler = MagicMock()
    engine.input_processor = MagicMock()

    def assign_internal_id(engine_request: EngineCoreRequest) -> None:
        engine_request.external_req_id = engine_request.request_id
        engine_request.request_id = f"{engine_request.request_id}-a1b2c3d4"

    engine.input_processor.assign_request_id.side_effect = assign_internal_id

    async def authenticate(ticket, request_id, prompt_token_ids):
        return {
            "ticket": ticket,
            "request_id": request_id,
            "cache_object_id": "mcp-builder:object",
            "segment_start": 100,
            "segment_end": 2019,
        }

    engine.authenticate_csk_request = AsyncMock(side_effect=authenticate)
    engine.cancel_csk_prefetch = AsyncMock()

    asyncio.run(
        engine.add_request(
            "chatcmpl-external", request, request.sampling_params
        )
    )

    engine.authenticate_csk_request.assert_awaited_once_with(
        "tool-call-1", "chatcmpl-external-a1b2c3d4", [10, 11, 12]
    )
    admitted_request = engine.engine_core.add_request_async.await_args.args[0]
    assert admitted_request.request_id == "chatcmpl-external-a1b2c3d4"
    verified = admitted_request.sampling_params.extra_args[
        "kv_transfer_params"
    ]["cskcache_verified"]
    assert verified["request_id"] == admitted_request.request_id


def test_parallel_candidate_is_cancelled_before_engine_admission() -> None:
    request = make_cskcache_candidate_request(n=2)
    engine = AsyncLLM.__new__(AsyncLLM)
    engine.authenticate_csk_request = AsyncMock()
    engine.cancel_csk_prefetch = AsyncMock()

    asyncio.run(engine._authenticate_cskcache_candidate(request))

    engine.authenticate_csk_request.assert_not_awaited()
    engine.cancel_csk_prefetch.assert_awaited_once_with(
        "tool-call-1", "parallel_sampling_unsupported"
    )
    transfer_params = request.sampling_params.extra_args["kv_transfer_params"]
    assert transfer_params == {"unrelated": "preserved"}
