"""Unit tests for Converse-based BedrockService invocation.

Uses botocore's Stubber, so each test asserts the exact Converse request.
"""

import os
from unittest.mock import MagicMock

import boto3
import pytest
from botocore.config import Config
from botocore.stub import Stubber

from common import aws_clients
from common.aws_clients import BedrockService
from common.exceptions import AWSServiceError, BedrockThrottlingException

PROMPT = "Summarize the policy."
NOVA_PRO = "amazon.nova-pro-v1:0"
SONNET_4_6 = "us.anthropic.claude-sonnet-4-6"


@pytest.fixture(autouse=True)
def clear_overrides(monkeypatch):
    for name in list(os.environ):
        if name.startswith(("PA_MODEL_ID", "PA_TEMPERATURE_", "PA_TOP_P_")):
            monkeypatch.delenv(name)


@pytest.fixture
def stubbed_service():
    client = boto3.client(
        "bedrock-runtime",
        region_name="us-east-1",
        aws_access_key_id="testing",
        aws_secret_access_key="testing",
        config=Config(retries={"max_attempts": 1}),
    )
    manager = MagicMock()
    manager.get_bedrock_client.return_value = client
    with Stubber(client) as stubber:
        yield BedrockService(manager), stubber
        stubber.assert_no_pending_responses()


@pytest.fixture
def observed():
    records = []
    aws_clients.register_usage_observer(records.append)
    yield records
    aws_clients.unregister_usage_observer(records.append)


def _request(model_id, inference_config):
    return {
        "modelId": model_id,
        "messages": [{"role": "user", "content": [{"text": PROMPT}]}],
        "inferenceConfig": inference_config,
    }


def _response(content, stop_reason="end_turn", input_tokens=12, output_tokens=5):
    return {
        "output": {"message": {"role": "assistant", "content": content}},
        "stopReason": stop_reason,
        "usage": {
            "inputTokens": input_tokens,
            "outputTokens": output_tokens,
            "totalTokens": input_tokens + output_tokens,
        },
        "metrics": {"latencyMs": 321},
    }


def test_invoke_model_keeps_signature_and_sampling(stubbed_service):
    service, stubber = stubbed_service
    stubber.add_response(
        "converse",
        _response([{"text": "Summary."}]),
        _request(NOVA_PRO, {"maxTokens": 2000, "temperature": 0.1, "topP": 0.9}),
    )

    assert service.invoke_model(model_id=NOVA_PRO, prompt=PROMPT, max_tokens=2000) == (
        "Summary."
    )


def test_invoke_model_still_rejects_temperature_argument(stubbed_service):
    service, _ = stubbed_service

    with pytest.raises(TypeError):
        service.invoke_model(
            model_id=NOVA_PRO, prompt=PROMPT, max_tokens=400, temperature=0.3
        )


def test_same_call_works_for_nova_and_claude(stubbed_service):
    service, stubber = stubbed_service
    stubber.add_response(
        "converse",
        _response([{"text": "From Nova."}]),
        _request(NOVA_PRO, {"maxTokens": 4000, "temperature": 0.1, "topP": 0.9}),
    )
    stubber.add_response(
        "converse",
        _response([{"text": "From Claude."}]),
        _request(SONNET_4_6, {"maxTokens": 4000, "temperature": 0.0}),
    )

    assert service.invoke_role("gap_analysis", PROMPT, 4000) == "From Nova."
    assert service.invoke_role("clinical_criteria_extraction", PROMPT, 4000) == (
        "From Claude."
    )


def test_unset_sampling_settings_are_omitted(stubbed_service, monkeypatch):
    service, stubber = stubbed_service
    monkeypatch.setenv("PA_MODEL_ID_GAP_ANALYSIS", "us.anthropic.claude-sonnet-5-5")
    stubber.add_response(
        "converse",
        _response([{"text": "ok"}]),
        _request("us.anthropic.claude-sonnet-5-5", {"maxTokens": 2000}),
    )

    assert service.invoke_role("gap_analysis", PROMPT, 2000) == "ok"


def test_reasoning_blocks_are_excluded(stubbed_service):
    service, stubber = stubbed_service
    stubber.add_response(
        "converse",
        _response(
            [
                {"reasoningContent": {"reasoningText": {"text": "thinking..."}}},
                {"text": "Answer."},
            ]
        ),
        _request(NOVA_PRO, {"maxTokens": 300, "temperature": 0.1, "topP": 0.9}),
    )

    assert service.invoke_model(NOVA_PRO, PROMPT, 300) == "Answer."


def test_truncated_response_without_text_blocks_raises(stubbed_service, observed):
    service, stubber = stubbed_service
    stubber.add_response(
        "converse",
        _response(
            [{"reasoningContent": {"reasoningText": {"text": "thinking..."}}}],
            stop_reason="max_tokens",
            output_tokens=20,
        ),
        _request(NOVA_PRO, {"maxTokens": 20, "temperature": 0.1, "topP": 0.9}),
    )

    with pytest.raises(AWSServiceError, match="max_tokens") as excinfo:
        service.invoke_model(NOVA_PRO, PROMPT, 20)
    assert type(excinfo.value) is AWSServiceError
    assert "Unexpected error" not in str(excinfo.value)
    assert observed[-1].stop_reason == "max_tokens"


def test_response_without_text_blocks_raises(stubbed_service):
    service, stubber = stubbed_service
    stubber.add_response(
        "converse",
        _response([], stop_reason="end_turn"),
        _request(SONNET_4_6, {"maxTokens": 100, "temperature": 0.0}),
    )

    with pytest.raises(AWSServiceError, match="end_turn"):
        service.invoke_role("clinical_criteria_extraction", PROMPT, 100)


@pytest.mark.parametrize(
    "content",
    [[{"text": ""}], [{"text": ""}, {"text": ""}]],
    ids=["one-empty-text", "all-empty-text"],
)
def test_empty_text_block_returns_empty_string_as_before(stubbed_service, content):
    """Parity with the previous content[0]['text'] parsing (returned "")."""
    service, stubber = stubbed_service
    stubber.add_response(
        "converse",
        _response(content, stop_reason="end_turn"),
        _request(SONNET_4_6, {"maxTokens": 100, "temperature": 0.0}),
    )

    assert service.invoke_role("clinical_criteria_extraction", PROMPT, 100) == ""


def test_missing_usage_fields_do_not_discard_text(observed):
    # botocore's response shape requires usage and metrics, so this guards
    # against non-standard clients rather than a documented API response.
    client = MagicMock()
    client.converse.return_value = {
        "output": {"message": {"role": "assistant", "content": [{"text": "ok"}]}},
        "stopReason": "end_turn",
    }

    assert aws_clients.converse_role(client, "gap_analysis", PROMPT, 100) == "ok"
    assert (observed[-1].input_tokens, observed[-1].latency_ms) == (None, None)


def test_partial_text_at_token_limit_is_returned(stubbed_service):
    service, stubber = stubbed_service
    stubber.add_response(
        "converse",
        _response([{"text": "Partial answ"}], stop_reason="max_tokens"),
        _request(NOVA_PRO, {"maxTokens": 5, "temperature": 0.1, "topP": 0.9}),
    )

    assert service.invoke_model(NOVA_PRO, PROMPT, 5) == "Partial answ"


def test_observer_receives_usage(stubbed_service, observed):
    service, stubber = stubbed_service
    stubber.add_response(
        "converse",
        _response([{"text": "ok"}], input_tokens=12, output_tokens=5),
        _request(SONNET_4_6, {"maxTokens": 2000, "temperature": 0.0}),
    )

    service.invoke_role("policy_metadata_extraction", PROMPT, 2000)

    assert len(observed) == 1
    record = observed[0]
    assert record.model_id == SONNET_4_6
    assert record.role == "policy_metadata_extraction"
    assert (record.input_tokens, record.output_tokens) == (12, 5)
    assert record.latency_ms == 321
    assert record.stop_reason == "end_turn"


def test_failing_observer_does_not_break_invocation(stubbed_service):
    service, stubber = stubbed_service
    stubber.add_response(
        "converse",
        _response([{"text": "ok"}]),
        _request(NOVA_PRO, {"maxTokens": 100, "temperature": 0.1, "topP": 0.9}),
    )

    def broken(record):
        raise RuntimeError("observer bug")

    aws_clients.register_usage_observer(broken)
    try:
        assert service.invoke_model(NOVA_PRO, PROMPT, 100) == "ok"
    finally:
        aws_clients.unregister_usage_observer(broken)


def test_throttling_surfaces_as_application_error(stubbed_service):
    service, stubber = stubbed_service
    stubber.add_client_error(
        "converse", service_error_code="ThrottlingException", http_status_code=429
    )

    with pytest.raises(BedrockThrottlingException):
        service.invoke_model(NOVA_PRO, PROMPT, 100)


def test_other_client_errors_surface_as_aws_service_error(stubbed_service):
    service, stubber = stubbed_service
    stubber.add_client_error(
        "converse", service_error_code="ValidationException", http_status_code=400
    )

    with pytest.raises(AWSServiceError) as excinfo:
        service.invoke_role("gap_analysis", PROMPT, 100)
    assert not isinstance(excinfo.value, BedrockThrottlingException)
