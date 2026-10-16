"""Call sites resolve their model through the registry.

Every generation call goes through BedrockService.invoke_role with the role
and max_tokens the call site used before the registry existed.
"""

import ast
import os
from pathlib import Path
from unittest.mock import create_autospec, patch

import pytest

from common.aws_clients import BedrockService

SRC = Path(__file__).resolve().parents[1] / "src"


@pytest.fixture(autouse=True)
def clear_overrides(monkeypatch):
    for name in list(os.environ):
        if name.startswith(("PA_MODEL_ID", "PA_TEMPERATURE_", "PA_TOP_P_")):
            monkeypatch.delenv(name)


EXPECTED_ROLE_CALLS = {
    "extractors/medical_extractor.py": [("medical_extraction", 4000)],
    "selector/policy_selector.py": [("policy_selection", 1000)],
    "selector/criteria_extractor.py": [("policy_criteria_extraction", 4000)],
    "reviewer/compliance_engine.py": [("compliance_assessment", 2000)],
    "reviewer/evidence_mapper.py": [("evidence_mapping", 3000)],
    "reviewer/gap_analyzer.py": [("gap_analysis", 2000)],
}


def _calls(path, method):
    tree = ast.parse((SRC / path).read_text())
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and (
            (isinstance(node.func, ast.Attribute) and node.func.attr == method)
            or (isinstance(node.func, ast.Name) and node.func.id == method)
        )
    ]


def _role_and_max_tokens(call):
    role = call.args[0].value
    max_tokens = next(kw.value.value for kw in call.keywords if kw.arg == "max_tokens")
    return role, max_tokens


@pytest.mark.parametrize("path", sorted(EXPECTED_ROLE_CALLS))
def test_call_site_uses_its_role_and_max_tokens(path):
    found = sorted(_role_and_max_tokens(c) for c in _calls(path, "invoke_role"))

    assert found == sorted(EXPECTED_ROLE_CALLS[path])


def test_invoke_model_remains_only_in_clinical_summary():
    remaining = {
        path: len(_calls(path, "invoke_model")) for path in EXPECTED_ROLE_CALLS
    }

    assert remaining == {
        **{path: 0 for path in EXPECTED_ROLE_CALLS},
        "selector/policy_selector.py": 1,
    }


def test_clinical_summary_still_uses_fallback():
    from selector import policy_selector

    bedrock_service = create_autospec(BedrockService, instance=True)
    with (
        patch.object(
            policy_selector, "get_bedrock_service", return_value=bedrock_service
        ),
        patch.object(policy_selector, "PolicyRetrievalService"),
    ):
        selector = policy_selector.IntelligentPolicySelector()

    # Long enough (>= 100 chars) to reach the model call
    raw_text = (
        "Chief complaint: low back pain radiating to the left leg. "
        "Failed physical therapy for 6 weeks. MRI shows L4-L5 disc herniation."
    )
    with (
        patch.object(
            selector, "_extract_clinical_summary_fallback", return_value="fallback"
        ) as fallback,
        patch.object(policy_selector, "logger") as logger,
    ):
        summary = selector._extract_clinical_summary(raw_text)

    assert summary == "fallback"
    fallback.assert_called_once_with(raw_text)
    # The existing temperature= argument raises TypeError before any model call
    logger.warning.assert_called_once()
    assert "temperature" in logger.warning.call_args.kwargs["error"]
    bedrock_service.invoke_role.assert_not_called()


def test_clinical_criteria_handler_uses_its_own_client_and_role():
    calls = _calls("extract_clinical_criteria/handler.py", "converse_role")
    assert len(calls) == 1
    client, role, _prompt, max_tokens = calls[0].args

    # Keeps the module's client, which has a 120s read timeout
    assert client.id == "bedrock_runtime"
    assert (role.value, max_tokens.value) == ("clinical_criteria_extraction", 4000)
    assert _calls("extract_clinical_criteria/handler.py", "invoke_model") == []


def test_metadata_extraction_call_uses_passed_client_and_role(monkeypatch):
    # The package's handler module creates boto3 clients at import time
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    import boto3
    from botocore.stub import Stubber

    from metadata_extractor.metadata_extractor import _call_bedrock_claude

    client = boto3.client(
        "bedrock-runtime",
        region_name="us-east-1",
        aws_access_key_id="testing",
        aws_secret_access_key="testing",
    )
    with Stubber(client) as stubber:
        stubber.add_response(
            "converse",
            {
                "output": {
                    "message": {"role": "assistant", "content": [{"text": "{}"}]}
                },
                "stopReason": "end_turn",
                "usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2},
                "metrics": {"latencyMs": 1},
            },
            {
                "modelId": "us.anthropic.claude-sonnet-4-6",
                "messages": [{"role": "user", "content": [{"text": "prompt"}]}],
                "inferenceConfig": {"maxTokens": 2000, "temperature": 0.0},
            },
        )

        assert _call_bedrock_claude(client, "prompt") == "{}"
