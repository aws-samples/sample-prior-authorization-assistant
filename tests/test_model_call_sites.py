"""Call sites resolve their model through the registry.

Every generation call goes through BedrockService.invoke_role with the role
and max_tokens the call site used before the registry existed.
"""

import ast
from pathlib import Path
from unittest.mock import create_autospec, patch

import pytest

from common.aws_clients import BedrockService

SRC = Path(__file__).resolve().parents[1] / "src"

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
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == method
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
