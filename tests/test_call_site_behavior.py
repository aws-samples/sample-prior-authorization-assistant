"""Generation call sites behave as on main when the model call fails or is empty.

Expected outcomes were observed by running each call site on main (InvokeModel
bodies through BedrockService.invoke_model for Nova, Anthropic bodies for the
two Claude sites) against a fake client returning the same failure or empty
text. The call counts include the call sites' own retry decorators and loops.
"""

import os
from datetime import datetime
from unittest.mock import create_autospec, patch

import boto3
import pytest
from botocore.exceptions import ClientError
from botocore.stub import ANY, Stubber

from common import decorators
from common.aws_clients import AWSClientManager, BedrockService
from common.exceptions import AssessmentError, AWSServiceError, PolicySelectionError
from common.models import ClinicalCriterion, ExtractedData, PatientInfo

NOVA_PRO = "amazon.nova-pro-v1:0"
SONNET_4_6 = "us.anthropic.claude-sonnet-4-6"
DENIED = "Bedrock model invocation failed: An error occurred (AccessDeniedException)"
MINIMAL_METADATA = {
    "payor": "Aetna",
    "title": None,
    "procedure_codes": [],
    "icd_codes": [],
    "effective_date": None,
    "policy_number": None,
    "service_category": None,
}


@pytest.fixture(autouse=True)
def clear_overrides(monkeypatch):
    for name in list(os.environ):
        if name.startswith(("PA_MODEL_ID", "PA_TEMPERATURE_", "PA_TOP_P_")):
            monkeypatch.delenv(name)
    # Some handler modules create boto3 clients at import time
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")


@pytest.fixture(autouse=True)
def no_backoff(monkeypatch):
    monkeypatch.setattr(decorators.time, "sleep", lambda _delay: None)


def _real_client(service_name):
    return boto3.client(
        service_name,
        region_name="us-east-1",
        aws_access_key_id="testing",
        aws_secret_access_key="testing",
    )


def _access_denied(**_kwargs):
    raise ClientError(
        {"Error": {"Code": "AccessDeniedException", "Message": "denied"}},
        "Converse",
    )


def _empty_text(**_kwargs):
    return {
        "output": {"message": {"role": "assistant", "content": [{"text": ""}]}},
        "stopReason": "end_turn",
        "usage": {"inputTokens": 10, "outputTokens": 0, "totalTokens": 10},
        "metrics": {"latencyMs": 5},
    }


def _service(client):
    manager = create_autospec(AWSClientManager, instance=True)
    manager.get_bedrock_client.return_value = client
    return BedrockService(manager)


def _extracted_data():
    return ExtractedData(
        document_id="doc-1",
        cpt_codes=["22630"],
        icd_codes=["M51.26"],
        service_type="Spinal Fusion",
        payor_name="Aetna",
        plan_type="PPO",
        patient_info=PatientInfo("patient-1", "50-60", "male", "M51.26"),
        extraction_confidence=0.9,
        # Under 100 chars, so policy selection skips the clinical summary call
        raw_text="Short record",
    )


def _criterion():
    return ClinicalCriterion(
        "crit_001", "Failed conservative care", ["Physical therapy notes"], []
    )


def _assessment_context():
    from reviewer.compliance_engine import AssessmentContext

    return AssessmentContext("50-60", "male", "M51.26", [], "Aetna", "Spinal Fusion")


def _medical_extraction(client):
    from extractors import medical_extractor

    with patch.object(
        medical_extractor, "get_bedrock_service", return_value=_service(client)
    ):
        extractor = medical_extractor.MedicalDataExtractor()
    return extractor._extract_with_bedrock("Patient record text")


def _policy_selection(client):
    from knowledge.policy_retrieval import PolicySearchResult
    from selector import policy_selector

    def policy(policy_id):
        return PolicySearchResult(
            policy_id,
            "Aetna",
            f"Policy {policy_id}",
            0.9,
            0.9,
            "Matched content",
            {},
            datetime(2024, 1, 1),
        )

    with (
        patch.object(
            policy_selector, "get_bedrock_service", return_value=_service(client)
        ),
        patch.object(policy_selector, "PolicyRetrievalService"),
    ):
        selector = policy_selector.IntelligentPolicySelector()
    return selector._ai_select_best_policy(
        [(policy("a"), 0.9), (policy("b"), 0.8)], _extracted_data()
    )


def _policy_criteria_extraction(client):
    from selector import criteria_extractor

    with patch.object(
        criteria_extractor, "get_bedrock_service", return_value=_service(client)
    ):
        extractor = criteria_extractor.ClinicalCriteriaExtractor()
    return extractor._ai_extract_criteria("Policy content", _extracted_data())


def _compliance_assessment(client):
    from reviewer import compliance_engine

    with patch.object(
        compliance_engine, "get_bedrock_service", return_value=_service(client)
    ):
        engine = compliance_engine.ComplianceAssessmentEngine()
    evaluation = engine._evaluate_single_criterion(
        _criterion(), [], _assessment_context()
    )
    return evaluation.decision.value, evaluation.confidence_score


def _evidence_mapping(client):
    from reviewer import evidence_mapper

    with patch.object(
        evidence_mapper, "get_bedrock_service", return_value=_service(client)
    ):
        mapper = evidence_mapper.ClinicalEvidenceMapper()
    return mapper._map_criterion_evidence_direct(_criterion(), "Records", "doc-1")


def _gap_analysis(client):
    from reviewer import gap_analyzer

    with patch.object(
        gap_analyzer, "get_bedrock_service", return_value=_service(client)
    ):
        engine = gap_analyzer.GapAnalysisEngine()
    return engine._ai_generate_recommendations([], [], _assessment_context())


def _clinical_criteria_extraction(client):
    from extract_clinical_criteria import handler

    with patch.object(handler, "bedrock_runtime", client):
        criteria = handler.extract_clinical_criteria(
            "Policy content", "Aetna", ["22630"], ["M51.26"], "Spinal Fusion"
        )
    return [criterion.criterion_id for criterion in criteria]


def _policy_metadata_extraction(client):
    from metadata_extractor.metadata_extractor import extract_metadata_from_text

    return extract_metadata_from_text(client, "Policy text", "Aetna")


def _returns(value):
    return ("returns", value)


def _raises(exc_type, message_prefix):
    return ("raises", exc_type, message_prefix)


# (runner, model, outcome, model calls) per call site when the model call is
# denied. The comment on each entry is main's observed behavior.
ACCESS_DENIED_CASES = [
    # main: _extract_with_bedrock retries once (2 calls), then raises
    # AWSServiceError "Bedrock extraction failed: Bedrock model invocation
    # failed: An error occurred (AccessDeniedException) ..."
    pytest.param(
        _medical_extraction,
        NOVA_PRO,
        _raises(AWSServiceError, "Bedrock extraction failed: " + DENIED),
        2,
        id="medical_extraction",
    ),
    # main: 1 call (PolicySelectionError is not retried), raises
    # PolicySelectionError "AI policy selection failed: Bedrock model
    # invocation failed: ... (AccessDeniedException) ..."
    pytest.param(
        _policy_selection,
        NOVA_PRO,
        _raises(PolicySelectionError, "AI policy selection failed: " + DENIED),
        1,
        id="policy_selection",
    ),
    # main: 3 calls (AWSServiceError is retried), raises AWSServiceError
    # "Criteria extraction failed: Bedrock model invocation failed: ..."
    pytest.param(
        _policy_criteria_extraction,
        NOVA_PRO,
        _raises(AWSServiceError, "Criteria extraction failed: " + DENIED),
        3,
        id="policy_criteria_extraction",
    ),
    # main: 1 call (AssessmentError is not retried), raises AssessmentError
    # "AI assessment failed for criterion crit_001: Bedrock model ..."
    pytest.param(
        _compliance_assessment,
        NOVA_PRO,
        _raises(
            AssessmentError, "AI assessment failed for criterion crit_001: " + DENIED
        ),
        1,
        id="compliance_assessment",
    ),
    # main: 1 call, error swallowed, returns []
    pytest.param(_evidence_mapping, NOVA_PRO, _returns([]), 1, id="evidence_mapping"),
    # main: 1 call, error swallowed, returns []
    pytest.param(_gap_analysis, NOVA_PRO, _returns([]), 1, id="gap_analysis"),
    # main: 1 call (raw ClientError caught), returns the default criterion
    pytest.param(
        _clinical_criteria_extraction,
        SONNET_4_6,
        _returns(["default_001"]),
        1,
        id="clinical_criteria_extraction",
    ),
    # main: 3 calls (max_retries=3), returns minimal metadata
    pytest.param(
        _policy_metadata_extraction,
        SONNET_4_6,
        _returns(MINIMAL_METADATA),
        3,
        id="policy_metadata_extraction",
    ),
]

# Same, when Converse returns a single {"text": ""} block with end_turn.
EMPTY_TEXT_CASES = [
    # main: 1 call, returns {}
    pytest.param(
        _medical_extraction, NOVA_PRO, _returns({}), 1, id="medical_extraction"
    ),
    # main: 1 call, raises PolicySelectionError "AI policy selection failed:
    # Failed to parse AI selection response: AI policy selection parsing failed"
    pytest.param(
        _policy_selection,
        NOVA_PRO,
        _raises(
            PolicySelectionError,
            "AI policy selection failed: Failed to parse AI selection response: "
            "AI policy selection parsing failed",
        ),
        1,
        id="policy_selection",
    ),
    # main: 1 call, returns []
    pytest.param(
        _policy_criteria_extraction,
        NOVA_PRO,
        _returns([]),
        1,
        id="policy_criteria_extraction",
    ),
    # main: 1 call, returns an insufficient_evidence evaluation, confidence 0.0
    pytest.param(
        _compliance_assessment,
        NOVA_PRO,
        _returns(("insufficient_evidence", 0.0)),
        1,
        id="compliance_assessment",
    ),
    # main: 1 call, returns []
    pytest.param(_evidence_mapping, NOVA_PRO, _returns([]), 1, id="evidence_mapping"),
    # main: 1 call, returns []
    pytest.param(_gap_analysis, NOVA_PRO, _returns([]), 1, id="gap_analysis"),
    # main: 1 call, unparseable text gives [] (not the default criterion)
    pytest.param(
        _clinical_criteria_extraction,
        SONNET_4_6,
        _returns([]),
        1,
        id="clinical_criteria_extraction",
    ),
    # main: 3 calls ("" fails JSON parsing each time), returns minimal metadata
    pytest.param(
        _policy_metadata_extraction,
        SONNET_4_6,
        _returns(MINIMAL_METADATA),
        3,
        id="policy_metadata_extraction",
    ),
]


def _assert_outcome(run, client, expected):
    if expected[0] == "raises":
        _, exc_type, message_prefix = expected
        with pytest.raises(exc_type) as excinfo:
            run(client)
        assert type(excinfo.value) is exc_type
        assert str(excinfo.value).startswith(message_prefix)
    else:
        assert run(client) == expected[1]


@pytest.mark.parametrize("run,model,expected,calls", ACCESS_DENIED_CASES)
def test_access_denied_matches_main(run, model, expected, calls):
    client = create_autospec(_real_client("bedrock-runtime"), instance=True)
    client.converse.side_effect = _access_denied

    _assert_outcome(run, client, expected)

    assert client.converse.call_count == calls
    assert {c.kwargs["modelId"] for c in client.converse.call_args_list} == {model}


@pytest.mark.parametrize("run,model,expected,calls", EMPTY_TEXT_CASES)
def test_empty_text_matches_main(run, model, expected, calls):
    client = create_autospec(_real_client("bedrock-runtime"), instance=True)
    client.converse.side_effect = _empty_text

    _assert_outcome(run, client, expected)

    assert client.converse.call_count == calls
    assert {c.kwargs["modelId"] for c in client.converse.call_args_list} == {model}


def test_clinical_criteria_client_keeps_timeouts_and_retries():
    from extract_clinical_criteria import handler

    config = handler.bedrock_runtime.meta.config

    assert (config.read_timeout, config.connect_timeout) == (120, 10)
    # botocore stores retries={"max_attempts": 2} as 2 retries + 1 initial call
    assert config.retries["total_max_attempts"] == 3


def test_shared_bedrock_client_uses_adaptive_retries():
    config = AWSClientManager(region="us-east-1").get_bedrock_client().meta.config

    assert config.retries["mode"] == "adaptive"
    # botocore stores retries={"max_attempts": 3} as 3 retries + 1 initial call
    assert config.retries["total_max_attempts"] == 4


def _kb_response():
    return {
        "sessionId": "session-1",
        "output": {"text": '{"selection_rationale": "Best match"}'},
        "citations": [
            {
                "retrievedReferences": [
                    {
                        "location": {
                            "type": "S3",
                            "s3Location": {"uri": "s3://policies/aetna/fusion.pdf"},
                        }
                    }
                ]
            }
        ],
    }


@pytest.mark.parametrize(
    "override,model_arn",
    [
        (None, SONNET_4_6),
        ("us.anthropic.claude-sonnet-5-5", "us.anthropic.claude-sonnet-5-5"),
    ],
    ids=["default", "PA_MODEL_ID_KB_GENERATION"],
)
def test_kb_generation_sends_resolved_model_arn(monkeypatch, override, model_arn):
    from intelligent_policy_selector import handler

    monkeypatch.setenv("KNOWLEDGE_BASE_ID", "KB12345678")
    if override:
        monkeypatch.setenv("PA_MODEL_ID_KB_GENERATION", override)
    client = _real_client("bedrock-agent-runtime")
    monkeypatch.setattr(handler, "bedrock_agent_runtime", client)

    with Stubber(client) as stubber:
        stubber.add_response(
            "retrieve_and_generate",
            _kb_response(),
            {
                "input": ANY,
                "retrieveAndGenerateConfiguration": {
                    "type": "KNOWLEDGE_BASE",
                    "knowledgeBaseConfiguration": {
                        "knowledgeBaseId": "KB12345678",
                        "modelArn": model_arn,
                        "retrievalConfiguration": ANY,
                    },
                },
            },
        )

        result = handler.select_policy_with_kb(_extracted_data(), ["22630"], {}, {})

        stubber.assert_no_pending_responses()
    assert result["selected_policy_s3_uri"] == "s3://policies/aetna/fusion.pdf"


def test_policy_selector_model_id_defaults_to_nova_pro():
    from selector import policy_selector

    with (
        patch.object(policy_selector, "get_bedrock_service"),
        patch.object(policy_selector, "PolicyRetrievalService"),
    ):
        selector = policy_selector.IntelligentPolicySelector()

    assert selector.model_id == NOVA_PRO
