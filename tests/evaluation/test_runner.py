"""The extraction comparison runner, with a fake Bedrock client."""

import json
import os
from unittest import mock

import pytest

from evaluation import ground_truth as gt
from evaluation.extraction import runner

CASES = {case["case_id"]: case for case in gt.load_cases()}
PDF_CASE = CASES["anthem-knee-surgery-pa-approved.pdf"]
FHIR_CASE = CASES["anthem-knee-surgery-pa-approved.fhir"]


class FakeBedrock:
    """Records Converse requests and answers with the case's expected values"""

    def __init__(self, expected):
        self.requests = []
        self._text = json.dumps(
            {
                "payor_name": "Anthem Blue Cross Blue Shield",
                "cpt_codes": expected["cpt_codes"],
                "icd_codes": expected["icd_codes"],
            }
        )

    def converse(self, **request):
        self.requests.append(request)
        return {
            "output": {"message": {"content": [{"text": self._text}]}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 1200, "outputTokens": 90, "totalTokens": 1290},
            "metrics": {"latencyMs": 850},
        }


@pytest.fixture
def fake_credentials(monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.delenv("AWS_PROFILE", raising=False)
    for name in list(os.environ):
        if name.startswith(("PA_MODEL_ID", "PA_TEMPERATURE_", "PA_TOP_P_")):
            monkeypatch.delenv(name)


def _run(case, row, bedrock):
    with (
        runner.mocked_aws(),
        mock.patch("time.sleep", lambda _s: None),
    ):
        from common import aws_clients
        from extractors import handler

        aws_clients._client_manager = None
        with mock.patch.object(
            aws_clients.AWSClientManager, "get_bedrock_client", lambda self: bedrock
        ):
            return runner.run_case(handler, aws_clients, row, case, trial=1)


def test_pdf_case_runs_through_handler_with_replayed_textract(fake_credentials):
    bedrock = FakeBedrock(PDF_CASE["expected"])
    row = runner.build_rows(["fake.model-v1"])[1]

    record = _run(PDF_CASE, row, bedrock)

    assert record["status"] == "ok", record["error"]
    assert record["result"]["cpt_codes"] == PDF_CASE["expected"]["cpt_codes"]
    assert record["result"]["icd_codes"] == PDF_CASE["expected"]["icd_codes"]
    recorded = json.loads(
        (runner.FIXTURES / "anthem-knee-surgery-pa-approved.json").read_text()
    )
    assert record["textract_calls_replayed"] == len(recorded["calls"])
    # The candidate gets the override and the role's temperature, with no top_p
    assert len(bedrock.requests) == 1
    assert bedrock.requests[0]["modelId"] == "fake.model-v1"
    assert bedrock.requests[0]["inferenceConfig"] == {
        "maxTokens": 4000,
        "temperature": 0.1,
    }
    assert record["model_calls"] == [
        {
            "model_id": "fake.model-v1",
            "input_tokens": 1200,
            "output_tokens": 90,
            "latency_ms": 850,
            "stop_reason": "end_turn",
        }
    ]


def test_as_shipped_row_sends_role_defaults(fake_credentials):
    bedrock = FakeBedrock(PDF_CASE["expected"])

    _run(PDF_CASE, runner.build_rows([])[0], bedrock)

    assert bedrock.requests[0]["modelId"] == "amazon.nova-pro-v1:0"
    assert bedrock.requests[0]["inferenceConfig"] == {
        "maxTokens": 4000,
        "temperature": 0.1,
        "topP": 0.9,
    }


def test_fhir_case_makes_no_model_call(fake_credentials):
    bedrock = FakeBedrock(FHIR_CASE["expected"])
    row = runner.build_rows(["fake.model-v1"])[1]

    record = _run(FHIR_CASE, row, bedrock)

    assert record["status"] == "ok", record["error"]
    assert bedrock.requests == []
    assert record["model_calls"] == []
    assert sorted(record["result"]["cpt_codes"]) == FHIR_CASE["expected"]["cpt_codes"]


def test_models_rejecting_custom_sampling_run_at_defaults():
    row = runner.build_rows(["us.anthropic.claude-sonnet-5-5"])[1]

    assert row["env"] == {
        "PA_MODEL_ID_MEDICAL_EXTRACTION": "us.anthropic.claude-sonnet-5-5"
    }
    assert row["sampling"] == {"temperature": None, "top_p": None}
    assert "rejects non-default sampling" in row["sampling_note"]


def test_non_bedrock_calls_stay_inside_moto(fake_credentials):
    import boto3

    with runner.mocked_aws():
        identity = boto3.client("sts", region_name="us-east-1").get_caller_identity()

    assert identity["Account"] == "123456789012"  # moto's account, not a real one


def test_dry_run_makes_no_aws_calls(capsys):
    with (
        mock.patch("boto3.client", side_effect=AssertionError("AWS call")),
        mock.patch("boto3.Session", side_effect=AssertionError("AWS session")),
    ):
        assert runner.main(["--dry-run"]) == 0

    output = capsys.readouterr().out
    assert "Dry run (no AWS calls)" in output
    assert "3 PDF exercise a model, 4 FHIR controls" in output
