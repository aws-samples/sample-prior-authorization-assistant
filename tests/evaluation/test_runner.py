"""The extraction comparison runner, with a fake Bedrock client."""

import contextlib
import json
import os
from unittest import mock

import pytest
from botocore.exceptions import ClientError

from evaluation import ground_truth as gt
from evaluation.extraction import runner, textract_replay

CASES = {case["case_id"]: case for case in gt.load_cases()}
PDF_CASE = CASES["anthem-knee-surgery-pa-approved.pdf"]
FHIR_CASE = CASES["anthem-knee-surgery-pa-approved.fhir"]
PDF_CASES = [case for case in CASES.values() if case["format"] == "pdf"]


class FakeBedrock:
    """Records Converse requests and answers with the case's expected values"""

    def __init__(self, expected, error=None):
        self.requests = []
        self._error = error
        self._text = json.dumps(
            {
                "payor_name": "Anthem Blue Cross Blue Shield",
                "cpt_codes": expected.get("cpt_codes", []),
                "icd_codes": expected.get("icd_codes", []),
            }
        )

    def converse(self, **request):
        self.requests.append(request)
        if self._error:
            raise self._error
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


def _row(model_id):
    rows, _ = runner.build_rows([model_id])
    return rows[-1]


def _fixture_calls(case):
    stem = case["source_file"].rsplit("/", 1)[1].rsplit(".", 1)[0]
    return json.loads((runner.FIXTURES / f"{stem}.json").read_text())["calls"]


def _run(case, row, bedrock, fixture=None):
    with runner.mocked_aws(), mock.patch("time.sleep", lambda _s: None):
        from common import aws_clients
        from extractors import handler

        aws_clients._client_manager = None
        with contextlib.ExitStack() as stack:
            stack.enter_context(
                mock.patch.object(
                    aws_clients.AWSClientManager,
                    "get_bedrock_client",
                    lambda self: bedrock,
                )
            )
            if fixture is not None:
                stack.enter_context(
                    mock.patch.object(
                        textract_replay.ReplayTextractClient,
                        "__init__",
                        _replay_init(fixture),
                    )
                )
            return runner.run_case(handler, aws_clients, row, case, trial=1)


def _replay_init(calls):
    def init(self, document, bucket):
        self._calls = list(calls)
        self._bucket = bucket
        self.calls_made = 0
        self.mismatch = None

    return init


# --- happy paths ---------------------------------------------------------


@pytest.mark.parametrize("case", PDF_CASES, ids=lambda c: c["case_id"])
def test_pdf_case_replays_its_own_fixture(fake_credentials, case):
    record = _run(case, _row("fake.model-v1"), FakeBedrock(case["expected"]))

    assert record["status"] == "ok", record["error"]
    assert record["textract_calls_replayed"] == len(_fixture_calls(case))


def test_candidate_sends_override_and_role_temperature(fake_credentials):
    bedrock = FakeBedrock(PDF_CASE["expected"])

    record = _run(PDF_CASE, _row("fake.model-v1"), bedrock)

    assert record["result"]["cpt_codes"] == PDF_CASE["expected"]["cpt_codes"]
    assert bedrock.requests[0]["modelId"] == "fake.model-v1"
    assert bedrock.requests[0]["inferenceConfig"] == {
        "maxTokens": 4000,
        "temperature": 0.1,
    }
    assert record["sampling"] == {"temperature": 0.1, "top_p": None}
    assert record["model_calls"][0]["model_id"] == "fake.model-v1"


def test_as_shipped_row_records_what_it_sends(fake_credentials):
    bedrock = FakeBedrock(PDF_CASE["expected"])
    as_shipped = runner.build_rows([])[0][0]

    record = _run(PDF_CASE, as_shipped, bedrock)

    assert bedrock.requests[0]["modelId"] == "amazon.nova-pro-v1:0"
    assert bedrock.requests[0]["inferenceConfig"] == {
        "maxTokens": 4000,
        "temperature": 0.1,
        "topP": 0.9,
    }
    assert record["sampling"] == {"temperature": 0.1, "top_p": 0.9}


def test_fhir_case_makes_no_model_call(fake_credentials):
    bedrock = FakeBedrock(FHIR_CASE["expected"])

    record = _run(FHIR_CASE, _row("fake.model-v1"), bedrock)

    assert record["status"] == "ok", record["error"]
    assert bedrock.requests == []
    assert record["model_calls"] == []


# --- rows and sampling ---------------------------------------------------


def test_candidate_equal_to_default_is_not_a_separate_row():
    rows, skipped = runner.build_rows(["amazon.nova-pro-v1:0", "fake.model-v1"])

    assert [r["row"] for r in rows] == ["as-shipped", "fake.model-v1"]
    assert "amazon.nova-pro-v1:0 is not run as a separate row" in skipped[0]


def test_models_rejecting_custom_sampling_run_at_defaults():
    row = _row("us.anthropic.claude-sonnet-5-5")

    assert row["env"] == {
        "PA_MODEL_ID_MEDICAL_EXTRACTION": "us.anthropic.claude-sonnet-5-5"
    }
    assert row["sampling"] == {"temperature": None, "top_p": None}
    assert "rejects non-default sampling" in row["sampling_note"]


def test_ambient_overrides_do_not_leak_into_rows(fake_credentials, monkeypatch):
    monkeypatch.setenv("PA_MODEL_ID", "us.anthropic.claude-sonnet-4-6")
    monkeypatch.setenv("PA_TOP_P_MEDICAL_EXTRACTION", "0.5")
    bedrock = FakeBedrock(PDF_CASE["expected"])
    rows, _ = runner.build_rows(["fake.model-v1"])

    assert rows[0]["model_id"] == "amazon.nova-pro-v1:0"
    assert rows[0]["sampling"] == {"temperature": 0.1, "top_p": 0.9}
    _run(PDF_CASE, rows[0], bedrock)
    _run(PDF_CASE, rows[1], bedrock)

    assert [r["modelId"] for r in bedrock.requests] == [
        "amazon.nova-pro-v1:0",
        "fake.model-v1",
    ]
    assert bedrock.requests[1]["inferenceConfig"] == {
        "maxTokens": 4000,
        "temperature": 0.1,
    }


# --- isolation from real AWS ---------------------------------------------


def test_moto_configuration_is_pinned():
    assert runner.MOTO_CONFIG == {
        "core": {
            "mock_credentials": False,
            "passthrough": {"services": ["bedrockruntime"]},
        }
    }


def test_non_bedrock_calls_stay_inside_moto(fake_credentials):
    import boto3

    with runner.mocked_aws():
        identity = boto3.client("sts", region_name="us-east-1").get_caller_identity()

    assert identity["Account"] == "123456789012"  # moto's account, not a real one


def test_unimplemented_services_fail_inside_moto(fake_credentials):
    import boto3

    with runner.mocked_aws():
        client = boto3.client("kendra-ranking", region_name="us-east-1")
        with pytest.raises(ClientError):
            client.list_rescore_execution_plans()


def test_mocked_aws_restores_environment(fake_credentials, monkeypatch):
    monkeypatch.setenv("DOCUMENTS_BUCKET", "outer-value")
    before = {k: os.environ.get(k) for k in runner._ENV_KEYS}

    with runner.mocked_aws():
        assert os.environ["DOCUMENTS_BUCKET"] == runner.DOCUMENTS_BUCKET

    assert {k: os.environ.get(k) for k in runner._ENV_KEYS} == before


# --- replay strictness and error paths -----------------------------------


def test_wrong_textract_parameters_are_an_error(fake_credentials):
    calls = _fixture_calls(PDF_CASE)
    calls[0] = {**calls[0], "params": {**calls[0]["params"], "ClientRequestToken": "x"}}

    record = _run(PDF_CASE, _row("fake.model-v1"), FakeBedrock({}), fixture=calls)

    assert record["status"] == "error"
    assert "parameters differ from the recording" in record["error"]


def test_unused_recorded_textract_calls_are_an_error(fake_credentials):
    calls = _fixture_calls(PDF_CASE) + [_fixture_calls(PDF_CASE)[-1]]
    bedrock = FakeBedrock(PDF_CASE["expected"])

    record = _run(PDF_CASE, _row("fake.model-v1"), bedrock, fixture=calls)

    assert record["status"] == "error"
    assert record["error"] == "FixtureMismatch: 1 recorded Textract calls unused"


def test_wrong_textract_operation_is_reported_as_the_first_mismatch(
    fake_credentials,
):
    calls = _fixture_calls(PDF_CASE)
    calls[0] = {**calls[0], "operation": "detect_document_text"}

    record = _run(PDF_CASE, _row("fake.model-v1"), FakeBedrock({}), fixture=calls)

    assert record["status"] == "error"
    assert record["error"].startswith(
        "FixtureMismatch: expected detect_document_text, "
        "got start_document_text_detection"
    )


def test_model_errors_are_recorded_as_failed_trials(fake_credentials):
    denied = ClientError(
        {"Error": {"Code": "AccessDeniedException", "Message": "denied"}}, "Converse"
    )
    bedrock = FakeBedrock(PDF_CASE["expected"], error=denied)

    record = _run(PDF_CASE, _row("fake.model-v1"), bedrock)

    assert record["status"] == "error"
    assert "AccessDeniedException" in record["error"]
    assert record["result"] is None


# --- dry run -------------------------------------------------------------


def test_dry_run_makes_no_aws_calls(capsys):
    with (
        mock.patch("boto3.client", side_effect=AssertionError("AWS call")),
        mock.patch("boto3.Session", side_effect=AssertionError("AWS session")),
    ):
        assert runner.main(["--dry-run", "--models", "amazon.nova-pro-v1:0"]) == 0

    output = capsys.readouterr().out
    assert "Dry run (no AWS calls)" in output
    assert "3 PDF exercise a model, 4 FHIR controls" in output
    rows_line = next(line for line in output.splitlines() if line.startswith("Rows:"))
    assert "us.anthropic.claude-haiku-4-5-20251001-v1:0" in rows_line
    assert "Not run separately: amazon.nova-pro-v1:0" in output
    assert "6 calls per trial" in output
