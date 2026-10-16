"""Compare models on the medical data extraction step.

Drives the real extraction Lambda handler in-process with Step Functions-shaped
events. S3, DynamoDB and CloudWatch are mocked with moto; Bedrock is real;
Textract is replayed from recorded fixtures.

Usage:
    python -m evaluation.extraction.runner --dry-run
    python -m evaluation.extraction.runner --trials 3 [--out DIR]

AWS credentials come from the environment (for example AWS_PROFILE).
"""

import argparse
import contextlib
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterator, List, Optional
from unittest import mock

from evaluation import ground_truth as gt
from evaluation.extraction.textract_replay import FIXTURES, ReplayTextractClient

REGION = "us-east-1"
DOCUMENTS_BUCKET = "pa-eval-documents"
EXTRACTION_TABLE = "pa-eval-extraction"
ROLE = "medical_extraction"
ROLE_TEMPERATURE = 0.1  # the medical_extraction default (src/common/model_config.py)
MAX_TOKENS = 4000  # the medical_extraction call site's max_tokens

CANDIDATES = [
    "us.anthropic.claude-sonnet-4-6",
    "amazon.nova-pro-v1:0",
    "openai.gpt-oss-120b-1:0",
]
LIFECYCLE_GATED = {
    # Included only if Bedrock reports the model ACTIVE on run day
    "us.anthropic.claude-haiku-4-5-20251001-v1:0": (
        "anthropic.claude-haiku-4-5-20251001-v1:0"
    ),
}
# Models that reject any non-default sampling value run at their own defaults
REJECTS_NON_DEFAULT_SAMPLING = {
    "us.anthropic.claude-sonnet-5-5",
    "us.anthropic.claude-haiku-5-5",
}
UNAVAILABLE_MODELS = [
    {
        "model_id": "us.anthropic.claude-sonnet-5-5",
        "reason": "AccessDeniedException: not available for this account (2026-10-16)",
    },
    {
        "model_id": "us.anthropic.claude-haiku-5-5",
        "reason": "AccessDeniedException: not available for this account (2026-10-16)",
    },
]
NOTES = [
    "Only the medical_extraction step runs; no other pipeline step is evaluated.",
    "Textract input is replayed from fixtures recorded 2026-10-16 through the "
    "production DocumentProcessor path. On all three PDFs, synchronous "
    "AnalyzeDocument (TABLES, FORMS) fails with UnsupportedDocumentException "
    "and the code continues on text detection only, as it does in production.",
    "FHIR cases make no model call (the FHIR route parses codes from the "
    "bundle). They run once per row as a control.",
    "Payor results pass through MedicalCodeValidator.normalize_payor_name "
    "(src/common/validators.py), which maps any name containing 'Blue Cross Blue "
    "Shield' to 'Blue Cross Blue Shield' and drops 'Anthem'. The Anthem PDF's "
    "payor therefore scores as wrong for every row, independent of the model.",
    "The FHIR route keeps only the first CPT code in a bundle "
    "(src/extractors/fhir_extractor.py, single service_info.cpt_code), so "
    "multi-procedure bundles lose codes in the FHIR control rows.",
]

MOTO_CONFIG = {
    "core": {"mock_credentials": False, "passthrough": {"services": ["bedrockruntime"]}}
}


def build_rows(models: List[str]) -> List[Dict[str, Any]]:
    rows = [
        {
            "row": "as-shipped",
            "model_id": "amazon.nova-pro-v1:0",
            "env": {},
            "sampling": {"temperature": 0.1, "top_p": 0.9},
            "sampling_note": "role defaults",
        }
    ]
    for model_id in models:
        if model_id in REJECTS_NON_DEFAULT_SAMPLING:
            env, sampling = {}, {"temperature": None, "top_p": None}
            note = "model rejects non-default sampling; runs at its own defaults"
        else:
            env = {"PA_TEMPERATURE_MEDICAL_EXTRACTION": str(ROLE_TEMPERATURE)}
            sampling = {"temperature": ROLE_TEMPERATURE, "top_p": None}
            note = None
        env["PA_MODEL_ID_MEDICAL_EXTRACTION"] = model_id
        rows.append(
            {
                "row": model_id,
                "model_id": model_id,
                "env": env,
                "sampling": sampling,
                "sampling_note": note,
            }
        )
    return rows


def check_lifecycle(checked_at: str) -> List[Dict[str, Any]]:
    """Return lifecycle records for gated models (real AWS call)"""
    import boto3

    bedrock = boto3.client("bedrock", region_name=REGION)
    records = []
    for profile_id, base_id in LIFECYCLE_GATED.items():
        details = bedrock.get_foundation_model(modelIdentifier=base_id)["modelDetails"]
        records.append(
            {
                "model_id": profile_id,
                "status": details.get("modelLifecycle", {}).get("status"),
                "source": f"bedrock:GetFoundationModel({base_id})",
                "checked_at": checked_at,
            }
        )
    return records


def plan(cases, rows, trials) -> List[Dict[str, Any]]:
    runs = []
    for row in rows:
        for case in cases:
            for trial in range(1, (trials if case["format"] == "pdf" else 1) + 1):
                runs.append({"row": row, "case": case, "trial": trial})
    return runs


def estimate_cost(runs, pricing) -> Dict[str, Any]:
    """Estimate model cost from fixture text length (4 characters per token)"""
    per_row: Dict[str, Dict[str, float]] = {}
    for run in runs:
        if run["case"]["format"] != "pdf":
            continue
        input_tokens = _fixture_chars(run["case"]) / 4 + 1500  # text + prompt template
        price = pricing["models"].get(run["row"]["model_id"])
        entry = per_row.setdefault(
            run["row"]["row"], {"calls": 0, "typical_usd": 0.0, "worst_usd": 0.0}
        )
        entry["calls"] += 1
        if price is None:
            entry["typical_usd"] = entry["worst_usd"] = math.nan
            continue
        cost_in = input_tokens / 1000 * price["input"]
        entry["typical_usd"] += cost_in + 800 / 1000 * price["output"]
        entry["worst_usd"] += 2 * (cost_in + MAX_TOKENS / 1000 * price["output"])
    return per_row


@contextlib.contextmanager
def mocked_aws() -> Iterator[None]:
    """moto for every service except Bedrock Runtime, with real credentials"""
    import boto3
    from moto import mock_aws

    credentials = boto3.Session().get_credentials()
    if credentials is None:
        raise SystemExit("No AWS credentials found (set AWS_PROFILE)")
    frozen = credentials.get_frozen_credentials()
    saved = {k: os.environ.get(k) for k in _ENV_KEYS}
    os.environ.update(
        AWS_ACCESS_KEY_ID=frozen.access_key,
        AWS_SECRET_ACCESS_KEY=frozen.secret_key,
        AWS_SESSION_TOKEN=frozen.token or "",
        AWS_DEFAULT_REGION=REGION,
        AWS_REGION=REGION,
        DOCUMENTS_BUCKET=DOCUMENTS_BUCKET,
        EXTRACTION_TABLE=EXTRACTION_TABLE,
    )
    os.environ.pop("AWS_PROFILE", None)
    try:
        with mock_aws(config=MOTO_CONFIG):
            s3 = boto3.client("s3", region_name=REGION)
            s3.create_bucket(Bucket=DOCUMENTS_BUCKET)
            for document in gt.sample_documents():
                s3.upload_file(
                    str(document), DOCUMENTS_BUCKET, f"samples/{document.name}"
                )
            boto3.client("dynamodb", region_name=REGION).create_table(
                TableName=EXTRACTION_TABLE,
                KeySchema=[{"AttributeName": "document_id", "KeyType": "HASH"}],
                AttributeDefinitions=[
                    {"AttributeName": "document_id", "AttributeType": "S"}
                ],
                BillingMode="PAY_PER_REQUEST",
            )
            yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def run_case(handler_module, aws_clients, row, case, trial) -> Dict[str, Any]:
    """Invoke the extraction handler once and return a trial record"""
    from common import bedrock_converse

    document = gt.ROOT / case["source_file"]
    usage: List[Any] = []

    def observe(record):
        if record.role == ROLE:
            usage.append(record)

    replay = ReplayTextractClient(document.name) if case["format"] == "pdf" else None
    document_id = f"eval-{document.stem}-{trial}"
    event = {
        "document_id": document_id,
        "bucket_name": DOCUMENTS_BUCKET,
        "object_key": f"samples/{document.name}",
        "filename": document.name,
        "content_type": (
            "application/pdf" if case["format"] == "pdf" else "application/json"
        ),
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    context = SimpleNamespace(
        aws_request_id=document_id, function_name="eval-extractor"
    )

    record = {
        "row": row["row"],
        "model_id": row["model_id"],
        "sampling": row["sampling"],
        "sampling_note": row["sampling_note"],
        "case_id": case["case_id"],
        "format": case["format"],
        "trial": trial,
    }
    start = time.monotonic()
    bedrock_converse.register_usage_observer(observe)
    env_patch = mock.patch.dict(os.environ, row["env"])
    textract_patch = mock.patch.object(
        aws_clients.AWSClientManager,
        "get_textract_client",
        lambda self: replay if replay is not None else _NoTextract(),
    )
    try:
        with env_patch, textract_patch:
            output = handler_module.lambda_handler(event, context)
        result = output["extraction_result"]
        record.update(
            status="ok",
            error=None,
            result={
                "payor_name": result.get("payor_name") or "",
                "cpt_codes": list(result.get("cpt_codes") or []),
                "icd_codes": list(result.get("icd_codes") or []),
            },
        )
    except Exception as e:  # every failure is recorded and counted
        record.update(
            status="error", error=f"{type(e).__name__}: {e}"[:500], result=None
        )
    finally:
        bedrock_converse.unregister_usage_observer(observe)
    record["duration_ms"] = int((time.monotonic() - start) * 1000)
    record["model_calls"] = [
        {
            "model_id": u.model_id,
            "input_tokens": u.input_tokens,
            "output_tokens": u.output_tokens,
            "latency_ms": u.latency_ms,
            "stop_reason": u.stop_reason,
        }
        for u in usage
    ]
    if replay is not None:
        record["textract_calls_replayed"] = replay.calls_made
    return record


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--models", default=",".join(CANDIDATES))
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-lifecycle-check", action="store_true")
    args = parser.parse_args(argv)

    cases = [c for c in gt.load_cases() if c["confirmed"]]
    pricing = json.loads((gt.ROOT / "evaluation" / "pricing.json").read_text())
    models = [m for m in args.models.split(",") if m]
    started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    lifecycle: List[Dict[str, Any]] = []
    if not args.dry_run and not args.skip_lifecycle_check:
        lifecycle = check_lifecycle(started_at)
        models += [r["model_id"] for r in lifecycle if r["status"] == "ACTIVE"]

    rows = build_rows(models)
    runs = plan(cases, rows, args.trials)

    if args.dry_run:
        print(_dry_run_text(cases, rows, runs, args.trials, pricing))
        return 0

    out = args.out or gt.ROOT / "evaluation" / "results" / started_at.replace(":", "")
    out.mkdir(parents=True, exist_ok=True)
    trials = []
    with mocked_aws(), mock.patch("time.sleep", lambda _s: None):
        from common import aws_clients
        from extractors import handler as handler_module

        for index, run in enumerate(runs, start=1):
            aws_clients._client_manager = None  # clients are created inside the mock
            trial = run_case(
                handler_module, aws_clients, run["row"], run["case"], run["trial"]
            )
            trials.append(trial)
            print(
                f"[{index}/{len(runs)}] {trial['row']} {trial['case_id']} "
                f"#{trial['trial']}: {trial['status']}",
                flush=True,
            )
    with open(out / "trials.jsonl", "w") as handle:
        for trial in trials:
            handle.write(json.dumps(trial) + "\n")

    manifest = {
        "git_commit": _git_commit(),
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "region": REGION,
        "rows": [
            {k: r[k] for k in ("row", "model_id", "sampling", "sampling_note")}
            for r in rows
        ],
        "trials": args.trials,
        "case_ids": [c["case_id"] for c in cases],
        "textract_input": {
            c["case_id"]: _fixture_label(c) for c in cases if c["format"] == "pdf"
        },
        "pricing": {"as_of": pricing["as_of"], "source": pricing["source"]},
        "lifecycle_checks": lifecycle,
        "unavailable_models": UNAVAILABLE_MODELS,
        "notes": NOTES,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    _write_report(out, trials, manifest, pricing)
    print(f"Results: {out}")
    return 0


def _write_report(out: Path, trials, manifest, pricing) -> None:
    try:
        from evaluation.extraction import report, scoring
    except ImportError:
        print("scoring/report modules not available; wrote trials and manifest only")
        return
    cases = gt.load_cases()
    expected = {c["case_id"]: c["expected"] for c in cases}
    scored = [scoring.score_trial(t, expected[t["case_id"]], pricing) for t in trials]
    summary = scoring.summarize(scored)
    markdown = report.render_markdown(scored, summary, manifest, pricing, cases)
    (out / "report.md").write_text(markdown)
    report.write_csv(scored, out / "trials.csv")


def _dry_run_text(cases, rows, runs, trials, pricing) -> str:
    estimate = estimate_cost(runs, pricing)
    pdf = sum(1 for c in cases if c["format"] == "pdf")
    lines = [
        "Dry run (no AWS calls)",
        f"Cases: {len(cases)} ({pdf} PDF exercise a model, "
        f"{len(cases) - pdf} FHIR controls)",
        f"Rows: {', '.join(r['row'] for r in rows)}",
        f"Trials per PDF case: {trials}; FHIR cases run once per row",
        f"Handler invocations: {len(runs)}",
        f"Lifecycle-gated (checked on a real run): {', '.join(LIFECYCLE_GATED)}",
        f"Prices: {pricing['as_of']}, {pricing['source']}",
        "Estimated model cost (typical: 800 output tokens per call; "
        f"worst: {MAX_TOKENS} output tokens and one retry per call):",
    ]
    total_typ = total_worst = 0.0
    for row, entry in estimate.items():
        lines.append(
            f"  {row:48s} {entry['calls']:3d} calls  "
            f"typical ${entry['typical_usd']:.4f}"
            f"  worst ${entry['worst_usd']:.4f}"
        )
        total_typ += entry["typical_usd"]
        total_worst += entry["worst_usd"]
    lines.append(
        f"  {'total':48s}            typical ${total_typ:.4f}  worst ${total_worst:.4f}"
    )
    return "\n".join(lines)


def _fixture_chars(case) -> int:
    fixture = json.loads(
        (FIXTURES / f"{Path(case['source_file']).stem}.json").read_text()
    )
    for call in fixture["calls"]:
        response = call.get("response", {})
        if (
            call["operation"] == "get_document_text_detection"
            and response.get("JobStatus") == "SUCCEEDED"
        ):
            return sum(
                len(b.get("Text", ""))
                for b in response.get("Blocks", [])
                if b["BlockType"] == "LINE"
            )
    return 0


def _fixture_label(case) -> str:
    stem = Path(case["source_file"]).stem
    return f"replayed from evaluation/fixtures/textract/{stem}.json"


class _NoTextract:
    """Textract client for FHIR cases: the handler builds one, but must not use it"""

    def __getattr__(self, operation):
        raise AssertionError(f"Textract {operation} must not be called for FHIR cases")


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=gt.ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


_ENV_KEYS = (
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "AWS_PROFILE",
    "AWS_DEFAULT_REGION",
    "AWS_REGION",
    "DOCUMENTS_BUCKET",
    "EXTRACTION_TABLE",
)


if __name__ == "__main__":
    sys.path.insert(0, str(gt.ROOT / "src"))
    sys.exit(main())
