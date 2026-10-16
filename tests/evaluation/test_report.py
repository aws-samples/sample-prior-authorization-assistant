"""Markdown and CSV reports for the extraction model comparison."""

import csv

from evaluation.extraction import report, scoring

EXPECTED = {
    "payor": "Aetna",
    "cpt_codes": ["97110", "97140"],
    "icd_codes": ["M25.561", "M54.5"],
}
PRICING = {
    "as_of": "2026-10-16",
    "source": "https://aws.amazon.com/bedrock/pricing/",
    "currency": "USD",
    "unit": "per 1K tokens",
    "models": {"model-a": {"input": 0.003, "output": 0.015}},
}
CASES = [
    {
        "case_id": "aetna.pdf",
        "format": "pdf",
        "review": {
            "payor": "read from the PDF's 'Payor:' line; not separately reviewed"
        },
    },
    {"case_id": "aetna.fhir", "format": "fhir"},
]
MANIFEST = {
    "git_commit": "abc1234",
    "trials": 3,
    "started_at": "2026-10-16T10:00:00Z",
    "finished_at": "2026-10-16T10:05:00Z",
    "lifecycle_checks": [
        {
            "model_id": "us.anthropic.claude-haiku-4-5",
            "status": "Active",
            "source": "https://docs.aws.amazon.com/bedrock/model-lifecycle",
            "checked_at": "2026-10-16",
        }
    ],
    "unavailable_models": [{"model_id": "model-z", "reason": "AccessDeniedException"}],
    "notes": ["AnalyzeDocument fails with UnsupportedDocumentException"],
}


def trial(row, sampling, note, case_id, fmt, n, status="ok", stop="end_turn"):
    ok = status == "ok"
    calls = (
        []
        if fmt == "fhir"
        else [
            {
                "model_id": row if row != "as-shipped" else "us.amazon.nova-pro-v1:0",
                "input_tokens": 1200,
                "output_tokens": 250,
                "latency_ms": 1000 + n,
                "stop_reason": stop,
            }
        ]
    )
    return {
        "row": row,
        "model_id": calls[0]["model_id"] if calls else row,
        "sampling": sampling,
        "sampling_note": note,
        "case_id": case_id,
        "format": fmt,
        "trial": n,
        "status": status,
        "error": None if ok else "ValueError: no JSON | in output",
        "result": (
            {
                "payor_name": "Aetna",
                "cpt_codes": ["97110"],
                "icd_codes": ["M54.5"],
            }
            if ok
            else None
        ),
        "model_calls": calls,
        "duration_ms": 2000,
    }


def build():
    t01 = {"temperature": 0.1, "top_p": None}
    default = {"temperature": None, "top_p": None}
    shipped = {"temperature": 0.1, "top_p": 0.9}
    trials = [
        trial("model-a", t01, None, "aetna.pdf", "pdf", 1),
        trial("model-a", t01, None, "aetna.pdf", "pdf", 2, status="error"),
        trial("model-a", t01, None, "aetna.pdf", "pdf", 3, stop="max_tokens"),
        trial("model-a", t01, None, "aetna.fhir", "fhir", 1),
        trial(
            "model-b",
            default,
            "model rejects non-default sampling",
            "aetna.pdf",
            "pdf",
            1,
        ),
        trial("as-shipped", shipped, None, "aetna.pdf", "pdf", 1),
    ]
    scored = [scoring.score_trial(t, EXPECTED, PRICING) for t in trials]
    return scored, scoring.summarize(scored)


def test_markdown_has_every_section_and_limit():
    scored, summary = build()

    text = report.render_markdown(scored, summary, MANIFEST, PRICING, CASES)

    assert text.startswith("# Extraction model comparison")
    for heading in (
        "## Limits",
        "## PDF results (model-dependent)",
        "## FHIR controls (model-independent)",
        "## Unavailable models",
        "## Lifecycle checks",
        "## Price table",
        "## Notes",
        f"## Every trial ({len(scored)})",
    ):
        assert heading in text
    assert "1 PDF cases exercise a model; each runs 3 trials per row" in text
    assert "1 FHIR cases are model-independent controls" in text
    assert "synthetic sample documents" in text
    assert "not clinical ground truth" in text
    assert "not separately reviewed" in text and "aetna.pdf" in text
    assert "does not name a winner" in text
    assert "Only the `medical_extraction` step runs" in text
    assert "As of 2026-10-16, source: https://aws.amazon.com/bedrock/pricing/" in text
    assert "| us.anthropic.claude-haiku-4-5 | Active |" in text
    assert "| model-z | AccessDeniedException |" in text
    assert "- AnalyzeDocument fails with UnsupportedDocumentException" in text


def test_markdown_shows_sampling_per_row():
    scored, summary = build()

    text = report.render_markdown(scored, summary, MANIFEST, PRICING, CASES)

    assert "| model-a | model-a | temperature 0.1 |" in text
    assert "model default (model rejects non-default sampling)" in text
    assert "temperature 0.1, top_p 0.9" in text


def test_markdown_lists_every_trial_with_formatted_numbers():
    scored, summary = build()

    text = report.render_markdown(scored, summary, MANIFEST, PRICING, CASES)
    trial_section = text.split("## Every trial")[1]
    trial_rows = [line for line in trial_section.splitlines() if line.startswith("| ")]

    assert len(trial_rows) == len(scored) + 1  # plus the header
    for t in scored:
        assert f"| {t['row']} | {t['case_id']} | {t['trial']} |" in trial_section
    assert "| model-a | aetna.pdf | 1 | ok | yes | 1.000 | 0.500 | 1.000 | 0.500 |" in (
        trial_section
    )
    assert "ok (truncated)" in trial_section
    assert "max_tokens" in trial_section
    assert "ValueError: no JSON \\| in output" in trial_section
    assert scored[0]["cost_usd"] == 1.2 * 0.003 + 0.25 * 0.015
    assert f"| ${scored[0]['cost_usd']:.4f} |" in trial_section
    assert "| n/a |" in trial_section  # unpriced model cost


def test_markdown_tolerates_empty_manifest_sections():
    scored, summary = build()

    text = report.render_markdown(scored, summary, {}, None, CASES)

    assert "## Unavailable models\n\nNone." in text
    assert "## Lifecycle checks\n\nNone recorded." in text
    assert "No price table was provided" in text
    assert "## Notes" not in text


def test_csv_has_one_row_per_trial(tmp_path):
    scored, _ = build()
    path = tmp_path / "trials.csv"

    report.write_csv(scored, path)
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == len(scored)
    assert rows[0]["row"] == "model-a" and rows[0]["temperature"] == "0.1"
    assert rows[2]["stop_reasons"] == "max_tokens"
    assert rows[1]["error"] == "ValueError: no JSON | in output"
